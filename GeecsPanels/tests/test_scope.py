"""The scope panel: the contract's allow-list, the capture's split and decimation, the feed, the routes — no CA, no IOC."""

from __future__ import annotations

import asyncio
import json
import math
from typing import Optional

import pytest
from fastapi.testclient import TestClient

from geecs_panels.app import create_app
from geecs_panels.feed import ArraysSource, WriteFailed
from geecs_panels.registry import Device
from geecs_panels.scope import (
    ENUM_SETTINGS,
    SCOPE_KIND,
    SETTLED_BY_READBACK,
    NotWritable,
    check_write,
    decimate,
    monitored,
    split_channels,
)
from geecs_panels.scope_feed import ScopeFeed, ScopeTraces, traces_payload

SCOPE = Device("picotest1", SCOPE_KIND, "undulator:picotest1:")
CAM = Device("ADTEST_CAM1", "aravis_camera", "undulator:adtest_cam1:")

#: What a 2-channel PicoScope's enum PVs report.
ENUMS = {
    "Resolution": ["8 bit", "12 bit", "14 bit", "15 bit", "16 bit"],
    "TriggerSource": ["A", "B", "C", "D", "External"],
    "A:Range": ["10 mV", "20 mV", "1 V", "20 V"],
    "A:Enable": ["Off", "On"],
    "B:Enable": ["Off", "On"],
}


class _Sub:
    def close(self) -> None:
        pass


def _feed() -> ScopeFeed:
    feed = ScopeFeed(SCOPE.prefix, lambda pv, s, cb: _Sub())
    feed.start()
    feed.enums.update(ENUMS)
    for key, value in {
        "model": "5242D", "state": "Acquire", "acquire": "Acquiring", "count": 3,
        "A.present": "Present", "A.enable": "On", "A.range": "1 V", "A.stat.total": 2.5,
        "B.present": "Present", "B.enable": "Off",
        "C.present": "Absent", "C.enable": "On",
        "D.present": "Absent",
    }.items():  # fmt: skip
        feed.update(key, value)
    return feed


def _capture(uid: float = 7.0) -> dict[str, list[float]]:
    """A 2-channel, 2-sample capture with channel B off, as LiveScopes reads it."""
    return {
        "t": [0.0, 1.0],
        "data": [0.1, math.nan, 0.3, math.nan],
        "uid": [uid],
        "channels": [2.0],
        "samples": [2.0],
    }


class FakeScopes:
    def __init__(self, refuse: bool = False) -> None:
        self.feeds: dict[str, ScopeFeed] = {}
        self.puts: list[tuple[str, object]] = []
        self.refuse = refuse
        self.refreshes = 0
        self._traces: Optional[ScopeTraces] = None

    def feed(self, device: Device) -> ScopeFeed:
        if device.name not in self.feeds:
            self.feeds[device.name] = _feed()
        return self.feeds[device.name]

    def traces(self, device: Device) -> ScopeTraces:
        if self._traces is None:

            async def fetch() -> dict[str, list[float]]:
                return _capture()

            self._traces = ScopeTraces(ArraysSource(fetch, 0.0))
        return self._traces

    def refresh(self, device: Device) -> None:
        self.refreshes += 1

    async def put(self, device: Device, suffix: str, value: object) -> None:
        if self.refuse:
            raise WriteFailed(f"{device.prefix}{suffix}: no write access")
        self.puts.append((suffix, value))


def _client(scopes: Optional[FakeScopes] = None, **kwargs) -> TestClient:
    return TestClient(
        create_app(
            "Undulator", [SCOPE, CAM], object(), scopes=scopes or FakeScopes(), **kwargs
        )
    )


# ---- the allow-list ---------------------------------------------------------


@pytest.mark.parametrize(
    "suffix, value, want",
    [
        ("A:Range", "1 V", "1 V"),
        ("B:Enable", "On", "On"),
        ("SampleInterval", 16, 16.0),
        ("NumSamples", 5000.0, 5000),
        ("TriggerSource", "External", "External"),
        ("Acquire", 1, 1),
    ],
)
def test_every_setting_is_writable_as_the_pv_takes_it(suffix, value, want) -> None:
    assert check_write(suffix, value, ENUMS) == want


@pytest.mark.parametrize(
    "suffix, value",
    [
        ("ARR:ArrayData", 0),  # a capture product, not a setting
        ("STAT_A:Total_RBV", 0),  # the chain's, not a setting
        ("A:Range", "3 V"),  # not one of this scope's ranges
        ("E:Range", "1 V"),  # no channel E in the contract
        ("Coupling", "AC"),  # choices not read yet: refused, not guessed
        ("NumSamples", 1.5),
        ("NumSamples", True),
        ("PreTrigger", 120),
        ("TriggerLevel", math.inf),
        ("Acquire", 2),
    ],
)
def test_the_allow_list_refuses_everything_else(suffix, value) -> None:
    with pytest.raises(NotWritable):
        check_write(suffix, value, ENUMS)


def test_the_choices_are_the_pvs_not_the_panels() -> None:
    # A vendor with other ranges: the panel takes whatever its PV offers.
    assert check_write("A:Range", "±5 V", {"A:Range": ["±5 V"]}) == "±5 V"
    assert "D:Coupling" in ENUM_SETTINGS


def test_every_channel_and_its_statistics_are_monitored() -> None:
    suffixes = monitored()
    assert suffixes["D:Range_RBV"] == ("D.range", True)
    assert suffixes["STAT_B:Total_RBV"] == ("B.stat.total", False)
    assert suffixes["DetectorState_RBV"] == ("state", True)


# ---- the capture ------------------------------------------------------------


def test_the_capture_splits_into_channels_fastest_first() -> None:
    assert split_channels([1, 10, 2, 20, 3, 30], 2, 3) == [[1, 2, 3], [10, 20, 30]]


def test_a_disabled_channel_is_left_out_of_the_traces() -> None:
    payload = traces_payload(_capture(), ["A", "B"])
    assert payload == {
        "traces": {"A": {"t": [0.0, 1.0], "v": [0.1, 0.3]}},
        "samples": 2,
    }


def test_a_capture_older_than_a_timebase_change_is_not_drawn_on_the_new_axis() -> None:
    # Stopped after capture 5, then the interval changes: the IOC's time axis
    # is the new one, the held capture is still the old one.
    feed = _feed()
    feed.update("arr_uid", 5)
    feed.update("interval_ns", 8.0)
    feed.update("interval_ns", 20.0)
    assert feed.timebase_uid == 5
    old = traces_payload(_capture(uid=5), ["A", "B"], feed.timebase_uid)
    assert old["traces"] == {} and "waiting" in old
    new = traces_payload(_capture(uid=6), ["A", "B"], feed.timebase_uid)
    assert list(new["traces"]) == ["A"]


def test_an_ioc_restart_drops_the_timebase_marker() -> None:
    feed = _feed()
    feed.update("arr_uid", 50)
    feed.update("samples", 1000)
    feed.update("samples", 2000)
    feed.update("arr_uid", None)  # disconnect: captures are renumbered from 0
    assert feed.timebase_uid is None


def test_a_capture_is_read_once_and_decimated_once() -> None:
    reads = []

    async def fetch() -> dict[str, list[float]]:
        reads.append(1)
        return _capture(uid=7)

    async def run() -> tuple[object, object]:
        traces = ScopeTraces(ArraysSource(fetch, 0.0))
        traces.kick(7)
        await asyncio.sleep(0.01)  # the background read lands
        first = traces.payload(["A", "B"], None)
        for _ in range(5):  # the same capture, monitored again and again
            traces.kick(7)
            await asyncio.sleep(0.01)
        return first, traces.payload(["A", "B"], None)

    first, again = asyncio.run(run())
    assert len(reads) == 1 and first is again


def test_a_short_trace_is_returned_whole() -> None:
    assert decimate([0, 1, 2], [5.0, 6.0, 7.0], 10) == ([0, 1, 2], [5.0, 6.0, 7.0])


def test_decimation_keeps_a_one_sample_spike() -> None:
    n = 10_000
    values = [0.0] * n
    values[4321] = 9.0
    xs, ys = decimate(list(range(n)), values, 100)
    assert len(ys) <= 200
    assert 9.0 in ys and 4321 in xs
    assert xs == sorted(xs)


def test_non_finite_samples_become_none() -> None:
    xs, ys = decimate([0, 1], [float("nan"), 1.0], 10)
    assert ys == [None, 1.0]


# ---- the feed ---------------------------------------------------------------


def test_present_and_enabled_channels() -> None:
    feed = _feed()
    assert feed.present_channels() == ["A", "B"]
    assert feed.enabled_channels() == ["A"]


def test_an_absent_channels_statistics_are_not_missing() -> None:
    missing = _feed().snapshot()["missing"]
    assert "C.stat.total" not in missing and "B.stat.total" in missing


def test_the_snapshot_carries_choices_and_statistics() -> None:
    snap = _feed().snapshot()
    assert snap["enums"]["TriggerSource"][-1] == "External"
    assert snap["channels"][0]["stats"]["total"] == 2.5


def test_a_capture_is_a_count_change_not_the_first_value() -> None:
    clock = iter([10.0, 12.5]).__next__
    feed = ScopeFeed(SCOPE.prefix, lambda pv, s, cb: _Sub(), clock=clock)
    feed.update("count", 7)  # (re)connected: the count's position
    assert feed.snapshot()["capture_age"] is None
    feed.update("count", 8)  # at 10.0
    assert feed.snapshot()["capture_age"] == 2.5


# ---- routes -----------------------------------------------------------------


def test_the_landing_page_links_the_scope_panel() -> None:
    assert 'href="/scope/picotest1"' in _client().get("/").text


def test_the_scope_page_renders_controls_without_a_vendors_choices() -> None:
    html = _client().get("/scope/picotest1").text
    assert 'id="screen"' in html and 'data-setting="C:Range"' in html
    assert 'data-writable="true"' in html and 'id="ch-A-stats"' in html
    assert "20 V" not in html  # the choices come from the PVs


def test_a_camera_is_not_a_scope() -> None:
    assert _client().get("/scope/ADTEST_CAM1").status_code == 404


def test_every_push_retries_what_the_feed_lacks() -> None:
    # An IOC down when the page opened: its choices are read again on every
    # push of the open stream, not only on a new request.
    scopes = FakeScopes()
    with _client(scopes).stream("GET", "/scope/picotest1/events") as r:
        pushes = 0
        for line in r.iter_lines():
            if line.startswith("data: "):
                pushes += 1
                if pushes == 3:
                    break
    assert scopes.refreshes >= 3


def test_a_viewer_is_sent_a_capture_once() -> None:
    with _client().stream("GET", "/scope/picotest1/events?traces=1") as r:
        snaps = []
        for line in r.iter_lines():
            if line.startswith("data: "):
                snaps.append(json.loads(line[6:]))
                if len(snaps) == 4:
                    break
    sent = [s for s in snaps if "traces" in s and s["traces"].get("traces")]
    assert len(sent) == 1  # the same capture is not resent every push


def test_run_settles_on_the_acquire_readback() -> None:
    # An areaDetector Acquire put completes only when acquiring ends.
    assert SETTLED_BY_READBACK == {"Acquire": "Acquire_RBV"}


def test_the_stream_carries_settings_and_finite_traces() -> None:
    with _client().stream("GET", "/scope/picotest1/events?once=1&traces=1") as r:
        body = "".join(r.iter_text())
    data = json.loads(body.split("data: ", 1)[1])
    assert data["scope"]["model"] == "5242D"
    assert [c["name"] for c in data["channels"]] == ["A", "B", "C", "D"]
    # The first push only starts the read; whatever it holds is valid JSON.
    assert "traces" in data


def test_a_put_goes_through_the_allow_list() -> None:
    scopes = FakeScopes()
    client = _client(scopes)
    ok = client.post("/scope/picotest1/put", json={"pv": "B:Enable", "value": "On"})
    assert ok.json()["ok"]
    assert scopes.puts == [("B:Enable", "On")]
    refused = client.post(
        "/scope/picotest1/put", json={"pv": "ARR:ArrayData", "value": 1}
    )
    assert refused.status_code == 422
    assert scopes.puts == [("B:Enable", "On")]


def test_the_ioc_refusing_is_a_502_and_read_only_a_403() -> None:
    put = {"pv": "Acquire", "value": 1}
    refusing = _client(FakeScopes(refuse=True))
    assert refusing.post("/scope/picotest1/put", json=put).status_code == 502
    read_only = _client(writable=False)
    assert read_only.post("/scope/picotest1/put", json=put).status_code == 403
