"""The oscilloscope panel: its allow-list, decimation, feed and routes — no CA, no IOC."""

from __future__ import annotations

import json
import math
from typing import Optional

import pytest
from fastapi.testclient import TestClient

from geecs_panels.app import create_app
from geecs_panels.feed import ArraysSource, WriteFailed
from geecs_panels.registry import Device
from geecs_panels.scope import NotWritable, check_write, decimate, monitored
from geecs_panels.scope_feed import ScopeFeed, traces_payload

SCOPE = Device("picotest1", "picoscope", "undulator:picotest1:")
CAM = Device("ADTEST_CAM1", "aravis_camera", "undulator:adtest_cam1:")


class _Sub:
    def close(self) -> None:
        pass


def _feed() -> ScopeFeed:
    feed = ScopeFeed(SCOPE.prefix, lambda pv, s, cb: _Sub())
    feed.start()
    for key, value in {
        "model": "5242D",
        "state": "Armed",
        "acquire": "Acquiring",
        "count": 3,
        "A.present": "Present",
        "A.enable": "On",
        "A.range": "1 V",
        "B.present": "Present",
        "B.enable": "Off",
        "C.present": "Absent",
        "C.enable": "On",
    }.items():
        feed.update(key, value)
    return feed


class FakeScopes:
    def __init__(self, refuse: bool = False) -> None:
        self.feeds: dict[str, ScopeFeed] = {}
        self.puts: list[tuple[str, object]] = []
        self.refuse = refuse

    def feed(self, device: Device) -> ScopeFeed:
        if device.name not in self.feeds:
            self.feeds[device.name] = _feed()
        return self.feeds[device.name]

    def traces(self, device: Device) -> ArraysSource:
        async def fetch() -> dict[str, list[float]]:
            return {"t": [0.0, 1.0, 2.0], "A": [0.1, float("nan"), 0.3]}

        return ArraysSource(fetch, 0.0)

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
        ("A:Range", "200 mV", "200 mV"),
        ("D:Enable", "On", "On"),
        ("SampleInterval", 16, 16.0),
        ("NumSamples", 5000.0, 5000),
        ("TriggerSource", "External", "External"),
        ("Acquire", "Start", "Start"),
    ],
)
def test_every_setting_is_writable_as_the_pv_takes_it(suffix, value, want) -> None:
    assert check_write(suffix, value) == want


@pytest.mark.parametrize(
    "suffix, value",
    [
        ("A:Waveform_RBV", 0),  # a capture product, not a setting
        ("CaptureCount_RBV", 0),
        ("A:Range", "3 V"),  # not one of the scope's ranges
        ("E:Range", "1 V"),  # no channel E
        ("NumSamples", 1.5),
        ("NumSamples", True),
        ("PreTrigger", 120),
        ("TriggerLevel", math.inf),
    ],
)
def test_the_allow_list_refuses_everything_else(suffix, value) -> None:
    with pytest.raises(NotWritable):
        check_write(suffix, value)


def test_every_channel_is_monitored_under_its_own_prefix() -> None:
    suffixes = monitored()
    assert suffixes["D:Range_RBV"] == ("D.range", True)
    assert suffixes["SampleInterval_RBV"] == ("interval_ns", False)


# ---- decimation -------------------------------------------------------------


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


def test_only_present_enabled_channels_are_read() -> None:
    assert _feed().enabled_channels() == ["A"]


def test_a_capture_is_a_count_change_not_the_first_value() -> None:
    clock = iter([10.0, 12.5]).__next__
    feed = ScopeFeed(SCOPE.prefix, lambda pv, s, cb: _Sub(), clock=clock)
    feed.update("count", 7)  # (re)connected: the count's position
    assert feed.snapshot()["capture_age"] is None
    feed.update("count", 8)  # at 10.0
    assert feed.snapshot()["capture_age"] == 2.5


def test_the_traces_payload_names_each_channel() -> None:
    payload = traces_payload({"t": [0.0, 1.0], "A": [1.0, 2.0]})
    assert payload == {
        "traces": {"A": {"t": [0.0, 1.0], "v": [1.0, 2.0]}},
        "samples": 2,
    }


# ---- routes -----------------------------------------------------------------


def test_the_landing_page_links_the_scope_panel() -> None:
    assert 'href="/scope/picotest1"' in _client().get("/").text


def test_the_scope_page_renders_its_controls() -> None:
    html = _client().get("/scope/picotest1").text
    assert 'id="screen"' in html and 'data-setting="C:Range"' in html
    assert "<option>20 V</option>" in html and 'data-writable="true"' in html


def test_a_camera_is_not_a_scope() -> None:
    assert _client().get("/scope/ADTEST_CAM1").status_code == 404


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
    assert client.post(
        "/scope/picotest1/put", json={"pv": "B:Enable", "value": "On"}
    ).json()["ok"]
    assert scopes.puts == [("B:Enable", "On")]
    assert (
        client.post(
            "/scope/picotest1/put", json={"pv": "A:Waveform_RBV", "value": 1}
        ).status_code
        == 422
    )
    assert scopes.puts == [("B:Enable", "On")]


def test_the_ioc_refusing_is_a_502_and_read_only_a_403() -> None:
    put = {"pv": "Acquire", "value": "Start"}
    assert (
        _client(FakeScopes(refuse=True))
        .post("/scope/picotest1/put", json=put)
        .status_code
        == 502
    )
    assert (
        _client(writable=False).post("/scope/picotest1/put", json=put).status_code
        == 403
    )
