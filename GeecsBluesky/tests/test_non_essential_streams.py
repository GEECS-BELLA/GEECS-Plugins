"""A non-essential device without a file plugin streams by stamp (slice 2b).

The 2026-09-26 ruling: a *triggered* device that has no file plugin — one
saving its own LabVIEW files (the HASO) or one with scalars only (a power
supply, a gauge with a stamp) — listed ``non_essential`` chugs along at its
own rate in its own ``<name>_stream``: one event per stamp it publishes,
with its scalars and (a native saver) its save path; its saving is switched
on by its own unbounded prepare and off by its unstage; it never throttles
the rep rate and never fails the run, in strict and gated alike.

Driven through the bound ``count`` on mocks: the fake box's every edge (a
strict fire, or a free-running SCAN edge) stamps every device on its
``cameras`` list, except the ``(name, edge)`` pairs in ``drop`` — which is
how a device three times slower than the rep rate is made.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import pytest

pytest.importorskip("aioca")

from bluesky import RunEngine  # noqa: E402
from ophyd_async.core import set_mock_value  # noqa: E402

from geecs_bluesky.devices.ca.snapshot import CaSnapshotReadable  # noqa: E402
from geecs_bluesky.devices.shot_control import ShotControl  # noqa: E402
from geecs_bluesky.exceptions import GeecsConfigurationError  # noqa: E402
from geecs_bluesky.plans import gated  # noqa: E402
from geecs_bluesky.plans.registry import TriggerProfiles, bind_plans  # noqa: E402
from tests.ca_mock_helpers import DocCollector, connect_mock  # noqa: E402
from tests.test_gated_plans import (  # noqa: E402
    GATED_WRITES,
    GatedBox,
    _events_from_pages,
    _saves,
)
from tests.test_strict_plans import _camera  # noqa: E402

MODES = ("strict", "gated")


class LateBox(GatedBox):
    """A box whose ``late`` devices publish their stamp *seconds* after the edge.

    The HASO's shape (26_0925): its 25 MB file is written first and its
    stamp PV reaches the worker ~0.9 s after its frame, where a strict row
    is read ~0.5 s after the fire.
    """

    def __init__(self, late: dict[str, float], **kw) -> None:
        super().__init__(**kw)
        self.late = late

    def edge(self) -> None:
        held = [c for c in self.cameras if c.name in self.late]
        self.cameras = [c for c in self.cameras if c.name not in self.late]
        try:
            super().edge()
        finally:
            self.cameras.extend(held)
        loop = asyncio.get_running_loop()
        for cam in held:
            loop.call_later(
                self.late[cam.name], set_mock_value, cam.acq_timestamp, self.stamp
            )


@pytest.fixture(autouse=True)
def _short_drain(monkeypatch):
    monkeypatch.setattr(gated, "TRIGGER_PERIOD_S", 0.08)
    monkeypatch.setattr(gated, "DRAIN_MARGIN_S", 0.04)


@pytest.fixture
def RE() -> RunEngine:
    return RunEngine()


def _count(RE: RunEngine, box: GatedBox):
    sc = ShotControl(
        GATED_WRITES, experiment="TestExp", name="shot_control", setter_factory=box
    )
    connect_mock(RE, sc)
    return bind_plans(TriggerProfiles({"HTU-Test": sc}, default="HTU-Test"))["count"]


def _rows(col: DocCollector, mode: str) -> list[dict]:
    return _events_from_pages(col, "primary" if mode == "strict" else "shots")


def _descriptor(col: DocCollector, name: str) -> dict:
    return next(d for d in col.docs["descriptor"] if d["name"] == name)


@pytest.mark.parametrize("mode", MODES)
def test_a_native_saver_and_a_scalar_device_stream_one_event_per_stamp(
    RE: RunEngine, tmp_path: Path, mode: str
) -> None:
    """The HASO's shape and a power supply's, non-essential beside an essential camera.

    Saving switches on exactly once (the unbounded prepare after
    ``open_run``) and off exactly once (unstage); each device's stream has
    one event per stamp it published — its stamp, its scalars and, the
    native saver, its save path — and neither device is in the rows.
    """
    (tmp_path / "Scan001").mkdir()
    box = GatedBox()
    count = _count(RE, box)
    a = _camera(RE, box, "UC_A")
    haso = _camera(RE, box, "U_HASO", tmp_path=tmp_path)
    ps = _camera(RE, box, "U_PS")
    set_mock_value(haso.meancounts, 7.0)
    set_mock_value(ps.meancounts, 2.5)
    saves = _saves(haso)
    col = DocCollector()
    RE.subscribe(col)
    RE(count([a], 4, acquisition=mode, non_essential=[haso, ps]))
    assert col.docs["stop"][-1]["exit_status"] == "success"
    assert col.docs["start"][0]["non_essential"] == ["u_haso", "u_ps"]
    assert saves == ["off", "on", "off"]
    rows = _rows(col, mode)
    assert len(rows) == 4
    row_stamps = [r["data"]["uc_a-acq_timestamp"] for r in rows]
    directory = str(tmp_path / "Scan001" / "U_HASO")
    for device, value in (("u_haso", 7.0), ("u_ps", 2.5)):
        events = _events_from_pages(col, f"{device}_stream")
        # one event per edge, the tick of no row among them
        assert len(events) == box.edges, device
        stamps = [e["data"][f"{device}-acq_timestamp"] for e in events]
        assert stamps == sorted(set(stamps)), device
        assert set(row_stamps) <= set(stamps), device
        assert [e["data"][f"{device}-meancounts"] for e in events] == [value] * len(
            events
        )
        assert not any(k.startswith(device) for k in rows[0]["data"]), device
    haso_events = _events_from_pages(col, "u_haso_stream")
    assert {e["data"]["u_haso-nonscalar_save_path"] for e in haso_events} == {directory}
    assert not any(
        "u_ps-nonscalar_save_path" in e["data"]
        for e in _events_from_pages(col, "u_ps_stream")
    )
    # the stream's descriptor carries the device's drain offset: the join's
    # correction for its stamps
    config = _descriptor(col, "u_haso_stream")["configuration"]
    assert config["u_haso"]["data"] == {"u_haso-drain_offset": 0.0}


@pytest.mark.parametrize("mode", MODES)
def test_a_device_three_times_slower_than_the_rep_rate_records_a_third(
    RE: RunEngine, mode: str
) -> None:
    """Every third edge stamps it: a third of the events, the rows untouched.

    No extra shot is taken for it (an essential's missed frame would add
    one) and no row waits for it — the essential camera's six rows are
    six edges in strict.
    """
    box = GatedBox()
    count = _count(RE, box)
    a = _camera(RE, box, "UC_A")
    slow = _camera(RE, box, "U_Slow")
    box.drop = {("u_slow", k) for k in range(1, 200) if k % 3}
    col = DocCollector()
    RE.subscribe(col)
    RE(count([a], 6, acquisition=mode, non_essential=[slow]))
    assert col.docs["stop"][-1]["exit_status"] == "success"
    assert len(_rows(col, mode)) == 6
    if mode == "strict":
        assert box.fires == 6
    else:
        assert box.scan_runs == 1
    events = _events_from_pages(col, "u_slow_stream")
    assert len(events) == box.edges // 3
    assert [e["data"]["u_slow-acq_timestamp"] for e in events] == [
        1000.0 + 3 * (i + 1) for i in range(len(events))
    ]


def test_a_late_stamping_non_essential_never_stretches_the_strict_cadence(
    RE: RunEngine, tmp_path: Path
) -> None:
    """The HASO's stamp lands 0.6 s after each fire; the strict rows do not wait.

    Were the device sampled at the row (or waited on), every shot would
    take the 0.6 s — the write time back in the rep rate, the very thing
    #999 removed.  The rows keep the run's own cadence (a 0.25 s throttle,
    so the run outlives the device's latency), and every
    stamp that landed before the run's close is an event of its own shot.
    """
    (tmp_path / "Scan001").mkdir()
    box = LateBox({"u_haso": 0.6})
    count = _count(RE, box)
    a = _camera(RE, box, "UC_A")
    haso = _camera(RE, box, "U_HASO", tmp_path=tmp_path)
    col = DocCollector()
    RE.subscribe(col)
    t0 = time.monotonic()
    RE(count([a], 5, non_essential=[haso], shot_period=0.25))
    elapsed = time.monotonic() - t0
    assert col.docs["stop"][-1]["exit_status"] == "success"
    rows = _rows(col, "strict")
    times = [e["time"] for e in col.primary_events()]
    gaps = [b - c for b, c in zip(times[1:], times[:-1])]
    # the throttle's 0.25 s per shot, never the device's 0.6 s
    assert max(gaps) < 0.45, gaps
    assert elapsed < 2.0, elapsed
    events = _events_from_pages(col, "u_haso_stream")
    row_stamps = [r["data"]["uc_a-acq_timestamp"] for r in rows]
    event_stamps = [e["data"]["u_haso-acq_timestamp"] for e in events]
    assert event_stamps == row_stamps[: len(event_stamps)]
    assert 1 <= len(events) < 5  # the last shots' stamps land after the close


@pytest.mark.parametrize("mode", MODES)
def test_a_non_essential_that_publishes_nothing_warns_and_never_fails(
    RE: RunEngine, mode: str, caplog
) -> None:
    """A power supply that never stamps (U_BCaveMagSpecPS today): zero events, success."""
    box = GatedBox()
    count = _count(RE, box)
    a = _camera(RE, box, "UC_A")
    dead = _camera(RE, box, "U_Dead")
    box.cameras.remove(dead)
    col = DocCollector()
    RE.subscribe(col)
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.plans.gated"):
        RE(count([a], 3, acquisition=mode, non_essential=[dead]))
    assert col.docs["stop"][-1]["exit_status"] == "success"
    assert len(_rows(col, mode)) == 3
    assert _events_from_pages(col, "u_dead_stream") == []
    assert "u_dead-acq_timestamp" in _descriptor(col, "u_dead_stream")["data_keys"]
    said = [r.getMessage() for r in caplog.records if "U_Dead" in r.getMessage()]
    assert len(said) == 1 and "published no shot stamp" in said[0]
    assert caplog.records[-1].levelno == logging.WARNING


def test_a_scalars_view_non_essential_streams_its_scalars_without_files(
    RE: RunEngine, tmp_path: Path
) -> None:
    """``save_images: false`` on a non-essential: its stamp and scalars, no saving."""
    (tmp_path / "Scan001").mkdir()
    box = GatedBox()
    count = _count(RE, box)
    a = _camera(RE, box, "UC_A")
    native = _camera(RE, box, "UC_Native", tmp_path=tmp_path)
    saves = _saves(native)
    col = DocCollector()
    RE.subscribe(col)
    RE(count([a], 3, non_essential=[native.scalars]))
    assert col.docs["stop"][-1]["exit_status"] == "success"
    # the start document names the stream the s-file must find (review
    # finding 1): the owner's, like the stream itself
    assert col.docs["start"][0]["non_essential"] == ["uc_native"]
    assert "on" not in saves and saves == ["off", "off"]  # stage, unstage
    assert not (tmp_path / "Scan001" / "UC_Native").exists()
    events = _events_from_pages(col, "uc_native_stream")
    assert len(events) == 3
    assert "uc_native-meancounts" in events[0]["data"]
    assert "uc_native-nonscalar_save_path" not in events[0]["data"]


def test_a_free_running_non_essential_is_refused_before_anything_moves(
    RE: RunEngine,
) -> None:
    """No stamp, nothing to join on: refused at bind (deferred), nothing claimed."""
    box = GatedBox()
    count = _count(RE, box)
    a = _camera(RE, box, "UC_A")
    gauge = CaSnapshotReadable(
        "U_Gauge", ["Pressure"], experiment="TestExp", name="u_gauge"
    )
    connect_mock(RE, gauge)
    col = DocCollector()
    RE.subscribe(col)
    for mode in MODES:
        with pytest.raises(GeecsConfigurationError, match="no shot stamp: u_gauge"):
            RE(count([a], 2, acquisition=mode, non_essential=[gauge]))
    assert box.states == [] and box.fires == 0
    assert not col.docs["start"]
