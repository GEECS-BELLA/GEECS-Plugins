"""End-to-end tests for geecs_free_run_step_scan (CA-mock devices).

A pacer coroutine on the RunEngine loop advances the reference's (and
optionally the contributor's) ``acq_timestamp`` — the fake free-running
trigger.  Pacing starts after a delay so the t0-sync stage sees the static
pre-scan caches.
"""

from __future__ import annotations

import math

import bluesky.plan_stubs as bps
import pytest
from bluesky import RunEngine

from geecs_bluesky.plans.free_run_step_scan import geecs_free_run_step_scan

pytest.importorskip("aioca")

from ophyd_async.core import set_mock_value  # noqa: E402

from geecs_bluesky.devices.ca import (  # noqa: E402
    CaGenericDetector,
    CaMotor,
    CaSnapshotReadable,
    CaTimestampedReadable,
)
from tests.ca_mock_helpers import (  # noqa: E402
    connect_mock,
    follow_setpoint,
    start_pacer,
)

REF_T0 = 1000.0
CAM_T0 = 1000.05


def _run_free_run_scan(
    fire_cam: bool,
    statistics: bool = False,
    quiesce_log: list | None = None,
) -> tuple[list, dict, list]:
    """Run a free-run scan; return (primary events, start doc, flush events).

    ``statistics=True`` runs the motorless path (motor=None, one no-move bin of
    4 shots) — the former NOSCAN — through the same free-run plan.
    ``quiesce_log`` (if given) receives a marker when the plan's
    ``quiesce_trigger`` runs, in order relative to the emitted events.
    """
    descriptors: dict[str, str] = {}  # descriptor uid → stream name
    primary_events: list[dict] = []
    flush_events: list[dict] = []
    start_docs: list[dict] = []

    def collect(name: str, doc: dict) -> None:
        if name == "start":
            start_docs.append(doc)
        elif name == "descriptor":
            descriptors[doc["uid"]] = doc.get("name", "")
        elif name == "event":
            stream = descriptors.get(doc["descriptor"], "")
            (flush_events if stream == "flush" else primary_events).append(doc)

    motor = CaMotor("U_Ref", "Position (mm)", name="scan_motor")
    ref = CaGenericDetector("U_Ref", ["Sig"], name="ref")
    cam = CaTimestampedReadable("U_Cam", ["Val"], name="cam")
    snapshot = CaSnapshotReadable("U_Ref", ["Position (mm)"], name="async_snapshot")
    ref.configure_shot_id(rep_rate_hz=1.0)
    cam.configure_shot_id(rep_rate_hz=1.0)
    if not fire_cam:
        cam.set_reference(ref, grace_wait_s=0.05)  # don't pay grace on a dead cam

    RE = RunEngine()
    RE.subscribe(collect)
    connect_mock(RE, motor, ref, cam, snapshot)
    follow_setpoint(motor)
    set_mock_value(ref.acq_timestamp, REF_T0)
    set_mock_value(cam.acq_timestamp, CAM_T0)
    set_mock_value(cam.val, 2.0)

    targets = [(ref, REF_T0)] + ([(cam, CAM_T0)] if fire_cam else [])
    pacer = start_pacer(RE, targets, initial_delay=1.0, interval=0.3)

    if statistics:
        plan_motor, positions, shots = None, [None], 4
    else:
        plan_motor, positions, shots = motor, [0.0, 1.0], 2

    quiesce = None
    if quiesce_log is not None:

        def quiesce():
            quiesce_log.append("quiesced")
            yield from bps.null()

    try:
        RE(
            geecs_free_run_step_scan(
                motor=plan_motor,
                positions=positions,
                reference=ref,
                detectors=[cam, snapshot],
                shots_per_step=shots,
                quiesce_trigger=quiesce,
            )
        )
    finally:
        pacer.cancel()
    return primary_events, start_docs[0], flush_events


def test_free_run_scan_with_live_contributor() -> None:
    """Contributor firing with the reference → complete, valid rows."""
    events, start, flush = _run_free_run_scan(fire_cam=True)

    assert len(events) == 4
    for ev in events:
        data = ev["data"]
        assert data["cam-valid"] is True
        assert data["cam-shot_offset"] == 0
        assert data["cam-shot_id"] == data["ref-shot_id"]
        assert data["cam-val"] == 2.0
        assert "async_snapshot-position_mm" in data
        assert "scan_motor-position" in data
    assert [ev["data"]["bin_number"] for ev in events] == [1, 1, 2, 2]
    assert [ev["data"]["shot_index_in_bin"] for ev in events] == [1, 2, 1, 2]
    assert [ev["data"]["scan_event_index"] for ev in events] == [1, 2, 3, 4]

    # Acquisition contract in the start document
    assert start["acquisition_mode"] == "free_run_time_sync"
    assert start["geecs_event_schema"] == 1
    assert start["reference_device"] == "U_Ref"
    assert start["device_t0s"]["U_Ref"] == REF_T0
    assert start["device_t0s"]["U_Cam"] == CAM_T0

    # Tail flush emitted to its own stream
    assert len(flush) == 1
    assert "cam-shot_id" in flush[0]["data"]


def test_free_run_quiesces_before_t0_sync() -> None:
    """The plan stops the trigger (quiesce_trigger) before establishing t0."""
    quiesce_log: list = []
    events, start, _flush = _run_free_run_scan(fire_cam=True, quiesce_log=quiesce_log)
    # quiesce ran before t0 sync (t0s captured afterward — sync succeeded)
    # and again at end of scan, closing the trigger before the tail
    # machinery (Gate-2: STANDBY frames leaked into native saving there).
    assert quiesce_log == ["quiesced", "quiesced"]
    assert start["device_t0s"]["U_Ref"] == REF_T0
    assert len(events) == 4


def test_free_run_statistics_collection() -> None:
    """Motorless free-run (former NOSCAN) — reference-paced rows, t0 sync, flush."""
    events, start, flush = _run_free_run_scan(fire_cam=True, statistics=True)

    assert len(events) == 4, "one bin × 4 shots"
    for ev in events:
        data = ev["data"]
        assert data["bin_number"] == 1
        assert data["cam-valid"] is True
        assert data["cam-shot_id"] == data["ref-shot_id"]
        assert "scan_motor-position" not in data  # no scan variable moved
    assert [ev["data"]["shot_index_in_bin"] for ev in events] == [1, 2, 3, 4]
    # Free-run contract still applies: t0 sync ran, flush emitted
    assert start["acquisition_mode"] == "free_run_time_sync"
    assert start["device_t0s"]["U_Ref"] == REF_T0
    assert len(flush) == 1


def test_free_run_scan_with_dead_contributor() -> None:
    """A contributor that never fires must not block or fake validity."""
    events, start, _flush = _run_free_run_scan(fire_cam=False)

    assert len(events) == 4, "rows must be emitted regardless of the dead camera"
    for ev in events:
        data = ev["data"]
        assert data["cam-valid"] is False
        assert data["cam-shot_offset"] < 0
        assert data["cam-shot_id"] == 1  # never advanced past its t0 seed
        assert data["cam-val"] == 2.0  # stale but real, truthfully labeled
        assert data["ref-valid"] is True
    # Reference rows advance even though the contributor is stuck
    ref_ids = [ev["data"]["ref-shot_id"] for ev in events]
    assert ref_ids == sorted(ref_ids)
    assert ref_ids[-1] > 1
    assert not math.isnan(start["device_t0s"]["U_Cam"])
