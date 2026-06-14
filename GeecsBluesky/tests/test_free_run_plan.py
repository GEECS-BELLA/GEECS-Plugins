"""End-to-end tests for geecs_free_run_step_scan (synchronous RunEngine).

Two fake devices on separate servers: a combined motor+reference device
(pacemaker) and a contributor camera.  A single firer thread advances both
fakes' ``acq_timestamp`` together — or only the reference's, to simulate a
dead contributor.  Firing starts after a delay so the t0-sync stage sees the
static pre-scan caches.
"""

from __future__ import annotations

import asyncio
import math
import threading

from bluesky import RunEngine

from geecs_bluesky.devices.generic_detector import GeecsGenericDetector
from geecs_bluesky.devices.motor import GeecsMotor
from geecs_bluesky.devices.snapshot import GeecsSnapshotReadable
from geecs_bluesky.devices.timestamped_readable import GeecsTimestampedReadable
from geecs_bluesky.plans.free_run_step_scan import geecs_free_run_step_scan
from geecs_bluesky.testing.fake_device_server import FakeGeecsDevice, FakeGeecsServer

REF_T0 = 1000.0
CAM_T0 = 1000.05


def _run_servers_with_firer(
    fake_ref: FakeGeecsDevice,
    fake_cam: FakeGeecsDevice,
    ready: threading.Event,
    host_port: list,
    fire_cam: bool = True,
    initial_delay: float = 1.0,
    interval: float = 0.3,
) -> None:
    """Thread target: serve both fakes; fire shots together after a delay."""

    async def _main() -> None:
        async with (
            FakeGeecsServer(fake_ref) as s_ref,
            FakeGeecsServer(fake_cam) as s_cam,
        ):
            host_port.extend([s_ref.host, s_ref.port, s_cam.host, s_cam.port])
            ready.set()
            try:
                await asyncio.sleep(initial_delay)
                while True:
                    fake_ref.fire_shot()
                    if fire_cam:
                        fake_cam.fire_shot()
                    await asyncio.sleep(interval)
            except asyncio.CancelledError:
                pass

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_main())
    except Exception:
        pass
    finally:
        loop.close()


def _run_free_run_scan(
    fire_cam: bool, statistics: bool = False
) -> tuple[list, dict, list]:
    """Run a free-run scan; return (primary events, start doc, flush events).

    ``statistics=True`` runs the motorless path (motor=None, one no-move bin of
    4 shots) — the former NOSCAN — through the same free-run plan.
    """
    fake_ref = FakeGeecsDevice(
        name="U_Ref",
        variables={"Position (mm)": 0.0, "Sig": 1.0, "acq_timestamp": REF_T0},
    )
    fake_cam = FakeGeecsDevice(
        name="U_Cam",
        variables={"Val": 2.0, "acq_timestamp": CAM_T0},
    )
    ready = threading.Event()
    host_port: list = []
    srv_thread = threading.Thread(
        target=_run_servers_with_firer,
        args=(fake_ref, fake_cam, ready, host_port),
        kwargs={"fire_cam": fire_cam},
        daemon=True,
    )
    srv_thread.start()
    ready.wait(timeout=5.0)
    assert len(host_port) == 4, "Servers failed to start"
    ref_host, ref_port, cam_host, cam_port = host_port

    motor = GeecsMotor("U_Ref", "Position (mm)", ref_host, ref_port, name="scan_motor")
    ref = GeecsGenericDetector("U_Ref", ["Sig"], ref_host, ref_port, name="ref")
    cam = GeecsTimestampedReadable("U_Cam", ["Val"], cam_host, cam_port, name="cam")
    snapshot = GeecsSnapshotReadable(
        "U_Ref", ["Position (mm)"], ref_host, ref_port, name="async_snapshot"
    )
    ref.configure_shot_id(rep_rate_hz=1.0)
    cam.configure_shot_id(rep_rate_hz=1.0)
    if not fire_cam:
        cam.set_reference(ref, grace_wait_s=0.05)  # don't pay grace on a dead camera

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

    RE = RunEngine()
    RE.subscribe(collect)
    for dev in (motor, ref, cam, snapshot):
        asyncio.run_coroutine_threadsafe(dev.connect(), RE._loop).result(timeout=10)

    if statistics:
        plan_motor, positions, shots = None, [None], 4
    else:
        plan_motor, positions, shots = motor, [0.0, 1.0], 2

    RE(
        geecs_free_run_step_scan(
            motor=plan_motor,
            positions=positions,
            reference=ref,
            detectors=[cam, snapshot],
            shots_per_step=shots,
        )
    )
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
        assert "async_snapshot-position__mm" in data
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
