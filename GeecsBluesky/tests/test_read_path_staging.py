"""Read-path hardening pins: staging, grace-wait cap, t0 window cap, telemetry bound.

The read-path contract (GeecsBluesky/CLAUDE.md "Read path: staging & shot
coherence"): every per-row read device is staged by the orchestration layer,
so per-shot reads are served from caching monitors instead of issuing one
network get per signal per row.  These tests pin the contract hermetically by
counting mock-backend ``get_reading`` calls — the mock analogue of a CA get.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from bluesky import RunEngine

pytest.importorskip("aioca")

from geecs_bluesky.devices.ca import (  # noqa: E402
    CaGenericDetector,
    CaSnapshotReadable,
    CaTelemetryReadable,
    CaTimestampedReadable,
)
from geecs_bluesky.plans.orchestration import build_step_scan_plan  # noqa: E402
from tests.ca_mock_helpers import connect_mock, start_pacer  # noqa: E402

REF_T0 = 1000.0
CAM_T0 = 1000.05


def _count_backend_gets(device: Any, counters: dict[str, int]) -> None:
    """Wrap every child signal backend's ``get_reading`` with a counter.

    A ``get_reading`` call is the mock analogue of an uncached CA get — the
    thing staging is supposed to eliminate from the per-row path.
    """
    for _, child in device.children():
        backend = getattr(child, "_connector", None)
        backend = getattr(backend, "backend", None)
        if backend is None or not hasattr(backend, "get_reading"):
            continue
        original = backend.get_reading
        key = child.name

        async def counted(_original=original, _key=key):
            counters[_key] = counters.get(_key, 0) + 1
            return await _original()

        backend.get_reading = counted


def _make_devices() -> tuple[Any, Any, Any]:
    ref = CaGenericDetector("U_Ref", ["Sig"], name="ref")
    cam = CaTimestampedReadable("U_Cam", ["Val"], name="cam")
    snapshot = CaSnapshotReadable("U_Ref", ["Sig"], name="async_snapshot")
    ref.configure_shot_id(rep_rate_hz=1.0)
    cam.configure_shot_id(rep_rate_hz=1.0)
    cam.set_reference(ref, grace_wait_s=0.05)
    return ref, cam, snapshot


def _noscan_plan(ref: Any, detectors: list[Any], shots: int = 3):
    return build_step_scan_plan(
        strict=False,
        motor=None,
        positions=[None],
        reference=ref,
        detectors=detectors,
        shots_per_step=shots,
        controller=None,
        experiment="",
        scan_number=None,
        scan_folder=None,
        saving_detectors=[],
    )


class TestStagingContract:
    """Per-row reads must be cache hits; stage/unstage must bracket the run."""

    def test_staged_scan_issues_zero_backend_gets(self) -> None:
        ref, cam, snapshot = _make_devices()
        RE = RunEngine()
        connect_mock(RE, ref, cam, snapshot)
        from ophyd_async.core import set_mock_value

        set_mock_value(ref.acq_timestamp, REF_T0)
        set_mock_value(cam.acq_timestamp, CAM_T0)

        counters: dict[str, int] = {}
        for dev in (ref, cam, snapshot):
            _count_backend_gets(dev, counters)

        pacer = start_pacer(
            RE, [(ref, REF_T0), (cam, CAM_T0)], initial_delay=1.0, interval=0.3
        )
        try:
            RE(_noscan_plan(ref, [ref, cam, snapshot]))
        finally:
            pacer.cancel()

        assert counters == {}, (
            "per-row reads issued uncached backend gets — the staging "
            f"contract is broken: {counters}"
        )

    def test_stage_and_unstage_bracket_the_run(self) -> None:
        ref, cam, snapshot = _make_devices()
        RE = RunEngine()
        commands: list[str] = []
        RE.msg_hook = lambda msg: commands.append(msg.command)
        connect_mock(RE, ref, cam, snapshot)
        from ophyd_async.core import set_mock_value

        set_mock_value(ref.acq_timestamp, REF_T0)
        set_mock_value(cam.acq_timestamp, CAM_T0)

        pacer = start_pacer(
            RE, [(ref, REF_T0), (cam, CAM_T0)], initial_delay=1.0, interval=0.3
        )
        try:
            RE(_noscan_plan(ref, [ref, cam, snapshot], shots=1))
        finally:
            pacer.cancel()

        # Staged exactly once per device (ophyd-async staging is a bool, not
        # a refcount — double-staging would break unstage).
        assert commands.count("stage") == 3
        assert commands.count("unstage") == 3
        assert commands.index("stage") < commands.index("open_run")
        assert commands[::-1].index("unstage") < commands[::-1].index("close_run")

    def test_devices_unstaged_after_mid_scan_failure(self) -> None:
        ref, cam, snapshot = _make_devices()
        RE = RunEngine()
        connect_mock(RE, ref, cam, snapshot)
        from ophyd_async.core import set_mock_value

        set_mock_value(ref.acq_timestamp, REF_T0)
        set_mock_value(cam.acq_timestamp, CAM_T0)

        def failing_per_step():
            raise RuntimeError("boom")
            yield  # pragma: no cover - marks this as a generator

        plan = build_step_scan_plan(
            strict=False,
            motor=None,
            positions=[None],
            reference=ref,
            detectors=[ref, cam, snapshot],
            shots_per_step=1,
            controller=None,
            experiment="",
            scan_number=None,
            scan_folder=None,
            saving_detectors=[],
            per_step=failing_per_step,
        )
        with pytest.raises(RuntimeError, match="boom"):
            RE(plan)

        # After the failed run the caches must be gone again: a fresh read of
        # a data signal (NOT acq_timestamp, whose persistent subscription
        # legitimately keeps its cache) goes back to the backend.
        counters: dict[str, int] = {}
        _count_backend_gets(snapshot, counters)
        asyncio.run_coroutine_threadsafe(snapshot.read(), RE._loop).result(timeout=5.0)
        assert sum(counters.values()) > 0, (
            "data-signal cache survived unstage — devices were not unstaged "
            "on the failure path"
        )


class TestRateDerivedBounds:
    """1 Hz-era constants must scale with the configured rep rate."""

    def test_grace_wait_capped_at_half_period(self) -> None:
        cam = CaTimestampedReadable("U_Cam", ["Val"], name="cam")
        cam.configure_shot_id(rep_rate_hz=5.0)
        assert cam._effective_grace_wait_s() == pytest.approx(0.1)

    def test_grace_wait_unchanged_at_one_hz(self) -> None:
        cam = CaTimestampedReadable("U_Cam", ["Val"], name="cam")
        cam.configure_shot_id(rep_rate_hz=1.0)
        assert cam._effective_grace_wait_s() == pytest.approx(0.3)

    def test_explicit_grace_below_cap_wins(self) -> None:
        ref = CaGenericDetector("U_Ref", ["Sig"], name="ref")
        ref.configure_shot_id(rep_rate_hz=5.0)
        cam = CaTimestampedReadable("U_Cam", ["Val"], name="cam")
        cam.configure_shot_id(rep_rate_hz=5.0)
        cam.set_reference(ref, grace_wait_s=0.05)
        assert cam._effective_grace_wait_s() == pytest.approx(0.05)

    def test_t0_sync_window_capped_in_start_doc(self) -> None:
        ref, cam, snapshot = _make_devices()
        ref.configure_shot_id(rep_rate_hz=5.0)
        cam.configure_shot_id(rep_rate_hz=5.0)
        start_docs: list[dict] = []
        RE = RunEngine()
        RE.subscribe(
            lambda name, doc: start_docs.append(doc) if name == "start" else None
        )
        connect_mock(RE, ref, cam, snapshot)
        from ophyd_async.core import set_mock_value

        set_mock_value(ref.acq_timestamp, REF_T0)
        set_mock_value(cam.acq_timestamp, CAM_T0)

        pacer = start_pacer(
            RE, [(ref, REF_T0), (cam, CAM_T0)], initial_delay=1.0, interval=0.3
        )
        try:
            RE(_noscan_plan(ref, [ref, cam, snapshot], shots=1))
        finally:
            pacer.cancel()

        assert start_docs[0]["t0_sync_window_s"] == pytest.approx(0.4 / 5.0)


class TestTelemetryReadBound:
    """One hung telemetry PV must cost at most the read budget, not 10 s."""

    def test_hung_signal_degrades_to_null_within_budget(self) -> None:
        telemetry = CaTelemetryReadable(
            "U_Slow", ["Fast", "Hung"], experiment="Test", name="telemetry_u_slow"
        )
        RE = RunEngine()
        connect_mock(RE, telemetry)
        telemetry._read_timeout_s = 0.2

        fast, hung = telemetry._telemetry_signals

        async def never_returns():
            await asyncio.sleep(30)

        hung.read = never_returns  # type: ignore[method-assign]

        t0 = time.monotonic()
        reading = asyncio.run_coroutine_threadsafe(telemetry.read(), RE._loop).result(
            timeout=5.0
        )
        elapsed = time.monotonic() - t0

        assert elapsed < 2.0, f"hung signal held the row for {elapsed:.1f}s"
        hung_cells = [k for k in reading if "hung" in k.lower()]
        assert hung_cells, f"no null cell emitted for the hung signal: {reading}"
        for key in hung_cells:
            assert reading[key]["alarm_severity"] == 3
        fast_cells = [k for k in reading if "fast" in k.lower()]
        assert fast_cells and all(reading[k]["alarm_severity"] == 0 for k in fast_cells)
