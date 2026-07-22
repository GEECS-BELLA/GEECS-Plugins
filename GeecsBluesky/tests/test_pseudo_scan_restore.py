"""End-of-scan baseline restore for relative pseudo (composite) axes.

Owner request (2026-07-22, from the first live composite scans): a relative
composite scan must hand its targets back where it found them.  The
orchestration adds a finalize — success AND abort, after closeout, inside
the stage wrapper (so ``unstage`` has not yet dropped the baselines) — that
puts every component back to its captured baseline, exactly (direct target
puts, no ``f(0) = 0`` assumption).

Absolute-mode pseudos and plain motors are deliberately NOT restored (an
absolute pseudo's pre-scan x is unknowable without an inverse; plain axes
match legacy end-at-last-position behavior).
"""

from __future__ import annotations

import bluesky.plan_stubs as bps
import pytest
from bluesky import RunEngine

from geecs_bluesky.forward_expr import compile_forward
from geecs_bluesky.models.shot_control import ShotControlConfig
from geecs_bluesky.plans.orchestration import build_step_scan_plan
from geecs_bluesky.shot_controller import ShotController

pytest.importorskip("aioca")

from ophyd_async.core import AsyncStatus, set_mock_value  # noqa: E402

from geecs_bluesky.devices.ca import CaGenericDetector, CaPseudoMovable  # noqa: E402
from tests.ca_mock_helpers import connect_mock, start_pacer  # noqa: E402

BASELINES = {"U_S3H": 3.0, "U_S4H": 10.0}


class _NoopSetter:
    def __init__(self):
        self.name = "trigger"

    def set(self, value):
        async def _ok():
            return None

        return AsyncStatus(_ok())


def _pseudo(mode: str) -> CaPseudoMovable:
    return CaPseudoMovable(
        [
            ("U_S3H", "Current", compile_forward("x * 1")),
            ("U_S4H", "Current", compile_forward("x * -2")),
        ],
        mode,
        variable_name="bump_x",
        name="bump_x",
    )


def _setpoints(run_engine: RunEngine, device: CaPseudoMovable) -> tuple[float, float]:
    import asyncio

    def read(sig):
        return asyncio.run_coroutine_threadsafe(
            sig.get_value(), run_engine._loop
        ).result(timeout=10.0)

    return read(device._setpoint_0), read(device._setpoint_1)


def _build_scan(mode: str, *, per_step=None):
    """Build a two-step free-run scan with a pseudo axis.

    Returns ``(RE, pseudo, plan, pacer)`` — the caller runs the plan, so
    abort tests can still reach the pseudo afterwards.
    """
    pseudo = _pseudo(mode)
    ref = CaGenericDetector("U_Ref", ["Sig"], name="ref")
    ref.configure_shot_id(rep_rate_hz=1.0)
    config = ShotControlConfig(
        device="U_DG645",
        variables={
            "Trigger.Source": {"OFF": "Single", "SCAN": "Ext", "STANDBY": "Ext"}
        },
    )
    controller = ShotController(config, {"Trigger.Source": _NoopSetter()})

    RE = RunEngine()
    connect_mock(RE, pseudo, ref)
    set_mock_value(pseudo._target_readback_0, BASELINES["U_S3H"])
    set_mock_value(pseudo._target_readback_1, BASELINES["U_S4H"])
    set_mock_value(ref.acq_timestamp, 1000.0)
    pacer = start_pacer(RE, [(ref, 1000.0)], initial_delay=1.0, interval=0.15)
    plan = build_step_scan_plan(
        strict=False,
        motor=pseudo,
        positions=[0.0, 0.5],
        reference=ref,
        detectors=[ref],
        shots_per_step=1,
        controller=controller,
        experiment="",
        scan_number=None,
        scan_folder=None,
        saving_detectors=[],
        per_step=per_step,
    )
    return RE, pseudo, plan, pacer


def _run_scan(mode: str, *, per_step=None) -> tuple[RunEngine, CaPseudoMovable]:
    RE, pseudo, plan, pacer = _build_scan(mode, per_step=per_step)
    try:
        RE(plan)
    finally:
        pacer.cancel()
    return RE, pseudo


def test_relative_pseudo_scan_restores_baselines_at_end() -> None:
    RE, pseudo = _run_scan("relative")
    # Last scan position was x=0.5 (targets 3.5 / 9.0); the finalize put
    # them back to the captured baselines, exactly.
    assert _setpoints(RE, pseudo) == (BASELINES["U_S3H"], BASELINES["U_S4H"])


def test_relative_pseudo_scan_restores_on_abort_too() -> None:
    calls = {"n": 0}

    def exploding_per_step():
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom mid-scan")
        yield from bps.null()

    RE, pseudo, plan, pacer = _build_scan("relative", per_step=exploding_per_step)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            RE(plan)
    finally:
        pacer.cancel()
    # The finalize chain ran on the abort path and put the targets back.
    assert _setpoints(RE, pseudo) == (BASELINES["U_S3H"], BASELINES["U_S4H"])


def test_abort_restore_at_the_finalize_level() -> None:
    """Directly: finalize restore runs even when the wrapped plan raises."""
    import bluesky.preprocessors as bpp

    pseudo = _pseudo("relative")
    RE = RunEngine()
    connect_mock(RE, pseudo)
    set_mock_value(pseudo._target_readback_0, BASELINES["U_S3H"])
    set_mock_value(pseudo._target_readback_1, BASELINES["U_S4H"])

    def exploding_scan():
        yield from bps.stage(pseudo)
        yield from bps.mv(pseudo, 0.5)
        raise RuntimeError("boom")

    plan = bpp.finalize_wrapper(exploding_scan(), pseudo.restore_baselines_plan())
    with pytest.raises(RuntimeError, match="boom"):
        RE(plan)
    assert _setpoints(RE, pseudo) == (BASELINES["U_S3H"], BASELINES["U_S4H"])


def test_absolute_pseudo_scan_is_not_restored() -> None:
    RE, pseudo = _run_scan("absolute")
    # Absolute mode: targets stay where the last position put them
    # (x=0.5 -> 0.5 / -1.0); no baseline to restore to.
    assert _setpoints(RE, pseudo) == (0.5, -1.0)


def test_restore_without_baselines_is_a_noop() -> None:
    pseudo = _pseudo("relative")
    RE = RunEngine()
    connect_mock(RE, pseudo)
    RE(pseudo.restore_baselines_plan())  # never staged/moved — must not raise
