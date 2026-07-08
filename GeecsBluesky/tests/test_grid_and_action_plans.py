"""RunEngine-level composition tests: setup / per_step / closeout hooks.

Uses the ``test_free_run_plan`` harness (CA mock devices, pacer coroutine)
to run the full ``build_step_scan_plan`` composition: setup before the
free-run quiesce/t0-sync and the first step, per_step at every bin,
closeout in a finalize wrapper that runs even when the plan dies mid-scan —
ordered *after* the trigger disarm (finalize nesting).  The message-level
grid tests (no CA needed) live in ``test_grid_plans_message_level.py``.
"""

from __future__ import annotations

from typing import Any

import bluesky.plan_stubs as bps
import pytest
from bluesky import RunEngine
from ophyd_async.core import AsyncStatus

from geecs_bluesky.models.shot_control import ShotControlConfig
from geecs_bluesky.plans.orchestration import build_step_scan_plan
from geecs_bluesky.shot_controller import ShotController

pytest.importorskip("aioca")

from ophyd_async.core import set_mock_value  # noqa: E402

from geecs_bluesky.devices.ca import CaGenericDetector, CaMotor  # noqa: E402
from tests.ca_mock_helpers import (  # noqa: E402
    connect_mock,
    follow_setpoint,
    start_pacer,
)


class _JournalSetter:
    """Mock shot-control setter appending to the shared journal."""

    def __init__(self, name: str, journal: list) -> None:
        self.name = name
        self._journal = journal

    def set(self, value: Any) -> AsyncStatus:
        self._journal.append((self.name, value))

        async def _noop() -> None:
            pass

        return AsyncStatus(_noop())


def _marker(journal: list, label: str, fail_on_call: int | None = None):
    """A plan-stub callable that records *label* (and can fail on call N)."""
    calls = {"n": 0}

    def _stub():
        calls["n"] += 1
        if fail_on_call is not None and calls["n"] == fail_on_call:
            raise RuntimeError(f"{label} failed on call {calls['n']}")
        journal.append(label)
        yield from bps.null()

    return _stub


def _run_composed(
    journal: list,
    *,
    per_step=None,
    setup=None,
    closeout=None,
    positions=(0.0, 1.0),
):
    """Run a free-run build_step_scan_plan with journaled hooks + trigger."""
    motor = CaMotor("U_Ref", "Position (mm)", name="scan_motor")
    ref = CaGenericDetector("U_Ref", ["Sig"], name="ref")
    ref.configure_shot_id(rep_rate_hz=1.0)

    config = ShotControlConfig(
        device="U_DG645",
        variables={
            "Trigger.Source": {"OFF": "Single", "SCAN": "Ext", "STANDBY": "Ext"}
        },
    )
    controller = ShotController(
        config, {"Trigger.Source": _JournalSetter("trigger", journal)}
    )

    RE = RunEngine()
    events: list[dict] = []
    RE.subscribe(lambda name, doc: events.append(doc) if name == "event" else None)
    connect_mock(RE, motor, ref)
    follow_setpoint(motor)
    set_mock_value(ref.acq_timestamp, 1000.0)
    pacer = start_pacer(RE, [(ref, 1000.0)], initial_delay=1.0, interval=0.15)
    plan = build_step_scan_plan(
        strict=False,
        motor=motor,
        positions=list(positions),
        reference=ref,
        detectors=[ref],
        shots_per_step=1,
        controller=controller,
        experiment="",
        scan_number=None,
        scan_folder=None,
        saving_detectors=[],
        setup=setup,
        per_step=per_step,
        closeout=closeout,
    )
    try:
        RE(plan)
    finally:
        pacer.cancel()
    return events


def test_setup_per_step_closeout_order_in_the_full_composition() -> None:
    journal: list = []
    _run_composed(
        journal,
        setup=_marker(journal, "setup"),
        per_step=_marker(journal, "per_step"),
        closeout=_marker(journal, "closeout"),
    )
    # setup runs before the free-run quiesce (OFF write) and the steps;
    # per_step at each of the 2 steps, bracketed by SCAN/STANDBY writes;
    # closeout LAST, after the finalize disarm (STANDBY).
    assert journal == [
        "setup",
        ("trigger", "Single"),  # quiesce (OFF) before t0 sync
        "per_step",
        ("trigger", "Ext"),  # arm (SCAN)
        ("trigger", "Ext"),  # disarm (STANDBY)
        "per_step",
        ("trigger", "Ext"),  # arm (SCAN)
        ("trigger", "Ext"),  # disarm (STANDBY)
        ("trigger", "Ext"),  # finalize disarm (STANDBY)
        "closeout",
    ]


def test_closeout_runs_even_when_the_scan_dies_mid_plan() -> None:
    journal: list = []
    with pytest.raises(RuntimeError, match="per_step failed on call 2"):
        _run_composed(
            journal,
            setup=_marker(journal, "setup"),
            per_step=_marker(journal, "per_step", fail_on_call=2),
            closeout=_marker(journal, "closeout"),
        )
    # The abort path still disarms (STANDBY) and then runs closeout — and
    # closeout comes after the disarm write (finalize nesting).
    assert "closeout" in journal
    assert journal[-1] == "closeout"
    assert journal[-2] == ("trigger", "Ext")  # the finalize disarm


def test_closeout_runs_when_setup_itself_fails() -> None:
    journal: list = []
    with pytest.raises(RuntimeError, match="setup failed"):
        _run_composed(
            journal,
            setup=_marker(journal, "setup", fail_on_call=1),
            closeout=_marker(journal, "closeout"),
        )
    assert journal[-1] == "closeout"
