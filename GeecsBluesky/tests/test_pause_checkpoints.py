"""Deferred-pause groundwork for G-actions v2 (issue #552, PR-1).

Pins three things:

* **Checkpoint placement** in both step plans — one before each step's
  move and one before every row, never between ``create`` and ``save`` —
  so ``request_pause(defer=True)`` always lands with an empty rewind
  cache and resume replays nothing (no re-move, no re-fire).
* **Resume replays nothing** at the RunEngine level: a deferred pause
  raised mid-scan and resumed yields exactly the expected event rows,
  each ``scan_event_index`` exactly once.
* **``ShotController.last_state``** records the last *standing* state
  actually driven (never the momentary ``SINGLESHOT`` fire) — what the
  #552 pause supervisor re-asserts after an operator action.

Also pins that the old hard-pause bridge API (``pause_scan`` /
``resume_scan``) is gone: with no checkpoints it was a
resume-replays-the-scan trap, and with checkpoints a bare hard pause
still replays a partial row (re-firing a shot in strict mode).
"""

from __future__ import annotations

from typing import Any

import bluesky.plan_stubs as bps
import pytest
from bluesky import RunEngine
from bluesky.utils import Msg, RunEngineInterrupted
from ophyd_async.core import AsyncStatus

from geecs_bluesky.models.shot_control import ShotControlWrites
from geecs_bluesky.plans.step_scan import geecs_step_scan
from geecs_bluesky.scanner_bridge.bluesky_scanner import BlueskyScanner
from geecs_bluesky.shot_controller import ShotController

# ---------------------------------------------------------------------------
# Message-level checkpoint placement (no RunEngine, no CA — CI-safe)
# ---------------------------------------------------------------------------


class _NamedFake:
    """Minimal named object for message-level plans (never actually set)."""

    parent = None  # bps.mv inspects .parent for coupled-device handling

    def __init__(self, name: str) -> None:
        self.name = name

    def read(self):  # pragma: no cover - message-level only
        return {}

    def describe(self):  # pragma: no cover - message-level only
        return {}


def _collect(plan) -> list[Msg]:
    messages: list[Msg] = []
    try:
        msg = plan.send(None)
        while True:
            messages.append(msg)
            msg = plan.send(None)
    except StopIteration:
        pass
    return messages


def _step_plan_messages(num_points: int = 3, shots_per_step: int = 2) -> list[Msg]:
    motor = _NamedFake("jet_z")
    det = _NamedFake("cam")
    return _collect(
        geecs_step_scan(
            motor=motor,
            positions=[float(i) for i in range(num_points)],
            detectors=[det],
            shots_per_step=shots_per_step,
        )
    )


def test_one_checkpoint_per_step_and_per_row() -> None:
    """3 steps × 2 shots → 3 pre-move + 6 pre-row = 9 checkpoints."""
    commands = [m.command for m in _step_plan_messages(3, 2)]
    assert commands.count("checkpoint") == 3 + 3 * 2


def test_checkpoint_precedes_the_move_and_every_row() -> None:
    """The first message of each step is a checkpoint, before any set."""
    messages = _step_plan_messages(2, 1)
    commands = [m.command for m in messages]
    first_checkpoint = commands.index("checkpoint")
    first_set = commands.index("set")
    assert first_checkpoint < first_set
    # Every create (row open) is preceded by a checkpoint later than the
    # previous save — i.e. one checkpoint per row, at the row boundary.
    last_boundary = -1
    for index, command in enumerate(commands):
        if command == "create":
            boundary = max(i for i in range(index) if commands[i] == "checkpoint")
            assert boundary > last_boundary
        elif command == "save":
            last_boundary = index


def test_no_checkpoint_between_create_and_save() -> None:
    """A checkpoint inside an open event bundle is IllegalMessageSequence."""
    commands = [m.command for m in _step_plan_messages(3, 2)]
    open_bundle = False
    for command in commands:
        if command == "create":
            open_bundle = True
        elif command == "save":
            open_bundle = False
        elif command == "checkpoint":
            assert not open_bundle, "checkpoint inside an open event bundle"


# ---------------------------------------------------------------------------
# RunEngine level: deferred pause + resume replays nothing (CA mocks)
# ---------------------------------------------------------------------------


def test_deferred_pause_and_resume_replays_no_row() -> None:
    """Pause mid-scan (deferred) + resume → every row exactly once."""
    pytest.importorskip("aioca")
    from ophyd_async.core import set_mock_value

    from geecs_bluesky.devices.ca import CaGenericDetector, CaMotor
    from tests.ca_mock_helpers import connect_mock, follow_setpoint, start_pacer

    events: list[dict] = []

    def collect(name: str, doc: dict) -> None:
        if name == "event":
            events.append(doc)

    motor = CaMotor("U_Ref", "Position (mm)", name="scan_motor")
    ref = CaGenericDetector("U_Ref", ["Sig"], name="ref")
    ref.configure_shot_id(rep_rate_hz=1.0)

    RE = RunEngine()
    RE.subscribe(collect)
    connect_mock(RE, motor, ref)
    follow_setpoint(motor)
    set_mock_value(ref.acq_timestamp, 1000.0)
    pacer = start_pacer(RE, [(ref, 1000.0)], initial_delay=0.2, interval=0.1)

    paused_once = {"done": False}

    def pausing_per_step():
        # Deterministic mid-scan deferred pause: the plan itself requests
        # it once; the RE then pauses at the NEXT checkpoint (the first
        # row of this step), where the rewind cache is empty.
        if not paused_once["done"]:
            paused_once["done"] = True
            yield from bps.deferred_pause()
        else:
            yield from bps.null()

    plan = geecs_step_scan(
        motor=motor,
        positions=[0.0, 1.0],
        detectors=[ref],
        shots_per_step=2,
        per_step=pausing_per_step,
    )
    try:
        with pytest.raises(RunEngineInterrupted):
            RE(plan)
        assert RE.state == "paused"
        RE.resume()
    finally:
        pacer.cancel()

    assert len(events) == 4  # 2 positions × 2 shots — nothing replayed
    indices = [e["data"]["scan_event_index"] for e in events]
    assert indices == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# ShotController.last_state
# ---------------------------------------------------------------------------


class _RecordingSetter:
    def __init__(self, name: str, journal: list) -> None:
        self.name = name
        self._journal = journal

    def set(self, value: Any) -> AsyncStatus:
        self._journal.append((self.name, value))

        async def _noop() -> None:
            pass

        return AsyncStatus(_noop())


_WRITES = ShotControlWrites(
    name="pause-test",
    states={
        "SCAN": [("U_DG645", "Amplitude.Ch AB", "4.0")],
        "STANDBY": [("U_DG645", "Amplitude.Ch AB", "0.5")],
        "ARMED": [("U_DG645", "Trigger.Source", "Single shot")],
        "SINGLESHOT": [("U_DG645", "Trigger.ExecuteSingleShot", "on")],
    },
)


def _controller() -> tuple[RunEngine, ShotController]:
    journal: list = []
    controller = ShotController.from_writes(
        _WRITES,
        setter_factory=lambda d, v: _RecordingSetter(f"{d}:{v}", journal),
    )
    return RunEngine(), controller


def test_last_state_tracks_standing_states() -> None:
    re, controller = _controller()
    assert controller.last_state is None
    re(controller.arm())
    assert controller.last_state == "SCAN"
    re(controller.disarm())
    assert controller.last_state == "STANDBY"


def test_singleshot_fire_never_recorded_as_last_state() -> None:
    """Re-asserting last_state must never refire a shot."""
    re, controller = _controller()
    re(controller.set_state("ARMED"))
    assert controller.last_state == "ARMED"
    re(controller.fire_shot())  # SINGLESHOT — momentary, not standing
    assert controller.last_state == "ARMED"


def test_stateless_transition_leaves_last_state_alone() -> None:
    """A state with no writes drives nothing, so nothing is recorded."""
    re, controller = _controller()
    re(controller.arm())
    re(controller.quiesce())  # OFF not declared in this profile → no-op
    assert controller.last_state == "SCAN"


# ---------------------------------------------------------------------------
# The hard-pause bridge API is gone
# ---------------------------------------------------------------------------


def test_hard_pause_bridge_api_deleted() -> None:
    """pause_scan/resume_scan (hard pause, replay trap) must not return."""
    assert not hasattr(BlueskyScanner, "pause_scan")
    assert not hasattr(BlueskyScanner, "resume_scan")
