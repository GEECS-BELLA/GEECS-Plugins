"""Queueserver pause semantics (issue #641): quiesce-on-pause + hard-pause replay.

Four contracts, one per section:

* **Replay-cache audit** — reconstructing the RunEngine's rewind cache over
  the step plans' message streams proves a completed event row (create+save)
  is replayable only in the single residual instant where the very next
  message is the post-row checkpoint.  Everything else a hard pause can
  replay is idempotent: absolute moves, trigger-state writes, an incomplete
  row (a re-fire — strict's bounded-refire semantics).
* **ShotControlPauseQuiescer** — the Pausable seam: OFF driven while the RE
  enters the paused state (both pause verbs), the interrupted standing state
  re-asserted inside ``RunEngine.resume()`` *before* the rewind replay;
  ARMED (strict) deliberately skipped — already quiescent by construction;
  a profile without OFF writes warns instead of pretending.
* **Hard pause mid-move (free-run, the flagship)** — an operator
  ``request_pause(defer=False)`` lands inside a move; resume must replay the
  move at the same absolute target, fire no extra trigger, duplicate no
  event row, and sequence quiesce/re-assert around the replay.  Also pins
  the issue's window claim: the free-run pre-move checkpoint sits in a
  disarmed window, but per-row checkpoints sit INSIDE the SCAN window — the
  quiescer, not checkpoint placement, is what makes those pauses quiescent.
* **Failed-move → pause (decision 4)** — a failed move status pauses the RE
  with the documented reason line; resume retries the move from the
  checkpoint (the replay working *for* us); a retry that fails pauses
  again; stop ends the run gracefully.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import bluesky.plan_stubs as bps
import pytest
from bluesky import RunEngine
from bluesky.utils import FailedStatus, Msg, RunEngineInterrupted
from ophyd_async.core import AsyncStatus

from geecs_bluesky.models.shot_control import ShotControlWrites
from geecs_bluesky.plans.orchestration import build_step_scan_plan
from geecs_bluesky.plans.pause_semantics import (
    FAILED_MOVE_LOG_PREFIX,
    QUIESCE_FROM,
    ShotControlPauseQuiescer,
)
from geecs_bluesky.plans.step_scan import geecs_step_scan
from geecs_bluesky.shot_controller import ShotController

# ---------------------------------------------------------------------------
# Shared minimal fakes (no CA, no network)
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


class _RecordingSetter:
    """Shot-control setter recording ``(target, value)`` into a journal."""

    parent = None

    def __init__(self, journal: list, key: tuple[str, str]) -> None:
        self._journal = journal
        self._key = key

    def set(self, value: Any) -> AsyncStatus:
        self._journal.append((self._key, str(value)))

        async def _done() -> None:
            return None

        return AsyncStatus(_done())


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


#: Write values name their state so tests can read the trigger state
#: straight off the journal / message stream.
_STATES = {
    "SCAN": [("U_DG645", "Amplitude.Ch AB", "SCAN")],
    "STANDBY": [("U_DG645", "Amplitude.Ch AB", "STANDBY")],
    # Two OFF writes pin the ordered, each-completes-first drive.
    "OFF": [
        ("U_DG645", "Trigger.Source", "OFF-1"),
        ("U_DG645", "Amplitude.Ch AB", "OFF-2"),
    ],
    "ARMED": [("U_DG645", "Trigger.Source", "ARMED")],
    "SINGLESHOT": [("U_DG645", "Trigger.Source", "FIRE")],
}


def _controller(journal: list, states: dict | None = None) -> ShotController:
    writes = ShotControlWrites(name="test profile", states=states or _STATES)
    return ShotController.from_writes(
        writes,
        setter_factory=lambda device, variable: _RecordingSetter(
            journal, (device, variable)
        ),
    )


def _journal_values(journal: list) -> list[str]:
    return [value for _key, value in journal]


# ---------------------------------------------------------------------------
# Replay-cache audit: what can a hard pause replay?  (message-level, CI-safe)
# ---------------------------------------------------------------------------


def _assert_no_replayable_completed_row(commands: list[str]) -> None:
    """Walk the RE's rewind-cache reconstruction over *commands*.

    Mirrors the RunEngine exactly: uncacheable commands are never cached, a
    processed ``checkpoint`` empties the cache.  A hard pause landing after
    message ``i`` replays the cache as reconstructed at ``i`` — so a cache
    containing a ``save`` (a completed event row: re-running it duplicates
    the row and, in strict mode, re-fires the shot) is tolerable only in the
    residual instant where the very next message is the checkpoint that
    clears it.  (That two-message window is irreducible: no message fits
    between a bundle's ``save`` and the next processed message.)
    """
    uncacheable = set(RunEngine._UNCACHEABLE_COMMANDS)
    cache: list[str] = []
    for index, command in enumerate(commands):
        if command == "checkpoint":
            cache = []
            continue
        if command not in uncacheable:
            cache.append(command)
        if "save" in cache:
            following = commands[index + 1] if index + 1 < len(commands) else None
            assert following == "checkpoint", (
                f"a completed event row is hard-pause-replayable at message "
                f"{index} ({command!r}): the next message is {following!r}, "
                "not the checkpoint that would clear it"
            )


def test_step_plan_hard_pause_never_replays_a_completed_row() -> None:
    """Strict step scan with full trigger bracketing passes the cache walk."""
    journal: list = []
    controller = _controller(journal)
    motor = _NamedFake("jet_z")
    det = _NamedFake("cam")
    messages = _collect(
        geecs_step_scan(
            motor=motor,
            positions=[0.0, 1.0, 2.0],
            detectors=[det],
            shots_per_step=2,
            setup_trigger=lambda: controller.arm_single_shot([]),
            fire_shot=controller.fire_shot,
        )
    )
    _assert_no_replayable_completed_row([m.command for m in messages])


def test_free_run_plan_hard_pause_never_replays_a_completed_row() -> None:
    """Free-run (arm/disarm per bin + tail flush) passes the cache walk.

    The bin's disarm writes and the end-of-scan quiesce + tail flush are the
    segments the #641 audit found unprotected: without the post-rows and
    post-flush checkpoints, a hard pause there replayed the last completed
    row (duplicate event + re-fired trigger).
    """
    pytest.importorskip("aioca")
    from ophyd_async.core import set_mock_value

    from geecs_bluesky.devices.ca import CaGenericDetector
    from geecs_bluesky.plans.free_run_step_scan import geecs_free_run_step_scan
    from tests.ca_mock_helpers import connect_mock, start_pacer

    journal: list = []
    controller = _controller(journal)
    ref = CaGenericDetector("U_Ref", ["Sig"], name="ref")
    ref.configure_shot_id(rep_rate_hz=1.0)

    RE = RunEngine()
    commands: list[str] = []
    RE.msg_hook = lambda msg: commands.append(msg.command)
    connect_mock(RE, ref)
    set_mock_value(ref.acq_timestamp, 1000.0)
    pacer = start_pacer(RE, [(ref, 1000.0)], initial_delay=0.2, interval=0.1)
    try:
        RE(
            geecs_free_run_step_scan(
                motor=None,
                positions=[None],
                reference=ref,
                detectors=[],
                shots_per_step=2,
                arm_trigger=controller.arm,
                disarm_trigger=controller.disarm,
                quiesce_trigger=controller.quiesce,
            )
        )
    finally:
        pacer.cancel()
    _assert_no_replayable_completed_row(commands)


def test_step_plan_checkpoints_immediately_after_per_step() -> None:
    """Pins P1 (#645 cross-vendor addendum): checkpoint right after per_step.

    A hard pause landing between per_step()'s (potentially non-idempotent)
    compiled-ActionPlan writes and arm must not replay those writes — the
    checkpoint must be the very next message once per_step yields control
    back, before arm_trigger runs.
    """
    marker = _NamedFake("action_marker")

    def per_step():
        yield Msg("null", marker)

    controller = _controller([])
    messages = _collect(
        geecs_step_scan(
            motor=_NamedFake("jet_z"),
            positions=[0.0],
            detectors=[_NamedFake("cam")],
            shots_per_step=1,
            setup_trigger=lambda: controller.arm_single_shot([]),
            fire_shot=controller.fire_shot,
            arm_trigger=controller.arm,
            per_step=per_step,
        )
    )
    marker_index = next(
        i for i, m in enumerate(messages) if m.command == "null" and m.obj is marker
    )
    assert messages[marker_index + 1].command == "checkpoint"


def test_free_run_plan_checkpoints_immediately_after_per_step() -> None:
    """Free-run counterpart of the per_step-checkpoint placement pin.

    Needs a real RE + CA-mock reference (t0-sync polls real readback
    values) — message-level ``_collect`` alone can't drive this plan, same
    constraint as the free-run replay-cache-audit test above.
    """
    pytest.importorskip("aioca")
    from ophyd_async.core import set_mock_value

    from geecs_bluesky.devices.ca import CaGenericDetector
    from geecs_bluesky.plans.free_run_step_scan import geecs_free_run_step_scan
    from tests.ca_mock_helpers import connect_mock, start_pacer

    marker = _NamedFake("action_marker")

    def per_step():
        yield Msg("null", marker)

    controller = _controller([])
    ref = CaGenericDetector("U_Ref", ["Sig"], name="ref")
    ref.configure_shot_id(rep_rate_hz=1.0)

    RE = RunEngine()
    messages: list[Msg] = []
    RE.msg_hook = lambda msg: messages.append(msg)
    connect_mock(RE, ref)
    set_mock_value(ref.acq_timestamp, 1000.0)
    pacer = start_pacer(RE, [(ref, 1000.0)], initial_delay=0.2, interval=0.1)
    try:
        RE(
            geecs_free_run_step_scan(
                motor=None,
                positions=[None],
                reference=ref,
                detectors=[],
                shots_per_step=1,
                arm_trigger=controller.arm,
                disarm_trigger=controller.disarm,
                per_step=per_step,
            )
        )
    finally:
        pacer.cancel()
    marker_index = next(
        i for i, m in enumerate(messages) if m.command == "null" and m.obj is marker
    )
    assert messages[marker_index + 1].command == "checkpoint"


# ---------------------------------------------------------------------------
# ShotControlPauseQuiescer: the Pausable seam
# ---------------------------------------------------------------------------


def _quiesced_plan(quiescer: ShotControlPauseQuiescer, controller: ShotController):
    yield from quiescer.register()
    yield from controller.arm()
    yield from bps.checkpoint()
    yield from bps.pause()
    yield from controller.disarm()


def test_hard_pause_quiesces_and_resume_reasserts_in_order() -> None:
    """SCAN → pause drives OFF (ordered); resume re-asserts SCAN first."""
    journal: list = []
    controller = _controller(journal)
    RE = RunEngine()
    with pytest.raises(RunEngineInterrupted):
        RE(_quiesced_plan(ShotControlPauseQuiescer(controller), controller))
    assert RE.state == "paused"
    # The two OFF writes ran in profile order, after the plan's SCAN.
    assert _journal_values(journal) == ["SCAN", "OFF-1", "OFF-2"]
    assert controller.last_state == "OFF"
    RE.resume()
    # SCAN re-asserted (before any replay could run), then the plan's disarm.
    assert _journal_values(journal) == [
        "SCAN",
        "OFF-1",
        "OFF-2",
        "SCAN",
        "STANDBY",
    ]
    assert controller.last_state == "STANDBY"


def test_deferred_pause_verb_quiesces_identically() -> None:
    """The operator's deferred verb rides the same seam as the hard pause."""
    journal: list = []
    controller = _controller(journal)

    def plan(quiescer):
        yield from quiescer.register()
        yield from controller.arm()
        yield from bps.deferred_pause()
        yield from bps.checkpoint()  # the deferred pause lands here
        yield from controller.disarm()

    RE = RunEngine()
    with pytest.raises(RunEngineInterrupted):
        RE(plan(ShotControlPauseQuiescer(controller)))
    assert RE.state == "paused"
    assert _journal_values(journal) == ["SCAN", "OFF-1", "OFF-2"]
    RE.resume()
    assert _journal_values(journal)[-2:] == ["SCAN", "STANDBY"]


def test_strict_armed_is_already_quiescent_no_writes_on_pause() -> None:
    """ARMED (strict) pauses without any quiesce write — pinned by design.

    The single-shot trigger source cannot free-run, so the paused state is
    quiescent by construction; driving OFF would be extra hardware traffic
    for nothing.  ``QUIESCE_FROM`` is the doctrine.
    """
    assert QUIESCE_FROM == {"SCAN", "STANDBY"}
    journal: list = []
    controller = _controller(journal)

    def plan(quiescer):
        yield from quiescer.register()
        yield from controller.set_state("ARMED")
        yield from bps.checkpoint()
        yield from bps.pause()

    RE = RunEngine()
    with pytest.raises(RunEngineInterrupted):
        RE(plan(ShotControlPauseQuiescer(controller)))
    assert RE.state == "paused"
    assert _journal_values(journal) == ["ARMED"]
    RE.resume()
    assert _journal_values(journal) == ["ARMED"]  # nothing re-asserted either
    assert controller.last_state == "ARMED"


def test_profile_without_off_warns_and_writes_nothing(caplog) -> None:
    """No OFF state → loud warning, no writes, pause still succeeds."""
    journal: list = []
    states = {key: writes for key, writes in _STATES.items() if key != "OFF"}
    controller = _controller(journal, states)

    def plan(quiescer):
        yield from quiescer.register()
        yield from controller.arm()
        yield from bps.checkpoint()
        yield from bps.pause()

    RE = RunEngine()
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.plans.pause_semantics"):
        with pytest.raises(RunEngineInterrupted):
            RE(plan(ShotControlPauseQuiescer(controller)))
    assert RE.state == "paused"
    assert _journal_values(journal) == ["SCAN"]
    assert any("defines no OFF" in record.message for record in caplog.records)
    RE.resume()


def test_build_step_scan_plan_registers_the_quiescer_first() -> None:
    """The composed plan's first message registers the quiescer (issue #641)."""
    journal: list = []
    controller = _controller(journal)
    plan = build_step_scan_plan(
        strict=True,
        motor=None,
        positions=[None],
        reference=None,
        detectors=[_NamedFake("cam")],
        shots_per_step=1,
        controller=controller,
        experiment="",
        scan_number=None,
        scan_folder=None,
        saving_detectors=[],
    )
    first = _collect(plan)[0]
    assert first.command == "null"
    assert isinstance(first.obj, ShotControlPauseQuiescer)


# ---------------------------------------------------------------------------
# Flagship: hard pause mid-move in a free-run scan (RE + CA mocks)
# ---------------------------------------------------------------------------


def test_hard_pause_mid_move_resume_replays_only_the_move() -> None:
    """Immediate operator pause mid-move: the issue-#641 acceptance triple.

    Resume must (1) duplicate no event document, (2) re-fire no trigger in
    the replayed segment, (3) re-issue the move to the same absolute target
    — with the quiescer's OFF/re-assert bracketing the pause, sequenced
    before the replay.  Also pins the window claim: pre-move checkpoints
    land disarmed (STANDBY/OFF), per-row checkpoints land inside SCAN.
    """
    pytest.importorskip("aioca")
    from ophyd_async.core import callback_on_mock_put, set_mock_value

    from geecs_bluesky.devices.ca import CaGenericDetector, CaMotor
    from geecs_bluesky.plans.free_run_step_scan import geecs_free_run_step_scan
    from tests.ca_mock_helpers import connect_mock, start_pacer

    journal: list = []
    controller = _controller(journal)
    quiescer = ShotControlPauseQuiescer(controller)
    motor = CaMotor("U_Stage", "Position (mm)", name="scan_motor", move_timeout=60.0)
    ref = CaGenericDetector("U_Ref", ["Sig"], name="ref")
    ref.configure_shot_id(rep_rate_hz=1.0)

    RE = RunEngine()
    messages: list[Msg] = []
    RE.msg_hook = messages.append
    docs: list[tuple[str, dict]] = []
    RE.subscribe(lambda name, doc: docs.append((name, doc)))
    connect_mock(RE, motor, ref)
    set_mock_value(ref.acq_timestamp, 1000.0)
    # No setpoint follower at first: the bin-2 move to 1.0 cannot converge,
    # so the RE is deterministically stuck in that move's wait when the
    # immediate pause arrives.  (Bin 1 moves to 0.0 = the initial readback.)
    # One combined mock-put callback — a second callback_on_mock_put would
    # replace the first, losing the replayed put from the record.
    setpoint_puts: list[float] = []
    following = {"on": False}

    def _on_setpoint_put(value, **kwargs):
        setpoint_puts.append(float(value))
        if following["on"]:
            set_mock_value(motor.position, value)

    callback_on_mock_put(motor._setpoint, _on_setpoint_put)

    def plan():
        yield from quiescer.register()
        yield from geecs_free_run_step_scan(
            motor=motor,
            positions=[0.0, 1.0],
            reference=ref,
            detectors=[],
            shots_per_step=2,
            arm_trigger=controller.arm,
            disarm_trigger=controller.disarm,
            quiesce_trigger=controller.quiesce,
        )

    def interrupt():
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if 1.0 in setpoint_puts and RE.state == "running":
                time.sleep(0.3)  # let the RE settle into the move's wait
                RE.request_pause(defer=False)
                return
            time.sleep(0.02)

    pacer = start_pacer(RE, [(ref, 1000.0)], initial_delay=0.2, interval=0.1)
    interrupter = threading.Thread(target=interrupt, daemon=True)
    interrupter.start()
    try:
        with pytest.raises(RunEngineInterrupted):
            RE(plan())
        interrupter.join(timeout=25.0)
        assert RE.state == "paused"
        # Paused mid-move: bin 1 had disarmed (STANDBY), so the quiescer
        # drove OFF — after the scan's own initial quiesce and bin 1.
        assert _journal_values(journal) == [
            "OFF-1",
            "OFF-2",  # scan-start quiesce (pre-t0)
            "SCAN",
            "STANDBY",  # bin 1 arm/disarm
            "OFF-1",
            "OFF-2",  # quiescer: pause landed
        ]
        # Let the replayed move converge, then resume.
        following["on"] = True
        RE.resume()
    finally:
        pacer.cancel()

    # Quiesce/re-assert bracketing: STANDBY re-asserted on resume (before
    # the replay), then bin 2 and the end-of-scan quiesce as normal.
    assert _journal_values(journal) == [
        "OFF-1",
        "OFF-2",
        "SCAN",
        "STANDBY",
        "OFF-1",
        "OFF-2",
        "STANDBY",  # resume re-assert, ahead of the replayed move
        "SCAN",
        "STANDBY",  # bin 2
        "OFF-1",
        "OFF-2",  # end-of-scan quiesce
    ]
    assert controller.last_state == "OFF"

    # (3) moves re-issue absolute targets only: bin-1 target, the paused
    # move, and its replay — same absolute value, nothing else.
    assert setpoint_puts == [0.0, 1.0, 1.0]

    # (1) no duplicate event documents: 4 primary rows, indices 1-4 once.
    streams = {
        doc["uid"]: doc.get("name") for name, doc in docs if name == "descriptor"
    }
    primary_rows = [
        doc
        for name, doc in docs
        if name == "event" and streams.get(doc["descriptor"]) == "primary"
    ]
    indices = sorted(row["data"]["scan_event_index"] for row in primary_rows)
    assert indices == [1, 2, 3, 4]
    seq_nums = [row["seq_num"] for row in primary_rows]
    assert len(seq_nums) == len(set(seq_nums))

    # (2) no re-fired trigger in the replayed segment: one trigger per row.
    commands = [m.command for m in messages]
    assert commands.count("trigger") == 4

    # Window claim (the issue's candidate (b), verified): walk the
    # plan-driven trigger state at each checkpoint.  Pre-move checkpoints sit
    # disarmed (OFF/STANDBY); per-row checkpoints sit INSIDE SCAN — which
    # is exactly why the quiescer exists.
    state = None
    states_at_checkpoints: list[str | None] = []
    for message in messages:
        if message.command == "set" and isinstance(message.obj, _RecordingSetter):
            state = str(message.args[0]).split("-")[0]
        elif message.command == "checkpoint":
            states_at_checkpoints.append(state)
    assert "SCAN" in states_at_checkpoints, (
        "expected per-row checkpoints inside the SCAN window - if this "
        "stops holding, the quiescer rationale needs re-auditing"
    )
    assert states_at_checkpoints[0] in (None, "OFF"), "pre-bin checkpoint armed?"


# ---------------------------------------------------------------------------
# Failed-move → pause (decision 4): retry from checkpoint, stop gracefully
# ---------------------------------------------------------------------------


class _FlakyMotor:
    """Movable failing its move to *fail_at* the first *failures* times."""

    parent = None

    def __init__(self, name: str, fail_at: float, failures: int) -> None:
        self.name = name
        self.fail_at = fail_at
        self.failures_left = failures
        self.attempts: list[float] = []
        self._position = 0.0

    def set(self, value: float) -> AsyncStatus:
        self.attempts.append(float(value))
        fail = value == self.fail_at and self.failures_left > 0
        if fail:
            self.failures_left -= 1

        async def _move() -> None:
            if fail:
                raise RuntimeError(
                    f"simulated move failure: {self.name} did not reach {value}"
                )
            self._position = float(value)

        return AsyncStatus(_move())

    def read(self) -> dict:
        return {self.name: {"value": self._position, "timestamp": 0.0}}

    def describe(self) -> dict:
        return {self.name: {"source": "sim", "dtype": "number", "shape": []}}


class _PlainDet(_NamedFake):
    def read(self) -> dict:
        return {self.name: {"value": 42.0, "timestamp": 0.0}}

    def describe(self) -> dict:
        return {self.name: {"source": "sim", "dtype": "number", "shape": []}}


def _failed_move_scan(motor: _FlakyMotor, failed_move_policy: str | None = None):
    """None means: omit the kwarg entirely, exercising the builder's default."""
    return geecs_step_scan(
        motor=motor,
        positions=[0.0, 1.0],
        detectors=[_PlainDet("cam")],
        shots_per_step=2,
        **({} if failed_move_policy is None else {"failed_move_policy": failed_move_policy}),
    )


def _primary_indices(docs: list[tuple[str, dict]]) -> list[int]:
    return sorted(
        doc["data"]["scan_event_index"] for name, doc in docs if name == "event"
    )


def test_failed_move_pauses_with_reason_and_resume_retries(caplog) -> None:
    """One failure: reason logged, RE paused; resume replays the move once."""
    motor = _FlakyMotor("flaky", fail_at=1.0, failures=1)
    RE = RunEngine()
    docs: list[tuple[str, dict]] = []
    RE.subscribe(lambda name, doc: docs.append((name, doc)))
    with caplog.at_level(logging.ERROR, logger="geecs_bluesky.plans.step_scan"):
        with pytest.raises(RunEngineInterrupted):
            RE(_failed_move_scan(motor, failed_move_policy="pause"))
    assert RE.state == "paused"
    reasons = [
        record.message
        for record in caplog.records
        if FAILED_MOVE_LOG_PREFIX in record.message
    ]
    assert len(reasons) == 1
    assert "flaky -> 1.0" in reasons[0]
    assert "simulated move failure" in reasons[0]

    RE.resume()
    assert RE.state == "idle"
    # The retry came from the checkpoint replay: same absolute target again.
    assert motor.attempts == [0.0, 1.0, 1.0]
    # No row lost, no row duplicated across the pause/replay.
    assert _primary_indices(docs) == [1, 2, 3, 4]


def test_failed_move_retry_that_fails_pauses_again(caplog) -> None:
    """Two failures: pause → resume (retry fails) → pause → resume → done."""
    motor = _FlakyMotor("flaky", fail_at=1.0, failures=2)
    RE = RunEngine()
    docs: list[tuple[str, dict]] = []
    RE.subscribe(lambda name, doc: docs.append((name, doc)))
    with caplog.at_level(logging.ERROR, logger="geecs_bluesky.plans.step_scan"):
        with pytest.raises(RunEngineInterrupted):
            RE(_failed_move_scan(motor, failed_move_policy="pause"))
        assert RE.state == "paused"
        with pytest.raises(RunEngineInterrupted):
            RE.resume()  # the replayed retry fails → paused again
        assert RE.state == "paused"
        RE.resume()
    assert RE.state == "idle"
    assert motor.attempts == [0.0, 1.0, 1.0, 1.0]
    assert _primary_indices(docs) == [1, 2, 3, 4]
    reasons = [
        record.message
        for record in caplog.records
        if FAILED_MOVE_LOG_PREFIX in record.message
    ]
    assert len(reasons) == 2


def test_failed_move_then_stop_ends_gracefully() -> None:
    """Stop from the failed-move pause: graceful stop document, RE idle."""
    motor = _FlakyMotor("flaky", fail_at=1.0, failures=10_000)
    RE = RunEngine()
    docs: list[tuple[str, dict]] = []
    RE.subscribe(lambda name, doc: docs.append((name, doc)))
    with pytest.raises(RunEngineInterrupted):
        RE(_failed_move_scan(motor, failed_move_policy="pause"))
    assert RE.state == "paused"
    RE.stop()
    assert RE.state == "idle"
    stop_docs = [doc for name, doc in docs if name == "stop"]
    assert len(stop_docs) == 1
    assert stop_docs[0]["exit_status"] == "success"  # graceful, not aborted
    # Bin 1 completed before the failure; its rows survive exactly once.
    assert _primary_indices(docs) == [1, 2]


def test_failed_move_policy_defaults_to_raise_like_legacy() -> None:
    """Default policy propagates FailedStatus — bridge/console untouched.

    Issue #645 F1: the pause-on-failed-move path coexists badly with the
    engine-side pause supervisor (auto-resume loop, stop-from-paused
    bypass).  Callers that don't opt in — the bridge/console path via
    ``scan_request_plan.py`` — must see exactly the pre-#641 behavior: the
    ``FailedStatus`` raises straight through ``RE()``, no pause.
    """
    import inspect

    from geecs_bluesky.plans.orchestration import build_step_scan_plan

    assert (
        inspect.signature(build_step_scan_plan).parameters["failed_move_policy"].default
        == "raise"
    )
    motor = _FlakyMotor("flaky", fail_at=1.0, failures=1)
    RE = RunEngine()
    with pytest.raises(FailedStatus):
        RE(_failed_move_scan(motor))
    assert RE.state == "idle"
    # One attempt only — no retry-by-replay under "raise".
    assert motor.attempts == [0.0, 1.0]
