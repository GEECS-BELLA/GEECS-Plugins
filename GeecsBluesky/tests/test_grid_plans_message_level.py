"""Message-level multi-axis grid tests (no RunEngine, no CA — CI-safe).

Pins the grid semantics of ``geecs_step_scan`` with fake motors: the move
sequence (only changed axes re-moved, innermost fastest under the
outer-product ordering), one bin per grid point with every motor read in
every event row, and the per-step hook placement (after the move, before
the shots).  The RunEngine-level composition tests (setup/closeout
finalize nesting) live in ``test_grid_and_action_plans.py`` and need the
``ca`` extra's mock backends.
"""

from __future__ import annotations

import pytest
from bluesky.utils import Msg

from geecs_bluesky.plans.step_scan import geecs_step_scan


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


def _grid_plan(per_step=None):
    outer = _NamedFake("jet_z")
    inner = _NamedFake("jet_x")
    det = _NamedFake("cam")
    grid = [(0.0, 4.0), (0.0, 5.0), (1.0, 4.0), (1.0, 5.0)]
    plan = geecs_step_scan(
        motor=[outer, inner],
        positions=grid,
        detectors=[det],
        shots_per_step=2,
        per_step=per_step,
    )
    return plan, outer, inner, det


def test_grid_moves_only_changed_axes_innermost_fastest() -> None:
    plan, outer, inner, det = _grid_plan()
    moves = [
        (msg.obj.name, msg.args[0]) for msg in _collect(plan) if msg.command == "set"
    ]
    # Grid point 1: both axes move; point 2: only the inner; point 3: outer
    # steps AND inner returns to its first value; point 4: only the inner.
    assert moves == [
        ("jet_z", 0.0),
        ("jet_x", 4.0),
        ("jet_x", 5.0),
        ("jet_z", 1.0),
        ("jet_x", 4.0),
        ("jet_x", 5.0),
    ]


def test_grid_one_bin_per_point_and_every_motor_in_every_row() -> None:
    plan, outer, inner, det = _grid_plan()
    messages = _collect(plan)
    saves = [m for m in messages if m.command == "save"]
    assert len(saves) == 8  # 4 grid points x 2 shots = 8 event rows
    # Every event row reads both motors (alongside the detector).
    reads_per_event: list[set[str]] = []
    current: set[str] = set()
    for msg in messages:
        if msg.command == "create":
            current = set()
        elif msg.command == "read":
            current.add(msg.obj.name)
        elif msg.command == "save":
            reads_per_event.append(current)
    assert all({"jet_z", "jet_x", "cam"} <= row for row in reads_per_event)


def test_per_step_runs_after_move_before_shots_at_every_grid_point() -> None:
    def per_step():
        yield Msg("per_step_marker", None)

    plan, *_ = _grid_plan(per_step=per_step)
    commands = [m.command for m in _collect(plan)]
    assert commands.count("per_step_marker") == 4  # every grid point
    # Placement: at each boundary the marker comes after the step's moves
    # (set/wait) and before its first shot (trigger/create).
    for index, command in enumerate(commands):
        if command != "per_step_marker":
            continue
        # No trigger/create between this step's moves and the marker.
        after = commands[index + 1]
        assert after in ("trigger", "create"), after
        before = commands[index - 1]
        assert before in ("wait", "open_run"), before


def test_grid_point_arity_mismatch_raises() -> None:
    plan = geecs_step_scan(
        motor=[_NamedFake("a"), _NamedFake("b")],
        positions=[(0.0,)],  # one value for two motors
        detectors=[_NamedFake("cam")],
        shots_per_step=1,
    )
    with pytest.raises(ValueError, match="positions must align"):
        _collect(plan)


# ---------------------------------------------------------------------------
# Native-save windowing (Gate-2): save-on only after the trigger is stopped
# ---------------------------------------------------------------------------


def test_strict_save_on_after_armed_before_first_shot() -> None:
    """Strict: enable_saving runs after setup_trigger (ARMED + quiescence).

    Gate-2 hardware evidence (Scan015): save-on before arming let free-run
    STANDBY frames be saved as orphans during the setup-action window.
    """

    def setup_trigger():
        yield Msg("armed_marker", None)

    def enable_saving():
        yield Msg("save_on_marker", None)

    plan = geecs_step_scan(
        motor=None,
        positions=[None],
        detectors=[_NamedFake("cam")],
        shots_per_step=2,
        setup_trigger=setup_trigger,
        enable_saving=enable_saving,
    )
    commands = [m.command for m in _collect(plan)]
    assert commands.index("open_run") < commands.index("armed_marker")
    assert commands.index("armed_marker") < commands.index("save_on_marker")
    # Saving is on before the first shot is taken.
    assert commands.index("save_on_marker") < commands.index("create")
    assert commands.count("save_on_marker") == 1  # once per run, not per step


def test_strict_per_step_actions_run_with_saving_already_on() -> None:
    """per_step comes after the one-time ARMED/save-on preamble."""

    def enable_saving():
        yield Msg("save_on_marker", None)

    def per_step():
        yield Msg("per_step_marker", None)

    plan = geecs_step_scan(
        motor=None,
        positions=[None, None],
        detectors=[_NamedFake("cam")],
        shots_per_step=1,
        enable_saving=enable_saving,
        per_step=per_step,
    )
    commands = [m.command for m in _collect(plan)]
    assert commands.index("save_on_marker") < commands.index("per_step_marker")
