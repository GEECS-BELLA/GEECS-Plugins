"""Hermetic tests for optimization-as-a-scan (CA mocks; no gateway, no xopt).

The suggester is scripted, shots are paced on the RE loop, and the settable's
mock readback follows its setpoint — so the full ask → move → bin → objective →
tell loop runs end to end with deterministic values.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

pytest.importorskip("aioca")

from ophyd_async.core import callback_on_mock_put, set_mock_value  # noqa: E402

from geecs_bluesky.exceptions import GeecsConfirmTimeoutError  # noqa: E402
from geecs_bluesky.optimize import BinData, RandomSuggester  # noqa: E402
from geecs_bluesky.session import GeecsSession  # noqa: E402
from tests.ca_mock_helpers import start_pacer  # noqa: E402


# ---------------------------------------------------------------------------
# BinData
# ---------------------------------------------------------------------------


def _rows() -> list[dict]:
    return [
        {"bin_number": 1, "cam-valid": True, "cam-acq_timestamp": 100.0, "v": 1.0},
        {"bin_number": 1, "cam-valid": False, "cam-acq_timestamp": 101.0, "v": 9.0},
        {"bin_number": 1, "cam-valid": True, "cam-acq_timestamp": 102.0, "v": 3.0},
    ]


def test_bindata_valid_rows_and_column() -> None:
    bin_data = BinData(1, _rows())
    assert len(bin_data.valid_rows("cam")) == 2
    assert list(bin_data.column("v", valid_for="cam")) == [1.0, 3.0]
    assert list(bin_data.column("v")) == [1.0, 9.0, 3.0]


# ---------------------------------------------------------------------------
# Suggesters
# ---------------------------------------------------------------------------


def test_random_suggester_bounds_and_best() -> None:
    sug = RandomSuggester({"x": (-1.0, 1.0)}, seed=1)
    for _ in range(5):
        value = sug.suggest()["x"]
        assert -1.0 <= value <= 1.0
    sug.observe({"x": 0.1}, 5.0, BinData(1, []))
    sug.observe({"x": 0.2}, float("nan"), BinData(2, []))
    sug.observe({"x": 0.3}, 7.0, BinData(3, []))
    assert sug.best == {"x": 0.3, "objective": 7.0}


class ScriptedSuggester:
    """Deterministic suggester for machinery tests."""

    def __init__(self, points: list[dict]) -> None:
        self._points = list(points)
        self.observed: list[tuple[dict, float, BinData]] = []

    def suggest(self):
        return self._points.pop(0) if self._points else None

    def observe(self, inputs, objective, bin_data):
        self.observed.append((inputs, objective, bin_data))


# ---------------------------------------------------------------------------
# session.optimize end-to-end (mock)
# ---------------------------------------------------------------------------


def test_optimize_runs_as_one_scan_with_bins() -> None:
    """ask -> move -> bin -> objective -> tell, all inside one Bluesky run."""
    s = GeecsSession("Test", tiled=False, mock=True)
    cam = s.detector("UC_Cam", ["Sig"], name="cam")
    knob = s.settable("U_S1H", "Current", name="s1h")
    callback_on_mock_put(
        knob._setpoint, lambda v, **kw: set_mock_value(knob.readback, v)
    )
    set_mock_value(cam.acq_timestamp, 1000.0)
    pacer = start_pacer(s.RE, [(cam, 1000.0)], initial_delay=0.8, interval=0.15)

    starts: list[dict] = []
    s.RE.subscribe(lambda n, d: starts.append(d) if n == "start" else None)

    suggester = ScriptedSuggester([{"s1h": 0.1}, {"s1h": 0.4}])

    def objective(bin_data: BinData) -> float:
        # The knob readback is recorded in every row like a scan motor.
        return -float(np.median(bin_data.column("s1h-readback")))

    try:
        uid, history = s.optimize(
            variables={"s1h": knob},
            detectors=[cam],
            objective=objective,
            suggester=suggester,
            shots_per_iteration=2,
            max_iterations=5,  # scripted suggester stops after 2
            save_data=False,
        )
    finally:
        pacer.cancel()

    assert uid is not None
    assert len(starts) == 1, "the whole optimization is ONE run"
    assert starts[0]["plan_name"] == "geecs_adaptive_scan"
    assert starts[0]["adaptive"] is True

    assert [h["inputs"] for h in history] == [{"s1h": 0.1}, {"s1h": 0.4}]
    assert [h["n_rows"] for h in history] == [2, 2]
    assert history[0]["objective"] == pytest.approx(-0.1)
    assert history[1]["objective"] == pytest.approx(-0.4)
    # The suggester saw both observations, with BinData attached
    assert len(suggester.observed) == 2
    assert suggester.observed[0][2].rows[0]["bin_number"] == 1


def test_optimize_on_finish_best_moves_to_winner() -> None:
    """on_finish='best' leaves the knob at the highest-objective inputs."""
    s = GeecsSession("Test", tiled=False, mock=True)
    cam = s.detector("UC_Cam", ["Sig"], name="cam")
    knob = s.settable("U_S1H", "Current", name="s1h")
    callback_on_mock_put(
        knob._setpoint, lambda v, **kw: set_mock_value(knob.readback, v)
    )
    set_mock_value(cam.acq_timestamp, 1000.0)
    pacer = start_pacer(s.RE, [(cam, 1000.0)], initial_delay=0.8, interval=0.15)

    suggester = ScriptedSuggester([{"s1h": 0.1}, {"s1h": 0.4}])
    try:
        _uid, history = s.optimize(
            variables={"s1h": knob},
            detectors=[cam],
            objective=lambda b: float(np.median(b.column("s1h-readback"))),
            suggester=suggester,
            shots_per_iteration=1,
            max_iterations=5,
            save_data=False,
            on_finish="best",
        )
    finally:
        pacer.cancel()

    # best objective = 0.4 -> knob moved back to 0.4's inputs (already there)
    assert history[-1]["objective"] == pytest.approx(0.4)
    assert s._read_movable(knob) == pytest.approx(0.4)


def test_optimize_on_finish_initial_restores() -> None:
    """on_finish='initial' restores the pre-optimization value."""
    s = GeecsSession("Test", tiled=False, mock=True)
    cam = s.detector("UC_Cam", ["Sig"], name="cam")
    knob = s.settable("U_S1H", "Current", name="s1h")
    callback_on_mock_put(
        knob._setpoint, lambda v, **kw: set_mock_value(knob.readback, v)
    )
    set_mock_value(knob.readback, 0.07)  # pre-optimization state
    set_mock_value(cam.acq_timestamp, 1000.0)
    pacer = start_pacer(s.RE, [(cam, 1000.0)], initial_delay=0.8, interval=0.15)

    try:
        s.optimize(
            variables={"s1h": knob},
            detectors=[cam],
            objective=lambda b: 1.0,
            suggester=ScriptedSuggester([{"s1h": 0.5}]),
            shots_per_iteration=1,
            max_iterations=3,
            save_data=False,
            on_finish="initial",
        )
    finally:
        pacer.cancel()

    assert s._read_movable(knob) == pytest.approx(0.07)


def test_move_movables_uses_movable_set_not_raw_setpoint(caplog) -> None:
    """_move_movables drives the movable's own set() — confirm semantics apply.

    Review finding (PR #477): ``_move_movables`` (used for optimize
    ``on_finish``) used to put directly to ``m._setpoint``, bypassing a
    movable's own completion semantics entirely — for a
    ``CaConfirmSettable``, that's the exact "the limit register converged
    but nothing physically moved" failure this device exists to catch. A
    fake movable whose ``set()`` always raises but whose ``_setpoint`` put
    always "succeeds" pins that ``_move_movables`` goes through ``set()``:
    with the old raw-signal bypass this test's move would succeed silently;
    with the fix it must log the failure.
    """
    from ophyd_async.core import AsyncStatus

    s = GeecsSession("Test", tiled=False, mock=True)

    class _RawPutOnly:
        """Bare signal stand-in: its own put always 'succeeds'."""

        async def set(self, value):
            return None

    class _NeverConfirmingMovable:
        """A CaConfirmSettable-shaped fake: set() always fails to confirm."""

        def __init__(self):
            self._setpoint = _RawPutOnly()
            self.set_called_with: list[float] = []

        def set(self, value: float) -> AsyncStatus:
            self.set_called_with.append(value)

            async def _fail():
                raise GeecsConfirmTimeoutError(
                    "U_EMQTripletBipolar",
                    "Current_Limit.Ch1",
                    "U_EMQTripletBipolar:Current.Ch1",
                    target=value,
                    current=0.0,
                    timeout=0.3,
                )

            return AsyncStatus(_fail())

    movable = _NeverConfirmingMovable()
    with caplog.at_level(logging.WARNING):
        s._move_movables({"emq1": movable}, {"emq1": 0.5})

    # The movable protocol was used (not a bypass straight to _setpoint) —
    # set() was actually called, and its confirm failure was surfaced.
    assert movable.set_called_with == [0.5]
    warnings = [r for r in caplog.records if "could not move" in r.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].exc_info is not None


def test_adaptive_scan_checkpoints_per_iteration_and_row() -> None:
    """Optimize scans carry the same deferred-pause checkpoints (issue #552).

    Without them, request_pause(defer=True) during an optimize scan would
    never fire — the console would sit in PAUSING while the scan ran on.
    2 iterations × (1 pre-iteration + 2 pre-row) = 6 checkpoints, never
    inside an open event bundle.
    """
    s = GeecsSession("Test", tiled=False, mock=True)
    cam = s.detector("UC_Cam", ["Sig"], name="cam")
    knob = s.settable("U_S1H", "Current", name="s1h")
    callback_on_mock_put(
        knob._setpoint, lambda v, **kw: set_mock_value(knob.readback, v)
    )
    set_mock_value(cam.acq_timestamp, 1000.0)
    commands: list[str] = []
    s.RE.msg_hook = lambda msg: commands.append(msg.command)
    pacer = start_pacer(s.RE, [(cam, 1000.0)], initial_delay=0.8, interval=0.15)
    try:
        s.optimize(
            variables={"s1h": knob},
            detectors=[cam],
            objective=lambda bin_data: 0.0,
            suggester=ScriptedSuggester([{"s1h": 0.1}, {"s1h": 0.4}]),
            shots_per_iteration=2,
            max_iterations=5,
            save_data=False,
        )
    finally:
        pacer.cancel()

    # 2 scripted iterations + the stopping call's own pre-iteration
    # checkpoint (the suggester answers None on the 3rd propose).
    assert commands.count("checkpoint") == 2 * (1 + 2) + 1
    open_bundle = False
    for command in commands:
        if command == "create":
            open_bundle = True
        elif command == "save":
            open_bundle = False
        elif command == "checkpoint":
            assert not open_bundle
