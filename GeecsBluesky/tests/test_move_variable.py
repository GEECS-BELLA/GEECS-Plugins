"""On-demand manual moves (`move_variable`) — hermetic.

Pins the manual-move contract, the counterpart of ``test_run_action``:

- ``GeecsSession.move_variable`` dispatches through the one
  ``build_movable`` seam (plain / confirm / pseudo — scan-identical
  completion semantics), builds a **fresh movable per call** (a relative
  pseudo re-baselines from the targets' current positions on every manual
  bump), and returns the ``{"variable", "kind", "value", "targets"}``
  summary.
- Refusal while a scan owns the engine uses the **exact** message
  ``"scan in progress — move not started"`` (the GUI surfaces it
  verbatim).
- A non-finite value is refused before any device is built.
- An **unserved** target (any pseudo component) is refused before any
  device is built, in words ending ``"— move not started"`` (#772) — the
  same served-set check the scan path runs, so a typo'd variable never
  surfaces as a 20 s ``NotConnectedError`` naming a PV.  Served set
  unknown (DB unreachable) → the move proceeds with a warning.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("aioca")  # session is CA-only

from geecs_bluesky import scan_request_runner as runner  # noqa: E402
from geecs_bluesky.exceptions import (  # noqa: E402
    GeecsConfigurationError,
    GeecsUnservedVariablesError,
)
from geecs_bluesky.session import GeecsSession  # noqa: E402
from geecs_schemas import PseudoScanVariable, ScanVariable  # noqa: E402

REFUSAL = "scan in progress — move not started"


class _CatalogResolver:
    """Minimal in-memory ConfigResolver over a scan-variable dict."""

    def __init__(self, variables: dict) -> None:
        self._variables = variables

    def resolve_scan_variable(self, name: str):
        try:
            return self._variables[name]
        except KeyError:
            raise GeecsConfigurationError(f"unknown scan variable {name!r}") from None


BUMP_X = PseudoScanVariable(
    kind="pseudo",
    mode="relative",
    targets=[
        {"target": "U_S3H:Current", "forward": "x * 1"},
        {"target": "U_S4H:Current", "forward": "x * -2"},
    ],
)


@pytest.fixture(autouse=True)
def _served_set_unknown(monkeypatch):
    """Default: no served-set provider (the check skips) — no DB in tests."""
    monkeypatch.setattr(runner, "make_served_set_provider", lambda session: None)


def _serve(monkeypatch, served: dict) -> None:
    provider = SimpleNamespace(served_by_device=lambda: served)
    monkeypatch.setattr(runner, "make_served_set_provider", lambda session: provider)


@pytest.fixture()
def session():
    s = GeecsSession("TestExp", tiled=False, mock=True)
    yield s


def _read_signal(s: GeecsSession, signal) -> float:
    return asyncio.run_coroutine_threadsafe(signal.get_value(), s.RE._loop).result(
        timeout=10.0
    )


def test_raw_device_variable_moves_plain_setpoint(session, monkeypatch) -> None:
    """A raw Device:Variable name moves via plain settable semantics."""
    built = []
    real = session.settable

    def spy(device, variable, **kwargs):
        movable = real(device, variable, **kwargs)
        built.append(movable)
        return movable

    monkeypatch.setattr(session, "settable", spy)
    result = session.move_variable("U_S1H:Current", 2.5)

    assert result == {
        "variable": "U_S1H:Current",
        "kind": "setpoint",
        "value": 2.5,
        "targets": {"U_S1H:Current": 2.5},
    }
    assert len(built) == 1
    assert _read_signal(session, built[0]._setpoint) == 2.5


def test_pseudo_move_fans_out_and_reports_commanded_targets(session) -> None:
    """A relative pseudo fans out baseline + forward(x) and reports it.

    Mock readbacks start at 0.0, so commanded = the pure offsets.
    """
    result = session.move_variable("bump_x", 0.5, _CatalogResolver({"bump_x": BUMP_X}))

    assert result["kind"] == "pseudo (relative)"
    assert result["value"] == 0.5
    assert result["targets"] == {"U_S3H:Current": 0.5, "U_S4H:Current": -1.0}


def test_each_move_builds_a_fresh_movable(session, monkeypatch) -> None:
    """Two manual moves = two movables (relative re-baselines every call)."""
    calls = []
    real = session.pseudo_movable

    def spy(*args, **kwargs):
        movable = real(*args, **kwargs)
        calls.append(movable)
        return movable

    monkeypatch.setattr(session, "pseudo_movable", spy)
    resolver = _CatalogResolver({"bump_x": BUMP_X})
    session.move_variable("bump_x", 0.5, resolver)
    session.move_variable("bump_x", -0.5, resolver)

    assert len(calls) == 2
    assert calls[0] is not calls[1]


def test_confirm_variable_dispatches_to_confirm_settable(session, monkeypatch) -> None:
    """A catalog entry with `confirm` moves through CaConfirmSettable."""
    from ophyd_async.core import AsyncStatus

    recorded = {}

    class _FakeConfirm:
        def set(self, value):
            recorded["value"] = value

            async def _ok():
                return None

            return AsyncStatus(_ok())

    def fake_confirm_settable(device, variable, *, confirm_device, confirm_variable):
        recorded["target"] = (device, variable, confirm_device, confirm_variable)
        return _FakeConfirm()

    monkeypatch.setattr(session, "confirm_settable", fake_confirm_settable)
    emq = ScanVariable(
        target="U_EMQTripletBipolar:Current_Limit.Ch1",
        confirm="U_EMQTripletBipolar:Current.Ch1",
    )
    result = session.move_variable("emq1", 1.5, _CatalogResolver({"emq1": emq}))

    assert recorded["target"] == (
        "U_EMQTripletBipolar",
        "Current_Limit.Ch1",
        "U_EMQTripletBipolar",
        "Current.Ch1",
    )
    assert recorded["value"] == 1.5
    assert result["kind"] == "confirm"


def test_session_refuses_when_run_engine_not_idle() -> None:
    s = GeecsSession("TestExp", tiled=False, mock=True)
    s.RE = SimpleNamespace(state="running")  # a scan owns the engine

    class _Untouchable:
        def resolve_scan_variable(self, name: str):
            raise AssertionError("refusal must come before resolution")

    with pytest.raises(RuntimeError) as excinfo:
        s.move_variable("anything", 1.0, _Untouchable())
    assert str(excinfo.value) == REFUSAL


def test_non_finite_value_refused_before_any_device(session) -> None:
    class _Untouchable:
        def resolve_scan_variable(self, name: str):
            raise AssertionError("refusal must come before resolution")

    with pytest.raises(GeecsConfigurationError, match="non-finite"):
        session.move_variable("bump_x", float("nan"), _Untouchable())


# ---------------------------------------------------------------------------
# Mutual exclusion + timeout + remaining dispatch (review findings, PR #597)
# ---------------------------------------------------------------------------


def test_motor_variable_dispatches_to_motor(session, monkeypatch) -> None:
    """A catalog entry with kind=motor moves through CaMotor semantics."""
    from ophyd_async.core import AsyncStatus

    recorded = {}

    class _FakeMotor:
        def set(self, value):
            recorded["value"] = value

            async def _ok():
                return None

            return AsyncStatus(_ok())

    def fake_motor(device, variable):
        recorded["target"] = (device, variable)
        return _FakeMotor()

    monkeypatch.setattr(session, "motor", fake_motor)
    jet = ScanVariable(target="U_ESP_JetXYZ:Position.Axis 3", kind="motor")
    result = session.move_variable("jet_z", 4.5, _CatalogResolver({"jet_z": jet}))

    assert recorded["target"] == ("U_ESP_JetXYZ", "Position.Axis 3")
    assert recorded["value"] == 4.5
    assert result["kind"] == "motor"


def test_second_move_refused_while_one_is_running(session) -> None:
    """The manual-move lock closes the double-click race."""
    assert session._manual_move_lock.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            session.move_variable("U_S1H:Current", 1.0)
        assert str(excinfo.value) == "manual move in progress — move not started"
    finally:
        session._manual_move_lock.release()


def test_scan_and_optimize_and_action_refused_during_manual_move(session) -> None:
    """The exclusion is mutual: nothing else starts while a move holds the lock.

    A manual move is not a plan, so the RE stays idle for its duration —
    without these guards a scan/action could start mid-move (review
    finding 1, PR #597).
    """
    assert session._manual_move_lock.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="manual move in progress — scan"):
            session.scan(detectors=[object()], shots_per_step=1)
        with pytest.raises(
            RuntimeError, match="manual move in progress — optimization"
        ):
            session.optimize(
                variables={},
                detectors=[],
                objective=lambda b: 0.0,
                suggester=None,
            )
        with pytest.raises(RuntimeError, match="manual move in progress — action"):
            session.run_action("anything", None)
    finally:
        session._manual_move_lock.release()


def test_timeout_cancels_the_move_and_raises_timeout_error(
    session, monkeypatch
) -> None:
    """A stalled move raises TimeoutError and cancels the in-flight coroutine."""
    import asyncio as _asyncio
    import time

    from ophyd_async.core import AsyncStatus

    cancelled = {"flag": False}

    class _StalledMovable:
        def set(self, value):
            async def _hang():
                try:
                    await _asyncio.sleep(3600)
                except _asyncio.CancelledError:
                    cancelled["flag"] = True
                    raise

            return AsyncStatus(_hang())

    monkeypatch.setattr(session, "settable", lambda device, variable: _StalledMovable())
    with pytest.raises(TimeoutError, match="did not complete within"):
        session.move_variable("U_S1H:Current", 1.0, timeout=0.2)
    # The lock must be released after the failure (next move can start)...
    assert not session._manual_move_lock.locked()
    # ...and the coroutine must actually have been cancelled.
    for _ in range(50):
        if cancelled["flag"]:
            break
        time.sleep(0.02)
    assert cancelled["flag"]


def test_non_numeric_value_refused_as_configuration_error(session) -> None:
    class _Untouchable:
        def resolve_scan_variable(self, name: str):
            raise AssertionError("refusal must come before resolution")

    with pytest.raises(GeecsConfigurationError, match="not a number"):
        session.move_variable("bump_x", "sideways", _Untouchable())


# ---------------------------------------------------------------------------
# Unserved targets refuse pre-build (#772 part 2)
# ---------------------------------------------------------------------------


def _forbid_device_build(session, monkeypatch) -> None:
    for factory in ("settable", "motor", "confirm_settable", "pseudo_movable"):
        monkeypatch.setattr(
            session,
            factory,
            lambda *a, **k: pytest.fail(
                "no device may be built for an unserved target"
            ),
        )


def test_unserved_raw_target_refused_before_any_device(session, monkeypatch) -> None:
    """The live #772 shape: a near-miss variable name, refused in words."""
    _serve(monkeypatch, {"U_Grating2Rotation": {"Position.Axis 1", "CONNECTED"}})
    _forbid_device_build(session, monkeypatch)

    with pytest.raises(GeecsUnservedVariablesError) as excinfo:
        session.move_variable("U_Grating2Rotation:Position.Axis1", 1.0)

    message = str(excinfo.value)
    assert "U_Grating2Rotation:Position.Axis1 is not served by the gateway" in message
    assert message.endswith("— move not started")
    assert "TestExp" in message
    assert excinfo.value.unserved == {"U_Grating2Rotation": ["Position.Axis1"]}
    assert not session._manual_move_lock.locked()


def test_unserved_pseudo_component_refused_naming_the_component(
    session, monkeypatch
) -> None:
    """A pseudo is vetted component by component; the bad one is named."""
    _serve(monkeypatch, {"U_S3H": {"Current"}, "U_S4H": {"Voltage"}})
    _forbid_device_build(session, monkeypatch)

    with pytest.raises(GeecsUnservedVariablesError) as excinfo:
        session.move_variable("bump_x", 0.5, _CatalogResolver({"bump_x": BUMP_X}))

    assert "U_S4H:Current is not served" in str(excinfo.value)
    assert "U_S3H" not in str(excinfo.value).split(" is not served")[0]


def test_unknown_device_refused_too(session, monkeypatch) -> None:
    """A device absent from the served set is unserved (not a KeyError)."""
    _serve(monkeypatch, {"U_S1H": {"Current"}})
    _forbid_device_build(session, monkeypatch)

    with pytest.raises(GeecsUnservedVariablesError, match="U_Nope:Current is not"):
        session.move_variable("U_Nope:Current", 1.0)


def test_served_target_moves(session, monkeypatch) -> None:
    _serve(monkeypatch, {"U_S1H": {"Current"}})
    result = session.move_variable("U_S1H:Current", 2.5)
    assert result["targets"] == {"U_S1H:Current": 2.5}


def test_unknown_served_set_proceeds_with_a_warning(
    session, monkeypatch, caplog
) -> None:
    """DB unreachable → served set None → the move is never refused for it."""
    provider = SimpleNamespace(served_by_device=lambda: None)
    monkeypatch.setattr(runner, "make_served_set_provider", lambda session: provider)
    with caplog.at_level("WARNING", logger="geecs_bluesky.scan_request_runner"):
        result = session.move_variable("U_S1H:Current", 1.0)
    assert result["value"] == 1.0
    assert any(
        "served set could not be determined" in r.message for r in caplog.records
    )


def test_movable_target_pairs_enumerates_writes() -> None:
    from geecs_bluesky.scan_request_runner import (
        PlainMovableTarget,
        movable_target_pairs,
        resolve_movable_target,
    )

    plain = PlainMovableTarget(
        "U_EMQ", "Current_Limit.Ch1", "setpoint", "U_EMQ:Current.Ch1"
    )
    assert movable_target_pairs(plain) == [
        ("U_EMQ", "Current_Limit.Ch1"),
        ("U_EMQ", "Current.Ch1"),  # the confirm readback connects too
    ]
    bare = PlainMovableTarget("U_S1H", "Current", "setpoint", None)
    assert movable_target_pairs(bare) == [("U_S1H", "Current")]
    pseudo = resolve_movable_target(BUMP_X, "bump_x")
    assert movable_target_pairs(pseudo) == [("U_S3H", "Current"), ("U_S4H", "Current")]


def test_unserved_confirm_readback_refused(session, monkeypatch) -> None:
    """The confirm variable is monitored, so it must be served as well."""
    _serve(monkeypatch, {"U_EMQTripletBipolar": {"Current_Limit.Ch1"}})
    _forbid_device_build(session, monkeypatch)
    emq = ScanVariable(
        target="U_EMQTripletBipolar:Current_Limit.Ch1",
        confirm="U_EMQTripletBipolar:Current.Ch1",
    )
    with pytest.raises(GeecsUnservedVariablesError, match="Current.Ch1 is not served"):
        session.move_variable("emq1", 1.5, _CatalogResolver({"emq1": emq}))
