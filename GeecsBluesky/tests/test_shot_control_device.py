"""ShotControl — the trigger box as a Movable + Pausable device (phase 0, #807)."""

from __future__ import annotations

import asyncio

import pytest
from bluesky import RunEngine
from bluesky.plan_stubs import mv
from geecs_schemas.trigger_profile import TriggerState

from geecs_bluesky.devices.shot_control import ShotControl
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.models.shot_control import ShotControlWrites
from tests.ca_mock_helpers import connect_mock

WRITES = ShotControlWrites(
    name="htu_test",
    states={
        # Ordered, multi-device: amplitude first, then the source (the
        # TriggerProfile semantics).
        "SCAN": [("DG", "Amplitude", "4.0"), ("DG", "Source", "edges")],
        "STANDBY": [("DG", "Amplitude", "0.5"), ("DG", "Source", "edges")],
        "OFF": [("DG", "Amplitude", "0.5"), ("DG", "Source", "single")],
        "ARMED": [("DG", "Amplitude", "4.0"), ("DG", "Source", "single")],
        "SINGLESHOT": [("DG", "Fire", "on")],
    },
)


class Recorder:
    """A setter recording every put, in order, across all targets."""

    log: list[tuple[str, str, str]] = []

    def __init__(self, device: str, variable: str) -> None:
        self.key = (device, variable)

    async def put(self, value: str) -> None:
        Recorder.log.append((*self.key, value))


@pytest.fixture
def shot_control(RE: RunEngine) -> ShotControl:
    Recorder.log = []
    sc = ShotControl(
        WRITES, experiment="TestExp", name="shot_control", setter_factory=Recorder
    )
    connect_mock(RE, sc)
    return sc


@pytest.fixture
def RE() -> RunEngine:
    return RunEngine()


def _run(RE: RunEngine, make_awaitable):
    """Call *make_awaitable* inside the RE loop and await what it returns.

    Device methods wrapped in ``AsyncStatus`` create their task at call time,
    so the call itself must happen on the running loop.
    """

    async def call():
        return await make_awaitable()

    return asyncio.run_coroutine_threadsafe(call(), RE._loop).result(timeout=10.0)


def test_set_replays_the_state_writes_in_order(
    RE: RunEngine, shot_control: ShotControl
) -> None:
    RE(mv(shot_control, "ARMED"))
    assert Recorder.log == [("DG", "Amplitude", "4.0"), ("DG", "Source", "single")]
    assert shot_control.standing_state == "ARMED"
    assert _run(RE, lambda: shot_control.state.get_value()) == "ARMED"


def test_enum_and_lowercase_are_accepted(
    RE: RunEngine, shot_control: ShotControl
) -> None:
    RE(mv(shot_control, TriggerState.SCAN))
    RE(mv(shot_control, "standby"))
    assert shot_control.standing_state == "STANDBY"


def test_singleshot_is_momentary(RE: RunEngine, shot_control: ShotControl) -> None:
    """The fire never becomes the standing state."""
    RE(mv(shot_control, "ARMED"))
    RE(mv(shot_control, "SINGLESHOT"))
    assert Recorder.log[-1] == ("DG", "Fire", "on")
    assert shot_control.standing_state == "ARMED"


def test_undefined_state_is_refused(RE: RunEngine) -> None:
    sc = ShotControl(
        ShotControlWrites(name="partial", states={"SCAN": [("DG", "Source", "edges")]}),
        experiment="TestExp",
        name="sc",
        setter_factory=Recorder,
    )
    connect_mock(RE, sc)
    assert sc.defines("SCAN") and not sc.defines("ARMED")
    with pytest.raises(GeecsConfigurationError, match="ARMED"):
        _run(RE, lambda: sc.set("ARMED"))


def test_standing_state_is_configuration(
    RE: RunEngine, shot_control: ShotControl
) -> None:
    RE(mv(shot_control, "SCAN"))
    config = _run(RE, lambda: shot_control.read_configuration())
    assert config["shot_control-state"]["value"] == "SCAN"
    assert _run(RE, lambda: shot_control.read()) == {}  # nothing per event


def test_pause_is_a_no_op_in_strict_mode(
    RE: RunEngine, shot_control: ShotControl
) -> None:
    """ARMED: the plan simply stops firing; the box is left alone (§10.3)."""
    RE(mv(shot_control, "ARMED"))
    n = len(Recorder.log)
    _run(RE, lambda: shot_control.pause())
    _run(RE, lambda: shot_control.resume())
    assert len(Recorder.log) == n
    assert shot_control.standing_state == "ARMED"


def test_pause_stops_edges_in_gated_mode(
    RE: RunEngine, shot_control: ShotControl
) -> None:
    """SCAN: edges flow on their own, so pause → OFF and resume → SCAN."""
    RE(mv(shot_control, "SCAN"))
    _run(RE, lambda: shot_control.pause())
    assert shot_control.standing_state == "OFF"
    assert Recorder.log[-1] == ("DG", "Source", "single")
    _run(RE, lambda: shot_control.resume())
    assert shot_control.standing_state == "SCAN"
    _run(RE, lambda: shot_control.resume())  # idempotent
    assert Recorder.log[-1] == ("DG", "Source", "edges")


def test_run_engine_pauses_the_box_it_set(
    RE: RunEngine, shot_control: ShotControl
) -> None:
    """Having been a `set` target is enough: RE.request_pause reaches pause()."""
    import bluesky.plan_stubs as bps

    def plan():
        yield from bps.open_run()
        yield from bps.mv(shot_control, "SCAN")
        yield from bps.checkpoint()
        yield from bps.pause()
        yield from bps.close_run()

    with pytest.raises(Exception):  # RunEngineInterrupted
        RE(plan())
    assert shot_control.standing_state == "OFF"
    RE.resume()
    assert shot_control.standing_state == "SCAN"


def test_stop_from_a_paused_gated_run_does_not_leak_into_the_next(
    RE: RunEngine, shot_control: ShotControl
) -> None:
    """Review P1: pause in SCAN, stop, then a strict run's pause/resume must not drive SCAN."""
    RE(mv(shot_control, "SCAN"))
    _run(RE, lambda: shot_control.pause())  # SCAN → OFF, remembers SCAN
    assert shot_control.standing_state == "OFF"
    # The run is stopped: finalize drives STANDBY, resume() is never called.
    RE(mv(shot_control, "STANDBY"))
    # Next run, strict: ARMED, pause (no-op), resume — must stay ARMED.
    RE(mv(shot_control, "ARMED"))
    _run(RE, lambda: shot_control.pause())
    _run(RE, lambda: shot_control.resume())
    assert shot_control.standing_state == "ARMED"
    assert Recorder.log[-2:] == [("DG", "Amplitude", "4.0"), ("DG", "Source", "single")]


def test_pause_from_standby_quiesces_too(
    RE: RunEngine, shot_control: ShotControl
) -> None:
    """STANDBY passes external edges (§11.1): a pause there drives OFF as well."""
    RE(mv(shot_control, "STANDBY"))
    _run(RE, lambda: shot_control.pause())
    assert shot_control.standing_state == "OFF"
    _run(RE, lambda: shot_control.resume())
    assert shot_control.standing_state == "STANDBY"


def test_pause_without_off_logs_and_never_raises(RE: RunEngine, caplog) -> None:
    """A profile with no OFF: the pause is logged, the box left alone, nothing raised."""
    sc = ShotControl(
        ShotControlWrites(name="no_off", states={"SCAN": [("DG", "Source", "edges")]}),
        experiment="TestExp",
        name="sc",
        setter_factory=Recorder,
    )
    connect_mock(RE, sc)
    RE(mv(sc, "SCAN"))
    n = len(Recorder.log)
    with caplog.at_level("WARNING"):
        _run(RE, lambda: sc.pause())
        _run(RE, lambda: sc.resume())
    assert len(Recorder.log) == n
    assert sc.standing_state == "SCAN"
    assert "defines no OFF" in caplog.text


def test_unknown_state_name_is_a_configuration_error(
    RE: RunEngine, shot_control: ShotControl
) -> None:
    assert shot_control.defines("armed") and not shot_control.defines("FOO")
    with pytest.raises(GeecsConfigurationError, match="not a trigger state"):
        _run(RE, lambda: shot_control.set("FOO"))


def test_standing_state_has_one_source(
    RE: RunEngine, shot_control: ShotControl
) -> None:
    """The composed controller's last_state is the field; the config signal mirrors it."""
    RE(mv(shot_control, "SCAN"))
    assert shot_control._controller.last_state == "SCAN"
    assert _run(RE, lambda: shot_control.state.get_value()) == "SCAN"
    RE(mv(shot_control, "SINGLESHOT"))
    assert shot_control._controller.last_state == "SCAN"
