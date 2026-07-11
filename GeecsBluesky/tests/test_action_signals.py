"""CaActionSignalFactory — the CA-backed production SettableFactory.

Pins the CA-native put convention (raw wire-string caput, so numeric ``:SP``
PVs work — the bug found by the first hardware set-step on a float PV) and
the dtype-inferred probe/readable behavior.
"""

from __future__ import annotations

import pytest

pytest.importorskip("aioca")  # devices are CA-backed

import bluesky.plan_stubs as bps  # noqa: E402
from bluesky.run_engine import RunEngine  # noqa: E402

from geecs_bluesky.devices.ca.action_signals import CaActionSignalFactory  # noqa: E402


class _Recorder:
    """Fake session connector: records signals, never touches CA."""

    def __init__(self, fail: bool = False) -> None:
        self.connected: list = []
        self.fail = fail

    def __call__(self, signal) -> None:
        if self.fail:
            raise ConnectionError(f"no such PV: {signal.name}")
        self.connected.append(signal)


def test_settable_pv_has_no_transport_scheme() -> None:
    """The raw-CA put must use the bare PV name — ophyd strips ``ca://``,
    aioca does not (a schemed name searches forever; issue #490)."""
    factory = CaActionSignalFactory("Undulator", _Recorder(), mock=True)
    settable = factory.get_settable("U_ESP_JetXYZ", "Position.Axis 1")
    assert not settable._pv.startswith("ca://")
    assert settable._pv == "Undulator:U_ESP_JetXYZ:Position_Axis_1:SP"
    # the probe keeps the ophyd signal-URI form (ophyd handles the scheme)


def test_settable_puts_wire_string_on_numeric_value() -> None:
    """A float action value is put as its wire string (works on float PVs)."""
    connect = _Recorder()
    factory = CaActionSignalFactory("Undulator", connect, mock=True)
    settable = factory.get_settable("U_ESP_JetXYZ", "Position.Axis 1")

    RE = RunEngine(context_managers=[])

    def plan():
        yield from bps.abs_set(settable, 5.0, wait=True)

    RE(plan())
    assert settable.last_mock_put == "5.0"
    # dtype-inferred probe was connected (the pre-claim fail-fast)
    assert len(connect.connected) == 1
    assert connect.connected[0].name.endswith("_probe")


def test_settable_puts_enum_label_unchanged() -> None:
    connect = _Recorder()
    factory = CaActionSignalFactory("Undulator", connect, mock=True)
    settable = factory.get_settable("U_DG645_ShotControl", "Trigger.Source")

    RE = RunEngine(context_managers=[])
    RE(bps.abs_set(settable, "External rising edges", wait=True))
    assert settable.last_mock_put == "External rising edges"


def test_settable_cached_per_target() -> None:
    connect = _Recorder()
    factory = CaActionSignalFactory("Undulator", connect, mock=True)
    a = factory.get_settable("U_A", "V")
    b = factory.get_settable("U_A", "V")
    assert a is b
    assert len(connect.connected) == 1  # one probe, cached


def test_settable_failfast_when_probe_cannot_connect() -> None:
    """An unreachable :SP fails at creation (pre-claim), not mid-plan."""
    factory = CaActionSignalFactory("Undulator", _Recorder(fail=True), mock=True)
    with pytest.raises(ConnectionError):
        factory.get_settable("U_Nope", "Missing")


async def test_readable_is_dtype_inferred() -> None:
    """get_readable infers the PV's native type (float stays float)."""
    from ophyd_async.testing import set_mock_value

    created: list = []
    factory = CaActionSignalFactory("Undulator", created.append, mock=True)
    signal = factory.get_readable("U_EMQTripletBipolar", "Current.Ch1")
    await signal.connect(mock=True)
    set_mock_value(signal, 2.5)
    assert await signal.get_value() == 2.5


async def test_disconnect_drops_caches() -> None:
    factory = CaActionSignalFactory("Undulator", _Recorder(), mock=True)
    factory.get_settable("U_A", "V")
    factory.get_readable("U_A", "V")
    await factory.disconnect()
    assert factory._settables == {} and factory._readables == {}
    assert factory._probes == {}
