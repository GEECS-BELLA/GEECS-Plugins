"""install_interlocks over a mock namespace: what is armed, what is not, and a trip's action plan."""

from __future__ import annotations

import logging
import threading
import time

import pytest

pytest.importorskip("aioca")

import bluesky.plans as bp
from bluesky import RunEngine
from geecs_schemas import ActionPlanLibrary, InterlockSet
from ophyd_async.core import get_mock_put, set_mock_value

from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.interlocks import TRIPPED_LOG_PREFIX, install_interlocks
from geecs_bluesky.namespace import DeviceRoster, GeecsNamespace
from geecs_bluesky.run_engine import make_run_engine
from tests.ca_mock_helpers import follow_setpoint
from tests.test_connect_on_demand import row

ROSTER = DeviceRoster(
    experiment="TestExp",
    variables={
        "U_Gauge": [row("Pressure"), row("acq_timestamp")],
        "U_Plain": [row("Pressure")],
        "U_S1H": [row("Current", settable=True, tolerance=0.01), row("Voltage")],
    },
    types={"U_Gauge": "Gauge", "U_Plain": "Gauge", "U_S1H": "Magnet PS"},
    subscribed={"U_Gauge": ["Pressure"], "U_Plain": ["Pressure"], "U_S1H": ["Voltage"]},
)

LIBRARY = ActionPlanLibrary.model_validate(
    {
        "plans": {
            "zero_magnet": {
                "steps": [
                    {"do": "set", "device": "U_S1H", "variable": "Current", "value": 0}
                ]
            },
            "typo": {
                "steps": [
                    {"do": "set", "device": "U_S1H", "variable": "Curent", "value": 0}
                ]
            },
        }
    }
)


class Resolver:
    def resolve_action_plan(self, name):
        try:
            return LIBRARY.plans[name]
        except KeyError:
            raise GeecsConfigurationError(f"no plan {name!r}") from None

    def action_plan_registry(self):
        return dict(LIBRARY.plans)


def _threshold(name: str, device: str = "U_Gauge", **kw) -> dict:
    return {
        "name": name,
        "condition": {
            "kind": "threshold",
            "device": device,
            "variable": "Pressure",
            "operator": ">",
            "limit": 1e-5,
        },
        **kw,
    }


@pytest.fixture
def RE() -> RunEngine:
    return make_run_engine(mock=True)


@pytest.fixture
def namespace() -> GeecsNamespace:
    return GeecsNamespace(ROSTER)


def _install(RE, namespace, *interlocks):
    return install_interlocks(
        RE,
        InterlockSet.model_validate({"interlocks": list(interlocks)}),
        namespace,
        Resolver(),
        mock=True,
        timeout=5.0,
    )


def test_what_is_armed_and_what_is_not(RE, namespace) -> None:
    registry = _install(
        RE,
        namespace,
        _threshold("vacuum", on_trip="zero_magnet"),
        {
            "name": "magnet tracks",
            "condition": {
                "kind": "alignment",
                "device": "U_S1H",
                "variable": "Current",
                "target_variable": "Voltage",
                "tolerance": 1.0,
            },
        },
        {
            "name": "gauge alive",
            "condition": {"kind": "stale", "device": "U_Gauge", "max_age_s": 5},
        },
        _threshold("off", enabled=False),
        _threshold("unknown device", device="U_Nowhere"),
        _threshold("unknown plan", on_trip="nope"),
        _threshold("bad target", on_trip="typo"),
        {
            "name": "no stamp",
            "condition": {"kind": "stale", "device": "U_Plain", "max_age_s": 5},
        },
    )
    assert [s.name for s in registry.suspenders] == [
        "vacuum",
        "magnet tracks",
        "gauge alive",
    ]
    assert list(registry.unarmed) == [
        "unknown device",
        "unknown plan",
        "bad target",
        "no stamp",
    ]
    assert set(RE.suspenders) == set(registry.suspenders)
    statuses = {s.name: s for s in registry.snapshot()}
    assert not statuses["vacuum"].unsafe  # mock Pressure 0.0
    assert not statuses["magnet tracks"].unsafe  # 0.0 vs 0.0
    assert statuses["no stamp"].unsafe and not statuses["no stamp"].armed
    assert "serves no acq_timestamp" in statuses["no stamp"].diagnostic
    assert "U_Nowhere" in registry.unarmed["unknown device"]
    assert "nope" in registry.unarmed["unknown plan"]
    assert "Curent" in registry.unarmed["bad target"]
    assert "off" not in statuses
    registry.remove(RE)
    assert RE.suspenders == ()


def test_a_trip_runs_the_action_plan_and_logs(RE, namespace, caplog) -> None:
    registry = _install(RE, namespace, _threshold("vacuum", on_trip="zero_magnet"))
    gauge, magnet = namespace["U_Gauge"], namespace["U_S1H"].current
    set_mock_value(magnet.readback_signal, 3.0)
    follow_setpoint(magnet)

    def trip_then_clear() -> None:
        time.sleep(0.15)
        set_mock_value(gauge.pressure, 1e-3)
        time.sleep(0.3)
        set_mock_value(gauge.pressure, 1e-6)

    threading.Thread(target=trip_then_clear, daemon=True).start()
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.interlocks"):
        RE(bp.count([namespace["U_Plain"]], num=8, delay=0.05))
    get_mock_put(magnet._setpoint).assert_called_once()
    assert get_mock_put(magnet._setpoint).call_args.args[0] == 0
    assert not registry.suspenders[0].tripped
    messages = [r.getMessage() for r in caplog.records]
    assert f"{TRIPPED_LOG_PREFIX}: vacuum: Pressure=0.001 (limit: >1e-05)" in messages
    assert any("running action plan 'zero_magnet'" in m for m in messages)
