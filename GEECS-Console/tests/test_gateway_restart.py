"""services/gateway_restart.py — the #773 restart verb's pure parts."""

from __future__ import annotations

import pytest

from geecs_console.services.gateway_restart import (
    RESTART_VALUE,
    request_gateway_restart,
    restart_pv,
)


class TestRestartPv:
    def test_is_the_contracts_prefixed_control_pv(self):
        # Through ca_pv/bare_pv (the #490 rule): lowercased, no transport
        # prefix, no :SP suffix — this PV is the control itself.
        assert restart_pv("TestExp") == "testexp:cagateway:restart"

    def test_no_experiment_is_refused_before_any_ca(self):
        with pytest.raises(ValueError, match="no experiment"):
            restart_pv("")

    def test_restart_value_is_the_contract_label(self):
        # PV_CONTRACT.md: enum ["Idle", "Restart"]; "Restart" (or index 1)
        # requests the clean exit-86 shutdown, "Idle" is a no-op.
        assert RESTART_VALUE == "Restart"


class TestRequestGatewayRestart:
    def test_puts_restart_through_the_blessed_primitive(self, monkeypatch):
        import geecs_bluesky.devices.ca.gateway_put as gateway_put

        puts = []

        class FakePut:
            def __init__(self, setpoint_pv=None, *, coerce=None, timeout=None, name=""):
                self.pv = setpoint_pv
                self.coerce = coerce
                self.timeout = timeout

            async def put(self, value):
                puts.append((self.pv, self.coerce(value), self.timeout))

        monkeypatch.setattr(gateway_put, "GatewaySetpointPut", FakePut)
        assert request_gateway_restart("TestExp", timeout=2.5) == (
            "testexp:cagateway:restart"
        )
        assert puts == [("testexp:cagateway:restart", "Restart", 2.5)]

    def test_put_failure_propagates(self, monkeypatch):
        import geecs_bluesky.devices.ca.gateway_put as gateway_put

        class DeadPut:
            def __init__(self, *args, **kwargs):
                pass

            async def put(self, value):
                raise TimeoutError("no gateway answered")

        monkeypatch.setattr(gateway_put, "GatewaySetpointPut", DeadPut)
        with pytest.raises(TimeoutError):
            request_gateway_restart("TestExp")
