"""services/gateway_restart.py — the #773 restart verb's pure parts."""

from __future__ import annotations

import inspect

import pytest

from geecs_console.services import gateway_restart as module
from geecs_console.services.device_panel import StubDevicePanel
from geecs_console.services.gateway_restart import (
    RESTART_VALUE,
    request_gateway_restart,
    restart_pv,
)


class RecordingBackend:
    """DevicePanelBackend stand-in recording ``put_pv`` calls."""

    def __init__(self, error: Exception | None = None):
        self.puts = []
        self.error = error

    def put_pv(self, pv, value, *, timeout=None, name=""):
        if self.error is not None:
            raise self.error
        self.puts.append((pv, value, timeout, name))


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
    def test_puts_restart_through_the_backend_seam(self):
        backend = RecordingBackend()
        assert request_gateway_restart("TestExp", backend=backend, timeout=2.5) == (
            "testexp:cagateway:restart"
        )
        assert backend.puts == [
            ("testexp:cagateway:restart", "Restart", 2.5, "cagateway:restart")
        ]

    def test_put_failure_propagates(self):
        backend = RecordingBackend(error=TimeoutError("no gateway answered"))
        with pytest.raises(TimeoutError):
            request_gateway_restart("TestExp", backend=backend)
        assert backend.puts == []

    def test_offline_stub_refuses_with_a_clear_message(self):
        with pytest.raises(RuntimeError, match="not wired"):
            request_gateway_restart("TestExp", backend=StubDevicePanel())

    def test_no_per_call_event_loop(self):
        # Review of PR #796: a put under ``asyncio.run`` strands its aioca
        # channel on a loop that is then closed; the gateway's own exit
        # (this click) fires CONN_DOWN on it and the CA thread prints
        # "RuntimeError: Event loop is closed".  The put must ride the
        # device-panel backend's persistent loop instead.
        source = inspect.getsource(module)
        assert "asyncio.run(" not in source
        assert "import asyncio" not in source
