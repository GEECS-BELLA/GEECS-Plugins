"""GatewayTiledDbHealth.poll(), hermetic: every dependency monkeypatched.

The real probe reaches CA (aioca), Tiled (httpx), and MySQL (GeecsDb); none
are reachable in the test environment, so each check's dependency is patched
to pin the OK / WARN / DOWN / UNKNOWN mapping and the never-raises contract.
"""

import aioca
import httpx
import pytest

from geecs_console.services.health import (
    GatewayTiledDbHealth,
    HealthStatus,
)


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code


@pytest.fixture
def probe():
    return GatewayTiledDbHealth(experiment="Undulator")


def _caget_returning(mapping, default):
    async def _caget(pv, *args, **kwargs):
        for key, value in mapping.items():
            if key in pv:
                return value
        return default

    return _caget


def _caget_raising():
    async def _caget(pv, *args, **kwargs):
        raise RuntimeError("no CA")

    return _caget


class TestGateway:
    def test_heartbeat_ok(self, probe, monkeypatch):
        monkeypatch.setattr(aioca, "caget", _caget_returning({}, 232))
        assert probe._check_gateway() is HealthStatus.OK

    def test_zero_devices_connected_is_warn(self, probe, monkeypatch):
        monkeypatch.setattr(
            aioca,
            "caget",
            _caget_returning({"DEVICES_CONNECTED": 0, "HEARTBEAT": 232}, 232),
        )
        assert probe._check_gateway() is HealthStatus.WARN

    def test_missing_devices_connected_pv_stays_ok(self, probe, monkeypatch):
        # Heartbeat reads; the DEVICES_CONNECTED read fails -> still OK
        # (heartbeat is the primary liveness signal).
        async def _caget(pv, *args, **kwargs):
            if "DEVICES_CONNECTED" in pv:
                raise RuntimeError("no such PV")
            return 232

        monkeypatch.setattr(aioca, "caget", _caget)
        assert probe._check_gateway() is HealthStatus.OK

    def test_read_failure_is_down(self, probe, monkeypatch):
        monkeypatch.setattr(aioca, "caget", _caget_raising())
        assert probe._check_gateway() is HealthStatus.DOWN

    def test_no_experiment_is_unknown(self, monkeypatch):
        probe = GatewayTiledDbHealth(experiment=None)
        # caget must never be reached; make it explode if it is.
        monkeypatch.setattr(aioca, "caget", _caget_raising())
        assert probe._check_gateway() is HealthStatus.UNKNOWN


class TestTiled:
    def test_2xx_is_ok(self, probe, monkeypatch):
        monkeypatch.setattr(probe, "_tiled_uri", lambda: "http://tiled.local:8000")
        monkeypatch.setattr(httpx, "get", lambda *a, **k: _Response(200))
        assert probe._check_tiled() is HealthStatus.OK

    def test_non_2xx_is_down(self, probe, monkeypatch):
        monkeypatch.setattr(probe, "_tiled_uri", lambda: "http://tiled.local:8000")
        monkeypatch.setattr(httpx, "get", lambda *a, **k: _Response(503))
        assert probe._check_tiled() is HealthStatus.DOWN

    def test_exception_is_down(self, probe, monkeypatch):
        monkeypatch.setattr(probe, "_tiled_uri", lambda: "http://tiled.local:8000")

        def _boom(*a, **k):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx, "get", _boom)
        assert probe._check_tiled() is HealthStatus.DOWN

    def test_no_uri_is_unknown(self, probe, monkeypatch):
        monkeypatch.setattr(probe, "_tiled_uri", lambda: None)
        assert probe._check_tiled() is HealthStatus.UNKNOWN


class TestDb:
    def test_query_ok(self, probe, monkeypatch):
        from geecs_ca_gateway.db.geecs_db import GeecsDb

        monkeypatch.setattr(
            GeecsDb, "get_subscribed_variables", classmethod(lambda cls, exp: {})
        )
        assert probe._check_db() is HealthStatus.OK

    def test_query_raise_is_down(self, probe, monkeypatch):
        from geecs_ca_gateway.db.geecs_db import GeecsDb

        def _boom(cls, exp):
            raise RuntimeError("no MySQL")

        monkeypatch.setattr(GeecsDb, "get_subscribed_variables", classmethod(_boom))
        assert probe._check_db() is HealthStatus.DOWN


class TestPollAggregate:
    def test_poll_reports_all_three(self, probe, monkeypatch):
        from geecs_ca_gateway.db.geecs_db import GeecsDb

        monkeypatch.setattr(aioca, "caget", _caget_returning({}, 232))
        monkeypatch.setattr(probe, "_tiled_uri", lambda: "http://tiled.local:8000")
        monkeypatch.setattr(httpx, "get", lambda *a, **k: _Response(200))
        monkeypatch.setattr(
            GeecsDb, "get_subscribed_variables", classmethod(lambda cls, exp: {})
        )
        report = probe.poll()
        assert report.gateway is HealthStatus.OK
        assert report.tiled is HealthStatus.OK
        assert report.db is HealthStatus.OK

    def test_poll_never_raises_when_all_fail(self, probe, monkeypatch):
        from geecs_ca_gateway.db.geecs_db import GeecsDb

        monkeypatch.setattr(aioca, "caget", _caget_raising())
        monkeypatch.setattr(probe, "_tiled_uri", lambda: "http://tiled.local:8000")

        def _boom(*a, **k):
            raise RuntimeError("down")

        monkeypatch.setattr(httpx, "get", _boom)
        monkeypatch.setattr(
            GeecsDb,
            "get_subscribed_variables",
            classmethod(lambda cls, exp: (_ for _ in ()).throw(RuntimeError("db"))),
        )
        report = probe.poll()  # must not raise
        assert report.gateway is HealthStatus.DOWN
        assert report.tiled is HealthStatus.DOWN
        assert report.db is HealthStatus.DOWN

    def test_set_experiment_updates_target(self, probe):
        probe.set_experiment("Bella")
        assert probe.experiment == "Bella"
        probe.experiment = None
        assert probe.experiment is None
