"""CLI seam: main() maps restart_requested to RESTART_EXIT_CODE (NSSM's input)."""

from __future__ import annotations

import asyncio

from geecs_pva_gateway import __main__ as cli
from geecs_pva_gateway.config import DeviceSpec, PvaGatewayConfig
from geecs_pva_gateway.server import (
    RESTART_EXIT_CODE,
    ROSTER_INTERVAL_S,
    GeecsPvaGateway,
)


def _fake_config(experiment: str) -> PvaGatewayConfig:
    return PvaGatewayConfig(
        experiment=experiment,
        devices=[
            DeviceSpec(
                device="UC_Cam",
                host="127.0.0.1",
                port=1,
                experiment=experiment,
                image_variables=["image"],
            )
        ],
    )


def _patch_config(monkeypatch) -> None:
    monkeypatch.setattr(
        PvaGatewayConfig,
        "from_geecs_experiment",
        classmethod(lambda cls, experiment, **kw: _fake_config(experiment)),
    )


def test_main_idles_with_no_devices_instead_of_exiting(monkeypatch, caplog, capsys):
    """A host with nothing to serve keeps the instance PVs up (exit 0, a WARNING),
    so the fleet screen sees it and :restart can pick up a newly enabled device."""
    import logging

    monkeypatch.setattr(
        PvaGatewayConfig,
        "from_geecs_experiment",
        classmethod(
            lambda cls, experiment, **kw: PvaGatewayConfig(
                experiment=experiment, host="192.168.7.168"
            )
        ),
    )
    with caplog.at_level(logging.WARNING):
        assert cli.main(["--experiment", "testexp", "--list"]) == 0
    assert capsys.readouterr().out == ""  # nothing served, nothing listed
    assert any(
        "serving the instance PVs only" in r.getMessage() for r in caplog.records
    )


def test_main_returns_restart_exit_code(monkeypatch):
    """A restart-requested run exits with the code NSSM restarts on."""
    _patch_config(monkeypatch)

    async def fake_run(self, *, isolate: bool = False) -> None:
        self._restart_event = asyncio.Event()
        self._restart_event.set()

    monkeypatch.setattr(GeecsPvaGateway, "run", fake_run)
    assert cli.main(["--experiment", "testexp"]) == RESTART_EXIT_CODE


def test_main_returns_zero_on_plain_exit(monkeypatch):
    _patch_config(monkeypatch)

    async def fake_run(self, *, isolate: bool = False) -> None:
        self._restart_event = asyncio.Event()

    monkeypatch.setattr(GeecsPvaGateway, "run", fake_run)
    assert cli.main(["--experiment", "testexp"]) == 0


def test_main_list_prints_pvs_and_exits_zero(monkeypatch, capsys):
    _patch_config(monkeypatch)
    assert cli.main(["--experiment", "testexp", "--list"]) == 0
    assert "testexp:uc_cam:image" in capsys.readouterr().out


def test_main_wires_the_roster_re_read_with_the_startup_scoping(monkeypatch):
    """The served set is re-read with the startup call — same --host and
    --devices scoping — at --roster-interval (default ROSTER_INTERVAL_S)."""
    calls: list[dict] = []

    def recording_build(cls, experiment: str, **kw) -> PvaGatewayConfig:
        calls.append(kw)
        return _fake_config(experiment)

    monkeypatch.setattr(
        PvaGatewayConfig, "from_geecs_experiment", classmethod(recording_build)
    )
    built: dict = {}
    original_init = GeecsPvaGateway.__init__

    def recording_init(self, config, **kw):
        built.update(kw)
        original_init(self, config, **kw)

    monkeypatch.setattr(GeecsPvaGateway, "__init__", recording_init)

    async def fake_run(self, *, isolate: bool = False) -> None:
        self._restart_event = asyncio.Event()

    monkeypatch.setattr(GeecsPvaGateway, "run", fake_run)
    argv = ["--experiment", "testexp", "--host", "10.0.0.1", "--devices", "UC_Cam"]
    assert cli.main([*argv, "--roster-interval", "5"]) == 0
    assert built["roster_interval_s"] == 5.0
    assert built["roster_resolver"]() == _fake_config("testexp").devices
    startup = {"host": "10.0.0.1", "devices": ["UC_Cam"]}
    # Verbatim, plus the re-read's strict scope anchored on the identity
    # host (True when the startup config has none, as _fake_config's).
    assert calls == [startup, {**startup, "strict_scope": True}]
    assert cli.main(argv) == 0
    assert built["roster_interval_s"] == ROSTER_INTERVAL_S == 60.0
