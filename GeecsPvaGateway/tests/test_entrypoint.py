"""CLI seam: main() maps restart_requested to RESTART_EXIT_CODE (NSSM's input)."""

from __future__ import annotations

import asyncio

from geecs_pva_gateway import __main__ as cli
from geecs_pva_gateway.config import DeviceSpec, PvaGatewayConfig
from geecs_pva_gateway.server import RESTART_EXIT_CODE, GeecsPvaGateway


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
