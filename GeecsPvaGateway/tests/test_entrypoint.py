"""CLI seam: main() maps restart_requested to RESTART_EXIT_CODE (NSSM's input)."""

from __future__ import annotations

import asyncio

from geecs_pva_gateway import __main__ as cli
from geecs_pva_gateway.config import CameraSpec, PvaGatewayConfig
from geecs_pva_gateway.server import RESTART_EXIT_CODE, GeecsPvaGateway


def _fake_config(experiment: str) -> PvaGatewayConfig:
    return PvaGatewayConfig(
        experiment=experiment,
        cameras=[
            CameraSpec(device="UC_Cam", host="127.0.0.1", port=1, experiment=experiment)
        ],
    )


def _patch_config(monkeypatch) -> None:
    monkeypatch.setattr(
        PvaGatewayConfig,
        "from_geecs_experiment",
        classmethod(lambda cls, experiment, **kw: _fake_config(experiment)),
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
