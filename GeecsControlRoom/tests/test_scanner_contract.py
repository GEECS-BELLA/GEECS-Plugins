"""The home's strip against GeecsScanner's own app, in process: the fields the strip reads.

The scanner is built the way its own suite builds it (``DemoQueueClient`` with
``period=0``, a manager in memory that moves only when the test calls
``step()``) and mounted on an ``httpx.ASGITransport`` that the Control Room's
scanner client is pointed at. A route or field change in ``/api/status`` or
``/api/progress`` fails this suite in the same PR. geecs-scanner is in the
optional ``contract`` group (skipped without it): ``geecs_control_room`` never
imports it (the ``control-room-edges`` contract).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from geecs_control_room.app import create_app
from geecs_control_room.module_client import ModuleClient
from geecs_control_room.settings import Settings

pytest.importorskip(
    "geecs_scanner",
    reason="the contract group is not installed: poetry install --with dev,contract",
)

FIRST_SCAN = 46
SHOTS = 5


@pytest.fixture
def scanner(tmp_path: Path):
    """The scanner's app over its demo manager, and the manager to step."""
    from geecs_scanner.service import ScannerService, StreamCache
    from geecs_scanner.service.demo import (
        DemoQueueClient,
        DemoReadback,
        DemoResolver,
        DemoSettables,
        demo_preflight,
    )
    from geecs_scanner.web import create_app as create_scanner_app

    streams = StreamCache(clock=lambda: 1_000.0)
    manager = DemoQueueClient(streams, period=0.0, first_scan=FIRST_SCAN, user="control-room test")  # fmt: skip
    service = ScannerService(
        manager,
        DemoResolver(),
        experiment="Demo",
        identity="control-room test",
        streams=streams,
        preflight=demo_preflight,
        version="0.0.0+test",
        settables=DemoSettables(),
        readback=DemoReadback(manager),
        heartbeat_path=tmp_path / "scanner-heartbeat.json",
    )
    return manager, create_scanner_app(service)


def _strip(settings: Settings, module: ModuleClient, scanner_app) -> dict:
    client = httpx.AsyncClient(base_url="http://scanner.test", transport=httpx.ASGITransport(app=scanner_app))  # fmt: skip
    with TestClient(create_app(settings, module, client)) as c:
        reply = c.get("/api/home/strip")
    assert reply.status_code == 200, reply.text
    return reply.json()


def test_an_idle_manager_is_the_ok_run_word(settings: Settings, module: ModuleClient, scanner) -> None:  # fmt: skip
    _, app = scanner
    s = _strip(settings, module, app)
    assert (s["scanner"], s["run"], s["kit"]) == ("ok", "idle", "ok")
    assert s["shots_done"] == 0


def test_a_running_scan_fills_the_strip(settings: Settings, module: ModuleClient, scanner) -> None:  # fmt: skip
    manager, app = scanner
    assert manager.submit_plan("count", kwargs={"detectors": ["U_ICT"], "num": SHOTS}).ok  # fmt: skip
    manager.step()
    manager.step()
    s = _strip(settings, module, app)
    assert (s["scanner"], s["run"], s["kit"]) == ("ok", "running", "running")
    assert s["scan_number"] == FIRST_SCAN + 1
    assert (s["shots_done"], s["planned_total"], s["state"]) == (2, SHOTS, "running")
