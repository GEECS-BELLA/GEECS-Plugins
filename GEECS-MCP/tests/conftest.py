"""The contract fixtures: the scanner's own app, in process, behind the tools.

The scanner is built the way its own suite builds it (``DemoQueueClient``
with ``period=0`` — a manager in memory that moves only when the test
calls ``step()``), mounted on an ``httpx.ASGITransport`` that
:mod:`geecs_mcp.clients` is pointed at.  A route or model change in
GeecsScanner therefore fails these tests in the same PR (DESIGN.md §7).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from geecs_mcp import clients, config
from geecs_mcp.scans import tools


@pytest.fixture
def scanner(tmp_path: Path):
    """The scanner service + its demo manager, wired behind the tools."""
    from geecs_scanner.service import ScannerService, StreamCache
    from geecs_scanner.service.demo import (
        DemoQueueClient,
        DemoReadback,
        DemoResolver,
        DemoSettables,
        demo_preflight,
    )
    from geecs_scanner.web import create_app

    streams = StreamCache(clock=lambda: 1_000.0)
    manager = DemoQueueClient(
        streams, period=0.0, first_scan=46, user="geecs-scanner test"
    )
    service = ScannerService(
        manager,
        DemoResolver(),
        experiment="Demo",
        identity="geecs-scanner test",
        streams=streams,
        preflight=demo_preflight,
        version="0.0.0+test",
        settables=DemoSettables(),
        readback=DemoReadback(manager),
        heartbeat_path=tmp_path / "heartbeat.json",
    )
    return service, manager, create_app(service)


@pytest.fixture
def wired(scanner, monkeypatch):
    """Point the clients at the in-process scanner; the other two stay unconfigured."""
    _, manager, app = scanner
    transport = httpx.ASGITransport(app=app)
    urls = {"scanner": "http://scanner", "portal": None, "logbook": None}
    monkeypatch.setattr(config, "service_url", lambda service: urls[service])
    monkeypatch.setattr(
        clients,
        "transport_for",
        lambda service: transport if service == "scanner" else None,
    )
    yield manager
    import anyio

    anyio.run(clients.reset_clients)


@pytest.fixture
def stepping(wired, monkeypatch):
    """Every pause in a wait verb advances the demo manager one shot."""

    async def step_instead(_seconds: float) -> None:
        wired.step()

    monkeypatch.setattr(tools, "_sleep", step_instead)
    return wired


@pytest.fixture
def frozen_clock(monkeypatch):
    """A clock the test advances by hand (the wait verbs' deadline)."""
    now = {"t": 0.0}
    monkeypatch.setattr(tools, "_clock", lambda: now["t"])
    return now
