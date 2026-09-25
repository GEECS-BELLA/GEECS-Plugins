"""Fixtures: a demo-backed service stepped by hand, and a TestClient over it."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from geecs_scanner.service import ProgressCache, ScannerService
from geecs_scanner.service.demo import (
    DemoQueueClient,
    DemoReadback,
    DemoResolver,
    DemoSettables,
    demo_preflight,
)
from geecs_scanner.web import create_app


@pytest.fixture
def streams() -> ProgressCache:
    return ProgressCache(clock=lambda: 1_000.0)


@pytest.fixture
def manager(streams: ProgressCache) -> DemoQueueClient:
    return DemoQueueClient(
        streams, period=0.0, first_scan=46, user="geecs-scanner test"
    )


@pytest.fixture
def service(
    manager: DemoQueueClient, streams: ProgressCache, tmp_path: Path
) -> ScannerService:
    # The writer's heartbeat is read from a path of the test's own, never
    # from the developer's ~/.local/state (or wherever GEECS_TILED_WRITER_STATE
    # points on this machine).
    return ScannerService(
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


@pytest.fixture
def client(service: ScannerService) -> TestClient:
    return TestClient(create_app(service))


@pytest.fixture
def preset_doc(service: ScannerService) -> dict:
    return service.preset("jet_pressure_sweep")
