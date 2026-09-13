"""Fixtures: a demo-backed service stepped by hand, and a TestClient over it."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from geecs_scanner.service import ProgressCache, ScannerService
from geecs_scanner.service.demo import DemoQueueClient, DemoResolver, demo_preflight
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
def service(manager: DemoQueueClient, streams: ProgressCache) -> ScannerService:
    return ScannerService(
        manager,
        DemoResolver(),
        experiment="Demo",
        identity="geecs-scanner test",
        streams=streams,
        preflight=demo_preflight,
        version="0.0.0+test",
    )


@pytest.fixture
def client(service: ScannerService) -> TestClient:
    return TestClient(create_app(service))


@pytest.fixture
def preset_doc(service: ScannerService) -> dict:
    return service.preset("jet_pressure_sweep")
