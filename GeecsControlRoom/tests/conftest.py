"""Settings and an app over a stubbed module (httpx's MockTransport): no Canopy."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from geecs_control_room.app import create_app
from geecs_control_room.module_client import ModuleClient
from geecs_control_room.settings import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        experiment="Bench",
        configs_root=tmp_path,
        canopy_url="http://canopy.test",
        read_key="read-key-for-tests",
    )


@pytest.fixture
def sent() -> list[httpx.Request]:
    return []


@pytest.fixture
def module(sent: list[httpx.Request]) -> ModuleClient:
    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if request.url.path.endswith("/put"):
            return httpx.Response(
                428, json={"outcome": "denied", "code": "confirm_required"}
            )
        return httpx.Response(200, json={"experiment": "Bench", "devices": []})

    return ModuleClient(
        "http://canopy.test",
        "read-key-for-tests",
        transport=httpx.MockTransport(answer),
    )


@pytest.fixture
def client(settings: Settings, module: ModuleClient) -> TestClient:
    with TestClient(create_app(settings, module)) as c:
        yield c


def body_of(request: httpx.Request) -> dict:
    return json.loads(request.content)
