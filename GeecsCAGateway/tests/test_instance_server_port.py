"""The CA server port of a per-experiment gateway instance comes from the env.

``geecs-ca-gateway@.service`` hands ``GEECS_CA_SERVER_PORT`` to this process
alone as ``EPICS_CA_SERVER_PORT`` — no gateway option exists for it, so the
contract is caproto's: its server context takes its port from that key, and
the main gateway (no key set) keeps the stock 5064.
"""

from __future__ import annotations

import asyncio

import pytest
from caproto.asyncio.server import Context


@pytest.fixture
def no_ca_env(monkeypatch):
    for key in ("EPICS_CA_SERVER_PORT", "EPICS_CAS_SERVER_PORT"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def _port() -> int:
    async def build() -> int:  # caproto's context must be built in a loop
        return Context({}).ca_server_port

    return asyncio.run(build())


def test_the_main_gateway_keeps_the_stock_port(no_ca_env) -> None:
    assert _port() == 5064


def test_an_instance_serves_on_its_own_port(no_ca_env) -> None:
    no_ca_env.setenv("EPICS_CA_SERVER_PORT", "5164")
    assert _port() == 5164
