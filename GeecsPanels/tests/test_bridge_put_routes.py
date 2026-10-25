"""The put routes: POST /api/put, the run lock's release, GET /api/journal."""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from geecs_panels.app import create_app
from geecs_panels.bridge.journal import Journal
from geecs_panels.bridge.puts import Puts
from geecs_panels.bridge.routes import Bridge
from geecs_panels.bridge.stream import DeviceStreams
from geecs_panels.contracts import JournalEntry, PutResult
from geecs_panels.registry import Device
from tests._bridge import SUPPLY, FakeCA, devices, healthy_supply
from tests.test_bridge_puts import FakeIO

OPERATOR = "S. Barber"


def _client(writable: bool = True) -> tuple[TestClient, FakeCA, FakeIO, DeviceStreams]:
    ca, io = FakeCA(), FakeIO()
    served = devices()
    streams = DeviceStreams(served, ca.subscribe, push_s=0.01)
    puts = Puts(streams, io, Journal(), writable=writable)
    bridge = Bridge("expt", list(served.values()), [], streams, puts=puts)
    return TestClient(create_app("expt", [], object(), bridge=bridge)), ca, io, streams


def _open_stream(streams: DeviceStreams, client: TestClient, name: str) -> None:
    """A page's stream of the device, open (the stale-page gate needs one)."""

    async def listen() -> None:
        hub = streams.hub(name)
        hub.listeners.add(asyncio.Queue())
        hub.tick()

    client.portal.call(listen)


def test_an_accepted_put_is_200_with_its_result_and_journal_id() -> None:
    client, ca, io, streams = _client()
    with client:
        _open_stream(streams, client, "PS1")
        healthy_supply(ca)
        response = client.post(
            "/api/put",
            json={
                "device": "PS1",
                "role": "current_sp",
                "value": 0.5,
                "operator": OPERATOR,
            },
        )
        journal = client.get("/api/journal?entries=5").json()
    assert response.status_code == 200
    result = PutResult.model_validate(response.json())
    assert result.outcome == "accepted"
    entry = JournalEntry.model_validate(journal[0])
    assert entry.id == result.journal_id and entry.operator == OPERATOR
    assert io.puts == [(SUPPLY + "current_ramp:SP", 0.5, True)]


def test_a_denied_put_answers_its_status_with_a_journaled_result() -> None:
    client, ca, io, streams = _client()
    with client:
        response = client.post(
            "/api/put",
            json={
                "device": "PS1",
                "role": "current_sp",
                "value": 0.5,
                "operator": OPERATOR,
            },
        )
        journal = client.get("/api/journal").json()
    assert response.status_code == 409  # no stream open
    assert response.json()["outcome"] == "denied"
    assert journal[0]["outcome"] == "denied"


def test_a_read_only_bridge_answers_403() -> None:
    client, ca, io, streams = _client(writable=False)
    with client:
        response = client.post(
            "/api/put",
            json={
                "device": "PS1",
                "role": "current_sp",
                "value": 0.5,
                "operator": OPERATOR,
            },
        )
    assert response.status_code == 403


def test_an_unknown_device_is_a_404_and_no_operator_a_422() -> None:
    client, *_ = _client()
    with client:
        unknown = client.post(
            "/api/put",
            json={"device": "Nope", "role": "x", "value": 1, "operator": OPERATOR},
        )
        anonymous = client.post(
            "/api/put", json={"device": "PS1", "role": "current_sp", "value": 0.5}
        )
        blank = client.post(
            "/api/put",
            json={"device": "PS1", "role": "current_sp", "value": 0.5, "operator": ""},
        )
        journal = client.get("/api/journal").json()
    assert unknown.status_code == 404
    assert anonymous.status_code == 422 and blank.status_code == 422
    assert journal == []  # nothing to journal: no device, or no one asking


def test_the_lock_release_route_names_the_device_by_its_path() -> None:
    client, ca, io, streams = _client()
    with client:
        response = client.post("/api/devices/Cam/unlock", json={"operator": OPERATOR})
        refused = client.post(
            "/api/devices/Cam/unlock", json={"operator": OPERATOR, "device": "PS1"}
        )
    assert response.status_code == 200 and response.json()["outcome"] == "accepted"
    assert io.puts == [("expt:cam:RUN:Lock", 0, True)]
    assert refused.status_code == 422  # the body cannot name another device


def test_the_journal_is_newest_first_and_bounded() -> None:
    client, ca, io, streams = _client(writable=False)
    with client:
        for value in (0.1, 0.2, 0.3):
            client.post(
                "/api/put",
                json={
                    "device": "PS1",
                    "role": "current_sp",
                    "value": value,
                    "operator": OPERATOR,
                },
            )
        two = client.get("/api/journal?entries=2").json()
        too_many = client.get("/api/journal?entries=100000")
    assert [e["value"] for e in two] == [0.3, 0.2]
    assert too_many.status_code == 422


def test_the_put_routes_have_stable_operation_ids() -> None:
    client, *_ = _client()
    with client:
        paths = client.get("/openapi.json").json()["paths"]
    assert paths["/api/put"]["post"]["operationId"] == "post_api_put"
    assert paths["/api/journal"]["get"]["operationId"] == "get_api_journal"
    assert (
        paths["/api/devices/{name}/unlock"]["post"]["operationId"]
        == "post_api_devices_name_unlock"
    )


def test_the_pages_writes_can_be_off_while_the_bridges_are_on() -> None:
    """--pages-read-only: the per-kind put routes (no operator, journal or confirm) refuse; /api/put still writes."""
    ca, io = FakeCA(), FakeIO()
    served = devices()
    streams = DeviceStreams(served, ca.subscribe, push_s=0.01)
    bridge = Bridge(
        "expt", list(served.values()), [], streams, puts=Puts(streams, io, Journal())
    )
    supply = Device(
        "PS1",
        "power_supply",
        SUPPLY,
        roles=tuple(
            (r, s)
            for r, s in {
                "current_sp": "current_ramp:SP",
                "current": "current",
                "voltage": "voltage",
                "output_sp": "enable_output:SP",
                "output": "enable_output",
                "reset": "reset:SP",
            }.items()
        ),  # fmt: skip
    )
    app = create_app(
        "expt", [supply], object(), writable=False, supplies=object(), bridge=bridge
    )
    with TestClient(app) as client:
        _open_stream(streams, client, "PS1")
        healthy_supply(ca)
        page = client.post("/supply/PS1/put", json={"action": "output", "value": 1})
        api = client.post(
            "/api/put",
            json={
                "device": "PS1",
                "role": "current_sp",
                "value": 0.5,
                "operator": OPERATOR,
            },
        )
    assert page.status_code == 403
    assert api.status_code == 200 and api.json()["outcome"] == "accepted"
