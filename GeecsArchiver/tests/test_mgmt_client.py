"""The management client against a fake appliance (httpx MockTransport)."""

import json

import httpx
import pytest

from geecs_archiver.mgmt_client import MgmtClient, MgmtError, PVStatus


class FakeAppliance:
    def __init__(self):
        self.calls = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        path = request.url.path
        if path.endswith("/getVersions"):
            return httpx.Response(
                200, json={"mgmt_version": "Archiver Appliance Version 2.4.1"}
            )
        if path.endswith("/getApplianceMetrics"):
            return httpx.Response(200, json=[{"pvCount": "9", "status": "Working"}])
        if path.endswith("/getAllPVs"):
            assert request.url.params["limit"] == "-1"
            return httpx.Response(200, json=["undulator:a:b", "undulator:c:d"])
        if path.endswith("/getPVStatus"):
            return httpx.Response(
                200,
                json=[
                    {
                        "pvName": pv,
                        "status": "Being archived",
                        "connectionState": "true",
                        "samplingPeriod": "1.0",
                        "appliance": "appliance0",
                    }
                    for pv in request.url.params.get_list("pv")
                ],
            )
        if path.endswith("/archivePV"):
            body = json.loads(request.content)
            return httpx.Response(
                200,
                json=[
                    {"pvName": r["pv"], "status": "Archive request submitted"}
                    for r in body
                ],
            )
        if path.endswith("/pauseArchivingPV"):
            return httpx.Response(
                200, json={"status": "ok", "pvName": request.url.params["pv"]}
            )
        if path.endswith("/changeArchivalParameters"):
            return httpx.Response(200, json=dict(request.url.params))
        if path.endswith("/boom"):
            return httpx.Response(500, text="no")
        return httpx.Response(404)


@pytest.fixture
def fake():
    return FakeAppliance()


@pytest.fixture
def client(fake):
    with MgmtClient(
        "http://appliance:17665/", transport=httpx.MockTransport(fake.handler)
    ) as c:
        yield c


def test_base_url_and_versions(client, fake):
    assert client.versions()["mgmt_version"].endswith("2.4.1")
    assert str(fake.calls[0].url) == "http://appliance:17665/mgmt/bpl/getVersions"


def test_metrics_is_the_single_appliance_row(client):
    assert client.appliance_metrics()["pvCount"] == "9"


def test_get_all_pvs_asks_for_everything(client):
    assert client.get_all_pvs() == ["undulator:a:b", "undulator:c:d"]


def test_status_is_batched_and_parsed(client, fake):
    pvs = [f"undulator:x:v{i}" for i in range(250)]
    statuses = client.get_pv_status(pvs)
    assert [s.pv for s in statuses] == pvs
    assert all(
        s.archived and s.connected is True and s.sampling_period == 1.0
        for s in statuses
    )
    assert sum(1 for c in fake.calls if c.url.path.endswith("/getPVStatus")) == 3


def test_archive_request_is_bulk_json(client, fake):
    out = client.archive_pvs(
        [{"pv": "undulator:a:b", "samplingperiod": "1", "samplingmethod": "MONITOR"}]
    )
    assert out[0]["status"] == "Archive request submitted"
    sent = [c for c in fake.calls if c.url.path.endswith("/archivePV")][0]
    assert sent.headers["content-type"].startswith("application/json")
    assert client.archive_pvs([]) == []


def test_pause_and_retune_pass_the_appliance_parameter_names(client):
    assert client.pause("undulator:a:b")["pvName"] == "undulator:a:b"
    params = client.change_archival_params("undulator:a:b", 10.0, "MONITOR")
    assert params == {
        "pv": "undulator:a:b",
        "samplingperiod": "10",
        "samplingmethod": "MONITOR",
    }


def test_http_failures_become_mgmt_error(fake):
    with MgmtClient(
        "http://appliance:17665", transport=httpx.MockTransport(fake.handler)
    ) as c:
        with pytest.raises(MgmtError, match="GET /boom"):
            c._get("/boom")


def test_pvstatus_parses_the_appliance_booleans_and_states():
    s = PVStatus.from_bpl(
        {
            "pvName": "p",
            "status": "Not being archived",
            "connectionState": None,
            "samplingPeriod": None,
        }
    )
    assert s.unknown and s.connected is None and s.sampling_period is None
    s = PVStatus.from_bpl(
        {
            "pvName": "p",
            "status": "Paused",
            "connectionState": "false",
            "samplingPeriod": "2.5",
        }
    )
    assert s.paused and s.connected is False and s.sampling_period == 2.5
    s = PVStatus.from_bpl(
        {"pvName": "p", "status": "Initial sampling", "connectionState": "true"}
    )
    assert s.pending and not s.archived
