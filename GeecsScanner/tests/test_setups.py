"""Setups through the scanner: the preview, the diff, the apply's questions and its item, over the demo manager."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from geecs_scanner.service import ScannerService
from geecs_scanner.service.demo import DemoDeviceTypes, DemoQueueClient
from geecs_scanner.web import create_app


@pytest.fixture
def service(service: ScannerService) -> ScannerService:
    service._device_types = DemoDeviceTypes()
    return service


@pytest.fixture
def client(service: ScannerService) -> TestClient:
    return TestClient(create_app(service))


def _checks(r) -> list[str]:
    return [q["check"] for q in r.json()["error"]["needs_acknowledgement"]]


def _apply(client: TestClient, name: str = "low_power", **body: Any):
    return client.post(f"/api/setups/{name}/apply", json={"operator": "ada", **body})


def _apply_all(client: TestClient, name: str = "low_power"):
    first = _apply(client, name)
    assert first.status_code == 409, first.text
    return _apply(client, name, acknowledged=_checks(first))


# ----------------------------------------------------------------- reads


def test_the_list_and_the_preview_inline_nested_setups(client: TestClient) -> None:
    rows = {r["name"]: r for r in client.get("/api/setups").json()}
    assert rows["low_power"]["steps"] == 6 and rows["low_power"]["writes"] == 4
    assert rows["low_power"]["nested"] == ["shutters_closed"]
    assert rows["shutters_closed"]["captured_by"] == "demo"
    assert "in a loop" in rows["loop_a"]["problem"]
    d = client.get("/api/setups/low_power").json()
    assert [s["do"] for s in d["steps"]] == ["set", "set", "set", "set", "wait", "check"]  # fmt: skip
    assert [s["from_setup"] for s in d["steps"]] == [None, None] + ["shutters_closed"] * 4  # fmt: skip
    assert d["steps"][2]["text"] == "set U_148_PLC:shutter1 = 'off'"
    assert d["path"] == "scanner_configs/experiments/Demo/setups/low_power.yaml"
    assert [(h["index"], h["device"]) for h in d["hazards"]] == [(2, "U_HP_Daq")]
    assert client.get("/api/setups/nope").status_code == 404
    assert client.get("/api/setups/loop_a").status_code == 400
    assert client.get("/api/configs/setups").json()["names"] == [
        "loop_a",
        "loop_b",
        "low_power",
        "shutters_closed",
    ]


def test_the_diff_reads_each_write_target_as_a_number_or_a_word(client: TestClient) -> None:  # fmt: skip
    d = client.get("/api/setups/low_power/diff").json()
    rows = {r["variable"]: r for r in d["rows"]}
    assert rows["current"]["reading"] == 0.0 and rows["current"]["will_change"] is False
    assert rows["Jet pressure"]["will_change"] is True
    assert rows["shutter1"]["reading"] == "off"  # read as a word
    assert (d["changes"], d["unread"]) == (1, 0)
    assert d["digest"] == client.get("/api/setups/low_power").json()["digest"]


# ----------------------------------------------------------------- apply


def test_the_apply_asks_the_writes_and_each_hazard_and_never_acknowledges(client: TestClient, manager: DemoQueueClient) -> None:  # fmt: skip
    digest = client.get("/api/setups/low_power").json()["digest"]
    r = _apply(client)
    assert r.status_code == 409
    assert r.json()["error"]["digest"] == digest
    tag = digest[:16]
    assert _checks(r) == [f"setup:{tag}", f"hazard:{tag}:2"]
    # The writes acknowledged, the hazard not: still refused, nothing queued.
    r = _apply(client, acknowledged=[f"setup:{tag}"])
    assert r.status_code == 409 and _checks(r) == [f"hazard:{tag}:2"]
    assert manager.queue_items() == [] and manager.running_item() is None


def test_an_acknowledged_apply_queues_one_item_carrying_the_confirmed_setup(client: TestClient, manager: DemoQueueClient) -> None:  # fmt: skip
    digest = client.get("/api/setups/low_power").json()["digest"]
    r = _apply_all(client)
    assert r.status_code == 200, r.text
    assert (r.json()["plan"], r.json()["summary"]) == ("apply_setup", "setup low_power")
    item = manager.running_item()
    assert item["args"] == ["low_power"]
    geecs = item["kwargs"]["md"]["geecs"]
    assert geecs["operator"] == "ada"
    assert geecs["setup"] == {
        "name": "low_power",
        "path": "scanner_configs/experiments/Demo/setups/low_power.yaml",
        "commit": None,
        "digest": digest,
    }
    outcomes = {o["check"]: o["result"] for o in geecs["submission"]["preflight"]}
    assert set(outcomes.values()) == {"continued"} and len(outcomes) == 2
    running = client.get("/api/queue").json()["running"]
    assert running["user"] == "ada"
    manager.step()
    assert "applied" in client.get("/api/queue").json()["finished"][0]["detail"]


def test_an_acknowledgement_never_covers_another_version_of_the_setup(client: TestClient, service: ScannerService, manager: DemoQueueClient) -> None:  # fmt: skip
    old = _checks(_apply(client))
    resolver = service.resolver
    edited = resolver.setups()
    low = edited["low_power"].model_copy(deep=True)
    low.steps[1].value = 3.0
    edited["low_power"] = low
    resolver.setups = lambda: edited  # the file changed after the operator read it
    r = _apply(client, acknowledged=old)
    assert r.status_code == 409 and set(_checks(r)).isdisjoint(old)
    assert manager.running_item() is None


def test_the_apply_refuses_without_an_operator_and_on_a_bad_setup(client: TestClient) -> None:  # fmt: skip
    r = client.post("/api/setups/low_power/apply", json={"operator": "  "})
    assert r.status_code == 400 and "operator" in r.json()["error"]["message"]
    assert _apply(client, "nope").status_code == 404
    assert _apply(client, "loop_a").status_code == 400


def test_an_unreadable_hazard_source_refuses_the_apply(client: TestClient, service: ScannerService) -> None:  # fmt: skip
    def broken() -> Any:
        raise ValueError("operator_policy.yaml: not a mapping")

    service.resolver.resolve_operator_policy = broken
    r = _apply(client, acknowledged=["anything"])
    assert r.status_code == 400 and "cannot be asked" in r.json()["error"]["message"]
    d = client.get("/api/setups/low_power").json()
    assert d["hazards"] is None and "not a mapping" in d["hazard_problem"]


def test_with_the_db_silent_a_family_rule_still_asks(client: TestClient, service: ScannerService) -> None:  # fmt: skip
    class Silent:
        def by_device(self) -> None:
            return None

    service._device_types = Silent()
    hazards = client.get("/api/setups/low_power").json()["hazards"]
    assert [h["index"] for h in hazards] == [2]
    assert "unknown" in hazards[0]["reasons"][0]


def test_the_apply_is_idle_only(client: TestClient, manager: DemoQueueClient, preset_doc: dict) -> None:  # fmt: skip
    r = client.post("/api/submit", json={"preset": preset_doc, "acknowledged": ["gateway_liveness"]})  # fmt: skip
    assert r.status_code == 200, r.text
    r = _apply_all(client)
    assert r.status_code == 409 and "idle-only" in r.json()["error"]["message"]


def test_a_reopen_that_ran_while_the_apply_was_judged_refuses_it(client: TestClient, service: ScannerService, manager: DemoQueueClient) -> None:  # fmt: skip
    checks = _checks(_apply(client))
    judged = service._db_families

    def a_reopen_meanwhile() -> Any:
        service._reopen_generation += 1  # what reopen_environment does under the lock
        return judged()

    service._db_families = a_reopen_meanwhile
    r = _apply(client, acknowledged=checks)
    assert r.status_code == 409 and r.json()["error"].get("reopening") is True
    assert manager.running_item() is None
