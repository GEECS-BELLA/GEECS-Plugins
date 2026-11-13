"""What the configurator asks of the scanner, and what the scanner asks of it.

The environment reopen (the apply of a trigger profile, the scan-variable
catalogue, the native-device registry, the shot offsets), the dry run of an
unsaved trigger profile, and "Save as preset" as a commit made by the
configurator — over the demo manager and a recorded configurator.
"""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest
from fastapi.testclient import TestClient

from geecs_scanner.service import ScannerService
from geecs_scanner.service.demo import DemoQueueClient
from geecs_scanner.service.errors import ScannerError
from geecs_scanner.service.models import ReopenIn, SavePresetIn
from geecs_scanner.service.preset_saver import ConfiguratorPresets, NoConfigurator

# ------------------------------------------------------------------ reopen


def test_a_reopen_when_idle_and_empty_closes_then_opens(client: TestClient, manager: DemoQueueClient) -> None:  # fmt: skip
    r = client.post("/api/environment/reopen", json={"operator": "configurator", "reason": "laser_off.yaml"})  # fmt: skip
    assert r.status_code == 200, r.text
    assert r.json()["ok"] and not r.json()["queue_started"]
    assert manager.reopens == 1
    assert client.get("/api/status").json()["reopening"] is False


def test_a_reopen_refuses_while_a_plan_runs_or_items_wait(client: TestClient, preset_doc: dict, manager: DemoQueueClient) -> None:  # fmt: skip
    r = client.post("/api/submit", json={"preset": preset_doc, "acknowledged": ["gateway_liveness"]})  # fmt: skip
    assert r.status_code == 200
    r = client.post("/api/environment/reopen", json={})
    assert r.status_code == 409 and r.json()["error"]["kind"] == "policy_refusal"
    assert manager.reopens == 0
    while client.get("/api/status").json()["re_state"] != "idle":
        manager.step()
    manager._queue.append({"name": "count", "args": [], "kwargs": {}, "item_uid": "w1"})
    r = client.post("/api/environment/reopen", json={})
    assert r.status_code == 409 and r.json()["error"]["items_in_queue"] == 1
    assert manager.reopens == 0


class _Recording:
    """A manager double for the reopen alone: scripted status, every verb recorded."""

    def __init__(self, **status: Any) -> None:
        from geecs_bluesky.qs_client import QueueStatus

        base = dict(connected=True, re_state="idle", manager_state="idle", worker_exists=True, worker_environment_state="idle")  # fmt: skip
        self.status_now = QueueStatus(**{**base, **status})
        self.calls: list[str] = []
        self.gate = threading.Event()
        self.gate.set()
        self.items_after_open = 0

    def status(self) -> Any:
        return self.status_now

    def readiness(self, expected_plans: Any = None) -> Any:
        from geecs_bluesky.qs_client import readiness_verdict

        return readiness_verdict(self.status_now, [], expected_plans)

    def environment_close(self, **_: Any) -> tuple[bool, str]:
        self.calls.append("close")
        return True, "closed"

    def environment_open(self, **_: Any) -> tuple[bool, str]:
        self.calls.append("open")
        self.gate.wait(10)
        from dataclasses import replace

        self.status_now = replace(self.status_now, items_in_queue=self.items_after_open)
        return True, "opened"

    def queue_start(self) -> tuple[bool, str]:
        self.calls.append("start")
        return True, "started"

    def submit_plan(self, name: str, **_: Any) -> Any:
        from geecs_bluesky.qs_client import SubmitResult

        self.calls.append(f"submit:{name}")
        return SubmitResult(ok=True, item_uid="u")


def _service(manager: Any, **kw: Any) -> ScannerService:
    from geecs_scanner.service.demo import DemoResolver, demo_preflight

    return ScannerService(manager, DemoResolver(), experiment="Demo", identity="t", preflight=demo_preflight, **kw)  # fmt: skip


@pytest.mark.parametrize(
    ("status", "word"),
    [
        ({"connected": False, "detail": "down"}, "manager_unreachable"),
        (
            {
                "worker_exists": False,
                "worker_environment_state": "closed",
                "re_state": None,
            },
            "policy_refusal",
        ),  # fmt: skip
        ({"worker_environment_state": "initializing"}, "policy_refusal"),
        ({"re_state": "paused", "manager_state": "paused"}, "policy_refusal"),
        ({"manager_state": "executing_queue", "re_state": "running"}, "policy_refusal"),
    ],
)
def test_a_reopen_is_refused_unless_the_environment_is_open_and_idle(status: dict, word: str) -> None:  # fmt: skip
    manager = _Recording(**status)
    with pytest.raises(ScannerError) as err:
        _service(manager).reopen_environment(ReopenIn())
    assert err.value.kind == word and manager.calls == []


def test_items_added_past_the_scanner_during_the_reopen_start_the_queue() -> None:
    manager = _Recording()
    manager.items_after_open = 1
    out = _service(manager).reopen_environment(ReopenIn())
    assert out.queue_started and manager.calls == ["close", "open", "start"]


def test_a_failed_close_is_a_refusal_and_never_opens() -> None:
    manager = _Recording()
    manager.environment_close = lambda **_: (False, "refused")  # type: ignore[method-assign]
    service = _service(manager)
    with pytest.raises(ScannerError) as err:
        service.reopen_environment(ReopenIn())
    assert err.value.kind == "policy_refusal" and "open" not in manager.calls
    assert not service.reopening


def test_mid_reopen_a_start_is_refused_with_the_reason_and_an_idle_item_waits_for_the_lock() -> None:  # fmt: skip
    manager = _Recording()
    manager.gate.clear()  # the open blocks until released
    service = _service(manager)
    done = threading.Thread(target=service.reopen_environment, args=(ReopenIn(),))
    done.start()
    for _ in range(500):
        if "open" in manager.calls:
            break
        threading.Event().wait(0.01)
    assert "open" in manager.calls
    assert service.status().reopening
    with pytest.raises(ScannerError) as err:
        service.preflight({"name": "p", "devices": [], "plan": {"name": "count", "kwargs": {"num": 1}}})  # fmt: skip
    assert err.value.extra.get("reopening") is True
    move = threading.Thread(target=lambda: service.move(_MoveIn()))
    move.start()
    move.join(0.3)
    assert move.is_alive() and "submit:mv" not in manager.calls  # the lock is held
    manager.gate.set()
    done.join(5)
    move.join(5)
    assert manager.calls == ["close", "open", "submit:mv"]
    assert not service.reopening


class _MoveIn:
    variable, value, operator = "U_S1H:current", 1.0, ""


# ------------------------------------------------- trigger-profile dry run

PROFILE = {
    "name": "laser_off",
    "states": {
        "SCAN": [{"device": "U_DG645", "variable": "Amplitude.Ch AB", "value": "4.0"}],
        "STANDBY": [
            {"device": "U_DG645", "variable": "Amplitude.Ch AB", "value": "0.5"}
        ],
    },
}


def test_a_draft_profile_is_checked_by_the_workers_rules(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    probed: list[list[str]] = []
    monkeypatch.setattr("geecs_bluesky.devices.ca.liveness.probe_disconnected", lambda e, d, *, timeout: probed.append(list(d)) or [])  # fmt: skip
    r = client.post("/api/preflight/trigger-profile", json={"document": PROFILE})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["refusal"] is None and probed == [["U_DG645"]]
    assert [o["check"] for o in out["outcomes"]] == ["validate", "gateway_liveness"]
    r = client.post("/api/preflight/trigger-profile", json={"document": {"name": "x", "states": {}}})  # fmt: skip
    assert "names no trigger device" in r.json()["refusal"]


def test_a_draft_profile_under_a_preset_runs_the_full_preflight(client: TestClient, preset_doc: dict) -> None:  # fmt: skip
    r = client.post("/api/preflight/trigger-profile", json={"document": PROFILE, "preset": preset_doc})  # fmt: skip
    assert r.status_code == 200 and r.json()["refusal"] is None
    r = client.post("/api/preflight/trigger-profile", json={"document": {"states": 3}, "preset": preset_doc})  # fmt: skip
    assert "not valid" in r.json()["refusal"]


# ------------------------------------------------------------ save preset


class _Configurator:
    """The configurator's document API, recorded: one preset ``old.yaml`` exists."""

    def __init__(self, put_status: int = 200, put_body: Any = None) -> None:
        self.requests: list[tuple[str, str, Any]] = []
        self.put_status = put_status
        self.put_body = put_body or {"outcome": "committed", "commit": "abcdef1234567", "path": "scanner_configs/experiments/Demo/presets/x.yaml"}  # fmt: skip

    def __call__(self, method: str, url: str, data: bytes | None) -> tuple[int, bytes]:
        body = json.loads(data) if data else None
        self.requests.append((method, url, body))
        if method == "GET" and "/documents/preset?" in url:
            return 200, json.dumps([{"id": "old.yaml", "valid": True}]).encode()
        if method == "GET":
            return 200, json.dumps({"etag": "e-old"}).encode()
        return self.put_status, json.dumps(self.put_body).encode()


def _preset_doc(name: str) -> dict:
    return {
        "name": name,
        "devices": [],
        "plan": {"name": "count", "kwargs": {"num": 1}},
    }


def test_a_new_preset_is_a_create_and_an_existing_one_saves_over_its_etag() -> None:
    fake = _Configurator()
    saver = ConfiguratorPresets("http://cfg.test/", "Demo", send=fake)
    out = saver.save("new", _preset_doc("new"), operator="Ada", overwrite=False)
    method, url, body = fake.requests[-1]
    assert (method, url) == ("PUT", "http://cfg.test/api/documents/preset/new.yaml?experiment=Demo")  # fmt: skip
    assert body["create"] is True and body["etag"] == "" and body["operator"] == "Ada"
    assert "abcdef1234" in out.message and out.commit == "abcdef1234567"
    saver.save("old", _preset_doc("old"), operator="Ada", overwrite=True)
    _, _, body = fake.requests[-1]
    assert body["create"] is False and body["etag"] == "e-old"


@pytest.mark.parametrize(
    ("status", "kind"),
    [(409, "conflict"), (422, "invalid_request"), (502, "manager_unreachable")],
)
def test_the_configurators_refusals_keep_their_meaning(status: int, kind: str) -> None:
    saver = ConfiguratorPresets("http://cfg.test", "Demo", send=_Configurator(status, {"detail": "no"}))  # fmt: skip
    with pytest.raises(ScannerError) as err:
        saver.save("old", _preset_doc("old"), operator="Ada", overwrite=True)
    assert err.value.kind == kind


def test_an_unreachable_configurator_is_said_so() -> None:
    def down(method: str, url: str, data: bytes | None) -> tuple[int, bytes]:
        raise OSError("connection refused")

    with pytest.raises(ScannerError) as err:
        ConfiguratorPresets("http://cfg.test", "Demo", send=down).save("n", {}, operator="A", overwrite=False)  # fmt: skip
    assert err.value.kind == "manager_unreachable"


def test_without_a_configurator_nothing_is_saved_and_the_answer_says_why() -> None:
    service = _service(_Recording(), presets=NoConfigurator())
    with pytest.raises(ScannerError) as err:
        service.save_preset("p", SavePresetIn(preset=_preset_doc("p"), operator="Ada"))
    assert "--configurator-url" in err.value.message


def test_a_save_needs_an_operator(client: TestClient, preset_doc: dict) -> None:
    r = client.post("/api/configs/presets/jet_v3", json={"preset": preset_doc})
    assert r.status_code == 400 and "operator" in r.json()["error"]["message"]


def test_without_overwrite_a_preset_the_configurator_holds_is_never_replaced() -> None:
    """The scanner's own checkout may lag main: the configurator's answer decides, and nothing is sent."""
    fake = _Configurator()
    saver = ConfiguratorPresets("http://cfg.test", "Demo", send=fake)
    with pytest.raises(ScannerError) as err:
        saver.save("old", _preset_doc("old"), operator="Ada", overwrite=False)
    assert err.value.kind == "policy_refusal" and err.value.extra["exists"] is True
    assert [m for m, _, _ in fake.requests] == ["GET"]  # the listing only, no PUT


def test_a_create_that_lost_a_race_is_the_exists_refusal() -> None:
    saver = ConfiguratorPresets("http://cfg.test", "Demo", send=_Configurator(409, {"detail": {"reason": "the document already exists"}}))  # fmt: skip
    with pytest.raises(ScannerError) as err:
        saver.save("new", _preset_doc("new"), operator="Ada", overwrite=False)
    assert err.value.extra.get("exists") is True


def test_a_submit_whose_preflight_straddled_a_reopen_is_refused(manager: DemoQueueClient, streams: Any, tmp_path: Any) -> None:  # fmt: skip
    """Judged on the configs before the apply: refused with the reason, never queued."""
    from geecs_scanner.service.demo import DemoResolver, demo_preflight
    from geecs_scanner.service.models import SubmitIn

    service: ScannerService

    def preflight_during_a_reopen(*args: Any, **kwargs: Any) -> Any:
        report = demo_preflight(*args, **kwargs)
        service.reopen_environment(ReopenIn())  # a reopen runs while this preflight did
        return report

    service = ScannerService(manager, DemoResolver(), experiment="Demo", identity="t", streams=streams, preflight=preflight_during_a_reopen, heartbeat_path=tmp_path / "hb.json")  # fmt: skip
    doc = {"name": "p", "devices": [{"device": "UC_ALineEBeam3"}], "plan": {"name": "count", "kwargs": {"num": 1}}}  # fmt: skip
    with pytest.raises(ScannerError) as err:
        service.submit(SubmitIn(preset=doc, acknowledged=["gateway_liveness"]))
    assert err.value.extra.get("reopening") is True
    assert manager.reopens == 1 and service.queue().waiting == [] and service.queue().running is None  # fmt: skip
