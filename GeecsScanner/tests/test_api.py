"""The JSON API over the demo manager: every route, the error mapping, the prefix."""

from __future__ import annotations

from fastapi.testclient import TestClient

from geecs_scanner.service.demo import DemoQueueClient


def test_health_and_index(client: TestClient) -> None:
    h = client.get("/health").json()
    assert h["ok"] is True and h["manager"] is True and h["readiness"] == "ready"
    assert h["experiment"] == "Demo" and h["version"] == "0.0.0+test"
    idx = client.get("/").json()
    assert idx["api"] == "/api/status" and idx["kit"] == "/theme/kit.html"


def test_status_shape(client: TestClient) -> None:
    r = client.get("/api/status")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
    body = r.json()
    assert body["connected"] is True
    assert body["re_state"] == "idle"
    assert body["readiness"] == "ready"
    assert body["identity"] == "geecs-scanner test"


def test_config_listings(client: TestClient) -> None:
    assert client.get("/api/configs/presets").json()["names"] == [
        "background_dark",
        "eb_align_1hz",
        "jet_pressure_sweep",
    ]
    assert "standard_1hz" in client.get("/api/configs/trigger_profiles").json()["names"]
    assert client.get("/api/configs/optimizer_configs").json()["names"] == [
        "xopt_beam_charge"
    ]
    assert "Gas jet 2-axis" in client.get("/api/configs/scan_variables").json()["names"]
    r = client.get("/api/configs/save_sets")
    assert r.status_code == 404
    assert r.json()["error"]["kind"] == "not_found"
    assert "presets" in r.json()["error"]["kinds"]


def test_scan_variables_list_pseudo_as_not_scannable(client: TestClient) -> None:
    rows = {v["name"]: v for v in client.get("/api/scan-variables").json()}
    assert rows["Jet pressure"]["scannable"] is True
    assert rows["Jet pressure"]["target"] == "U_HP_Daq:Jet pressure"
    pseudo = rows["Gas jet 2-axis"]
    assert pseudo["kind"] == "pseudo" and pseudo["scannable"] is False
    assert "pseudo" in pseudo["reason"]


def test_preset_document_and_devices(client: TestClient) -> None:
    doc = client.get("/api/configs/presets/jet_pressure_sweep").json()
    assert doc["plan"]["name"] == "scan"
    assert client.get("/api/configs/presets/nope").status_code == 404
    devices = client.get("/api/devices").json()
    assert "UC_ALineEBeam3" in devices and "U_S1H.current" in devices


def test_preflight_expands_and_asks(client: TestClient, preset_doc: dict) -> None:
    out = client.post("/api/preflight", json=preset_doc).json()
    assert out["refusal"] is None
    assert [q["check"] for q in out["questions"]] == ["gateway_liveness"]
    assert out["plan"]["name"] == "scan"
    # the catalog name became a namespace reference (expand_preset's spelling
    # of U_HP_Daq:Jet pressure); the scanner bound nothing itself
    assert out["plan"]["args"][1].startswith("U_HP_Daq.")
    assert out["plan"]["args"][1] in out["plan"]["references"]
    assert out["plan"]["args"][0] == ["UC_ALineEBeam3", "U_ICT.scalars"]
    assert out["plan"]["kwargs"]["non_essential"] == ["UC_TC_Phosphor"]
    assert out["planned_shots"] == 70
    assert "7 steps" in out["summary"] and "strict" in out["summary"]


def test_preflight_refuses_a_pseudo_axis(client: TestClient, preset_doc: dict) -> None:
    preset_doc["plan"]["args"][0] = "Gas jet 2-axis"
    out = client.post("/api/preflight", json=preset_doc).json()
    assert out["refusal"] and "pseudo" in out["refusal"]
    assert out["plan"] is None


def test_submit_requires_every_acknowledgement(
    client: TestClient, preset_doc: dict
) -> None:
    r = client.post("/api/submit", json={"preset": preset_doc})
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["kind"] == "policy_refusal"
    assert err["needs_acknowledgement"][0]["check"] == "gateway_liveness"
    assert client.get("/api/queue").json()["running"] is None


def test_submit_invalid_document_is_400(client: TestClient) -> None:
    r = client.post("/api/submit", json={"preset": {"name": "x", "devices": "nope"}})
    assert r.status_code == 400
    assert r.json()["error"]["kind"] == "invalid_request"
    assert r.json()["error"]["errors"]


def test_submit_then_run_pause_resume_stop(
    client: TestClient, preset_doc: dict, manager: DemoQueueClient
) -> None:
    r = client.post(
        "/api/submit",
        json={
            "preset": preset_doc,
            "acknowledged": ["gateway_liveness"],
            "operator": "sam",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["item_uid"] and body["planned_shots"] == 70
    assert body["submitted_as"] == "geecs-scanner test"

    q = client.get("/api/queue").json()
    assert (
        q["running"]["plan"] == "scan" and q["running"]["user"] == "geecs-scanner test"
    )
    assert q["running"]["state"] == "running" and q["summary"].startswith("Running")
    assert client.get("/api/status").json()["re_state"] == "running"

    for _ in range(12):
        manager.step()
    p = client.get("/api/progress").json()
    assert p["available"] and p["scan_number"] == 47
    assert (
        p["planned_total"] == 70 and p["shots_done"] == 12 and p["state"] == "running"
    )

    assert client.post("/api/pause", json={"operator": "sam"}).json()["ok"] is True
    for _ in range(8):  # to the next step boundary (shot 20)
        manager.step()
    # pause is the manager's word; the progress picture keeps saying running
    assert client.get("/api/status").json()["re_state"] == "paused"
    assert client.get("/api/progress").json()["state"] == "running"
    manager.step()
    assert client.get("/api/progress").json()["shots_done"] == 20

    assert client.post("/api/resume").json()["ok"] is True
    manager.step()
    assert client.get("/api/progress").json()["shots_done"] == 21

    assert (
        client.post("/api/stop", json={"force": True, "operator": "sam"}).json()["ok"]
        is True
    )
    q = client.get("/api/queue").json()
    assert q["running"] is None
    done = q["finished"][0]
    assert done["state"] == "failed" and done["word"] == "stopped"
    assert done["scan_numbers"] == [47] and done["detail"] == "stopped by operator"
    # RunEngine.stop() marks the run successful; the history row is what says stopped
    assert client.get("/api/progress").json()["state"] == "done"


def test_queue_runs_items_in_order_and_clear_drops_the_waiting(
    client: TestClient, service, manager: DemoQueueClient
) -> None:
    first = service.preset("eb_align_1hz")
    second = service.preset("background_dark")
    ack = ["gateway_liveness"]
    # first item: idle queue -> starts at once; second: queues behind it
    assert (
        client.post(
            "/api/submit", json={"preset": first, "acknowledged": ack}
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/submit", json={"preset": second, "acknowledged": ack}
        ).status_code
        == 200
    )
    q = client.get("/api/queue").json()
    assert q["running"]["plan"] == "count" and q["running"]["planned_shots"] == 20
    assert [w["position"] for w in q["waiting"]] == [1]
    assert q["waiting"][0]["summary"].startswith("background · count · 50 shots")
    # a third while one WAITS is the failed-item-at-front trap: refused, pending items named
    r = client.post("/api/submit", json={"preset": first, "acknowledged": ack})
    assert r.status_code == 409
    err = r.json()["error"]
    assert (
        err["kind"] == "policy_refusal" and err["pending_items"][0]["plan"] == "count"
    )
    assert len(client.get("/api/queue").json()["waiting"]) == 1
    # clear_pending replaces the waiting item
    r = client.post(
        "/api/submit",
        json={"preset": first, "acknowledged": ack, "clear_pending": True},
    )
    assert r.status_code == 200
    q = client.get("/api/queue").json()
    assert len(q["waiting"]) == 1 and q["waiting"][0]["plan"] == "count"
    assert not q["waiting"][0]["summary"].startswith("background")
    for _ in range(20):
        manager.step()
    q = client.get("/api/queue").json()
    assert q["finished"][0]["state"] == "ok" and q["finished"][0]["scan_numbers"] == [
        47
    ]
    # the manager keeps going: the next waiting item is running already
    assert q["running"]["plan"] == "count" and q["waiting"] == []
    assert client.get("/api/progress").json()["scan_number"] == 48
    manager.stop_scan()
    assert (
        client.post(
            "/api/submit", json={"preset": second, "acknowledged": ack}
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/submit", json={"preset": first, "acknowledged": ack}
        ).status_code
        == 200
    )
    assert len(client.get("/api/queue").json()["waiting"]) == 1
    out = client.post("/api/clear").json()
    assert out["ok"] is True and "1 item" in out["message"]
    assert client.get("/api/queue").json()["waiting"] == []


def test_verbs_refuse_when_idle(client: TestClient) -> None:
    assert client.post("/api/pause").json() == {
        "ok": False,
        "message": "nothing is running",
    }
    assert client.post("/api/resume").json()["ok"] is False
    assert client.post("/api/stop").json()["ok"] is False


def test_forwarded_prefix_is_adopted(client: TestClient) -> None:
    # the proxy strips the prefix before forwarding and names it in the header
    r = client.get("/health", headers={"X-Forwarded-Prefix": "/scan"})
    assert r.status_code == 200 and r.json()["ok"] is True
    # the events route builds nothing path-shaped, but the index must still answer
    assert client.get("/", headers={"X-Forwarded-Prefix": "/scan"}).status_code == 200
    # a malformed prefix is ignored rather than propagated
    assert (
        client.get("/health", headers={"X-Forwarded-Prefix": "//bad path"}).status_code
        == 200
    )


def test_theme_is_served_from_this_origin(client: TestClient) -> None:
    css = client.get("/theme/kit.css")
    assert css.status_code == 200 and ".kit .panel" in css.text
    assert client.get("/theme/kit.html").status_code == 200
