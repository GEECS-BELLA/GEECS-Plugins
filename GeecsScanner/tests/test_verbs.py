"""The idle-only items (move, actions, calibration) and the preset write, over the demo manager."""

from __future__ import annotations

from fastapi.testclient import TestClient

from geecs_scanner.service.demo import DemoQueueClient


def _start_a_scan(client: TestClient, preset_doc: dict) -> None:
    r = client.post(
        "/api/submit", json={"preset": preset_doc, "acknowledged": ["gateway_liveness"]}
    )
    assert r.status_code == 200, r.text


# ------------------------------------------------------------------ move


def test_move_resolves_the_catalog_name_and_runs_as_an_mv_item(
    client: TestClient, manager: DemoQueueClient
) -> None:
    r = client.post("/api/move", json={"variable": "S1H current", "value": 1.5})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["plan"] == "mv" and out["reference"] == "U_S1H.current"
    assert out["summary"] == "move U_S1H.current → 1.5"
    running = client.get("/api/queue").json()["running"]
    assert running["plan"] == "mv" and running["summary"] == "move U_S1H.current → 1.5"
    manager.step()  # the fake worker finishes a non-run item in one step
    q = client.get("/api/queue").json()
    assert q["running"] is None
    assert q["finished"][0]["plan"] == "mv"
    assert q["finished"][0]["detail"] == "moved U_S1H.current = 1.5"
    assert q["finished"][0]["scan_numbers"] == []  # no run was opened
    # no scan number was claimed: the next real scan is still 47
    assert client.get("/api/progress").json()["scan_number"] is None


def test_move_accepts_device_colon_variable_and_refuses_pseudo(
    client: TestClient,
) -> None:
    r = client.post("/api/move", json={"variable": "U_Hexapod:xpos", "value": 2})
    assert r.status_code == 200 and r.json()["reference"] == "U_Hexapod.xpos"
    client.post("/api/clear")
    r = client.post("/api/move", json={"variable": "Gas jet 2-axis", "value": 0})
    assert r.status_code == 400
    assert "pseudo" in r.json()["error"]["message"]
    r = client.post("/api/move", json={"variable": "  ", "value": 0})
    assert r.status_code == 400


def test_idle_only_items_refuse_while_a_plan_runs_or_waits(
    client: TestClient, preset_doc: dict, manager: DemoQueueClient
) -> None:
    _start_a_scan(client, preset_doc)
    for path, body in (
        ("/api/move", {"variable": "S1H current", "value": 0}),
        ("/api/actions/close_shutters/run", {}),
        ("/api/calibration/check", {"devices": ["UC_ALineEBeam3", "U_ICT"]}),
        ("/api/calibration/measure", {"devices": ["UC_ALineEBeam3", "U_ICT"]}),
    ):
        r = client.post(path, json=body)
        assert r.status_code == 409, (path, r.text)
        err = r.json()["error"]
        assert err["kind"] == "policy_refusal" and err["re_state"] == "running"
        assert "idle-only" in err["message"]
    # nothing got past the check onto the queue
    assert client.get("/api/status").json()["items_in_queue"] == 0
    # a WAITING item is refused too: the queue is started, so an item added
    # behind it would run by itself the moment the scan ends. The real
    # manager reaches "idle with an item waiting" when a failed item returns
    # to the queue front; the demo only ever waits while something runs, so
    # arrange the state by hand.
    while manager.status().re_state == "running":
        manager.step()
    assert client.get("/api/status").json()["re_state"] == "idle"
    manager._queue.append(  # noqa: SLF001 — the failed-item-at-front trap, arranged
        {
            "name": "count",
            "args": [],
            "kwargs": {},
            "item_type": "plan",
            "user": "t",
            "item_uid": "w1",
        }
    )
    r = client.post("/api/move", json={"variable": "S1H current", "value": 0})
    assert r.status_code == 409
    assert r.json()["error"]["items_in_queue"] == 1
    assert "clear the queue" in r.json()["error"]["message"]


# --------------------------------------------------------------- actions


def test_actions_list_and_preview_inline_nested_runs(client: TestClient) -> None:
    rows = {a["name"]: a for a in client.get("/api/actions").json()}
    assert (
        rows["close_shutters"]["steps"] == 4
        and rows["close_shutters"]["problem"] is None
    )
    assert rows["experiment_closeout"]["steps"] == 5
    assert rows["experiment_closeout"]["nested"] == ["close_shutters"]
    assert "no_such_plan" in rows["broken_reference"]["problem"]
    d = client.get("/api/actions/experiment_closeout").json()
    assert [s["do"] for s in d["steps"]] == ["set", "set", "set", "wait", "check"]
    assert [s["from_plan"] for s in d["steps"]] == [None] + ["close_shutters"] * 4
    assert d["steps"][0]["text"] == "set U_S1H:current = 0.0"
    assert d["steps"][1]["text"] == "set U_148_PLC:shutter1 = 'off'"
    assert d["steps"][3]["text"] == "wait 2 s"
    assert d["writes"] == 3
    assert client.get("/api/actions/no_such").status_code == 404
    r = client.get("/api/actions/broken_reference")
    assert r.status_code == 400 and "no_such_plan" in r.json()["error"]["message"]
    # a loop: the library validator lets it through, the flatten must not
    assert "loop" in rows["loop_a"]["problem"] and "loop" in rows["loop_b"]["problem"]
    r = client.get("/api/actions/loop_a")
    assert r.status_code == 400 and "in a loop" in r.json()["error"]["message"]
    assert client.post("/api/actions/loop_a/run").status_code == 400


def test_run_action_queues_run_action_and_refuses_an_unresolvable_one(
    client: TestClient, manager: DemoQueueClient
) -> None:
    r = client.post("/api/actions/experiment_closeout/run", json={"operator": "sam"})
    assert r.status_code == 200, r.text
    assert r.json()["plan"] == "run_action"
    assert r.json()["summary"] == "action experiment_closeout"
    manager.step()
    done = client.get("/api/queue").json()["finished"][0]
    assert done["plan"] == "run_action" and "ran to completion" in done["detail"]
    assert client.post("/api/actions/broken_reference/run").status_code == 400
    assert client.post("/api/actions/nope/run").status_code == 404
    assert client.get("/api/status").json()["items_in_queue"] == 0


# ----------------------------------------------------------- calibration


def test_calibration_summarizes_the_stored_offsets(client: TestClient) -> None:
    c = client.get("/api/calibration").json()
    assert c["stored"] is True and c["reference"] == "uc_alineebeam3"
    assert c["max_offset_s"] == 0.031 and c["max_offset_device"] == "uc_tc_phosphor"
    assert [d["name"] for d in c["devices"]] == [
        "u_ict",
        "uc_alineebeam3",
        "uc_tc_phosphor",
    ]
    assert c["path"].endswith("shot_offsets.yaml")


def test_calibration_verbs_need_two_devices_and_carry_their_kwargs(
    client: TestClient, manager: DemoQueueClient
) -> None:
    r = client.post("/api/calibration/check", json={"devices": ["UC_ALineEBeam3"]})
    assert r.status_code == 400 and "at least two" in r.json()["error"]["message"]
    r = client.post(
        "/api/calibration/check",
        json={
            "devices": ["UC_ALineEBeam3", "U_ICT"],
            "tolerance_s": 0.08,
            "trigger_profile": "no_gas",
        },
    )
    assert r.status_code == 200, r.text
    item = manager.running_item()
    assert item["name"] == "check_shot_sync"
    assert item["args"] == [["UC_ALineEBeam3", "U_ICT"]]
    assert item["kwargs"] == {"trigger_profile": "no_gas", "tolerance_s": 0.08}
    manager.step()
    assert "in tolerance" in client.get("/api/queue").json()["finished"][0]["detail"]
    r = client.post(
        "/api/calibration/measure",
        json={
            "devices": ["UC_ALineEBeam3", "UC_TC_Phosphor", "U_ICT"],
            "shots": 5,
            "write": True,
        },
    )
    assert r.status_code == 200, r.text
    item = manager.running_item()
    assert item["name"] == "measure_shot_offsets"
    assert item["kwargs"] == {"write": True, "shots": 5}
    assert client.post(
        "/api/calibration/measure", json={"devices": ["a", "b"], "shots": 0}
    ).status_code in (400, 409)  # 0 shots is refused; the running item refuses first


# ---------------------------------------------------------- save preset


def test_save_preset_writes_then_refuses_to_overwrite_unless_asked(
    client: TestClient, preset_doc: dict
) -> None:
    doc = dict(preset_doc)
    doc["description"] = "saved from the page"
    r = client.post("/api/configs/presets/jet_v2", json={"preset": doc})
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "jet_v2" and r.json()["path"].endswith("jet_v2.yaml")
    assert "commit" in r.json()["message"]
    assert "jet_v2" in client.get("/api/configs/presets").json()["names"]
    saved = client.get("/api/configs/presets/jet_v2").json()
    assert saved["name"] == "jet_v2"  # the URL names the file, not the body
    assert saved["description"] == "saved from the page"
    r = client.post("/api/configs/presets/jet_v2", json={"preset": doc})
    assert r.status_code == 409 and r.json()["error"]["exists"] is True
    doc["description"] = "second"
    r = client.post(
        "/api/configs/presets/jet_v2", json={"preset": doc, "overwrite": True}
    )
    assert r.status_code == 200
    assert client.get("/api/configs/presets/jet_v2").json()["description"] == "second"


def test_save_preset_exists_is_decided_from_the_listing_not_a_message(
    client: TestClient, preset_doc: dict
) -> None:
    """The page's replace dialog keys on 409 + exists; that must not hang on the resolver's wording."""
    r = client.post("/api/configs/presets/eb_align_1hz", json={"preset": preset_doc})
    assert r.status_code == 409 and r.json()["error"]["exists"] is True
    r = client.post(
        "/api/configs/presets/eb_align_1hz.yaml", json={"preset": preset_doc}
    )
    assert r.status_code == 409  # the suffix names the same file


def test_save_preset_refuses_an_invalid_document(client: TestClient) -> None:
    r = client.post("/api/configs/presets/bad", json={"preset": {"devices": "nope"}})
    assert r.status_code == 400
    assert r.json()["error"]["kind"] == "invalid_request"
    assert "bad" not in client.get("/api/configs/presets").json()["names"]
