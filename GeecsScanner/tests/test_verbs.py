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


def test_move_accepts_device_colon_variable_and_a_pseudo(
    client: TestClient, manager: DemoQueueClient
) -> None:
    r = client.post("/api/move", json={"variable": "U_Hexapod:xpos", "value": 2})
    assert r.status_code == 200 and r.json()["reference"] == "U_Hexapod.xpos"
    manager.step()  # the fake worker finishes the move
    # a pseudo moves as its namespace noun (a manual bump)
    r = client.post("/api/move", json={"variable": "Gas jet 2-axis", "value": 0})
    assert r.status_code == 200 and r.json()["reference"] == "gas_jet_2_axis"
    manager.step()
    r = client.post("/api/move", json={"variable": "  ", "value": 0})
    assert r.status_code == 400


def test_idle_only_items_refuse_while_a_plan_runs_or_waits(
    client: TestClient, preset_doc: dict, manager: DemoQueueClient
) -> None:
    _start_a_scan(client, preset_doc)
    for path, body in (
        ("/api/move", {"variable": "S1H current", "value": 0}),
        (
            "/api/setups/shutters_closed/apply",
            {"operator": "ada", "acknowledged": ["*"]},
        ),
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
    manager._queue.append(
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
    r = client.post(
        "/api/configs/presets/jet_v2", json={"preset": doc, "operator": "Op"}
    )
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "jet_v2" and r.json()["path"].endswith("jet_v2.yaml")
    assert "commit" in r.json()["message"]
    assert "jet_v2" in client.get("/api/configs/presets").json()["names"]
    saved = client.get("/api/configs/presets/jet_v2").json()
    assert saved["name"] == "jet_v2"  # the URL names the file, not the body
    assert saved["description"] == "saved from the page"
    r = client.post(
        "/api/configs/presets/jet_v2", json={"preset": doc, "operator": "Op"}
    )
    assert r.status_code == 409 and r.json()["error"]["exists"] is True
    doc["description"] = "second"
    r = client.post(
        "/api/configs/presets/jet_v2",
        json={"preset": doc, "overwrite": True, "operator": "Op"},
    )
    assert r.status_code == 200
    assert client.get("/api/configs/presets/jet_v2").json()["description"] == "second"


def test_save_preset_exists_is_decided_from_the_listing_not_a_message(
    client: TestClient, preset_doc: dict
) -> None:
    """The page's replace dialog keys on 409 + exists; that must not hang on the resolver's wording."""
    r = client.post(
        "/api/configs/presets/eb_align_1hz",
        json={"preset": preset_doc, "operator": "Op"},
    )
    assert r.status_code == 409 and r.json()["error"]["exists"] is True
    r = client.post(
        "/api/configs/presets/eb_align_1hz.yaml",
        json={"preset": preset_doc, "operator": "Op"},
    )
    assert r.status_code == 409  # the suffix names the same file


def test_save_preset_refuses_an_invalid_document(
    client: TestClient, preset_doc: dict
) -> None:
    r = client.post(
        "/api/configs/presets/bad",
        json={"preset": {"devices": "nope"}, "operator": "Op"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["kind"] == "invalid_request"
    assert "bad" not in client.get("/api/configs/presets").json()["names"]
    # a name that is only a suffix strips to nothing and is refused, not written as ''
    r = client.post(
        "/api/configs/presets/.yaml", json={"preset": preset_doc, "operator": "Op"}
    )
    assert r.status_code == 400 and "not a file name" in r.json()["error"]["message"]
    assert "" not in client.get("/api/configs/presets").json()["names"]


def test_save_cannot_overwrite_with_schema_invalid_sweep(client, preset_doc):
    original = client.get("/api/configs/presets/jet_pressure_sweep").json()
    preset_doc["plan"] = {
        "name": "sweep",
        "kwargs": {
            "sweep": {
                "trajectory": {
                    "kind": "axes",
                    "axes": [
                        {"kind": "list", "axis": "A", "positions": [1, 2]},
                        {"kind": "list", "axis": "B", "positions": [1, 2, 3]},
                    ],
                }
            }
        },
    }
    response = client.post(
        "/api/configs/presets/jet_pressure_sweep",
        json={"preset": preset_doc, "overwrite": True, "operator": "Op"},
    )
    assert response.status_code == 400
    assert "equal point counts" in str(response.json())
    assert client.get("/api/configs/presets/jet_pressure_sweep").json() == original


def test_save_valid_sweep_does_not_require_preview(client, preset_doc, monkeypatch):
    from geecs_scanner.service import trajectory

    def no_preview(*a, **kw):
        raise AssertionError("saving must not need a numerical preview")

    monkeypatch.setattr(trajectory, "preview", no_preview)
    response = client.post(
        "/api/configs/presets/no_preview", json={"preset": preset_doc, "operator": "Op"}
    )
    assert response.status_code == 200, response.text
