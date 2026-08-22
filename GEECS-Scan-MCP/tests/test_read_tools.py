"""Hermetic tests for the v0 read tools.

Each tool's ``_*_impl`` is the tested surface (the async wrappers are
transport glue).  Singletons are monkeypatched on ``runtime`` — the
documented patch seam (tools call ``runtime.get_*()`` through the module
attribute).  Every assertion parses the JSON envelope: ``ok`` plus
payload, or ``{ok: false, error_kind, message}``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from geecs_scan_mcp import runtime
from geecs_scan_mcp.tools import read_tools


@pytest.fixture(autouse=True)
def _fresh_runtime():
    runtime.clear_runtime_cache()
    yield
    runtime.clear_runtime_cache()


def _load(payload: str) -> dict:
    return json.loads(payload)


# ---------------------------------------------------------------------------
# scan_status
# ---------------------------------------------------------------------------


@dataclass
class _FakeClient:
    connected: bool = True
    re_state: str = "idle"
    queue: list = field(default_factory=list)
    history: list = field(default_factory=list)
    raise_on_history: Exception | None = None

    def status(self):
        return SimpleNamespace(
            connected=self.connected,
            re_state=self.re_state if self.connected else None,
            manager_state="idle" if self.connected else None,
            worker_exists=self.connected,
            items_in_queue=len(self.queue),
            running_item_uid=None,
            detail="" if self.connected else "timeout occurred",
        )

    def queue_items(self):
        return list(self.queue)

    def history_items(self):
        if self.raise_on_history is not None:
            raise self.raise_on_history
        return list(self.history)


def test_scan_status_reports_pending_items(monkeypatch):
    client = _FakeClient(
        queue=[{"item_uid": "u1", "name": "geecs_scan_request_plan", "user": "console"}]
    )
    monkeypatch.setattr(runtime, "get_queue_client", lambda: client)
    result = _load(read_tools._scan_status_impl())
    assert result["ok"] and result["connected"]
    assert result["pending_items"] == [
        {"item_uid": "u1", "plan": "geecs_scan_request_plan", "user": "console"}
    ]


def test_scan_status_disconnected_still_answers(monkeypatch):
    monkeypatch.setattr(
        runtime, "get_queue_client", lambda: _FakeClient(connected=False)
    )
    result = _load(read_tools._scan_status_impl())
    assert result["ok"] and not result["connected"]
    assert "timeout" in result["detail"]
    assert result["pending_items"] == []


# ---------------------------------------------------------------------------
# scan_history
# ---------------------------------------------------------------------------


def test_scan_history_maps_fields_defensively(monkeypatch):
    history = [
        {
            "name": "geecs_scan_request_plan",
            "user": "console",
            "result": {"exit_status": "completed", "scan_ids": [3]},
        },
        {
            "name": "geecs_scan_request_plan",
            "user": "mcp",
            "result": {
                "exit_status": "failed",
                "traceback": "...\nGeecsDeviceDownError: gateway reports UC_X as Disconnected",
            },
        },
        {},  # a shape we did not predict must not crash the tool
    ]
    monkeypatch.setattr(
        runtime, "get_queue_client", lambda: _FakeClient(history=history)
    )
    result = _load(read_tools._scan_history_impl(limit=10))
    assert result["ok"] and result["total_in_history"] == 3
    assert result["items"][0]["exit_status"] == "completed"
    assert "UC_X" in result["items"][1]["error"]
    assert result["items"][2]["plan"] is None


def test_scan_history_unreachable_is_an_error_envelope(monkeypatch):
    client = _FakeClient(raise_on_history=RuntimeError("no manager"))
    monkeypatch.setattr(runtime, "get_queue_client", lambda: client)
    result = _load(read_tools._scan_history_impl(limit=5))
    assert not result["ok"] and result["error_kind"] == "manager_unreachable"


# ---------------------------------------------------------------------------
# get_scan_result
# ---------------------------------------------------------------------------


class _FakeCatalog:
    def __init__(self, runs, detail):
        self._runs = runs
        self._detail = detail

    def list_runs(self, experiment, day):
        return self._runs

    def load_run(self, uid):
        if self._detail is None or self._detail.summary.uid != uid:
            raise KeyError(uid)
        return self._detail


def _detail(scan_number=7, uid="uid-7"):
    import pandas as pd

    summary = SimpleNamespace(
        uid=uid,
        scan_number=scan_number,
        start_time=123.0,
        mode="NOSCAN",
        shots=5,
        exit_status="success",
        experiment="Test",
        description="smoke",
        save_sets=("Amp4In",),
    )
    frame = pd.DataFrame({"cam-MeanCounts": [1.0, 2.0, 3.0], "label": ["a", "b", "c"]})
    return SimpleNamespace(
        summary=summary,
        start_doc={
            "scan_folder": "/data/Scan007",
            "submission": {"client": "geecs-scan-mcp 0.1.0"},
        },
        stop_doc={},
        data=frame,
    )


def test_get_scan_result_by_number(monkeypatch):
    detail = _detail()
    monkeypatch.setattr(runtime, "get_experiment", lambda: "Test")
    monkeypatch.setattr(
        runtime, "get_catalog", lambda: _FakeCatalog([detail.summary], detail)
    )
    result = _load(read_tools._get_scan_result_impl(7, "2026-08-22", None))
    assert result["ok"] and result["scan_number"] == 7
    assert result["submission"]["client"].startswith("geecs-scan-mcp")
    assert result["data"]["rows"] == 3
    assert "cam-MeanCounts" in result["data"]["stats"]
    assert result["data"]["stats"]["cam-MeanCounts"]["mean"] == pytest.approx(2.0)
    assert "label" not in result["data"]["stats"]  # non-numeric: named, not summarized
    assert "label" in result["data"]["columns"]


def test_get_scan_result_unknown_number_is_not_found(monkeypatch):
    monkeypatch.setattr(runtime, "get_experiment", lambda: "Test")
    monkeypatch.setattr(runtime, "get_catalog", lambda: _FakeCatalog([], None))
    result = _load(read_tools._get_scan_result_impl(99, None, None))
    assert not result["ok"] and result["error_kind"] == "not_found"
    assert "Scan099" in result["message"]


def test_get_scan_result_needs_number_or_uid(monkeypatch):
    monkeypatch.setattr(runtime, "get_catalog", lambda: _FakeCatalog([], None))
    result = _load(read_tools._get_scan_result_impl(None, None, None))
    assert not result["ok"] and result["error_kind"] == "invalid_request"


def test_get_scan_result_bad_day_is_invalid_request(monkeypatch):
    monkeypatch.setattr(runtime, "get_experiment", lambda: "Test")
    monkeypatch.setattr(runtime, "get_catalog", lambda: _FakeCatalog([], None))
    result = _load(read_tools._get_scan_result_impl(7, "yesterday", None))
    assert not result["ok"] and result["error_kind"] == "invalid_request"


def test_get_scan_result_catalog_failure_is_tiled_unreachable(monkeypatch):
    class _Boom:
        def load_run(self, uid):
            raise ConnectionError("tiled down")

    monkeypatch.setattr(runtime, "get_catalog", lambda: _Boom())
    result = _load(read_tools._get_scan_result_impl(None, None, "uid-1"))
    assert not result["ok"] and result["error_kind"] == "tiled_unreachable"


# ---------------------------------------------------------------------------
# list_scan_configs
# ---------------------------------------------------------------------------


class _FakeResolver:
    def list_save_sets(self):
        return ["Amp4In"]

    def list_trigger_profiles(self):
        return ["HTU-LaserOFF"]

    def list_presets(self):
        return ["basic test"]

    def list_optimizer_configs(self):
        return []

    def scan_variable_catalog(self):
        spec = SimpleNamespace(
            kind=SimpleNamespace(value="pseudo"),
            device=None,
            variable=None,
            min=-1.0,
            max=1.0,
            units="mm",
        )
        return SimpleNamespace(variables={"jet_z": spec})

    def action_plan_registry(self):
        return {"close_shutters": object()}


def test_list_scan_configs_all_kinds(monkeypatch):
    monkeypatch.setattr(runtime, "get_resolver", lambda: _FakeResolver())
    monkeypatch.setattr(runtime, "get_experiment", lambda: "Test")
    assert _load(read_tools._list_scan_configs_impl("save_sets"))["names"] == ["Amp4In"]
    assert _load(read_tools._list_scan_configs_impl("trigger_profiles"))["names"] == [
        "HTU-LaserOFF"
    ]
    assert _load(read_tools._list_scan_configs_impl("presets"))["names"] == [
        "basic test"
    ]
    assert _load(read_tools._list_scan_configs_impl("optimizer_configs"))["names"] == []
    assert _load(read_tools._list_scan_configs_impl("actions"))["names"] == [
        "close_shutters"
    ]
    rows = _load(read_tools._list_scan_configs_impl("scan_variables"))["names"]
    assert rows == [
        {"name": "jet_z", "kind": "pseudo", "min": -1.0, "max": 1.0, "units": "mm"}
    ]


def test_list_scan_configs_bad_kind(monkeypatch):
    monkeypatch.setattr(runtime, "get_resolver", lambda: _FakeResolver())
    result = _load(read_tools._list_scan_configs_impl("nope"))
    assert not result["ok"] and result["error_kind"] == "invalid_request"


def test_list_scan_configs_without_experiment(monkeypatch):
    monkeypatch.setattr(runtime, "get_resolver", lambda: None)
    result = _load(read_tools._list_scan_configs_impl("save_sets"))
    assert not result["ok"] and result["error_kind"] == "invalid_request"


# ---------------------------------------------------------------------------
# validate_scan_request
# ---------------------------------------------------------------------------


def test_validate_bad_shape_is_invalid_not_error(monkeypatch):
    result = _load(read_tools._validate_scan_request_impl({"mode": "no-such-mode"}))
    assert result["ok"] and result["valid"] is False
    assert result["refusal"]


def test_validate_runs_preflight_and_maps_questions(monkeypatch):
    from geecs_bluesky import qs_client

    monkeypatch.setattr(runtime, "get_experiment", lambda: "Test")
    report = qs_client.PreflightReport(
        outcomes=[("validate", "passed", "")],
        questions=[
            qs_client.PreflightQuestion(
                check="gateway_liveness",
                title="Devices disconnected",
                message="UC_X is Disconnected. Continue anyway?",
            )
        ],
    )
    monkeypatch.setattr(
        "geecs_bluesky.qs_client.run_submit_preflight", lambda req, exp: report
    )
    request = {
        "mode": "noscan",
        "shots_per_step": 2,
        "acquisition": "free_run",
        "save_sets": ["Amp4In"],
    }
    result = _load(read_tools._validate_scan_request_impl(request))
    assert result["ok"] and result["valid"] is True
    assert result["warnings"] == [
        {
            "check": "gateway_liveness",
            "title": "Devices disconnected",
            "message": "UC_X is Disconnected. Continue anyway?",
        }
    ]
    assert {"check": "validate", "result": "passed", "detail": ""} in result["outcomes"]


def test_validate_engine_refusal(monkeypatch):
    from geecs_bluesky import qs_client

    monkeypatch.setattr(runtime, "get_experiment", lambda: "Test")
    monkeypatch.setattr(
        "geecs_bluesky.qs_client.run_submit_preflight",
        lambda req, exp: qs_client.PreflightReport(
            refusal="save set 'Nope' is unknown"
        ),
    )
    request = {
        "mode": "noscan",
        "shots_per_step": 2,
        "acquisition": "free_run",
        "save_sets": ["Nope"],
    }
    result = _load(read_tools._validate_scan_request_impl(request))
    assert result["ok"] and result["valid"] is False
    assert "Nope" in result["refusal"]


# ---------------------------------------------------------------------------
# registration + envelope hygiene
# ---------------------------------------------------------------------------


def test_every_v0_tool_is_registered():
    import anyio

    from geecs_scan_mcp import tool_names
    from geecs_scan_mcp.server import create_server

    server = create_server()
    registered = {tool.name for tool in anyio.run(server.list_tools)}
    for name in tool_names.READ_TOOLS:
        assert name in registered, f"{name} not registered on the server"


def test_make_error_rejects_unknown_kind():
    from geecs_scan_mcp.errors import make_error

    with pytest.raises(ValueError):
        make_error("surprise", "boom")
