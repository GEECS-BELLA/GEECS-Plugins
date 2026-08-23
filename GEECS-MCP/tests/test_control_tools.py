"""Hermetic tests for the v1 control tools.

Same seams as the read-tool suite: fakes patched on ``runtime``, JSON
envelopes asserted.  The submission flow's engine seams
(``run_submit_preflight`` / ``stamp_submission``) are patched at their
``geecs_bluesky.qs_client`` home — the impl from-imports at call time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from geecs_bluesky import qs_client
from geecs_mcp import runtime
from geecs_mcp.scans import control_tools


@pytest.fixture(autouse=True)
def _fresh_runtime():
    runtime.clear_runtime_cache()
    yield
    runtime.clear_runtime_cache()


def _load(payload: str) -> dict:
    return json.loads(payload)


GOOD_REQUEST = {
    "mode": "noscan",
    "shots_per_step": 5,
    "acquisition": "free_run",
    "save_sets": ["Amp4In"],
}


@dataclass
class _FakeClient:
    re_state: str = "idle"
    connected: bool = True
    queue: list = field(default_factory=list)
    running: dict | None = None
    history: list = field(default_factory=list)
    submitted: list = field(default_factory=list)
    submit_ok: bool = True
    stop_result: tuple = (True, "stop requested (from paused)")
    cleared: int = 0

    def status(self):
        return SimpleNamespace(
            connected=self.connected,
            re_state=self.re_state if self.connected else None,
            manager_state="idle" if self.connected else None,
            worker_exists=self.connected,
            items_in_queue=len(self.queue),
            running_item_uid=(self.running or {}).get("item_uid"),
            detail="" if self.connected else "timeout occurred",
        )

    def queue_items(self):
        return list(self.queue)

    def running_item(self):
        return dict(self.running) if self.running else None

    def history_items(self):
        return list(self.history)

    def submit_scan(self, request, *, clear_pending=False):
        assert clear_pending is False  # the doctrine: never clear implicitly
        self.submitted.append(request)
        if self.submit_ok:
            return SimpleNamespace(
                ok=True, message="queued", item_uid="uid-9", pending_items=[]
            )
        return SimpleNamespace(
            ok=False, message="refused", item_uid=None, pending_items=[]
        )

    def stop_scan(self):
        return self.stop_result

    def clear_queue(self):
        self.cleared += 1
        return True, "queue cleared"


@pytest.fixture
def wired(monkeypatch):
    """A connected idle manager + experiment + pass-through preflight."""
    client = _FakeClient()
    monkeypatch.setattr(runtime, "get_queue_client", lambda: client)
    monkeypatch.setattr(runtime, "get_experiment", lambda: "Test")
    monkeypatch.setattr(
        "geecs_bluesky.qs_client.run_submit_preflight",
        lambda req, exp: qs_client.PreflightReport(
            outcomes=[("validate", "passed", "")]
        ),
    )
    return client


# ---------------------------------------------------------------------------
# submit_scan
# ---------------------------------------------------------------------------


def test_submit_happy_path_stamps_and_queues(wired):
    result = _load(control_tools._submit_scan_impl(GOOD_REQUEST, None, None, None))
    assert result["ok"] and result["item_uid"] == "uid-9"
    assert result["planned_shots"] == 5
    (submitted,) = wired.submitted
    record = submitted["submission"]
    assert record["client"] == runtime.client_identity()
    assert [o["check"] for o in record["preflight"]] == ["validate"]


def test_submit_requires_exactly_one_selector(wired):
    both = _load(control_tools._submit_scan_impl(GOOD_REQUEST, "preset", None, None))
    neither = _load(control_tools._submit_scan_impl(None, None, None, None))
    assert both["error_kind"] == "invalid_request"
    assert neither["error_kind"] == "invalid_request"


def test_submit_preset_resolves_via_resolver(wired, monkeypatch):
    from geecs_schemas import ScanRequest

    preset_request = ScanRequest.model_validate(GOOD_REQUEST)
    resolver = SimpleNamespace(resolve_preset=lambda name: preset_request)
    monkeypatch.setattr(runtime, "get_resolver", lambda: resolver)
    result = _load(
        control_tools._submit_scan_impl(None, "smoke", "override text", None)
    )
    assert result["ok"]
    assert wired.submitted[0]["description"] == "override text"


def test_submit_enforces_the_shot_cap(wired, monkeypatch):
    monkeypatch.setattr(runtime, "max_shots", lambda: 10)
    big = dict(
        GOOD_REQUEST,
        mode="step",
        axes=[{"variable": "jet_z", "positions": {"start": 0, "end": 9, "step": 1}}],
        shots_per_step=5,
    )
    result = _load(control_tools._submit_scan_impl(big, None, None, None))
    assert result["error_kind"] == "policy_refusal"
    assert "50" in result["message"] and "10" in result["message"]
    assert wired.submitted == []


def _valid_optimization_spec(**overrides) -> dict:
    """A minimal schema-VALID OptimizationSpec (review finding: the old
    fake was invalid on three counts, so the policy branch was never
    genuinely pinned)."""
    spec = {
        "variables": {"jet_z": (0.0, 1.0)},
        "objectives": {"counts": "MAXIMIZE"},
        "evaluator": {"module": "geecs.eval", "class_name": "CountsEval"},
        "generator": {"name": "random"},
    }
    spec.update(overrides)
    return spec


def test_submit_optimize_without_iterations_refused(wired):
    optimize = dict(
        GOOD_REQUEST, mode="optimize", optimization=_valid_optimization_spec()
    )
    result = _load(control_tools._submit_scan_impl(optimize, None, None, None))
    assert result["error_kind"] == "policy_refusal"
    assert "max_iterations" in result["message"]
    assert wired.submitted == []


def test_submit_optimize_with_iterations_counts_against_cap(wired, monkeypatch):
    monkeypatch.setattr(runtime, "max_shots", lambda: 100)
    optimize = dict(
        GOOD_REQUEST,
        mode="optimize",
        optimization=_valid_optimization_spec(max_iterations=30),
        shots_per_step=5,
    )
    result = _load(control_tools._submit_scan_impl(optimize, None, None, None))
    assert result["error_kind"] == "policy_refusal"
    assert "150" in result["message"]
    assert wired.submitted == []


def test_submit_pathological_range_is_counted_not_expanded(wired):
    # Review HIGH: {start: 0, end: 1e15, step: 1e-9} validates cleanly;
    # the cap must refuse it arithmetically — expanding it to count it
    # would OOM the server inside its own guard. Completing at all IS the
    # assertion (the old code would hang here).
    huge = dict(
        GOOD_REQUEST,
        mode="step",
        axes=[
            {
                "variable": "jet_z",
                "positions": {"start": 0.0, "end": 1.0e15, "step": 1.0e-9},
            }
        ],
    )
    result = _load(control_tools._submit_scan_impl(huge, None, None, None))
    assert result["error_kind"] == "policy_refusal"
    assert wired.submitted == []


def test_submit_unknown_acknowledgement_names_refused(wired):
    result = _load(
        control_tools._submit_scan_impl(GOOD_REQUEST, None, None, ["not_a_check"])
    )
    assert result["error_kind"] == "invalid_request"
    assert "not_a_check" in result["message"]
    assert wired.submitted == []


def test_submit_refuses_while_running_or_queued(wired):
    wired.re_state = "running"
    result = _load(control_tools._submit_scan_impl(GOOD_REQUEST, None, None, None))
    assert result["error_kind"] == "policy_refusal" and "active" in result["message"]

    wired.re_state = "idle"
    wired.queue = [
        {"item_uid": "old", "name": "geecs_scan_request_plan", "user": "console"}
    ]
    result = _load(control_tools._submit_scan_impl(GOOD_REQUEST, None, None, None))
    assert result["error_kind"] == "policy_refusal"
    assert result["pending_items"][0]["item_uid"] == "old"
    assert wired.submitted == []


def test_submit_warnings_need_acknowledgement(wired, monkeypatch):
    report = qs_client.PreflightReport(
        outcomes=[("validate", "passed", "")],
        questions=[
            qs_client.PreflightQuestion(
                check="free_run_staleness",
                title="Trigger looks stopped",
                message="acq_timestamp did not advance. Continue anyway?",
            )
        ],
    )
    monkeypatch.setattr(
        "geecs_bluesky.qs_client.run_submit_preflight", lambda req, exp: report
    )
    first = _load(control_tools._submit_scan_impl(GOOD_REQUEST, None, None, None))
    assert first["error_kind"] == "policy_refusal"
    assert first["needs_acknowledgement"][0]["check"] == "free_run_staleness"
    assert wired.submitted == []

    second = _load(
        control_tools._submit_scan_impl(
            GOOD_REQUEST, None, None, ["free_run_staleness"]
        )
    )
    assert second["ok"]
    record = wired.submitted[0]["submission"]
    by_check = {o["check"]: o["result"] for o in record["preflight"]}
    assert by_check["free_run_staleness"] == "continued"


def test_submit_engine_refusal_verbatim(wired, monkeypatch):
    monkeypatch.setattr(
        "geecs_bluesky.qs_client.run_submit_preflight",
        lambda req, exp: qs_client.PreflightReport(
            refusal="save set 'Nope' is unknown"
        ),
    )
    result = _load(control_tools._submit_scan_impl(GOOD_REQUEST, None, None, None))
    assert result["error_kind"] == "invalid_request" and "Nope" in result["message"]


# ---------------------------------------------------------------------------
# stop_scan
# ---------------------------------------------------------------------------


def test_stop_own_scan_proceeds(wired, monkeypatch):
    wired.running = {"item_uid": "r1", "user": runtime.client_identity()}
    result = _load(control_tools._stop_scan_impl(False))
    assert result["ok"] and "stop requested" in result["message"]


def test_stop_foreign_scan_refused_by_name(wired):
    wired.running = {"item_uid": "r1", "user": "geecs-console"}
    result = _load(control_tools._stop_scan_impl(False))
    assert result["error_kind"] == "policy_refusal"
    assert "geecs-console" in result["message"]


def test_stop_foreign_scan_with_force(wired):
    wired.running = {"item_uid": "r1", "user": "geecs-console"}
    result = _load(control_tools._stop_scan_impl(True))
    assert result["ok"] and result["forced"] is True


def test_stop_own_scan_with_force_is_not_marked_forced(wired):
    # The audit marker means "an operator authorized stopping ANOTHER
    # client's scan" — habitual force=true on our own scan must not
    # pollute it (review finding).
    wired.running = {"item_uid": "r1", "user": runtime.client_identity()}
    result = _load(control_tools._stop_scan_impl(True))
    assert result["ok"] and result["forced"] is False


def test_stop_failure_is_worker_refused(wired):
    wired.stop_result = (False, "pause did not land within 120 s")
    result = _load(control_tools._stop_scan_impl(False))
    assert result["error_kind"] == "worker_refused" and "120" in result["message"]


# ---------------------------------------------------------------------------
# clear_queue + scan_progress
# ---------------------------------------------------------------------------


def test_clear_queue_lists_what_it_removed(wired):
    wired.queue = [
        {"item_uid": "old", "name": "geecs_scan_request_plan", "user": "mcp"}
    ]
    result = _load(control_tools._clear_queue_impl())
    assert result["ok"] and result["cleared"][0]["item_uid"] == "old"
    assert wired.cleared == 1

    wired.queue = []
    result = _load(control_tools._clear_queue_impl())
    assert result["ok"] and result["cleared"] == []
    assert wired.cleared == 1  # empty queue: no clear call issued


def test_scan_progress_shapes(wired):
    wired.re_state = "running"
    wired.running = {"item_uid": "r1", "name": "geecs_scan_request_plan", "user": "mcp"}
    wired.history = [
        {
            "name": "geecs_scan_request_plan",
            "user": "console",
            "result": {"exit_status": "completed", "scan_ids": [4]},
        },
    ]
    result = _load(control_tools._scan_progress_impl())
    assert result["state"] == "running"
    assert result["running_item"]["user"] == "mcp"
    assert result["last_completed"]["scan_ids"] == [4]

    wired.connected = False
    result = _load(control_tools._scan_progress_impl())
    assert result["state"] == "unknown" and "timeout" in result["detail"]


def test_all_v1_tools_registered():
    import anyio

    from geecs_mcp import tool_names
    from geecs_mcp.server import create_server

    server = create_server()
    registered = {tool.name for tool in anyio.run(server.list_tools)}
    for name in (
        *tool_names.QUEUE_TOOLS,
        *tool_names.STOP_TOOLS,
        tool_names.SCAN_PROGRESS,
    ):
        assert name in registered, f"{name} not registered"
