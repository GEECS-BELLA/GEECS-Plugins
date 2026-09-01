"""Hermetic tests for the v1 control tools.

Same seams as the read-tool suite: fakes patched on ``runtime``, JSON
envelopes asserted.  The submission flow's engine seams
(``run_submit_preflight`` / ``build_submission_record``) are patched at
their ``geecs_bluesky.qs_client`` home — the impl from-imports at call
time.
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
    # v2 seams
    doc_addr: str | None = None
    info_addr: str | None = None
    actions_submitted: list = field(default_factory=list)
    action_ok: bool = True
    action_pending: list = field(default_factory=list)
    describe_steps: list = field(default_factory=list)
    describe_error: str | None = None
    moves: list = field(default_factory=list)
    move_error: str | None = None
    move_result: dict = field(default_factory=lambda: {"variable": "jet_z"})
    pause_result: tuple = (True, "pause requested")
    resume_result: tuple = (True, "resumed")

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

    def submit_scan(self, request, *, submission=None, clear_pending=False):
        assert clear_pending is False  # the doctrine: never clear implicitly
        self.submitted.append(request)
        self.submissions = getattr(self, "submissions", [])
        self.submissions.append(submission)
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

    def submit_action(self, name):
        self.actions_submitted.append(name)
        if self.action_pending:
            return SimpleNamespace(
                ok=False,
                message="queue not empty",
                item_uid=None,
                pending_items=list(self.action_pending),
            )
        if self.action_ok:
            return SimpleNamespace(
                ok=True, message="queued", item_uid="act-1", pending_items=[]
            )
        return SimpleNamespace(
            ok=False, message="unknown action", item_uid=None, pending_items=[]
        )

    def describe_action(self, name):
        if self.describe_error:
            raise RuntimeError(self.describe_error)
        return list(self.describe_steps)

    def move_variable(self, name, value):
        if self.move_error:
            raise RuntimeError(self.move_error)
        self.moves.append((name, value))
        return dict(self.move_result)

    def request_pause(self):
        return self.pause_result

    def request_resume(self):
        return self.resume_result


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
    assert "submission" not in submitted  # request/record split (schemas 0.14.0)
    (record,) = wired.submissions
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
    record = wired.submissions[0]
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


# ---------------------------------------------------------------------------
# run_action / describe_action (v2)
# ---------------------------------------------------------------------------


def test_run_action_happy_path(wired):
    result = _load(control_tools._run_action_impl("Insert Screen"))
    assert result["ok"] and result["item_uid"] == "act-1"
    assert wired.actions_submitted == ["Insert Screen"]
    assert result["submitted_as"] == runtime.client_identity()


def test_run_action_refused_while_scan_active(wired):
    # Submitting mid-scan would silently queue the action to auto-run the
    # moment the scan finishes — the guard this pins.
    wired.re_state = "running"
    result = _load(control_tools._run_action_impl("Insert Screen"))
    assert result["error_kind"] == "policy_refusal" and "idle-only" in result["message"]
    assert wired.actions_submitted == []


def test_run_action_pending_items_surfaced(wired):
    wired.action_pending = [
        {"item_uid": "old", "name": "geecs_scan_request_plan", "user": "console"}
    ]
    result = _load(control_tools._run_action_impl("Insert Screen"))
    assert result["error_kind"] == "policy_refusal"
    assert result["pending_items"][0]["item_uid"] == "old"


def test_run_action_blank_name_refused(wired):
    result = _load(control_tools._run_action_impl("  "))
    assert result["error_kind"] == "invalid_request"


def test_run_action_worker_refusal_verbatim(wired):
    wired.action_ok = False
    result = _load(control_tools._run_action_impl("Bogus"))
    assert result["error_kind"] == "worker_refused"
    assert result["message"] == "unknown action"


def test_describe_action_returns_steps(wired):
    wired.describe_steps = [
        {"kind": "set", "device": "U_Screen4", "variable": "position", "value": "IN"}
    ]
    result = _load(control_tools._describe_action_impl("Insert Screen"))
    assert result["ok"] and result["step_count"] == 1
    assert result["steps"][0]["device"] == "U_Screen4"


def test_describe_action_failure_is_worker_refused(wired):
    wired.describe_error = "manager busy: RE state is running"
    result = _load(control_tools._describe_action_impl("Insert Screen"))
    assert result["error_kind"] == "worker_refused" and "busy" in result["message"]


# ---------------------------------------------------------------------------
# move_scan_variable (v2)
# ---------------------------------------------------------------------------


def test_move_variable_happy_path(wired):
    result = _load(control_tools._move_scan_variable_impl("jet_z", 12.5))
    assert result["ok"] and result["requested"] == 12.5
    assert result["result"] == {"variable": "jet_z"}
    assert wired.moves == [("jet_z", 12.5)]


def test_move_variable_nonfinite_refused(wired):
    for bad in (float("nan"), float("inf")):
        result = _load(control_tools._move_scan_variable_impl("jet_z", bad))
        assert result["error_kind"] == "invalid_request"
    assert wired.moves == []


def test_move_variable_non_number_refused(wired):
    result = _load(control_tools._move_scan_variable_impl("jet_z", "twelve"))
    assert result["error_kind"] == "invalid_request"
    assert wired.moves == []


def test_move_variable_worker_failure_verbatim(wired):
    wired.move_error = "manual-move lock held"
    result = _load(control_tools._move_scan_variable_impl("jet_z", 1.0))
    assert result["error_kind"] == "worker_refused" and "lock" in result["message"]


def test_move_variable_timeout_is_task_timeout(wired):
    wired.move_error = "worker task did not finish within 120 s"
    result = _load(control_tools._move_scan_variable_impl("jet_z", 1.0))
    assert result["error_kind"] == "task_timeout"


def test_names_are_submitted_stripped(wired):
    # Validated stripped ⇒ submitted stripped — " Insert Screen " must
    # not reach the worker as an unknown padded name.
    result = _load(control_tools._run_action_impl("  Insert Screen  "))
    assert result["ok"] and wired.actions_submitted == ["Insert Screen"]
    result = _load(control_tools._move_scan_variable_impl(" jet_z ", 1.0))
    assert result["ok"] and wired.moves[-1] == ("jet_z", 1.0)


# ---------------------------------------------------------------------------
# pause_scan / resume_scan (v2)
# ---------------------------------------------------------------------------


def test_pause_own_scan_proceeds(wired):
    wired.re_state = "running"
    wired.running = {"item_uid": "r1", "user": runtime.client_identity()}
    result = _load(control_tools._pause_scan_impl(False))
    assert result["ok"] and result["forced"] is False


def test_pause_nothing_running(wired):
    result = _load(control_tools._pause_scan_impl(False))
    assert result["error_kind"] == "invalid_request" and "idle" in result["message"]


def test_pause_foreign_scan_refused_then_forced(wired):
    wired.re_state = "running"
    wired.running = {"item_uid": "r1", "user": "geecs-console"}
    result = _load(control_tools._pause_scan_impl(False))
    assert result["error_kind"] == "policy_refusal"
    assert "geecs-console" in result["message"]
    result = _load(control_tools._pause_scan_impl(True))
    assert result["ok"] and result["forced"] is True


def test_pause_failure_is_worker_refused(wired):
    wired.re_state = "running"
    wired.pause_result = (False, "no plan is running")
    result = _load(control_tools._pause_scan_impl(False))
    assert result["error_kind"] == "worker_refused"


def test_resume_requires_paused(wired):
    wired.re_state = "running"
    result = _load(control_tools._resume_scan_impl(False))
    assert result["error_kind"] == "invalid_request" and "running" in result["message"]


def test_resume_own_paused_scan(wired):
    wired.re_state = "paused"
    wired.running = {"item_uid": "r1", "user": runtime.client_identity()}
    result = _load(control_tools._resume_scan_impl(False))
    assert result["ok"] and result["message"] == "resumed"


def test_resume_foreign_scan_refused_then_forced(wired):
    wired.re_state = "paused"
    wired.running = {"item_uid": "r1", "user": "geecs-console"}
    result = _load(control_tools._resume_scan_impl(False))
    assert result["error_kind"] == "policy_refusal"
    result = _load(control_tools._resume_scan_impl(True))
    assert result["ok"] and result["forced"] is True


def test_resume_fails_closed_on_unreadable_ownership(wired, monkeypatch):
    # Review finding #683-2: resume is a GO verb — a transient
    # running-item read failure must not let this client restart another
    # client's paused scan unforced (the halt family stays fail-open).
    wired.re_state = "paused"

    def boom():
        raise RuntimeError("recv timeout")

    monkeypatch.setattr(wired, "running_item", boom)
    result = _load(control_tools._resume_scan_impl(False))
    assert result["error_kind"] == "policy_refusal"
    assert "could not be read" in result["message"]
    # force past unknown ownership works and is audit-marked forced.
    result = _load(control_tools._resume_scan_impl(True))
    assert result["ok"] and result["forced"] is True


def test_pause_stays_fail_open_on_unreadable_ownership(wired, monkeypatch):
    wired.re_state = "running"

    def boom():
        raise RuntimeError("recv timeout")

    monkeypatch.setattr(wired, "running_item", boom)
    result = _load(control_tools._pause_scan_impl(False))
    assert result["ok"] and result["forced"] is False


# ---------------------------------------------------------------------------
# scan_progress + the stream picture (v2)
# ---------------------------------------------------------------------------


class _FakeCache:
    def __init__(self, snapshot):
        self._snapshot = snapshot
        self.started_with = None

    def ensure_started(self, doc_addr, info_addr):
        self.started_with = (doc_addr, info_addr)

    def snapshot(self):
        return dict(self._snapshot)


def test_scan_progress_merges_stream_picture(wired, monkeypatch):
    from geecs_mcp.scans import progress_stream

    wired.re_state = "running"
    wired.doc_addr = "localhost:5568"
    wired.info_addr = "tcp://localhost:60625"
    cache = _FakeCache(
        {
            "available": True,
            "detail": "",
            "scan_number": 42,
            "shots_done": 30,
            "shots_total": 55,
            "paused_reason": "U_Hexapod move failed",
        }
    )
    monkeypatch.setattr(progress_stream, "get_progress_cache", lambda: cache)
    result = _load(control_tools._scan_progress_impl())
    assert cache.started_with == ("localhost:5568", "tcp://localhost:60625")
    assert result["stream"]["shots_done"] == 30
    assert result["stream"]["shots_total"] == 55
    # The failed-move reason is only shown while actually paused (the
    # cache itself clears it on resumed progress — pinned in
    # test_progress_stream.py).
    assert "paused_reason" not in result["stream"]

    wired.re_state = "paused"
    result = _load(control_tools._scan_progress_impl())
    assert result["stream"]["paused_reason"] == "U_Hexapod move failed"


def test_scan_progress_survives_stream_failure(wired, monkeypatch):
    from geecs_mcp.scans import progress_stream

    def boom():
        raise RuntimeError("stream exploded")

    monkeypatch.setattr(progress_stream, "get_progress_cache", boom)
    result = _load(control_tools._scan_progress_impl())
    assert result["ok"] and result["stream"]["available"] is False


def test_all_control_tools_registered():
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
