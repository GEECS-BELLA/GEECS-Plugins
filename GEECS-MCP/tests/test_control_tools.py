"""Hermetic tests for the control tools: halt, clear-queue, progress.

Same seams as the read-tool suite: fakes patched on ``runtime``, JSON
envelopes asserted.  The write verbs these tests used to cover were
deleted in 0.9.0 — ``_FakeClient`` is now pinned against the real
``QueueClient`` protocol so it cannot outlive the seam again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from geecs_mcp import runtime
from geecs_mcp.scans import control_tools


@pytest.fixture(autouse=True)
def _fresh_runtime():
    runtime.clear_runtime_cache()
    yield
    runtime.clear_runtime_cache()


def _load(payload: str) -> dict:
    return json.loads(payload)


@dataclass
class _FakeClient:
    re_state: str = "idle"
    connected: bool = True
    queue: list = field(default_factory=list)
    running: dict | None = None
    history: list = field(default_factory=list)
    stop_result: tuple = (True, "stop requested (from paused)")
    cleared: int = 0
    doc_addr: str | None = None
    info_addr: str | None = None
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

    def stop_scan(self):
        return self.stop_result

    def clear_queue(self):
        self.cleared += 1
        return True, "queue cleared"

    def request_pause(self):
        return self.pause_result

    def request_resume(self):
        return self.resume_result


@pytest.fixture
def wired(monkeypatch):
    """A connected, idle manager behind the runtime's client seam."""
    client = _FakeClient()
    monkeypatch.setattr(runtime, "get_queue_client", lambda: client)
    return client


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
