"""Hermetic tests for the R8 queue panel (app/queue_panel.py).

The summarizers are pure functions over the manager's item shapes; the
controller runs against an in-memory fake client with the same read
verbs as ``QueueClient`` (fetches still ride the daemon-thread worker,
so results are awaited with ``qtbot.waitUntil``).
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QLabel, QPushButton, QTableWidget

from geecs_console.app.queue_panel import (
    COLUMNS,
    QueuePanelController,
    QueueSnapshot,
    build_rows,
    summarize_counts,
    summarize_item,
    summarize_request,
)
from geecs_bluesky.qs_client import QueueStatus


def _scan_item(request: dict, *, user: str = "geecs-console", uid: str = "u1") -> dict:
    return {
        "name": "geecs_scan_request_plan",
        "args": [request],
        "kwargs": {},
        "item_type": "plan",
        "user": user,
        "item_uid": uid,
    }


def _finished(
    item: dict, exit_status: str, *, msg: str = "", stop: float = 0.0
) -> dict:
    return {
        **item,
        "result": {"exit_status": exit_status, "msg": msg, "time_stop": stop},
    }


STEP_REQUEST = {
    "mode": "step",
    "axes": [{"variable": "jet_x", "positions": {"start": 0, "end": 5, "step": 0.5}}],
    "capture": {"shots_per_step": 10},
    "description": "focus check",
}


class TestSummarizeRequest:
    def test_step_scan(self):
        assert (
            summarize_request(STEP_REQUEST)
            == 'jet_x 0 → 5 step 0.5 · 10 shots/step — "focus check"'
        )

    def test_grid_joins_axes_and_values_list_counts(self):
        request = {
            "mode": "step",
            "axes": [
                {"variable": "jet_x", "positions": {"start": 0, "end": 1, "step": 1}},
                {"variable": "jet_z", "positions": {"values": [1.0, 2.0, 4.0]}},
            ],
            "capture": {"shots_per_step": 3},
        }
        assert summarize_request(request) == (
            "jet_x 0 → 1 step 1 × jet_z [3 values] · 3 shots/step"
        )

    def test_noscan_says_shots_not_shots_per_step(self):
        request = {"mode": "noscan", "axes": [], "capture": {"shots_per_step": 50}}
        assert summarize_request(request) == "No-scan · 50 shots"

    def test_background_flag_leads(self):
        request = {
            "mode": "noscan",
            "capture": {"shots_per_step": 5},
            "background": True,
        }
        assert summarize_request(request) == "Background · No-scan · 5 shots"

    def test_optimize_names_the_objectives(self):
        request = {
            "mode": "optimize",
            "optimization": {"objectives": {"charge": "MAXIMIZE"}},
            "capture": {"shots_per_step": 4},
        }
        assert summarize_request(request) == "Optimize charge · 4 shots/step"

    def test_v1_flat_shots_key_still_reads(self):
        # A history item from before the capture block (geecs-schemas < 0.14)
        # must still render — the manager keeps old submissions verbatim.
        request = {"mode": "noscan", "shots_per_step": 7}
        assert summarize_request(request) == "No-scan · 7 shots"

    def test_garbage_never_raises(self):
        assert summarize_request({}) == "No-scan"
        assert summarize_request({"axes": ["nope"], "capture": "x"}) == "No-scan"


class TestSummarizeItem:
    def test_scan_and_action_plans_render_by_argument(self):
        assert summarize_item(_scan_item(STEP_REQUEST)).startswith("jet_x 0 → 5")
        item = {"name": "geecs_run_action_plan", "args": ["shutdown_laser"]}
        assert summarize_item(item) == "Action: shutdown_laser"

    def test_unknown_plan_renders_its_name(self):
        assert summarize_item({"name": "count", "item_type": "plan"}) == "count (plan)"
        assert summarize_item({}) == "? (item)"


class TestBuildRows:
    def test_order_is_running_then_waiting_then_newest_history(self):
        snapshot = QueueSnapshot(
            running=_scan_item(STEP_REQUEST, user="osprey"),
            queued=[
                {"name": "geecs_run_action_plan", "args": ["a1"], "user": "u"},
                {"name": "geecs_run_action_plan", "args": ["a2"], "user": "u"},
            ],
            history=[
                _finished(_scan_item(STEP_REQUEST), "completed", stop=1.0),
                _finished(
                    _scan_item(STEP_REQUEST),
                    "failed",
                    msg="boom\ntraceback...",
                    stop=2.0,
                ),
            ],
        )
        rows = build_rows(snapshot)
        assert [r.state for r in rows] == [
            "running",
            "queued",
            "queued",
            "failed",
            "completed",
        ]
        assert rows[0].user == "osprey"
        assert [r.detail for r in rows[1:3]] == ["#1", "#2"]
        # First message line only, after the finish clock.
        assert rows[3].detail.endswith(" · boom")
        assert "traceback" not in rows[3].detail

    def test_history_limit_keeps_the_newest(self):
        history = [
            _finished(_scan_item({"description": f"s{i}"}), "completed")
            for i in range(5)
        ]
        rows = build_rows(QueueSnapshot(history=history), history_rows=2)
        assert [r.item for r in rows] == ['No-scan — "s4"', 'No-scan — "s3"']
        assert build_rows(QueueSnapshot(history=history), history_rows=0) == []

    def test_counts_line(self):
        assert summarize_counts(QueueSnapshot()) == "Queue empty"
        assert summarize_counts(QueueSnapshot(running={"name": "x"})) == "Running"
        assert (
            summarize_counts(QueueSnapshot(running={"name": "x"}, queued=[{}, {}]))
            == "Running · 2 waiting"
        )


class FakeQueueClient:
    """The QueueClient read verbs + clear, with call counting."""

    def __init__(self):
        self.running = None
        self.queued: list[dict] = []
        self.history: list[dict] = []
        self.fetches = 0
        self.clears = 0
        self.clear_result = (True, "queue cleared")
        self.raise_on_fetch: Exception | None = None

    def running_item(self):
        self.fetches += 1
        if self.raise_on_fetch is not None:
            raise self.raise_on_fetch
        return self.running

    def queue_items(self):
        return list(self.queued)

    def history_items(self):
        return list(self.history)

    def clear_queue(self):
        self.clears += 1
        return self.clear_result


@pytest.fixture
def panel(qtbot):
    table = QTableWidget()
    label = QLabel()
    button = QPushButton()
    qtbot.addWidget(table)
    qtbot.addWidget(label)
    qtbot.addWidget(button)
    client = FakeQueueClient()
    answers = {"confirm": True}
    reports: list[str] = []
    controller = QueuePanelController(
        table=table,
        summary_label=label,
        clear_button=button,
        client_provider=lambda: client,
        confirm=lambda title, message: answers["confirm"],
        report=reports.append,
        fallback_refresh_s=3600.0,
    )
    yield controller, client, table, label, button, answers, reports
    controller.dispose()


def _status(items=0, running_uid=None, re_state="idle", connected=True):
    return QueueStatus(
        connected=connected,
        re_state=re_state,
        worker_exists=connected,
        items_in_queue=items,
        running_item_uid=running_uid,
    )


class TestQueuePanelController:
    def test_table_shape_and_initial_message(self, panel):
        _, _, table, label, button, _, _ = panel
        assert table.columnCount() == len(COLUMNS)
        assert [
            table.horizontalHeaderItem(i).text() for i in range(len(COLUMNS))
        ] == list(COLUMNS)
        assert "waiting for the manager" in label.text()
        assert not button.isEnabled()

    def test_status_change_fetches_and_renders(self, panel, qtbot):
        controller, client, table, label, button, _, _ = panel
        client.running = _scan_item(STEP_REQUEST, user="osprey", uid="r")
        client.queued = [{"name": "geecs_run_action_plan", "args": ["a1"], "user": "c"}]
        controller.on_status(_status(items=1, running_uid="r", re_state="running"))
        qtbot.waitUntil(lambda: table.rowCount() == 2, timeout=3000)
        assert table.item(0, 0).text() == "running"
        assert table.item(0, 2).text() == "osprey"
        assert table.item(1, 1).text() == "Action: a1"
        assert label.text() == "Running · 1 waiting"
        assert button.isEnabled()

    def test_unchanged_status_does_not_refetch(self, panel, qtbot):
        controller, client, table, _, _, _, _ = panel
        controller.on_status(_status())
        qtbot.waitUntil(lambda: client.fetches == 1, timeout=3000)
        qtbot.waitUntil(lambda: not controller._fetch_inflight, timeout=3000)
        controller.on_status(_status())
        controller.on_status(_status())
        assert client.fetches == 1
        controller.on_status(_status(items=1))
        qtbot.waitUntil(lambda: client.fetches == 2, timeout=3000)

    def test_disconnected_status_reads_unreachable_and_disables_clear(
        self, panel, qtbot
    ):
        controller, client, table, label, button, _, _ = panel
        client.queued = [{"name": "x"}]
        controller.on_status(_status(items=1))
        qtbot.waitUntil(lambda: table.rowCount() == 1, timeout=3000)
        assert button.isEnabled()
        controller.on_status(_status(connected=False))
        assert table.rowCount() == 0
        assert "unreachable" in label.text()
        assert not button.isEnabled()

    def test_fetch_failure_renders_unavailable_not_empty(self, panel, qtbot):
        controller, client, table, label, _, _, _ = panel
        client.raise_on_fetch = RuntimeError("socket wedged")
        controller.on_status(_status(items=2))
        qtbot.waitUntil(lambda: "unavailable" in label.text(), timeout=3000)
        assert "socket wedged" in label.text()
        # The failure released the in-flight guard: a later change refetches.
        client.raise_on_fetch = None
        client.queued = [{"name": "x"}]
        controller.on_status(_status(items=1))
        qtbot.waitUntil(lambda: table.rowCount() == 1, timeout=3000)

    def test_clear_declined_leaves_the_queue(self, panel, qtbot):
        controller, client, _, _, button, answers, reports = panel
        controller.on_status(_status(items=1))
        answers["confirm"] = False
        button.click()
        assert client.clears == 0
        assert reports == []

    def test_clear_confirmed_runs_the_verb_and_reports(self, panel, qtbot):
        controller, client, _, _, button, _, reports = panel
        client.queued = [{"name": "x"}]
        controller.on_status(_status(items=1))
        qtbot.waitUntil(lambda: client.fetches == 1, timeout=3000)
        button.click()
        assert not button.isEnabled()  # in flight
        qtbot.waitUntil(lambda: reports == ["queue cleared"], timeout=3000)
        assert client.clears == 1
        # The clear forced a re-read of the queue (a second fetch).
        qtbot.waitUntil(lambda: client.fetches == 2, timeout=3000)

    def test_clear_failure_is_reported(self, panel, qtbot):
        controller, client, _, _, button, _, reports = panel
        client.clear_result = (False, "manager refused")
        controller.on_status(_status(items=1))
        button.click()
        qtbot.waitUntil(lambda: bool(reports), timeout=3000)
        assert reports == ["Clear queue failed: manager refused"]

    def test_dispose_severs_the_window_edges(self, panel):
        controller, client, _, _, _, _, _ = panel
        controller.dispose()
        controller.dispose()  # idempotent
        controller.on_status(_status(items=1))
        assert client.fetches == 0
        assert controller._client_provider() is None
