"""Hermetic tests for the queueserver client seam (qs_client/client.py).

No manager, no network: the real client's lazy ``REManagerAPI`` is replaced
by setting the ``_api`` cache directly (the documented injection point for
tests), and the config reader runs against a temp home.  The submit paths
build ``BPlan``/``BFunc`` items, so the module needs the ``qs-client``
extra (skipped whole without it — the optimize-extra pattern).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("bluesky_queueserver_api")

from geecs_bluesky.qs_client import (  # noqa: E402
    QserverConfig,
    QueueClient,
    QueueStatus,
    StubQueueClient,
    ZmqQueueClient,
    make_queue_client,
    read_qserver_config,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    # The reader expands the "~/..." config path — point the tilde at a
    # temp home on both POSIX and Windows.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


def _write_config(home: Path, body: str) -> None:
    cfg = home / ".config" / "geecs_python_api"
    cfg.mkdir(parents=True)
    (cfg / "config.ini").write_text(body)


class TestReadQserverConfig:
    def test_missing_file_reads_none(self, home):
        assert read_qserver_config() is None

    def test_missing_section_reads_none(self, home):
        _write_config(home, "[Experiment]\nexpt = Undulator\n")
        assert read_qserver_config() is None

    def test_host_derives_the_standard_ports(self, home):
        _write_config(home, "[qserver]\nhost = 192.168.6.99\n")
        config = read_qserver_config()
        assert config == QserverConfig(
            control_addr="tcp://192.168.6.99:60615",
            info_addr="tcp://192.168.6.99:60625",
            doc_addr="192.168.6.99:5568",
        )

    def test_explicit_addresses_win(self, home):
        _write_config(
            home,
            "[qserver]\nhost = h\ncontrol_addr = tcp://h:1\n"
            "info_addr = tcp://h:2\ndoc_addr = h:3\n",
        )
        config = read_qserver_config()
        assert config.control_addr == "tcp://h:1"
        assert config.info_addr == "tcp://h:2"
        assert config.doc_addr == "h:3"

    def test_empty_section_reads_none(self, home):
        _write_config(home, "[qserver]\n")
        assert read_qserver_config() is None

    def test_factory_returns_stub_without_config(self, home):
        client = make_queue_client()
        assert isinstance(client, StubQueueClient)
        assert isinstance(client, QueueClient)  # protocol conformance


class TestStubQueueClient:
    def test_status_disconnected_names_the_config(self):
        status = StubQueueClient().status()
        assert not status.connected
        assert "[qserver]" in status.detail

    def test_verbs_refuse_clearly(self):
        stub = StubQueueClient()
        assert not stub.submit_scan({}).ok
        assert not stub.request_pause()[0]
        with pytest.raises(RuntimeError, match=r"\[qserver\]"):
            stub.move_variable("x", 1.0)
        with pytest.raises(RuntimeError, match=r"\[qserver\]"):
            stub.run_action("close_shutters")
        with pytest.raises(RuntimeError, match=r"\[qserver\]"):
            stub.queue_items()

    def test_stream_addresses_are_none(self):
        stub = StubQueueClient()
        assert stub.info_addr is None and stub.doc_addr is None


class _FakeManagerAPI:
    """Recording stand-in for REManagerAPI (the ``_api`` injection point)."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.status_payloads: list[dict] = []
        self.queue_items: list[dict] = []
        self.running: dict | None = None
        self.task_results: list[dict] = []
        self.raise_on_status: Exception | None = None

    def status(self) -> dict:
        self.calls.append(("status",))
        if self.raise_on_status is not None:
            raise self.raise_on_status
        if self.status_payloads:
            return self.status_payloads.pop(0)
        return {
            "re_state": "idle",
            "manager_state": "idle",
            "worker_environment_exists": True,
            "items_in_queue": len(self.queue_items),
        }

    def queue_get(self) -> dict:
        self.calls.append(("queue_get",))
        return {
            "items": list(self.queue_items),
            "running_item": dict(self.running) if self.running else {},
        }

    def queue_clear(self) -> None:
        self.calls.append(("queue_clear",))
        self.queue_items = []

    def item_add(self, item, *, user=None) -> dict:
        self.calls.append(("item_add", item.to_dict(), user))
        return {"item": {"item_uid": "uid-1"}}

    def queue_start(self) -> None:
        self.calls.append(("queue_start",))

    def queue_stop(self) -> None:
        self.calls.append(("queue_stop",))

    def re_pause(self, option=None) -> None:
        self.calls.append(("re_pause", option))

    def re_resume(self) -> None:
        self.calls.append(("re_resume",))

    def re_stop(self) -> None:
        self.calls.append(("re_stop",))

    def function_execute(self, item, *, user=None) -> dict:
        self.calls.append(("function_execute", item.to_dict(), user))
        return {"task_uid": "task-1"}

    def task_result(self, uid) -> dict:
        self.calls.append(("task_result", uid))
        if self.task_results:
            return self.task_results.pop(0)
        return {"status": "running"}

    def plans_allowed(self, *, reload=False, user_group=None) -> dict:
        self.calls.append(("plans_allowed",))
        return {
            "success": True,
            "plans_allowed": {
                "geecs_scan_request_plan": {"name": "geecs_scan_request_plan"},
                "geecs_run_action_plan": {"name": "geecs_run_action_plan"},
            },
        }

    def close(self) -> None:
        self.calls.append(("close",))


def _client(fake: _FakeManagerAPI) -> ZmqQueueClient:
    client = ZmqQueueClient(
        QserverConfig("tcp://x:1", "tcp://x:2", "x:3"), user="test-console"
    )
    client._api = fake
    return client


class TestZmqQueueClient:
    def test_status_maps_the_manager_payload(self):
        fake = _FakeManagerAPI()
        fake.status_payloads = [
            {
                "re_state": "paused",
                "manager_state": "paused",
                "worker_environment_exists": True,
                "items_in_queue": 2,
                "running_item_uid": "abc",
            }
        ]
        status = _client(fake).status()
        assert status == QueueStatus(
            connected=True,
            re_state="paused",
            manager_state="paused",
            worker_exists=True,
            items_in_queue=2,
            running_item_uid="abc",
        )

    def test_status_failure_reads_disconnected(self):
        fake = _FakeManagerAPI()
        fake.raise_on_status = OSError("timeout occurred")
        status = _client(fake).status()
        assert not status.connected
        assert "timeout" in status.detail

    def test_submit_scan_adds_and_starts(self):
        fake = _FakeManagerAPI()
        result = _client(fake).submit_scan({"mode": "noscan"})
        assert result.ok and result.item_uid == "uid-1"
        names = [c[0] for c in fake.calls]
        assert names == ["queue_get", "item_add", "queue_start"]
        added = fake.calls[1][1]
        assert added["name"] == "geecs_scan_request_plan"
        assert added["args"] == [{"mode": "noscan"}]
        assert fake.calls[1][2] == "test-console"

    def test_submit_scan_surfaces_pending_items_without_clearing(self):
        # The #648 item-3 trap: a failed item returns to the queue FRONT;
        # blind add-and-start would re-run it.
        fake = _FakeManagerAPI()
        fake.queue_items = [{"name": "geecs_scan_request_plan", "item_uid": "old"}]
        result = _client(fake).submit_scan({"mode": "noscan"})
        assert not result.ok
        assert result.pending_items[0]["item_uid"] == "old"
        assert ("queue_clear",) not in fake.calls
        assert not any(c[0] == "item_add" for c in fake.calls)

    def test_submit_scan_clear_pending_clears_then_submits(self):
        fake = _FakeManagerAPI()
        fake.queue_items = [{"item_uid": "old"}]
        result = _client(fake).submit_scan({"mode": "noscan"}, clear_pending=True)
        assert result.ok
        names = [c[0] for c in fake.calls]
        assert names == ["queue_get", "queue_clear", "item_add", "queue_start"]

    def test_submit_action_queues_the_action_plan(self):
        fake = _FakeManagerAPI()
        result = _client(fake).submit_action("close_shutters")
        assert result.ok
        added = next(c[1] for c in fake.calls if c[0] == "item_add")
        assert added["name"] == "geecs_run_action_plan"
        assert added["args"] == ["close_shutters"]

    def test_run_action_raises_the_refusal_with_pending_items(self):
        # The raise-mapping the console's QueueSubmitter used to own.
        fake = _FakeManagerAPI()
        fake.queue_items = [{"item_uid": "old"}]
        with pytest.raises(RuntimeError, match="queue not empty"):
            _client(fake).run_action("close_shutters")

    def test_stream_addresses_come_from_the_config(self):
        client = _client(_FakeManagerAPI())
        assert client.info_addr == "tcp://x:2"
        assert client.doc_addr == "x:3"

    def test_running_item_maps_empty_to_none(self):
        fake = _FakeManagerAPI()
        client = _client(fake)
        assert client.running_item() is None  # manager reports {} while idle
        fake.running = {"item_uid": "r1", "user": "geecs-console"}
        assert client.running_item() == {"item_uid": "r1", "user": "geecs-console"}

    def test_clear_queue_reports_ok_and_failure(self):
        fake = _FakeManagerAPI()
        client = _client(fake)
        assert client.clear_queue() == (True, "queue cleared")
        assert ("queue_clear",) in fake.calls

        def boom():
            raise RuntimeError("manager busy")

        fake.queue_clear = boom
        ok, message = client.clear_queue()
        assert not ok and "busy" in message

    def test_queue_and_history_read_verbs(self):
        fake = _FakeManagerAPI()
        fake.queue_items = [{"item_uid": "a"}]
        fake.history_get = lambda: {"items": [{"item_uid": "done"}]}
        client = _client(fake)
        assert client.queue_items() == [{"item_uid": "a"}]
        assert client.history_items() == [{"item_uid": "done"}]

    def test_stop_from_paused_stops_directly(self):
        fake = _FakeManagerAPI()
        fake.status_payloads = [{"re_state": "paused"}]
        ok, message = _client(fake).stop_scan()
        assert ok and "paused" in message
        assert ("re_stop",) in fake.calls
        assert not any(c[0] == "re_pause" for c in fake.calls)

    def test_stop_from_running_pauses_then_stops(self):
        fake = _FakeManagerAPI()
        fake.status_payloads = [
            {"re_state": "running"},  # initial
            {"re_state": "running"},  # first poll
            {"re_state": "paused"},  # pause landed
        ]
        ok, message = _client(fake).stop_scan()
        assert ok, message
        names = [c[0] for c in fake.calls]
        assert names.index("re_pause") < names.index("re_stop")

    def test_stop_when_idle_reports_nothing_to_stop(self):
        fake = _FakeManagerAPI()
        fake.status_payloads = [{"re_state": "idle"}]
        ok, message = _client(fake).stop_scan()
        assert not ok and "nothing to stop" in message

    def test_move_variable_runs_the_worker_function(self):
        fake = _FakeManagerAPI()
        fake.task_results = [
            {"status": "running"},
            {
                "status": "completed",
                "result": {
                    "success": True,
                    "return_value": {"variable": "u_s1h", "value": 1.0},
                },
            },
        ]
        result = _client(fake).move_variable("u_s1h", 1.0)
        assert result == {"variable": "u_s1h", "value": 1.0}
        executed = next(c[1] for c in fake.calls if c[0] == "function_execute")
        assert executed["name"] == "geecs_move_variable"
        assert executed["args"] == ["u_s1h", 1.0]

    def test_worker_task_failure_raises_with_the_message(self):
        fake = _FakeManagerAPI()
        fake.task_results = [
            {
                "status": "completed",
                "result": {"success": False, "msg": "scan in progress"},
            }
        ]
        with pytest.raises(RuntimeError, match="scan in progress"):
            _client(fake).move_variable("u_s1h", 1.0)

    def test_task_timeout_raises(self, monkeypatch):
        fake = _FakeManagerAPI()  # always "running"
        client = _client(fake)
        # Collapse the polling clock so the timeout path runs instantly.
        ticks = iter([0.0, 0.1, 1000.0])
        monkeypatch.setattr(time, "monotonic", lambda: next(ticks, 2000.0))
        monkeypatch.setattr(time, "sleep", lambda s: None)
        with pytest.raises(RuntimeError, match="did not finish"):
            client.move_variable("u_s1h", 1.0)


class TestQueueStartFailure:
    """#653 review finding 2: a start failure must never silently leave the
    item queued while reporting plain failure."""

    def test_start_refusal_removes_the_item_and_says_so(self):
        fake = _FakeManagerAPI()
        removed: list[str] = []
        fake.queue_start = lambda: (_ for _ in ()).throw(RuntimeError("busy"))

        def item_remove(*, uid=None, pos=None):
            removed.append(uid)

        fake.item_remove = item_remove
        result = _client(fake).submit_scan({"mode": "noscan"})
        assert not result.ok
        assert removed == ["uid-1"]
        assert "removed again" in result.message

    def test_start_refusal_with_failed_removal_names_the_stuck_item(self):
        fake = _FakeManagerAPI()
        fake.queue_start = lambda: (_ for _ in ()).throw(RuntimeError("busy"))

        def item_remove(*, uid=None, pos=None):
            raise RuntimeError("remove refused")

        fake.item_remove = item_remove
        result = _client(fake).submit_scan({"mode": "noscan"})
        assert not result.ok
        assert "REMAINS queued" in result.message
        assert result.item_uid == "uid-1"


class TestPlanListAndClose:
    """#793 part 2 plumbing: the plan-list read verb and connection release."""

    def test_allowed_plan_names_are_sorted_keys(self):
        fake = _FakeManagerAPI()
        assert _client(fake).allowed_plan_names() == [
            "geecs_run_action_plan",
            "geecs_scan_request_plan",
        ]
        assert ("plans_allowed",) in fake.calls

    def test_closed_environment_reads_as_no_plans(self):
        fake = _FakeManagerAPI()
        fake.plans_allowed = lambda **kw: {"success": True, "plans_allowed": {}}
        assert _client(fake).allowed_plan_names() == []

    def test_close_releases_the_api_once_and_is_idempotent(self):
        fake = _FakeManagerAPI()
        client = _client(fake)
        client.close()
        client.close()
        assert fake.calls.count(("close",)) == 1
        assert client._api is None

    def test_stub_refuses_plan_list_and_closes_quietly(self):
        stub = StubQueueClient()
        with pytest.raises(RuntimeError, match="no queueserver configured"):
            stub.allowed_plan_names()
        stub.close()

    def test_submit_verbs_use_the_canonical_plan_names(self):
        from geecs_bluesky.plan_names import RUN_ACTION_PLAN, SCAN_REQUEST_PLAN

        fake = _FakeManagerAPI()
        client = _client(fake)
        client.submit_scan({"mode": "noscan"})
        client.submit_action("reset_plc")
        added = [c[1]["name"] for c in fake.calls if c[0] == "item_add"]
        assert added == [SCAN_REQUEST_PLAN, RUN_ACTION_PLAN]
