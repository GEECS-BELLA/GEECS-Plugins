"""The ONE readiness definition (#793): ``readiness_verdict`` and its wrappers.

Pure, hermetic — no ``bluesky-queueserver-api`` needed for the verdict
itself; the client wrappers are exercised through the same fake manager
API shape ``test_queue_client.py`` uses.
"""

from __future__ import annotations


from geecs_bluesky.plan_names import GEECS_PLAN_NAMES, SCAN_REQUEST_PLAN
from geecs_bluesky.qs_client.client import (
    QserverConfig,
    QueueStatus,
    StubQueueClient,
    ZmqQueueClient,
    queue_status_from_manager,
    readiness_verdict,
)

UP = QueueStatus(
    connected=True, re_state="idle", manager_state="idle", worker_exists=True
)
CLOSED = QueueStatus(
    connected=True, re_state=None, manager_state="idle", worker_exists=False
)
DOWN = QueueStatus(connected=False, detail="timeout")
PLANS = {name: {"name": name} for name in GEECS_PLAN_NAMES}


class TestReadinessVerdict:
    def test_ready_needs_environment_plans_and_the_expected_plan(self):
        verdict = readiness_verdict(UP, PLANS, SCAN_REQUEST_PLAN)
        assert verdict.ready and verdict.state == "ready"
        assert verdict.allowed_plans == tuple(sorted(GEECS_PLAN_NAMES))
        assert readiness_verdict(UP, PLANS, list(GEECS_PLAN_NAMES)).ready
        assert readiness_verdict(UP, list(PLANS), None).ready  # names alone suffice

    def test_unreachable(self):
        verdict = readiness_verdict(DOWN, PLANS, SCAN_REQUEST_PLAN)
        assert not verdict.ready and verdict.state == "unreachable"
        assert "timeout" in verdict.detail

    def test_closed_environment_names_the_recovery_gesture(self):
        verdict = readiness_verdict(CLOSED, {}, SCAN_REQUEST_PLAN)
        assert not verdict.ready and verdict.state == "environment_closed"
        assert "geecs-qserver-ready" in verdict.detail
        assert "qserver environment open" in verdict.detail

    def test_unanswered_plan_list_is_unknown_never_ready(self):
        """The coordination rule: no answer ≠ ready, even with the env up."""
        verdict = readiness_verdict(UP, None, None)
        assert not verdict.ready and verdict.state == "plans_unknown"
        assert verdict.allowed_plans == ()
        assert not readiness_verdict(UP, None, SCAN_REQUEST_PLAN).ready

    def test_empty_plan_list_is_not_ready(self):
        verdict = readiness_verdict(UP, {}, None)
        assert not verdict.ready and verdict.state == "plans_empty"
        assert "Troubleshooting" in verdict.detail

    def test_missing_expected_plan_lists_what_is_allowed(self):
        verdict = readiness_verdict(
            UP, {"geecs_run_action_plan": {}}, SCAN_REQUEST_PLAN
        )
        assert not verdict.ready and verdict.state == "plan_missing"
        assert SCAN_REQUEST_PLAN in verdict.detail
        assert "listed: geecs_run_action_plan" in verdict.detail
        # several expected: every missing one is named
        verdict = readiness_verdict(UP, {"geecs_run_action_plan": {}}, GEECS_PLAN_NAMES)
        assert "geecs_noscan_plan" in verdict.detail
        assert "geecs_run_action_plan" not in verdict.detail.split("(listed")[0]

    def test_precedence_unreachable_before_closed_before_plans(self):
        assert readiness_verdict(DOWN, None, None).state == "unreachable"
        assert readiness_verdict(CLOSED, None, None).state == "environment_closed"
        assert readiness_verdict(UP, None, None).state == "plans_unknown"


def test_queue_status_from_manager_maps_the_payload():
    status = queue_status_from_manager(
        {
            "re_state": "running",
            "manager_state": "executing_queue",
            "worker_environment_exists": True,
            "items_in_queue": "2",
            "running_item_uid": "u1",
        }
    )
    assert status == QueueStatus(
        connected=True,
        re_state="running",
        manager_state="executing_queue",
        worker_exists=True,
        items_in_queue=2,
        running_item_uid="u1",
    )
    assert queue_status_from_manager({}).worker_exists is False


def test_stub_readiness_is_unreachable_naming_the_config():
    verdict = StubQueueClient().readiness(SCAN_REQUEST_PLAN)
    assert not verdict.ready and verdict.state == "unreachable"
    assert "[qserver]" in verdict.detail


class _Api:
    def __init__(self, *, status, plans=None, plans_error=None):
        self._status = status
        self._plans = plans
        self._plans_error = plans_error
        self.calls = []

    def status(self):
        self.calls.append("status")
        return self._status

    def plans_allowed(self, **kw):
        self.calls.append("plans_allowed")
        if self._plans_error:
            raise self._plans_error
        return {"success": True, "plans_allowed": self._plans}


def _client(api) -> ZmqQueueClient:
    client = ZmqQueueClient(QserverConfig("tcp://x:1", "tcp://x:2", "x:3"))
    client._api = api
    return client


class TestZmqReadiness:
    def test_ready_reads_status_then_plans(self):
        api = _Api(
            status={"worker_environment_exists": True, "manager_state": "idle"},
            plans=PLANS,
        )
        assert _client(api).readiness(SCAN_REQUEST_PLAN).ready
        assert api.calls == ["status", "plans_allowed"]

    def test_closed_environment_skips_the_plan_read(self):
        api = _Api(status={"worker_environment_exists": False, "manager_state": "idle"})
        verdict = _client(api).readiness(SCAN_REQUEST_PLAN)
        assert verdict.state == "environment_closed"
        assert api.calls == ["status"]

    def test_plan_read_failure_is_unknown_not_ready(self):
        api = _Api(
            status={"worker_environment_exists": True, "manager_state": "idle"},
            plans_error=RuntimeError("boom"),
        )
        verdict = _client(api).readiness(SCAN_REQUEST_PLAN)
        assert not verdict.ready and verdict.state == "plans_unknown"

    def test_status_failure_is_unreachable(self):
        class _Down:
            def status(self):
                raise RuntimeError("no route")

        verdict = _client(_Down()).readiness(SCAN_REQUEST_PLAN)
        assert verdict.state == "unreachable" and "no route" in verdict.detail
