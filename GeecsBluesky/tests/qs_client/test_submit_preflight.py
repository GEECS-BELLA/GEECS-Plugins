"""The client-side pre-submit checks over a preset (validate, worker_ready, liveness)."""

from __future__ import annotations

import pytest

from geecs_bluesky.plan_names import GEECS_PLAN_NAMES
from geecs_bluesky.qs_client import submit_preflight
from geecs_bluesky.qs_client.client import QueueStatus, StubQueueClient
from geecs_bluesky.qs_client.submit_preflight import (
    PreflightReport,
    build_submission_record,
    run_submit_preflight,
)
from geecs_schemas import Preset


class _FakeQueueClient:
    """A manager client the worker_ready check reads (status + plan list)."""

    def __init__(self, status=None, plans=None, plans_error=None):
        self._status = status or QueueStatus(
            connected=True, re_state="idle", manager_state="idle", worker_exists=True
        )
        self._plans = list(GEECS_PLAN_NAMES) if plans is None else list(plans)
        self._plans_error = plans_error
        self.closed = 0
        self.status_calls = 0

    def status(self):
        self.status_calls += 1
        return self._status

    def allowed_plan_names(self):
        if self._plans_error is not None:
            raise self._plans_error
        return list(self._plans)

    def close(self):
        self.closed += 1


def _preset(**overrides) -> Preset:
    base = {
        "name": "p",
        "devices": [{"device": "UC_Cam1"}, {"device": "UC_Cam2", "save_images": False}],
        "plan": {"name": "count", "kwargs": {"num": 2}},
    }
    base.update(overrides)
    return Preset.model_validate(base)


@pytest.fixture
def engine(monkeypatch):
    """Patch the seams: the liveness probe is canned, the manager is ready."""
    reads = {"CONNECTED": "Connected"}

    # The liveness check delegates to the shared probe (its own DBR_ENUM
    # datatype=str contract is pinned in tests/test_preflight_connected.py);
    # here the probe outcome is faked from reads["CONNECTED"] — None means
    # unreadable, which the real probe reads as fail-open (not down).
    def fake_probe(experiment, device_names, *, timeout):
        if reads["CONNECTED"] == "Disconnected":
            return list(device_names)
        return []

    monkeypatch.setattr(
        "geecs_bluesky.devices.ca.liveness.probe_disconnected", fake_probe
    )
    monkeypatch.setattr(
        submit_preflight, "_make_default_client", lambda experiment: _FakeQueueClient()
    )
    return reads


class TestRunSubmitPreflight:
    def test_all_green_records_every_pass(self, engine):
        report = run_submit_preflight(_preset(), "Undulator")
        assert report.refusal is None
        assert report.questions == []
        assert report.outcomes == [
            ("validate", "passed", ""),
            ("worker_ready", "passed", ""),
            ("gateway_liveness", "passed", ""),
        ]

    def test_validation_failure_is_a_refusal(self, engine, monkeypatch):
        monkeypatch.setattr(
            submit_preflight,
            "_make_default_client",
            lambda e: pytest.fail("manager must not be asked about an invalid preset"),
        )
        report = run_submit_preflight(_preset(plan=None), "Undulator")
        assert report.refusal is not None and "no plan call" in report.refusal
        assert report.questions == [] and report.outcomes == []

    def test_disconnected_device_raises_a_question(self, engine):
        engine["CONNECTED"] = "Disconnected"
        report = run_submit_preflight(_preset(), "Undulator")
        questions = [q for q in report.questions if q.check == "gateway_liveness"]
        assert len(questions) == 1
        assert "UC_Cam1" in questions[0].message and "UC_Cam2" in questions[0].message

    def test_unreadable_liveness_is_fail_open(self, engine):
        engine["CONNECTED"] = None  # CA read failed — not a verdict
        report = run_submit_preflight(_preset(), "Undulator")
        assert not [q for q in report.questions if q.check == "gateway_liveness"]

    def test_device_less_preset_skips_the_liveness_check(self, engine):
        report = run_submit_preflight(_preset(devices=[]), "Undulator")
        assert report.refusal is None
        assert report.outcomes == [
            ("validate", "passed", ""),
            ("worker_ready", "passed", ""),
        ]


class TestWorkerReady:
    """#793 part 2: the execution surface is checked before queueing."""

    def test_closed_environment_is_a_refusal_naming_the_recovery(
        self, engine, monkeypatch
    ):
        fake = _FakeQueueClient(
            status=QueueStatus(connected=True, re_state=None, worker_exists=False)
        )
        monkeypatch.setattr(submit_preflight, "_make_default_client", lambda e: fake)
        report = run_submit_preflight(_preset(), "Undulator")
        assert report.refusal is not None
        assert "worker environment is closed" in report.refusal
        assert "geecs-qserver-ready" in report.refusal
        assert "qserver environment open" in report.refusal
        assert fake.closed == 1

    def test_missing_plan_is_a_refusal_listing_what_is_allowed(
        self, engine, monkeypatch
    ):
        fake = _FakeQueueClient(plans=["mv", "scan"])
        monkeypatch.setattr(submit_preflight, "_make_default_client", lambda e: fake)
        report = run_submit_preflight(_preset(), "Undulator")
        assert report.refusal is not None
        assert "count" in report.refusal and "mv" in report.refusal

    def test_the_presets_own_plan_is_what_is_expected(self, engine, monkeypatch):
        fake = _FakeQueueClient(plans=["scan"])
        monkeypatch.setattr(submit_preflight, "_make_default_client", lambda e: fake)
        preset = _preset(plan={"name": "scan", "args": ["U_S1H:Current", 0, 1, 2]})
        report = run_submit_preflight(preset, "Undulator")
        assert report.refusal is None
        assert ("worker_ready", "passed", "") in report.outcomes

    def test_unreachable_manager_is_skipped_not_refused(self, engine, monkeypatch):
        fake = _FakeQueueClient(status=QueueStatus(connected=False, detail="timeout"))
        monkeypatch.setattr(submit_preflight, "_make_default_client", lambda e: fake)
        report = run_submit_preflight(_preset(), "Undulator")
        assert report.refusal is None
        skipped = [o for o in report.outcomes if o[0] == "worker_ready"]
        assert skipped and skipped[0][1] == "skipped"

    def test_stub_client_is_skipped(self, engine, monkeypatch):
        monkeypatch.setattr(
            submit_preflight, "_make_default_client", lambda e: StubQueueClient()
        )
        report = run_submit_preflight(_preset(), "Undulator")
        assert report.refusal is None
        assert (
            "worker_ready",
            "skipped",
            "no [qserver] config — submission is off",
        ) in (report.outcomes)

    def test_unanswered_plan_list_is_skipped_with_a_note_not_passed(
        self, engine, monkeypatch
    ):
        fake = _FakeQueueClient(plans_error=TimeoutError("vpn"))
        monkeypatch.setattr(submit_preflight, "_make_default_client", lambda e: fake)
        report = run_submit_preflight(_preset(), "Undulator")
        assert report.refusal is None
        outcome = next(o for o in report.outcomes if o[0] == "worker_ready")
        assert outcome[1] == "skipped" and outcome[2]

    def test_opening_environment_is_a_refusal_saying_retry(self, engine, monkeypatch):
        fake = _FakeQueueClient(
            status=QueueStatus(
                connected=True,
                re_state=None,
                manager_state="creating_environment",
                worker_exists=False,
            )
        )
        monkeypatch.setattr(submit_preflight, "_make_default_client", lambda e: fake)
        report = run_submit_preflight(_preset(), "Undulator")
        assert report.refusal is not None and "retry" in report.refusal.lower()

    def test_empty_plan_list_is_a_refusal(self, engine, monkeypatch):
        fake = _FakeQueueClient(plans=[])
        monkeypatch.setattr(submit_preflight, "_make_default_client", lambda e: fake)
        report = run_submit_preflight(_preset(), "Undulator")
        assert report.refusal is not None

    def test_owned_client_is_closed_but_a_passed_one_is_not(self, engine, monkeypatch):
        owned = _FakeQueueClient()
        monkeypatch.setattr(submit_preflight, "_make_default_client", lambda e: owned)
        run_submit_preflight(_preset(), "Undulator")
        assert owned.closed == 1
        passed = _FakeQueueClient()
        report = run_submit_preflight(_preset(), "Undulator", client=passed)
        assert passed.status_calls == 1 and passed.closed == 0
        assert ("worker_ready", "passed", "") in report.outcomes


class TestBuildSubmissionRecord:
    def test_builds_an_aware_timestamp_and_outcomes(self):
        record = build_submission_record(
            [("validate", "passed", ""), ("gateway_liveness", "continued", "x down")],
            client="geecs-console 0.21.0",
        )
        assert record.client == "geecs-console 0.21.0"
        from datetime import datetime

        assert datetime.fromisoformat(record.submitted_at).tzinfo is not None
        assert [o.check for o in record.preflight] == ["validate", "gateway_liveness"]
        assert record.preflight[1].result.value == "continued"

    def test_record_survives_the_queue_dump(self):
        from geecs_schemas import SubmissionRecord

        record = build_submission_record([], client="c")
        again = SubmissionRecord.model_validate(record.model_dump(mode="json"))
        assert again.client == "c"


class TestReportShape:
    def test_default_report_is_empty(self):
        report = PreflightReport()
        assert report.refusal is None
        assert report.outcomes == [] and report.questions == []
