"""Hermetic tests for the client-side pre-submit preflight (#648 decision 3).

The engine seams (`validate_scan_request`, the resolver, the served-set
provider) are monkeypatched at their `geecs_bluesky` homes — the lazy
imports inside `submit_preflight` resolve at call time, so patching the
source modules is enough.  CA reads are patched at `_read_pv`; the
manager client the ``worker_ready`` check builds is patched at
`_make_default_client` (a ready fake by default — never the real
``[qserver]`` config of the machine running the tests).
"""

from __future__ import annotations

import pytest

from geecs_bluesky.plan_names import GEECS_PLAN_NAMES, SCAN_REQUEST_PLAN
from geecs_bluesky.qs_client import submit_preflight
from geecs_bluesky.qs_client.client import QueueStatus, StubQueueClient
from geecs_bluesky.qs_client.submit_preflight import (
    PreflightReport,
    build_submission_record,
    run_submit_preflight,
)
from geecs_schemas import ScanRequest


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


def _request(**overrides) -> ScanRequest:
    base = dict(
        mode="noscan",
        shots_per_step=2,
        acquisition="free_run",
        save_sets=["UC_Test"],
    )
    base.update(overrides)
    return ScanRequest.model_validate(base)


_DEVICES_CONFIG = {
    "UC_Cam1": {"variable_list": ["MeanCounts"], "synchronous": True},
    "UC_Cam2": {"variable_list": ["Exposure"], "synchronous": False},
}


@pytest.fixture
def engine(monkeypatch):
    """Patch the engine seams: validation passes, devices resolve canned."""

    class _Resolver:
        def __init__(self, experiment):
            self.experiment = experiment

    monkeypatch.setattr("geecs_bluesky.config_resolver.ConfigsRepoResolver", _Resolver)
    monkeypatch.setattr(
        "geecs_bluesky.scan_request_runner.validate_scan_request",
        lambda request, resolver: (request, {}),
    )
    monkeypatch.setattr(
        submit_preflight,
        "_resolve_devices_config",
        lambda request, resolver: dict(_DEVICES_CONFIG),
    )
    # Default: everything served, everything connected, trigger alive.
    monkeypatch.setattr(
        "geecs_bluesky.db_runtime.GeecsDbServedSetProvider",
        _make_provider({"UC_Cam1": {"MeanCounts"}, "UC_Cam2": {"Exposure"}}),
    )
    reads = {"CONNECTED": "Connected", "acq_timestamp": [100.0, 101.5]}

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
    monkeypatch.setattr(submit_preflight, "_read_pv", _make_pv_reader(reads))
    monkeypatch.setattr(submit_preflight, "_STALENESS_WINDOW_S", 0.0)
    # Default: a ready manager (environment open, every GEECS plan listed).
    monkeypatch.setattr(
        submit_preflight, "_make_default_client", lambda experiment: _FakeQueueClient()
    )
    return reads


def _make_provider(served):
    class _Provider:
        def __init__(self, experiment):
            self.experiment = experiment

        def served_by_device(self):
            return served

    return _Provider


def _make_pv_reader(reads):
    state = {"ts_calls": 0}

    def _read(pv, timeout, datatype=None):
        # _read_pv's remaining consumer is the staleness sample (native
        # reads); CONNECTED goes through the shared probe, faked above.
        if pv.endswith(":acq_timestamp"):
            values = reads["acq_timestamp"]
            value = values[min(state["ts_calls"], len(values) - 1)]
            state["ts_calls"] += 1
            return value
        return None

    return _read


class TestRunSubmitPreflight:
    def test_all_green_records_three_passes(self, engine):
        report = run_submit_preflight(_request(), "Undulator")
        assert report.refusal is None
        assert report.questions == []
        assert ("validate", "passed", "") in report.outcomes
        assert ("snapshot_images", "passed", "") in report.outcomes
        assert ("gateway_liveness", "passed", "") in report.outcomes
        assert ("free_run_staleness", "passed", "") in report.outcomes
        assert ("unserved_variables", "passed", "") in report.outcomes
        assert ("worker_ready", "passed", "") in report.outcomes

    def test_validation_failure_is_a_refusal(self, engine, monkeypatch):
        def _boom(request, resolver):
            raise ValueError("save set 'Nope' is unknown")

        monkeypatch.setattr(
            "geecs_bluesky.scan_request_runner.validate_scan_request", _boom
        )
        report = run_submit_preflight(_request(), "Undulator")
        assert report.refusal is not None
        assert "Nope" in report.refusal
        # A refusal short-circuits — no other checks ran.
        assert report.questions == []

    def test_snapshot_images_raises_a_question_not_a_refusal(self, engine, monkeypatch):
        """#754: images: true on a snapshot-role entry surfaces pre-submit as a warning."""
        devices = dict(_DEVICES_CONFIG)
        devices["UC_Slow"] = {
            "variable_list": ["p"],
            "synchronous": False,
            "save_nonscalar_data": True,
        }
        monkeypatch.setattr(
            submit_preflight,
            "_resolve_devices_config",
            lambda request, resolver: devices,
        )
        report = run_submit_preflight(_request(), "Undulator")
        assert report.refusal is None
        questions = [q for q in report.questions if q.check == "snapshot_images"]
        assert len(questions) == 1
        assert "UC_Slow" in questions[0].message
        assert "UC_Cam1" not in questions[0].message
        assert "#754" in questions[0].message
        assert not any(check == "snapshot_images" for check, _, _ in report.outcomes)

    def test_unserved_variable_raises_a_question(self, engine, monkeypatch):
        monkeypatch.setattr(
            "geecs_bluesky.db_runtime.GeecsDbServedSetProvider",
            _make_provider({"UC_Cam1": {"MeanCounts"}, "UC_Cam2": set()}),
        )
        report = run_submit_preflight(_request(), "Undulator")
        questions = [q for q in report.questions if q.check == "unserved_variables"]
        assert len(questions) == 1
        assert "Exposure" in questions[0].message

    def test_unknown_served_set_is_skipped_not_blocking(self, engine, monkeypatch):
        monkeypatch.setattr(
            "geecs_bluesky.db_runtime.GeecsDbServedSetProvider",
            _make_provider(None),
        )
        report = run_submit_preflight(_request(), "Undulator")
        assert report.refusal is None
        assert not [q for q in report.questions if q.check == "unserved_variables"]
        assert any(
            check == "unserved_variables" and result == "skipped"
            for check, result, _ in report.outcomes
        )

    def test_disconnected_device_raises_a_question(self, engine):
        engine["CONNECTED"] = "Disconnected"
        report = run_submit_preflight(_request(), "Undulator")
        questions = [q for q in report.questions if q.check == "gateway_liveness"]
        assert len(questions) == 1
        assert "UC_Cam1" in questions[0].message

    def test_unreadable_liveness_is_fail_open(self, engine):
        engine["CONNECTED"] = None  # CA read failed — not a verdict
        report = run_submit_preflight(_request(), "Undulator")
        assert not [q for q in report.questions if q.check == "gateway_liveness"]

    def test_stale_trigger_raises_a_question(self, engine):
        engine["acq_timestamp"] = [100.0, 100.0]  # not advancing
        report = run_submit_preflight(_request(), "Undulator")
        questions = [q for q in report.questions if q.check == "free_run_staleness"]
        assert len(questions) == 1
        assert "UC_Cam1" in questions[0].message

    def test_strict_request_skips_staleness(self, engine):
        report = run_submit_preflight(_request(acquisition="strict"), "Undulator")
        assert not any(check == "free_run_staleness" for check, _, _ in report.outcomes)
        assert not [q for q in report.questions if q.check == "free_run_staleness"]

    def test_saveset_less_request_skips_device_checks(self, engine, monkeypatch):
        monkeypatch.setattr(
            submit_preflight,
            "_resolve_devices_config",
            lambda request, resolver: {},
        )
        report = run_submit_preflight(_request(), "Undulator")
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
        report = run_submit_preflight(_request(), "Undulator")
        assert report.refusal is not None
        assert "worker environment is closed" in report.refusal
        assert "geecs-qserver-ready" in report.refusal
        assert "qserver environment open" in report.refusal
        # the device checks never ran — nothing to ask about a dead surface
        assert report.questions == []
        assert not any(check == "gateway_liveness" for check, _, _ in report.outcomes)

    def test_missing_plan_is_a_refusal_listing_what_is_allowed(
        self, engine, monkeypatch
    ):
        fake = _FakeQueueClient(plans=["geecs_run_action_plan"])
        monkeypatch.setattr(submit_preflight, "_make_default_client", lambda e: fake)
        report = run_submit_preflight(_request(), "Undulator")
        assert report.refusal is not None
        assert SCAN_REQUEST_PLAN in report.refusal
        assert "geecs_run_action_plan" in report.refusal

    def test_unreachable_manager_is_skipped_not_refused(self, engine, monkeypatch):
        fake = _FakeQueueClient(status=QueueStatus(connected=False, detail="timeout"))
        monkeypatch.setattr(submit_preflight, "_make_default_client", lambda e: fake)
        report = run_submit_preflight(_request(), "Undulator")
        assert report.refusal is None
        assert (
            "worker_ready",
            "skipped",
            "manager unreachable (timeout)",
        ) in report.outcomes
        # the rest of the pipeline still ran
        assert ("gateway_liveness", "passed", "") in report.outcomes

    def test_stub_client_is_skipped(self, engine, monkeypatch):
        monkeypatch.setattr(
            submit_preflight, "_make_default_client", lambda e: StubQueueClient()
        )
        report = run_submit_preflight(_request(), "Undulator")
        assert report.refusal is None
        assert any(
            check == "worker_ready" and result == "skipped" and "[qserver]" in detail
            for check, result, detail in report.outcomes
        )

    def test_plan_list_failure_is_skipped(self, engine, monkeypatch):
        fake = _FakeQueueClient(plans_error=RuntimeError("boom"))
        monkeypatch.setattr(submit_preflight, "_make_default_client", lambda e: fake)
        report = run_submit_preflight(_request(), "Undulator")
        assert report.refusal is None
        assert ("worker_ready", "skipped", "boom") in report.outcomes

    def test_owned_client_is_closed_but_a_passed_one_is_not(self, engine, monkeypatch):
        owned = _FakeQueueClient()
        monkeypatch.setattr(submit_preflight, "_make_default_client", lambda e: owned)
        run_submit_preflight(_request(), "Undulator")
        assert owned.closed == 1

        passed = _FakeQueueClient()
        monkeypatch.setattr(
            submit_preflight,
            "_make_default_client",
            lambda e: pytest.fail("a passed client must be used, not a new one"),
        )
        report = run_submit_preflight(_request(), "Undulator", client=passed)
        assert passed.status_calls == 1
        assert passed.closed == 0
        assert ("worker_ready", "passed", "") in report.outcomes

    def test_validation_refusal_wins_before_the_manager_is_asked(
        self, engine, monkeypatch
    ):
        monkeypatch.setattr(
            "geecs_bluesky.scan_request_runner.validate_scan_request",
            lambda request, resolver: (_ for _ in ()).throw(ValueError("bad request")),
        )
        monkeypatch.setattr(
            submit_preflight,
            "_make_default_client",
            lambda e: pytest.fail("manager must not be asked about an invalid request"),
        )
        report = run_submit_preflight(_request(), "Undulator")
        assert report.refusal == "bad request"


class TestBuildSubmissionRecord:
    def test_builds_an_aware_timestamp_and_outcomes(self):
        record = build_submission_record(
            [("validate", "passed", ""), ("gateway_liveness", "continued", "x down")],
            client="geecs-console 0.21.0",
        )
        assert record.client == "geecs-console 0.21.0"
        # The schema's documented contract: ISO 8601 WITH timezone — a naive
        # datetime.now().isoformat() has no offset and must never appear.
        from datetime import datetime

        assert datetime.fromisoformat(record.submitted_at).tzinfo is not None
        assert [o.check for o in record.preflight] == [
            "validate",
            "gateway_liveness",
        ]
        assert record.preflight[1].result.value == "continued"

    def test_record_survives_the_queue_dump(self):
        # The record travels beside the request as its own JSON dict
        # (request/record split, geecs-schemas 0.14.0).
        from geecs_schemas import SubmissionRecord

        record = build_submission_record([], client="c")
        again = SubmissionRecord.model_validate(record.model_dump(mode="json"))
        assert again.client == "c"


class TestReportShape:
    def test_default_report_is_empty(self):
        report = PreflightReport()
        assert report.refusal is None
        assert report.outcomes == [] and report.questions == []
