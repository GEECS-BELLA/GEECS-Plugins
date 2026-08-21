"""Hermetic tests for the client-side pre-submit preflight (#648 decision 3).

The engine seams (`validate_scan_request`, the resolver, the served-set
provider) are monkeypatched at their `geecs_bluesky` homes — the lazy
imports inside `submit_preflight` resolve at call time, so patching the
source modules is enough.  CA reads are patched at `_read_pv`.
"""

from __future__ import annotations

import pytest

from geecs_console.services import submit_preflight
from geecs_console.services.submit_preflight import (
    PreflightReport,
    run_submit_preflight,
    stamp_submission,
)
from geecs_schemas import ScanRequest


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
    monkeypatch.setattr(submit_preflight, "_read_pv", _make_pv_reader(reads))
    monkeypatch.setattr(submit_preflight, "_STALENESS_WINDOW_S", 0.0)
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

    def _read(pv, timeout):
        if pv.endswith(":connected"):
            return reads["CONNECTED"]
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
        assert ("gateway_liveness", "passed", "") in report.outcomes
        assert ("free_run_staleness", "passed", "") in report.outcomes
        assert ("unserved_variables", "passed", "") in report.outcomes

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
        assert report.outcomes == [("validate", "passed", "")]


class TestStampSubmission:
    def test_stamps_an_aware_timestamp_and_outcomes(self):
        request = _request()
        stamped = stamp_submission(
            request,
            [("validate", "passed", ""), ("gateway_liveness", "continued", "x down")],
            client="geecs-console 0.21.0",
        )
        record = stamped.submission
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
        # The original request is untouched (model_copy semantics).
        assert request.submission is None

    def test_stamped_request_survives_the_queue_dump(self):
        stamped = stamp_submission(_request(), [], client="c")
        again = ScanRequest.model_validate(stamped.model_dump(mode="json"))
        assert again.submission.client == "c"


class TestReportShape:
    def test_default_report_is_empty(self):
        report = PreflightReport()
        assert report.refusal is None
        assert report.outcomes == [] and report.questions == []
