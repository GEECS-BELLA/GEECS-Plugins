"""The client-side pre-submit checks over a preset (validate, worker_ready, liveness)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from geecs_bluesky.plan_names import GEECS_PLAN_NAMES
from geecs_bluesky.qs_client import submit_preflight
from geecs_bluesky.qs_client.client import QueueStatus, StubQueueClient
from geecs_bluesky.qs_client.submit_preflight import (
    PreflightReport,
    build_submission_record,
    run_submit_preflight,
)
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_schemas import Preset
from geecs_schemas.trigger_profile import TriggerProfile


class _FakeQueueClient:
    """A manager client the worker_ready check reads (status + plan list)."""

    def __init__(self, status=None, plans=None, plans_error=None, devices=None):
        self._status = status or QueueStatus(
            connected=True, re_state="idle", manager_state="idle", worker_exists=True
        )
        self._plans = list(GEECS_PLAN_NAMES) if plans is None else list(plans)
        self._plans_error = plans_error
        self._devices = (
            [
                "UC_Cam1",
                "UC_Cam1.scalars",
                "UC_Cam1.acq_timestamp",
                "UC_Cam2",
                "UC_Cam2.scalars",
                "UC_Cam2.acq_timestamp",
                "U_S1H",
                "U_S1H.current",
            ]
            if devices is None
            else list(devices)
        )
        self.closed = 0
        self.status_calls = 0

    def status(self):
        self.status_calls += 1
        return self._status

    def allowed_plan_names(self):
        if self._plans_error is not None:
            raise self._plans_error
        return list(self._plans)

    def allowed_device_names(self):
        return list(self._devices)

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
    # No configs repo behind the default resolver: no default profile, so the
    # liveness list is the preset's devices alone unless a test says otherwise.
    monkeypatch.setattr(
        submit_preflight, "_make_default_resolver", lambda experiment: _FakeResolver()
    )
    return reads


class _FakeResolver:
    """A configs-repo resolver double: named trigger profiles + the defaults."""

    def __init__(self, profiles=None, default=None):
        self._profiles = dict(profiles or {})
        self._default = default

    def resolve_experiment_defaults(self):
        if self._default is None:
            return None
        return SimpleNamespace(trigger_profile=self._default)

    def resolve_trigger_profile(self, name):
        try:
            return self._profiles[name]
        except KeyError:
            raise GeecsConfigurationError(
                f"trigger profile {name!r} not found"
            ) from None


def _profile(
    name: str = "HTU-Test", device: str = "U_DG645_ShotControl"
) -> TriggerProfile:
    return TriggerProfile.model_validate(
        {
            "name": name,
            "states": {
                "ARMED": [
                    {"device": device, "variable": "Trigger.Source", "value": "single"}
                ],
                "STANDBY": [
                    {"device": device, "variable": "Trigger.Source", "value": "edges"},
                    {"device": "U_Shutter", "variable": "State", "value": "open"},
                ],
                "SINGLESHOT": [{"device": device, "variable": "Fire", "value": "on"}],
            },
        }
    )


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

    def test_the_question_says_which_down_devices_refuse_and_which_are_left_out(
        self, engine
    ):
        # The worker's rule (registry.drop_down_devices): required refuses,
        # the rest is left out of the scan.
        engine["CONNECTED"] = "Disconnected"
        preset = _preset(
            devices=[
                {"device": "UC_Cam1"},
                {"device": "UC_Cam2", "save_images": False, "required": False},
            ]
        )
        (question,) = run_submit_preflight(preset, "Undulator").questions
        refuses, _, left_out = question.message.partition("Not required")
        assert "UC_Cam1" in refuses and "refuses the run" in refuses
        assert "UC_Cam2" not in refuses
        assert "UC_Cam2" in left_out and "leaves them out" in left_out

    def test_a_down_device_the_run_does_not_need_is_a_note_not_a_question(
        self, engine, monkeypatch
    ):
        """The worker leaves it out by itself: nothing for the operator to acknowledge."""

        def probe(experiment, device_names, *, timeout):
            return ["UC_Cam2"]

        monkeypatch.setattr(
            "geecs_bluesky.devices.ca.liveness.probe_disconnected", probe
        )
        preset = _preset(
            devices=[
                {"device": "UC_Cam1"},
                {"device": "UC_Cam2", "save_images": False, "required": False},
            ]
        )
        report = run_submit_preflight(preset, "Undulator")
        assert report.refusal is None and report.questions == []
        assert (
            "gateway_liveness",
            "passed",
            "left out, disconnected and not required: UC_Cam2",
        ) in report.outcomes

    def test_every_listed_device_down_is_still_a_question(self, engine, monkeypatch):
        """None needed, all down: the worker refuses the run, so the operator is told."""
        monkeypatch.setattr(
            "geecs_bluesky.devices.ca.liveness.probe_disconnected",
            lambda experiment, device_names, *, timeout: list(device_names),
        )
        preset = _preset(
            devices=[
                {"device": "UC_Cam1", "required": False},
                {"device": "UC_Cam2", "save_images": False, "required": False},
            ]
        )
        (question,) = run_submit_preflight(preset, "Undulator").questions
        assert "Every device the scan lists is Disconnected" in question.message
        assert "refuses the run" in question.message

    def test_a_sweep_axis_device_is_probed_and_refuses(self, engine, monkeypatch):
        calls: list[list[str]] = []

        def probe(experiment, device_names, *, timeout):
            calls.append(list(device_names))
            return ["U_S1H"]

        monkeypatch.setattr(
            "geecs_bluesky.devices.ca.liveness.probe_disconnected", probe
        )
        sweep = {
            "trajectory": {
                "kind": "axes",
                "axes": [
                    {
                        "kind": "range",
                        "axis": "U_S1H:Current",
                        "start": 0,
                        "stop": 1,
                        "num": 2,
                    }
                ],
            }
        }
        preset = _preset(
            devices=[{"device": "UC_Cam1"}],
            plan={"name": "sweep", "kwargs": {"sweep": sweep}},
        )
        (question,) = run_submit_preflight(preset, "Undulator").questions
        assert calls == [["UC_Cam1", "U_S1H"]]
        assert "U_S1H" in question.message and "Not required" not in question.message

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


class TestTriggerProfileLiveness:
    """GEECS-Plugins#852 part 2: the box's devices are on the liveness list."""

    @staticmethod
    def _probed(monkeypatch) -> list[list[str]]:
        calls: list[list[str]] = []

        def probe(experiment, device_names, *, timeout):
            calls.append(list(device_names))
            return list(device_names)  # everything down: the question names all

        monkeypatch.setattr(
            "geecs_bluesky.devices.ca.liveness.probe_disconnected", probe
        )
        return calls

    def test_the_presets_profile_devices_join_the_list(self, engine, monkeypatch):
        calls = self._probed(monkeypatch)
        resolver = _FakeResolver({"HTU-LaserOFF": _profile("HTU-LaserOFF")})
        report = run_submit_preflight(
            _preset(trigger_profile="HTU-LaserOFF"), "Undulator", resolver=resolver
        )
        # Every device the profile writes, after the preset's, no duplicates.
        assert calls == [["UC_Cam1", "UC_Cam2", "U_DG645_ShotControl", "U_Shutter"]]
        (question,) = [q for q in report.questions if q.check == "gateway_liveness"]
        assert "U_DG645_ShotControl" in question.message

    def test_the_experiment_default_profile_is_used_when_the_preset_names_none(
        self, engine, monkeypatch
    ):
        calls = self._probed(monkeypatch)
        resolver = _FakeResolver({"HTU-Test": _profile()}, default="HTU-Test")
        run_submit_preflight(_preset(), "Undulator", resolver=resolver)
        assert calls == [["UC_Cam1", "UC_Cam2", "U_DG645_ShotControl", "U_Shutter"]]

    def test_a_device_less_preset_still_probes_the_box(self, engine, monkeypatch):
        calls = self._probed(monkeypatch)
        resolver = _FakeResolver({"HTU-Test": _profile()}, default="HTU-Test")
        report = run_submit_preflight(
            _preset(devices=[]), "Undulator", resolver=resolver
        )
        assert calls == [["U_DG645_ShotControl", "U_Shutter"]]
        assert [q.check for q in report.questions] == ["gateway_liveness"]

    def test_an_unresolvable_profile_is_fail_open(self, engine, monkeypatch, caplog):
        calls = self._probed(monkeypatch)
        resolver = _FakeResolver({})  # the named profile does not exist here
        with caplog.at_level("WARNING", logger="geecs_bluesky.qs_client"):
            report = run_submit_preflight(
                _preset(trigger_profile="HTU-Missing"), "Undulator", resolver=resolver
            )
        assert calls == [["UC_Cam1", "UC_Cam2"]]  # the preset's devices alone
        assert report.refusal is None
        assert any("HTU-Missing" in r.message for r in caplog.records)

    def test_the_default_resolver_seam_is_used_when_none_is_given(
        self, engine, monkeypatch
    ):
        calls = self._probed(monkeypatch)
        monkeypatch.setattr(
            submit_preflight,
            "_make_default_resolver",
            lambda experiment: _FakeResolver(
                {"HTU-Test": _profile()}, default="HTU-Test"
            ),
        )
        run_submit_preflight(_preset(), "Undulator")
        assert calls == [["UC_Cam1", "UC_Cam2", "U_DG645_ShotControl", "U_Shutter"]]


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
        fake = _FakeQueueClient(plans=["sweep"])
        monkeypatch.setattr(submit_preflight, "_make_default_client", lambda e: fake)
        preset = _preset(
            plan={
                "name": "sweep",
                "kwargs": {
                    "sweep": {
                        "trajectory": {
                            "kind": "axes",
                            "axes": [
                                {
                                    "kind": "range",
                                    "axis": "U_S1H:Current",
                                    "start": 0,
                                    "stop": 1,
                                    "num": 2,
                                }
                            ],
                        }
                    }
                },
            }
        )
        report = run_submit_preflight(preset, "Undulator")
        assert report.refusal is None
        assert ("worker_ready", "passed", "") in report.outcomes

    def test_an_unknown_device_not_required_is_left_out_not_refused(
        self, engine, monkeypatch
    ):
        """As the submit leaves it out: the outcome names it, nothing refuses."""
        fake = _FakeQueueClient(devices=["UC_Cam1", "UC_Cam1.acq_timestamp"])
        monkeypatch.setattr(submit_preflight, "_make_default_client", lambda e: fake)
        preset = _preset(
            devices=[
                {"device": "UC_Cam1"},
                {"device": "UC_Off", "required": False, "save_images": False},
            ]
        )
        report = run_submit_preflight(preset, "Undulator")
        assert report.refusal is None
        assert (
            "worker_ready",
            "passed",
            "left out, unknown to the worker and not required: UC_Off",
        ) in report.outcomes

    def test_an_unknown_required_device_is_still_refused(self, engine, monkeypatch):
        fake = _FakeQueueClient(devices=["UC_Cam1", "UC_Cam1.acq_timestamp"])
        monkeypatch.setattr(submit_preflight, "_make_default_client", lambda e: fake)
        preset = _preset(
            devices=[
                {"device": "UC_Cam1", "required": False},
                {"device": "UC_Need", "save_images": False},
            ]
        )
        report = run_submit_preflight(preset, "Undulator")
        assert report.refusal and "UC_Need.scalars" in report.refusal

    def test_unknown_device_reference_is_a_refusal(self, engine, monkeypatch):
        fake = _FakeQueueClient(devices=["UC_Cam1", "UC_Cam1.scalars"])
        monkeypatch.setattr(submit_preflight, "_make_default_client", lambda e: fake)
        preset = _preset(
            devices=[
                {"device": "UC_Cam1"},
                {"device": "UC_Typo", "save_images": False},
            ],
            plan={
                "name": "sweep",
                "kwargs": {
                    "sweep": {
                        "trajectory": {
                            "kind": "axes",
                            "axes": [
                                {
                                    "kind": "range",
                                    "axis": "U_S1H:Current",
                                    "start": 0,
                                    "stop": 1,
                                    "num": 2,
                                }
                            ],
                        }
                    }
                },
            },
            trigger_profile="HTU-Normal",
        )
        report = run_submit_preflight(preset, "Undulator")
        assert report.refusal is not None
        assert "UC_Typo.scalars" in report.refusal and "U_S1H.current" in report.refusal
        assert "HTU-Normal" not in report.refusal

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


class TestAcquisitionRules:
    """The expansion over the manager's device tree, and the retired mode."""

    TREE = [
        "UC_Plugin",
        "UC_Plugin.scalars",
        "UC_Plugin.hdf",
        "UC_Plugin.acq_timestamp",
        "UC_Native",
        "UC_Native.scalars",
        "UC_Native.save",
        "UC_Native.acq_timestamp",
        "U_ICT",
        "U_ICT.scalars",
        "U_ICT.acq_timestamp",
        "U_Gauge",
        "U_Gauge.scalars",
    ]

    def test_a_condition_is_left_out_of_required(self, engine):
        """A device with no stamp (a free-running gauge) is a condition, read in
        every row, whatever its ``required`` says: the client leaves it out.
        """
        from geecs_bluesky.qs_client.presets import expand_preset

        preset = _preset(
            devices=[{"device": "UC_Plugin", "required": False}, {"device": "U_Gauge"}]
        )
        item = expand_preset(preset, device_tree=set(self.TREE))
        assert item.args[0] == ["UC_Plugin", "U_Gauge"]
        assert item.kwargs["required"] == []

    def test_a_gated_preset_is_refused_before_the_manager_is_asked(
        self, engine, monkeypatch
    ):
        """The retired mode is refused at ``validate``, in its own words."""
        monkeypatch.setattr(
            submit_preflight,
            "_make_default_client",
            lambda e: pytest.fail("the manager must not be asked about it"),
        )
        preset = _preset(
            devices=[{"device": "UC_Plugin"}],
            plan={"name": "count", "kwargs": {"num": 3, "acquisition": "gated"}},
        )
        report = run_submit_preflight(preset, "TestExp")
        assert report.refusal and "acquisition='gated' is retired" in report.refusal
        assert "use acquisition='strict'" in report.refusal
        assert report.outcomes == [] and report.questions == []


class TestInlineTriggerProfile:
    """A draft profile checked by the worker's own rules before it is saved (the configurator's dry run)."""

    def test_the_inline_document_replaces_the_one_on_disk(self, engine, monkeypatch):
        calls = TestTriggerProfileLiveness._probed(monkeypatch)
        on_disk = _FakeResolver({"HTU-LaserOFF": _profile("HTU-LaserOFF")})
        draft = _profile("HTU-LaserOFF", device="U_DG645_New")
        run_submit_preflight(
            _preset(trigger_profile="HTU-LaserOFF"),
            "Undulator",
            resolver=on_disk,
            trigger_profile=draft,
        )
        assert calls == [["UC_Cam1", "UC_Cam2", "U_DG645_New", "U_Shutter"]]

    def test_a_scan_mode_draft_refuses_a_fly_preset_naming_it(self, engine):
        draft = _profile("HTU-LaserOFF").model_copy(update={"use_scan_mode": True})
        on_disk = _FakeResolver({"HTU-LaserOFF": _profile("HTU-LaserOFF")})
        preset = _preset(
            trigger_profile="HTU-LaserOFF",
            plan={"name": "count", "kwargs": {"num": 2, "acquisition": "fly"}},
        )
        report = run_submit_preflight(
            preset,
            "Undulator",
            resolver=on_disk,
            trigger_profile=draft,
            client=_FakeQueueClient(),
        )
        assert report.refusal and "scan mode" in report.refusal
        # the profile on disk (no scan mode) is not what was checked
        report = run_submit_preflight(
            preset, "Undulator", resolver=on_disk, client=_FakeQueueClient()
        )
        assert report.refusal is None

    def test_the_profile_alone_probes_every_device_it_writes(self, engine, monkeypatch):
        from geecs_bluesky.qs_client import run_trigger_profile_preflight

        calls = TestTriggerProfileLiveness._probed(monkeypatch)
        report = run_trigger_profile_preflight(
            _profile().model_dump(mode="json"), "Undulator"
        )
        assert report.refusal is None
        assert report.outcomes[0] == ("validate", "passed", "")
        assert calls == [["U_DG645_ShotControl", "U_Shutter"]]
        assert [q.check for q in report.questions] == ["gateway_liveness"]

    def test_a_profile_the_worker_cannot_build_is_refused(self, engine, monkeypatch):
        from geecs_bluesky.qs_client import run_trigger_profile_preflight

        calls = TestTriggerProfileLiveness._probed(monkeypatch)
        report = run_trigger_profile_preflight({"name": "empty", "states": {}}, "X")
        assert report.refusal and "names no trigger device" in report.refusal
        assert calls == []  # refused before any read

    def test_a_document_that_does_not_validate_is_refused(self, engine):
        from geecs_bluesky.qs_client import run_trigger_profile_preflight

        report = run_trigger_profile_preflight({"states": "nonsense"}, "X")
        assert report.refusal and report.outcomes == []


def test_a_preset_whose_every_device_is_unknown_is_refused(engine, monkeypatch):
    """The expansion's refusal is the preflight's, never a fail-open skip."""
    fake = _FakeQueueClient(devices=["UC_Other", "UC_Other.acq_timestamp"])
    monkeypatch.setattr(submit_preflight, "_make_default_client", lambda e: fake)
    preset = _preset(devices=[{"device": "UC_Gone", "required": False}])
    report = run_submit_preflight(preset, "Undulator")
    assert report.refusal and "knows none of its devices" in report.refusal
