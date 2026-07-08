"""Model tests for ExperimentDefaults."""

import inspect

import pytest
from pydantic import ValidationError

from geecs_schemas import SCHEMA_REGISTRY, ExperimentDefaults


def make_defaults():
    return ExperimentDefaults.model_validate(
        {
            "trigger_profile": "htu_shot_control",
            "actions": {
                "setup": ["pre_scan_checklist"],
                "closeout": ["experiment_closeout"],
            },
            "description": "HTU standing defaults",
        }
    )


class TestExperimentDefaults:
    def test_round_trip(self):
        defaults = make_defaults()
        again = ExperimentDefaults.model_validate(defaults.model_dump(mode="json"))
        assert again == defaults

    def test_minimal_document(self):
        defaults = ExperimentDefaults.model_validate({})
        assert defaults.trigger_profile is None
        assert defaults.actions.setup == []
        assert defaults.actions.closeout == []
        assert defaults.schema_version == 1

    def test_registered_for_generic_tooling(self):
        assert SCHEMA_REGISTRY["experiment_defaults"] is ExperimentDefaults

    def test_unknown_field_fails_loudly(self):
        with pytest.raises(ValidationError, match="trigger_profil\\b"):
            ExperimentDefaults.model_validate({"trigger_profil": "x"})

    def test_no_per_step_defaults(self):
        # Defaults cover setup/closeout only — per-step actions are a
        # per-scan decision, never an experiment-wide ambient one.
        with pytest.raises(ValidationError, match="per_step"):
            ExperimentDefaults.model_validate({"actions": {"per_step": ["surprise"]}})

    def test_apply_db_scan_defaults_on_by_default(self):
        # MC parity: the DB's scan-start/end writes are honored unless the
        # experiment explicitly opts out.
        assert ExperimentDefaults.model_validate({}).apply_db_scan_defaults is True

    def test_apply_db_scan_defaults_opt_out_round_trips(self):
        defaults = ExperimentDefaults.model_validate({"apply_db_scan_defaults": False})
        assert defaults.apply_db_scan_defaults is False
        again = ExperimentDefaults.model_validate(defaults.model_dump(mode="json"))
        assert again.apply_db_scan_defaults is False

    def test_background_telemetry_on_by_default(self):
        # Soft recording is free and read-only, so nothing is silently lost
        # unless the experiment explicitly opts out.
        assert ExperimentDefaults.model_validate({}).background_telemetry is True

    def test_background_telemetry_opt_out_round_trips(self):
        defaults = ExperimentDefaults.model_validate({"background_telemetry": False})
        assert defaults.background_telemetry is False
        again = ExperimentDefaults.model_validate(defaults.model_dump(mode="json"))
        assert again.background_telemetry is False

    def test_merge_rule_documented_in_operator_language(self):
        # The resolver contract is part of the schema's documentation:
        # defaults run first on setup and last on closeout (teardown
        # mirrors setup — ratified with the action-execution milestone).
        doc = inspect.getdoc(ExperimentDefaults) or ""
        module_doc = inspect.getmodule(ExperimentDefaults).__doc__ or ""
        assert "defaults run first" in (doc + module_doc).lower()
        assert "mirrors" in (doc + module_doc).lower()
        actions_field = ExperimentDefaults.model_fields["actions"]
        assert "run first" in (actions_field.description or "")
        assert "run last" in (actions_field.description or "")
        # The mirrored closeout is pinned on the closeout field itself, so
        # an editor tooltip states the ordering an operator will observe.
        from geecs_schemas.experiment_defaults import DefaultActions

        closeout_field = DefaultActions.model_fields["closeout"]
        assert "after any" in (closeout_field.description or "")
        # provenance requirement is stated for resolver implementers
        assert "provenance" in module_doc
