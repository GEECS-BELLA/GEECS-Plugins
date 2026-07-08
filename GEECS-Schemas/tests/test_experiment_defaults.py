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

    def test_merge_rule_documented_in_operator_language(self):
        # The resolver contract is part of the schema's documentation:
        # defaults run first, then the scan's own.
        doc = inspect.getdoc(ExperimentDefaults) or ""
        module_doc = inspect.getmodule(ExperimentDefaults).__doc__ or ""
        assert "defaults run first" in (doc + module_doc).lower()
        actions_field = ExperimentDefaults.model_fields["actions"]
        assert "run first" in (actions_field.description or "")
        # provenance requirement is stated for resolver implementers
        assert "provenance" in module_doc
