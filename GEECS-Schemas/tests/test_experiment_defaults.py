"""Model tests for ExperimentDefaults."""

import pytest
from pydantic import ValidationError

from geecs_schemas import SCHEMA_REGISTRY, ExperimentDefaults


def make_defaults():
    return ExperimentDefaults.model_validate(
        {
            "trigger_profile": "htu_shot_control",
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
        assert defaults.schema_version == 1

    def test_registered_for_generic_tooling(self):
        assert SCHEMA_REGISTRY["experiment_defaults"] is ExperimentDefaults

    def test_unknown_field_fails_loudly(self):
        with pytest.raises(ValidationError, match="trigger_profil\\b"):
            ExperimentDefaults.model_validate({"trigger_profil": "x"})

    def test_actions_are_refused(self):
        # The action plans went with the action library: a named device
        # state is a setup, applied as its own queue item.
        with pytest.raises(ValidationError, match="actions"):
            ExperimentDefaults.model_validate({"actions": {"setup": ["x"]}})

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


def test_native_image_save_defaults_on_and_is_read():
    """The experiment-wide fallback for the preset's run-level switch (GEECS-Plugins#738).

    On by default (the dual-write stays the rollout's parity evidence);
    ``false`` is read, not dropped — the 0.30.0 drop-validator went with
    the rebuild, so a defaults file saying ``false`` means it.
    """
    assert make_defaults().native_image_save is True
    defaults = ExperimentDefaults.model_validate(
        {
            "schema_version": 1,
            "trigger_profile": "HTU-NoGas",
            "native_image_save": False,
        }
    )
    assert defaults.trigger_profile == "HTU-NoGas"
    assert defaults.native_image_save is False
    assert defaults.model_dump(mode="json")["native_image_save"] is False
