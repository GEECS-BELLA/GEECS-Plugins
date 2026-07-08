"""Model tests for TriggerProfile."""

import pytest
from pydantic import ValidationError

from geecs_schemas import TriggerProfile, TriggerState


def make_profile():
    return TriggerProfile.model_validate(
        {
            "name": "htu_shot_control",
            "device": "U_DG645_ShotControl",
            "states": {
                "OFF": {
                    "Amplitude.Ch AB": "0.5",
                    "Trigger.Source": "Single shot external rising edges",
                },
                "STANDBY": {
                    "Amplitude.Ch AB": "0.5",
                    "Trigger.Source": "External rising edges",
                },
                "SCAN": {
                    "Amplitude.Ch AB": "4.0",
                    "Trigger.Source": "External rising edges",
                },
                "ARMED": {
                    "Amplitude.Ch AB": "4.0",
                    "Trigger.Source": "Single shot external rising edges",
                },
                "SINGLESHOT": {"Trigger.ExecuteSingleShot": "on"},
            },
            "variants": {
                "laser_off": {
                    "states": {
                        "SCAN": {"Trigger.Source": "Internal"},
                        "ARMED": {"Trigger.Source": "Single shot"},
                        "OFF": {"Trigger.Source": "Single shot"},
                    }
                }
            },
        }
    )


class TestTriggerProfile:
    def test_round_trip(self):
        profile = make_profile()
        again = TriggerProfile.model_validate(profile.model_dump(mode="json"))
        assert again == profile

    def test_writes_for_base_state(self):
        writes = make_profile().writes_for("SCAN")
        assert writes == {
            "Amplitude.Ch AB": "4.0",
            "Trigger.Source": "External rising edges",
        }

    def test_variant_overlays_base(self):
        profile = make_profile()
        writes = profile.writes_for(TriggerState.SCAN, variant="laser_off")
        assert writes["Trigger.Source"] == "Internal"
        assert writes["Amplitude.Ch AB"] == "4.0"  # inherited from base

    def test_variant_leaves_untouched_states_alone(self):
        profile = make_profile()
        assert profile.writes_for("SINGLESHOT", variant="laser_off") == {
            "Trigger.ExecuteSingleShot": "on"
        }

    def test_unknown_variant_raises(self):
        with pytest.raises(KeyError, match="laser_on"):
            make_profile().writes_for("SCAN", variant="laser_on")

    def test_defines_state_semantics(self):
        # A state with no writes is "not defined" — legacy
        # ShotControlConfig.defines_state semantics.
        profile = TriggerProfile.model_validate(
            {"name": "bare", "device": "CAM-1BL-DM", "states": {}}
        )
        assert not profile.defines_state("ARMED")
        assert make_profile().defines_state("ARMED")

    def test_unknown_state_name_fails_loudly(self):
        with pytest.raises(ValidationError):
            TriggerProfile.model_validate(
                {
                    "name": "typo",
                    "device": "D",
                    "states": {"SCNA": {"Var": "1"}},
                }
            )

    def test_empty_string_write_rejected(self):
        # Legacy used "" for "no-op in this state"; v1 expresses no-ops by
        # omission and rejects empty writes.
        with pytest.raises(ValidationError, match="omit"):
            TriggerProfile.model_validate(
                {
                    "name": "x",
                    "device": "D",
                    "states": {"SCAN": {"Trigger.Source": ""}},
                }
            )

    def test_unquoted_yaml_off_key_is_normalized(self):
        # YAML 1.1 parses a bare `OFF:` key as boolean False; the model
        # forgives that so operators don't have to remember to quote it.
        profile = TriggerProfile.model_validate(
            {"name": "x", "device": "D", "states": {False: {"Var": "1"}}}
        )
        assert profile.writes_for("OFF") == {"Var": "1"}

    def test_empty_device_rejected(self):
        with pytest.raises(ValidationError):
            TriggerProfile.model_validate({"name": "x", "device": ""})
