"""Model tests for TriggerProfile."""

import pytest
from pydantic import ValidationError

from geecs_schemas import TriggerProfile, TriggerState, TriggerWrite


def write(device, variable, value):
    return {"device": device, "variable": variable, "value": value}


DG = "U_DG645_ShotControl"
JET = "U_GasJetPLC"


def make_profile():
    return TriggerProfile.model_validate(
        {
            "name": "htu_shot_control",
            "states": {
                "OFF": [
                    write(DG, "Amplitude.Ch AB", "0.5"),
                    write(DG, "Trigger.Source", "Single shot external rising edges"),
                ],
                "STANDBY": [
                    write(DG, "Amplitude.Ch AB", "0.5"),
                    write(DG, "Trigger.Source", "External rising edges"),
                ],
                "SCAN": [
                    write(DG, "Amplitude.Ch AB", "4.0"),
                    write(DG, "Trigger.Source", "External rising edges"),
                    write(JET, "DO.Jet", "on"),
                ],
                "ARMED": [
                    write(DG, "Amplitude.Ch AB", "4.0"),
                    write(DG, "Trigger.Source", "Single shot external rising edges"),
                ],
                "SINGLESHOT": [write(DG, "Trigger.ExecuteSingleShot", "on")],
            },
            "variants": {
                "laser_off": {
                    "states": {
                        "SCAN": [write(DG, "Trigger.Source", "Internal")],
                        "ARMED": [write(DG, "Trigger.Source", "Single shot")],
                        "OFF": [write(DG, "Trigger.Source", "Single shot")],
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

    def test_multi_device_state_and_write_order_preserved(self):
        # A transition may write several devices; writes stay in declared
        # order (applied top to bottom).
        writes = make_profile().writes_for("SCAN")
        assert [(w.device, w.variable, w.value) for w in writes] == [
            (DG, "Amplitude.Ch AB", "4.0"),
            (DG, "Trigger.Source", "External rising edges"),
            (JET, "DO.Jet", "on"),
        ]

    def test_devices_property_lists_distinct_devices(self):
        assert make_profile().devices == [DG, JET]

    def test_variant_overlays_in_place_and_keeps_order(self):
        profile = make_profile()
        writes = profile.writes_for(TriggerState.SCAN, variant="laser_off")
        assert [(w.device, w.variable, w.value) for w in writes] == [
            (DG, "Amplitude.Ch AB", "4.0"),  # inherited from base
            (DG, "Trigger.Source", "Internal"),  # replaced in place
            (JET, "DO.Jet", "on"),  # inherited from base
        ]

    def test_variant_appends_new_writes_after_base(self):
        profile = TriggerProfile.model_validate(
            {
                "name": "x",
                "states": {"SCAN": [write(DG, "Trigger.Source", "External")]},
                "variants": {
                    "with_jet": {"states": {"SCAN": [write(JET, "DO.Jet", "on")]}}
                },
            }
        )
        writes = profile.writes_for("SCAN", variant="with_jet")
        assert [(w.device, w.variable) for w in writes] == [
            (DG, "Trigger.Source"),
            (JET, "DO.Jet"),
        ]

    def test_variant_leaves_untouched_states_alone(self):
        profile = make_profile()
        [only] = profile.writes_for("SINGLESHOT", variant="laser_off")
        assert isinstance(only, TriggerWrite)
        assert (only.variable, only.value) == ("Trigger.ExecuteSingleShot", "on")

    def test_unknown_variant_raises(self):
        with pytest.raises(KeyError, match="laser_on"):
            make_profile().writes_for("SCAN", variant="laser_on")

    def test_defines_state_semantics(self):
        # A state with no writes is "not defined" — legacy
        # ShotControlConfig.defines_state semantics.
        profile = TriggerProfile.model_validate({"name": "bare", "states": {}})
        assert not profile.defines_state("ARMED")
        assert make_profile().defines_state("ARMED")

    def test_unknown_state_name_fails_loudly(self):
        with pytest.raises(ValidationError):
            TriggerProfile.model_validate(
                {"name": "typo", "states": {"SCNA": [write(DG, "Var", "1")]}}
            )

    def test_empty_string_write_rejected(self):
        # Legacy used "" for "no-op in this state"; v1 expresses no-ops by
        # omission and rejects empty writes.
        with pytest.raises(ValidationError, match="omit"):
            TriggerProfile.model_validate(
                {
                    "name": "x",
                    "states": {"SCAN": [write(DG, "Trigger.Source", "")]},
                }
            )

    def test_duplicate_device_variable_in_state_rejected(self):
        with pytest.raises(ValidationError, match="more than once"):
            TriggerProfile.model_validate(
                {
                    "name": "x",
                    "states": {
                        "SCAN": [
                            write(DG, "Trigger.Source", "Internal"),
                            write(DG, "Trigger.Source", "External"),
                        ]
                    },
                }
            )

    def test_unquoted_yaml_off_key_is_normalized(self):
        # YAML 1.1 parses a bare `OFF:` key as boolean False; the model
        # forgives that so operators don't have to remember to quote it.
        profile = TriggerProfile.model_validate(
            {"name": "x", "states": {False: [write(DG, "Var", "1")]}}
        )
        [only] = profile.writes_for("OFF")
        assert only.value == "1"

    def test_unknown_field_fails_loudly(self):
        with pytest.raises(ValidationError, match="devcie"):
            TriggerProfile.model_validate(
                {
                    "name": "x",
                    "states": {"SCAN": [{"devcie": DG, "variable": "V", "value": "1"}]},
                }
            )
