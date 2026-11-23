"""Model tests for TriggerProfile."""

import pytest
from pydantic import ValidationError

from geecs_schemas import TriggerProfile


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

    @pytest.mark.parametrize("version", [1, 3])
    def test_a_schema_version_other_than_2_is_refused(self, version):
        # v2 is the only format: an older or newer stamp is never lifted.
        with pytest.raises(ValidationError, match="schema_version") as info:
            TriggerProfile.model_validate(
                {
                    "schema_version": version,
                    "name": "x",
                    "states": {"SCAN": [write(DG, "Trigger.Source", "External")]},
                }
            )
        assert "Input should be 2" in str(info.value)

    def test_a_v1_variants_block_is_refused_as_an_unknown_key(self):
        with pytest.raises(ValidationError, match="variants") as info:
            TriggerProfile.model_validate(
                {
                    "schema_version": 2,
                    "name": "x",
                    "states": {"SCAN": [write(DG, "Trigger.Source", "External")]},
                    "variants": {},
                }
            )
        assert "Extra inputs are not permitted" in str(info.value)

    def test_v2_document_round_trips_untouched(self):
        profile = make_profile()
        assert profile.schema_version == 2
        again = TriggerProfile.model_validate(profile.model_dump(mode="json"))
        assert again == profile

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

    def test_use_scan_mode_is_off_by_default_and_round_trips(self):
        # The scan-mode toggle (#1084) is additive: an existing document
        # reads as before, and a flagged one survives a JSON round trip.
        assert make_profile().use_scan_mode is False
        flagged = TriggerProfile.model_validate(
            {**make_profile().model_dump(mode="json"), "use_scan_mode": True}
        )
        assert flagged.use_scan_mode is True
        again = TriggerProfile.model_validate(flagged.model_dump(mode="json"))
        assert again == flagged

    def test_a_scan_mode_profile_with_no_writes_is_the_box_less_case(self):
        # The shot message is the only trigger: no state is defined, and the
        # document still loads — the worker's adapter turns it into a no-box
        # controller under the flag (GeecsBluesky).
        profile = TriggerProfile.model_validate(
            {"name": "bench", "states": {}, "use_scan_mode": True}
        )
        assert profile.use_scan_mode and profile.devices == []
        assert not profile.defines_state("ARMED")
