"""Model tests for InterlockSet."""

import pytest
from pydantic import ValidationError

from geecs_schemas import (
    KIND_STORAGE,
    SCHEMA_REGISTRY,
    AlignmentCondition,
    InterlockSet,
    StaleCondition,
    ThresholdCondition,
)


def make_set():
    return InterlockSet.model_validate(
        {
            "interlocks": [
                {
                    "name": "vacuum",
                    "condition": {
                        "kind": "threshold",
                        "device": "U_HP_Vacuum",
                        "variable": "Pressure",
                        "operator": ">",
                        "limit": 1e-5,
                        "hysteresis": 2e-6,
                    },
                    "settle_s": 5,
                },
                {
                    "name": "pointing",
                    "condition": {
                        "kind": "alignment",
                        "device": "CAM-PL1-Pointing",
                        "variable": "CentroidX",
                        "target_variable": "TargetX",
                        "tolerance": 10,
                    },
                },
                {
                    "name": "gauge alive",
                    "condition": {
                        "kind": "stale",
                        "device": "U_HP_Vacuum",
                        "max_age_s": 3,
                    },
                },
            ]
        }
    )


class TestInterlockSet:
    def test_round_trip(self):
        interlocks = make_set()
        again = InterlockSet.model_validate(interlocks.model_dump(mode="json"))
        assert again == interlocks

    def test_discriminator_picks_each_kind(self):
        kinds = [i.condition for i in make_set().interlocks]
        assert isinstance(kinds[0], ThresholdCondition)
        assert isinstance(kinds[1], AlignmentCondition)
        assert isinstance(kinds[2], StaleCondition)

    def test_empty_document(self):
        assert InterlockSet.model_validate({}).interlocks == []

    def test_defaults(self):
        entry = make_set().interlocks[1]
        assert entry.enabled is True
        assert entry.settle_s == 0
        assert entry.condition.target_device is None

    def test_registered(self):
        assert SCHEMA_REGISTRY["interlocks"] is InterlockSet

    def test_operator_tier(self):
        assert InterlockSet.model_json_schema()["tier"] == "operator"

    def test_stored_at_the_experiment_top(self):
        storage = KIND_STORAGE["interlocks"]
        assert storage.path_pattern().endswith("<experiment>/interlocks.yaml")


class TestRefusals:
    def test_duplicate_names(self):
        data = make_set().model_dump(mode="json")
        data["interlocks"][1]["name"] = "vacuum"
        with pytest.raises(ValidationError, match="duplicate interlock name"):
            InterlockSet.model_validate(data)

    def test_unknown_operator(self):
        data = make_set().model_dump(mode="json")
        data["interlocks"][0]["condition"]["operator"] = "=="
        with pytest.raises(ValidationError):
            InterlockSet.model_validate(data)

    def test_unknown_kind(self):
        data = make_set().model_dump(mode="json")
        data["interlocks"][0]["condition"]["kind"] = "custom"
        with pytest.raises(ValidationError):
            InterlockSet.model_validate(data)

    @pytest.mark.parametrize("field", ["on_trip", "before_resume"])
    def test_no_actions_report_only(self, field):
        data = make_set().model_dump(mode="json")
        data["interlocks"][0][field] = "close_shutters"
        with pytest.raises(ValidationError):
            InterlockSet.model_validate(data)

    def test_typo_field(self):
        data = make_set().model_dump(mode="json")
        data["interlocks"][0]["settle"] = 5
        with pytest.raises(ValidationError):
            InterlockSet.model_validate(data)

    def test_hysteresis_must_be_inside_tolerance(self):
        data = make_set().model_dump(mode="json")
        data["interlocks"][1]["condition"]["hysteresis"] = 10
        with pytest.raises(ValidationError, match="never resume"):
            InterlockSet.model_validate(data)

    @pytest.mark.parametrize("field,value", [("settle_s", -1), ("name", "")])
    def test_bad_values(self, field, value):
        data = make_set().model_dump(mode="json")
        data["interlocks"][0][field] = value
        with pytest.raises(ValidationError):
            InterlockSet.model_validate(data)
