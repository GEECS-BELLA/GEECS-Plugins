"""Model tests for InterlockSet."""

import pytest
from pydantic import ValidationError

from geecs_schemas import (
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
                    "on_trip": "close_shutters",
                    "before_resume": "open_shutters",
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
        assert entry.on_trip is None
        assert entry.settle_s == 0
        assert entry.condition.target_device is None

    def test_registered(self):
        assert SCHEMA_REGISTRY["interlocks"] is InterlockSet


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
