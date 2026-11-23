"""Model tests for InterlockSet."""

import pytest
from pydantic import ValidationError

from geecs_schemas import (
    KIND_STORAGE,
    SCHEMA_REGISTRY,
    AlignmentCondition,
    ConfigChangeCondition,
    ExternalCondition,
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
                    "conditions": [
                        {
                            "kind": "threshold",
                            "device": "U_HP_Vacuum",
                            "variable": "Pressure",
                            "operator": ">",
                            "limit": 1e-5,
                            "hysteresis": 2e-6,
                        }
                    ],
                    "settle_s": 5,
                },
                {
                    "name": "tape camera",
                    "conditions": [
                        {
                            "kind": "alignment",
                            "device": "CAM-PL1-Pointing",
                            "variable": "CentroidX",
                            "target_variable": "TargetX",
                            "tolerance": 10,
                        },
                        {
                            "kind": "stale",
                            "device": "U_HP_Vacuum",
                            "max_age_s": 3,
                        },
                        {"kind": "external", "port": 5050, "source": "stage"},
                        {
                            "kind": "config_change",
                            "device": "U_Windmill",
                            "variables": ["Position", "Speed"],
                        },
                    ],
                },
            ]
        }
    )


def first_condition(data):
    return data["interlocks"][0]["conditions"][0]


class TestInterlockSet:
    def test_round_trip(self):
        interlocks = make_set()
        again = InterlockSet.model_validate(interlocks.model_dump(mode="json"))
        assert again == interlocks

    def test_discriminator_picks_each_kind(self):
        sets = make_set().interlocks
        assert isinstance(sets[0].conditions[0], ThresholdCondition)
        kinds = [type(c) for c in sets[1].conditions]
        assert kinds == [
            AlignmentCondition,
            StaleCondition,
            ExternalCondition,
            ConfigChangeCondition,
        ]

    def test_empty_document(self):
        document = InterlockSet.model_validate({})
        assert document.interlocks == []
        assert document.stale_after_s == 5.0

    def test_staleness_can_be_turned_off(self):
        assert (
            InterlockSet.model_validate({"stale_after_s": None}).stale_after_s is None
        )

    def test_defaults(self):
        entry = make_set().interlocks[1]
        assert entry.enabled is True
        assert entry.settle_s == 0
        assert entry.conditions[0].target_device is None
        assert entry.conditions[2].max_age_s is None

    @pytest.mark.parametrize("op", ["==", "!="])
    def test_equality_operators(self, op):
        data = make_set().model_dump(mode="json")
        first_condition(data).update(operator=op, hysteresis=0)
        assert InterlockSet.model_validate(data)

    def test_registered(self):
        assert SCHEMA_REGISTRY["interlocks"] is InterlockSet

    def test_maintainer_tier(self):
        """A limit change is a pull request, not a direct commit (the maintainer's call)."""
        assert InterlockSet.model_json_schema()["tier"] == "maintainer"

    def test_one_file_per_named_configuration(self):
        storage = KIND_STORAGE["interlocks"]
        assert storage.path_pattern().endswith("<experiment>/interlocks/<name>.yaml")


class TestRefusals:
    def test_duplicate_names(self):
        data = make_set().model_dump(mode="json")
        data["interlocks"][1]["name"] = "vacuum"
        with pytest.raises(ValidationError, match="duplicate interlock name"):
            InterlockSet.model_validate(data)

    def test_no_conditions(self):
        data = make_set().model_dump(mode="json")
        data["interlocks"][0]["conditions"] = []
        with pytest.raises(ValidationError):
            InterlockSet.model_validate(data)

    def test_single_condition_key_is_gone(self):
        data = make_set().model_dump(mode="json")
        entry = data["interlocks"][0]
        entry["condition"] = entry.pop("conditions")[0]
        with pytest.raises(ValidationError):
            InterlockSet.model_validate(data)

    def test_unknown_operator(self):
        data = make_set().model_dump(mode="json")
        first_condition(data)["operator"] = "~="
        with pytest.raises(ValidationError):
            InterlockSet.model_validate(data)

    @pytest.mark.parametrize("op", ["==", "!="])
    def test_no_hysteresis_on_equality(self, op):
        data = make_set().model_dump(mode="json")
        first_condition(data)["operator"] = op
        with pytest.raises(ValidationError, match="no meaning"):
            InterlockSet.model_validate(data)

    def test_unknown_kind(self):
        data = make_set().model_dump(mode="json")
        first_condition(data)["kind"] = "custom"
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
        data["interlocks"][1]["conditions"][0]["hysteresis"] = 10
        with pytest.raises(ValidationError, match="never resume"):
            InterlockSet.model_validate(data)

    @pytest.mark.parametrize("port", [0, 70000])
    def test_external_port_range(self, port):
        data = make_set().model_dump(mode="json")
        data["interlocks"][1]["conditions"][2]["port"] = port
        with pytest.raises(ValidationError):
            InterlockSet.model_validate(data)

    @pytest.mark.parametrize("variables", [[], ["Speed", "Speed"]])
    def test_config_change_variables(self, variables):
        data = make_set().model_dump(mode="json")
        data["interlocks"][1]["conditions"][3]["variables"] = variables
        with pytest.raises(ValidationError):
            InterlockSet.model_validate(data)

    @pytest.mark.parametrize("field,value", [("settle_s", -1), ("name", "")])
    def test_bad_values(self, field, value):
        data = make_set().model_dump(mode="json")
        data["interlocks"][0][field] = value
        with pytest.raises(ValidationError):
            InterlockSet.model_validate(data)

    def test_bad_staleness(self):
        with pytest.raises(ValidationError):
            InterlockSet.model_validate({"stale_after_s": 0})
