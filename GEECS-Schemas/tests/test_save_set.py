"""Model tests for SaveSet."""

import pytest
from pydantic import ValidationError

from geecs_schemas import SaveRole, SaveSet


def make_save_set(**overrides):
    base = {
        "name": "undulator_baseline",
        "entries": [
            {
                "device": "UC_Amp4_IR_input",
                "images": True,
                "scalars": ["MaxCounts", "centroidx"],
            },
            {"device": "U_HP_Daq", "scalars": ["AnalogOutput.Channel 1"]},
            {"device": "U_BCaveHallProbe", "scalars": ["Field"], "role": "snapshot"},
        ],
    }
    base.update(overrides)
    return SaveSet.model_validate(base)


class TestSaveSet:
    def test_round_trip(self):
        save_set = make_save_set()
        assert save_set.entries[2].role is SaveRole.SNAPSHOT
        again = SaveSet.model_validate(save_set.model_dump(mode="json"))
        assert again == save_set

    def test_role_defaults_to_derived(self):
        save_set = make_save_set()
        assert save_set.entries[0].role is None  # derived by the engine

    def test_unknown_field_fails_loudly(self):
        with pytest.raises(ValidationError, match="synchronous"):
            make_save_set(entries=[{"device": "X", "synchronous": True}])

    def test_duplicate_device_rejected(self):
        with pytest.raises(ValidationError, match="more than once"):
            make_save_set(entries=[{"device": "X"}, {"device": "X", "images": True}])

    def test_empty_entries_rejected(self):
        with pytest.raises(ValidationError):
            make_save_set(entries=[])

    def test_bad_role_rejected(self):
        with pytest.raises(ValidationError, match="role"):
            make_save_set(entries=[{"device": "X", "role": "pacemaker"}])
