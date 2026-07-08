"""Model tests for SaveSet."""

import pytest
from pydantic import ValidationError

from geecs_schemas import SaveRole, SaveSet, SaveSetEntry


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

    def test_entry_action_references(self):
        # setup/closeout are ActionPlan NAME references travelling with the
        # device, not inline plans.
        save_set = make_save_set(
            entries=[
                {
                    "device": "UC_UndulatorRad2",
                    "images": True,
                    "setup": ["visa1_setup", "spectrometer_warmup"],
                    "closeout": ["visa1_closeout"],
                }
            ]
        )
        [entry] = save_set.entries
        assert entry.setup == ["visa1_setup", "spectrometer_warmup"]
        assert entry.closeout == ["visa1_closeout"]

    def test_entry_action_references_default_empty(self):
        save_set = make_save_set()
        assert all(e.setup == [] and e.closeout == [] for e in save_set.entries)

    def test_inline_plan_in_setup_rejected(self):
        # References are names; an inline plan dict must fail loudly.
        with pytest.raises(ValidationError, match="setup"):
            make_save_set(
                entries=[
                    {
                        "device": "X",
                        "setup": [{"steps": [{"do": "wait", "seconds": 1}]}],
                    }
                ]
            )


class TestDbScanDefaults:
    def test_override_three_cases(self):
        # value = replace the DB's write; null = suppress it; absent =
        # inherit the DB behavior untouched.
        save_set = make_save_set(
            entries=[
                {
                    "device": "UC_ALineEbeam1",
                    "at_scan_start": {"Analysis": "on", "exposure": None},
                    "at_scan_end": {"Analysis": None},
                }
            ]
        )
        [entry] = save_set.entries
        assert entry.at_scan_start["Analysis"] == "on"  # replace
        assert entry.at_scan_start["exposure"] is None  # suppress
        assert "trigger" not in entry.at_scan_start  # inherit (absent)
        assert entry.at_scan_end == {"Analysis": None}

    def test_null_suppression_round_trips_through_yaml(self):
        yaml = pytest.importorskip("yaml")
        document = yaml.safe_load(
            """
            device: UC_ALineEbeam1
            at_scan_start:
              Analysis: "on"
              exposure: null
            at_scan_end:
              save: null
            """
        )
        entry = SaveSetEntry.model_validate(document)
        assert entry.at_scan_start == {"Analysis": "on", "exposure": None}
        assert entry.at_scan_end == {"save": None}
        # dump → YAML → reload keeps the explicit nulls distinct from absence
        dumped = yaml.safe_load(yaml.safe_dump(entry.model_dump(mode="json")))
        again = SaveSetEntry.model_validate(dumped)
        assert again == entry
        assert again.at_scan_start["exposure"] is None

    def test_defaults_are_empty_and_off(self):
        save_set = make_save_set()
        for entry in save_set.entries:
            assert entry.db_scalars is False
            assert entry.at_scan_start == {}
            assert entry.at_scan_end == {}

    def test_db_scalars_opt_in(self):
        save_set = make_save_set(entries=[{"device": "X", "db_scalars": True}])
        assert save_set.entries[0].db_scalars is True

    def test_non_string_override_value_rejected(self):
        with pytest.raises(ValidationError, match="at_scan_start"):
            make_save_set(
                entries=[{"device": "X", "at_scan_start": {"Analysis": ["on"]}}]
            )
