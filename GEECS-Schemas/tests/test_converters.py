"""Converter tests against hermetic fixture copies of real legacy configs.

Golden snapshots live in ``tests/golden/`` — regenerate them with
``python tests/generate_golden.py`` after an intentional schema change and
review the diff.
"""

import json
from pathlib import Path

import pytest

from geecs_schemas import SaveRole, TriggerState
from geecs_schemas.convert import (
    SchemaConversionError,
    convert_action_library,
    convert_assigned_actions,
    convert_optimizer_config,
    convert_save_element,
    convert_scan_preset,
    convert_scan_variables,
    convert_shot_control,
    merge_trigger_variant,
)

# Defined locally (not imported from conftest) so the module imports cleanly
# under any pytest import mode, including the monorepo's importlib mode.
FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = Path(__file__).parent / "golden"


def assert_matches_golden(payload: dict, name: str):
    expected = json.loads((GOLDEN / name).read_text())
    assert payload == expected, (
        f"Converted output no longer matches golden {name} — if the change "
        "is intentional, regenerate with tests/generate_golden.py."
    )


class TestSaveElements:
    def test_uc_aline1_with_setup_action(self):
        result = convert_save_element(FIXTURES / "save_elements/UC_Aline1.yaml")
        assert result.save_set.name == "UC_Aline1"
        [entry] = result.save_set.entries
        assert entry.device == "UC_ALineEbeam1"
        assert entry.images and entry.scalars == [] and entry.role is None
        assert set(result.actions) == {"UC_Aline1_setup"}
        assert_matches_golden(
            {
                "save_set": result.save_set.model_dump(mode="json"),
                "actions": {
                    k: v.model_dump(mode="json") for k, v in result.actions.items()
                },
                "notes": result.notes,
            },
            "UC_Aline1.converted.json",
        )

    def test_bcave_mixed_sync_and_deprecated_class(self):
        result = convert_save_element(FIXTURES / "save_elements/BCaveMagSpec.yaml")
        by_device = {e.device: e for e in result.save_set.entries}
        assert by_device["U_BCaveHallProbe"].role is SaveRole.SNAPSHOT
        assert by_device["UC_BCaveMagSpecCam1"].role is None
        assert any("post_analysis_class" in note for note in result.notes)

    def test_visa1_scan_setup_becomes_actions(self):
        result = convert_save_element(
            FIXTURES / "save_elements/visa1_spectrometer_setup.yaml"
        )
        setup = result.actions["visa1_spectrometer_setup_setup"]
        closeout = result.actions["visa1_spectrometer_setup_closeout"]
        # the scan_setup [pre, post] pair became set-steps
        assert any(
            getattr(s, "variable", None) == "Analysis" and s.value == "on"
            for s in setup.steps
        )
        assert any(
            getattr(s, "variable", None) == "Analysis" and s.value == "off"
            for s in closeout.steps
        )
        assert_matches_golden(
            {
                "save_set": result.save_set.model_dump(mode="json"),
                "actions": {
                    k: v.model_dump(mode="json") for k, v in result.actions.items()
                },
            },
            "visa1_spectrometer_setup.converted.json",
        )

    def test_action_only_element(self):
        result = convert_save_element(FIXTURES / "save_elements/WedgeInsert.yaml")
        assert result.save_set is None
        assert set(result.actions) == {
            "WedgeInsert_setup",
            "WedgeInsert_closeout",
        }

    def test_acq_timestamp_dropped_with_note(self):
        result = convert_save_element(
            {
                "Devices": {
                    "X": {
                        "synchronous": True,
                        "variable_list": ["acq_timestamp", "MaxCounts"],
                    }
                }
            },
            name="timestamped",
        )
        assert result.save_set.entries[0].scalars == ["MaxCounts"]
        assert any("acq_timestamp" in note for note in result.notes)

    def test_unknown_device_key_fails_loudly(self):
        with pytest.raises(SchemaConversionError, match="mystery_flag"):
            convert_save_element({"Devices": {"X": {"mystery_flag": True}}}, name="bad")


class TestScanVariables:
    def test_thomson_pair_converts(self):
        catalog = convert_scan_variables(
            FIXTURES / "scan_devices/scan_devices.yaml",
            FIXTURES / "scan_devices/composite_variables.yaml",
        )
        assert catalog.variables["Jet z"].target == "HTT-ESP01:Position.Axis 3"
        pseudo = catalog.variables["ebeam_profile_zScan"]
        assert pseudo.kind == "pseudo"
        assert pseudo.targets[1].forward == "100 - composite_var"
        assert_matches_golden(
            catalog.model_dump(mode="json"), "thomson_scan_variables.json"
        )

    def test_name_collision_fails_loudly(self):
        with pytest.raises(SchemaConversionError, match="also defined"):
            convert_scan_variables(
                {"single_scan_devices": {"x": "A:B"}},
                {
                    "composite_variables": {
                        "x": {
                            "mode": "absolute",
                            "components": [
                                {
                                    "device": "A",
                                    "variable": "B",
                                    "relation": "composite_var",
                                }
                            ],
                        }
                    }
                },
            )

    def test_relation_without_composite_var_fails(self):
        with pytest.raises(SchemaConversionError, match="composite_var"):
            convert_scan_variables(
                None,
                {
                    "composite_variables": {
                        "x": {
                            "mode": "absolute",
                            "components": [
                                {"device": "A", "variable": "B", "relation": "42"}
                            ],
                        }
                    }
                },
            )


class TestTriggerProfiles:
    def test_htu_normal_converts(self):
        profile = convert_shot_control(FIXTURES / "shot_control/HTU-Normal.yaml")
        assert profile.device == "U_DG645_ShotControl"
        # empty-string legacy no-ops were omitted, not stored
        assert profile.writes_for("SINGLESHOT") == {"Trigger.ExecuteSingleShot": "on"}
        assert profile.writes_for("SCAN")["Amplitude.Ch AB"] == "4.0"

    def test_empty_and_deviceless_convert_to_none(self):
        assert convert_shot_control(FIXTURES / "shot_control/Bella Normal.yaml") is None
        assert convert_shot_control(FIXTURES / "shot_control/No Device.yaml") is None

    def test_laser_off_pair_becomes_variant(self):
        base = convert_shot_control(FIXTURES / "shot_control/HTU-Normal.yaml")
        off = convert_shot_control(FIXTURES / "shot_control/HTU-LaserOFF.yaml")
        profile = merge_trigger_variant(base, off, "laser_off")
        assert set(profile.variants) == {"laser_off"}
        # the variant carries only the differing writes
        overlay = profile.variants["laser_off"].states
        assert overlay[TriggerState.SCAN] == {"Trigger.Source": "Internal"}
        # resolving through the variant reproduces the parallel file exactly
        for state in TriggerState:
            assert profile.writes_for(state, variant="laser_off") == (
                off.writes_for(state)
            )
        assert_matches_golden(
            profile.model_dump(mode="json"), "htu_trigger_profile.json"
        )

    def test_unknown_state_fails_loudly(self):
        with pytest.raises(SchemaConversionError, match="BLASTOFF"):
            convert_shot_control(
                {"device": "D", "variables": {"V": {"BLASTOFF": "1"}}},
                name="bad",
            )


class TestActions:
    def test_thomson_library_converts(self):
        library = convert_action_library(FIXTURES / "actions/actions.yaml")
        assert set(library.plans) == {"Quad-In_Long0", "Quad-In_Long300"}

    def test_undulator_library_nested_references(self):
        library = convert_action_library(FIXTURES / "actions/actions_undulator.yaml")
        outer = library.plans["experiment_CLOSEOUT"]
        assert all(step.do == "run" for step in outer.steps)
        assert_matches_golden(
            library.plans["Amp4_DUMP_HP"].model_dump(mode="json"),
            "amp4_dump_hp_plan.json",
        )

    def test_assigned_actions_extract_and_validate(self):
        library = convert_action_library(FIXTURES / "actions/actions_undulator.yaml")
        names = convert_assigned_actions(
            FIXTURES / "actions/assigned_actions.yaml", library=library
        )
        assert "experiment_CLOSEOUT" in names

    def test_script_run_step_fails_loudly(self):
        with pytest.raises(SchemaConversionError, match="script"):
            convert_action_library(
                {
                    "actions": {
                        "bad": {
                            "steps": [
                                {
                                    "action": "run",
                                    "file_name": "x.py",
                                    "class_name": "X",
                                }
                            ]
                        }
                    }
                }
            )


class TestPresets:
    def test_focuscan_converts(self):
        conversion = convert_scan_preset(FIXTURES / "presets/00_focuscan.yaml")
        request = conversion.scan_request
        assert request.mode.value == "step"
        # legacy 1-D presets become a single-axis list
        [axis] = request.axes
        assert axis.variable == "Mode Imager Stage"
        assert axis.positions.start == -18.0
        assert request.grid_shape() == (17,)
        assert request.shots_per_step == 10
        assert conversion.element_names == ["LP-FocusDiagnostics"]
        assert_matches_golden(
            request.model_dump(mode="json"), "focuscan_scan_request.json"
        )

    def test_noscan_preset(self):
        conversion = convert_scan_preset(FIXTURES / "presets/VHEE_Probe_BG.yaml")
        request = conversion.scan_request
        assert request.mode.value == "noscan"
        assert request.shots_per_step == 50
        assert not request.background

    def test_background_preset_sets_flag(self):
        conversion = convert_scan_preset(FIXTURES / "presets/HasoBackground.yaml")
        assert conversion.scan_request.mode.value == "noscan"
        assert conversion.scan_request.background is True

    def test_composition_with_save_sets(self):
        element = convert_save_element(FIXTURES / "save_elements/HP_Daq.yaml").save_set
        conversion = convert_scan_preset(
            {
                "Devices": ["HP_Daq"],
                "Info": "test",
                "Scan Mode": "No Scan",
                "Num Shots": 10,
            },
            name="composed",
            save_sets={"HP_Daq": element},
        )
        assert conversion.composed_save_set.name == "composed"
        assert conversion.composed_save_set.entries[0].device == "U_HP_Daq"

    def test_composition_merges_duplicate_devices_like_legacy(self):
        # Legacy DeviceManager extends an existing device's subscription;
        # composition unions scalars and ORs flags accordingly.
        from geecs_schemas import SaveSet

        first = SaveSet.model_validate(
            {
                "name": "a",
                "entries": [
                    {
                        "device": "PS",
                        "scalars": ["Current_Limit.Ch1"],
                        "role": "snapshot",
                    },
                ],
            }
        )
        second = SaveSet.model_validate(
            {
                "name": "b",
                "entries": [
                    {
                        "device": "PS",
                        "scalars": ["Field", "Current_Limit.Ch1"],
                        "images": True,
                    },
                ],
            }
        )
        from geecs_schemas.convert import compose_save_sets

        composed = compose_save_sets("merged", [first, second])
        [entry] = composed.entries
        assert entry.scalars == ["Current_Limit.Ch1", "Field"]
        assert entry.images is True
        assert entry.role is SaveRole.SNAPSHOT

    def test_composition_conflicting_roles_fail_loudly(self):
        from geecs_schemas import SaveSet
        from geecs_schemas.convert import compose_save_sets

        first = SaveSet.model_validate(
            {"name": "a", "entries": [{"device": "PS", "role": "snapshot"}]}
        )
        second = SaveSet.model_validate(
            {"name": "b", "entries": [{"device": "PS", "role": "reference"}]}
        )
        with pytest.raises(SchemaConversionError, match="conflicting role"):
            compose_save_sets("merged", [first, second])

    def test_unknown_scan_mode_fails_loudly(self):
        with pytest.raises(SchemaConversionError, match="Scan Mode"):
            convert_scan_preset({"Scan Mode": "2D Scan", "Num Shots": 1}, name="bad")


class TestOptimizerConfigs:
    def test_hexapod_alignment_converts(self):
        conversion = convert_optimizer_config(
            FIXTURES / "optimizer_configs/hexapod_alignment.yaml"
        )
        spec = conversion.optimization
        assert spec.variables == {"U_Hexapod:ypos": (17.0, 19.0)}
        assert spec.objectives == {"f": "MINIMIZE"}
        assert spec.evaluator.class_name == "MaxCountsEvaluator"
        assert conversion.save_set is None
        assert_matches_golden(
            spec.model_dump(mode="json"), "hexapod_optimization_spec.json"
        )

    def test_bax_overrides_become_generator_options(self):
        conversion = convert_optimizer_config(
            FIXTURES / "optimizer_configs/bax_alignment_S1H.yaml"
        )
        spec = conversion.optimization
        assert spec.objectives == {}
        assert spec.observables == ["x_CoM"]
        assert spec.generator.name == "multipoint_bax_alignment_l2"
        assert spec.generator.options["probe_nominal"] == 1.5

    def test_device_requirements_preserved_as_save_set(self):
        conversion = convert_optimizer_config(
            FIXTURES / "optimizer_configs/hi_res_mag_cam_max_counts.yaml"
        )
        devices = {e.device for e in conversion.save_set.entries}
        assert devices == {"UC_HiResMagCam", "U_BCaveICT", "U_BCaveMagSpec"}
        assert any("device_requirements" in n for n in conversion.notes)

    def test_mismatched_overrides_fail_loudly(self):
        with pytest.raises(SchemaConversionError, match="do not match"):
            convert_optimizer_config(
                {
                    "vocs": {
                        "variables": {"A:B": [0, 1]},
                        "objectives": {"f": "MINIMIZE"},
                    },
                    "evaluator": {"module": "m", "class": "C"},
                    "generator": {"name": "bayes_default"},
                    "xopt_config_overrides": {"some_other_generator": {}},
                },
                name="bad",
            )
