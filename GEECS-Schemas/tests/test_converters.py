"""Converter tests against hermetic fixture copies of real legacy configs.

Golden snapshots live in ``tests/golden/`` — regenerate them with
``python tests/generate_golden.py`` after an intentional schema change and
review the diff.
"""

import json
from pathlib import Path

import pytest

from geecs_schemas.convert import (
    SchemaConversionError,
    convert_optimizer_config,
    convert_shot_control,
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


def as_tuples(writes):
    return [(w.device, w.variable, w.value) for w in writes]


class TestTriggerProfiles:
    def test_htu_normal_converts(self):
        profile = convert_shot_control(FIXTURES / "shot_control/HTU-Normal.yaml")
        # the legacy file's single device was emitted into every write
        assert profile.devices == ["U_DG645_ShotControl"]
        # empty-string legacy no-ops were omitted, not stored
        assert as_tuples(profile.writes_for("SINGLESHOT")) == [
            ("U_DG645_ShotControl", "Trigger.ExecuteSingleShot", "on")
        ]
        scan = {(w.device, w.variable): w.value for w in profile.writes_for("SCAN")}
        assert scan[("U_DG645_ShotControl", "Amplitude.Ch AB")] == "4.0"

    def test_empty_and_deviceless_convert_to_none(self):
        assert convert_shot_control(FIXTURES / "shot_control/Bella Normal.yaml") is None
        assert convert_shot_control(FIXTURES / "shot_control/No Device.yaml") is None

    def test_laser_off_file_converts_as_its_own_profile(self):
        off = convert_shot_control(FIXTURES / "shot_control/HTU-LaserOFF.yaml")
        scan = {(w.device, w.variable): w.value for w in off.writes_for("SCAN")}
        assert scan[("U_DG645_ShotControl", "Trigger.Source")] == "Internal"
        assert "variants" not in off.model_dump(mode="json")
        base = convert_shot_control(FIXTURES / "shot_control/HTU-Normal.yaml")
        assert_matches_golden(base.model_dump(mode="json"), "htu_trigger_profile.json")

    def test_unknown_state_fails_loudly(self):
        with pytest.raises(SchemaConversionError, match="BLASTOFF"):
            convert_shot_control(
                {"device": "D", "variables": {"V": {"BLASTOFF": "1"}}},
                name="bad",
            )


class TestOptimizerConfigs:
    def test_hexapod_alignment_converts(self):
        conversion = convert_optimizer_config(
            FIXTURES / "optimizer_configs/hexapod_alignment.yaml"
        )
        spec = conversion.optimization
        assert spec.variables == {"U_Hexapod:ypos": (17.0, 19.0)}
        assert spec.objectives == {"f": "MINIMIZE"}
        assert spec.evaluator.class_name == "MaxCountsEvaluator"
        assert conversion.devices == []
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

    def test_device_requirements_preserved_as_a_device_group(self):
        conversion = convert_optimizer_config(
            FIXTURES / "optimizer_configs/hi_res_mag_cam_max_counts.yaml"
        )
        devices = {d.device for d in conversion.devices}
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
