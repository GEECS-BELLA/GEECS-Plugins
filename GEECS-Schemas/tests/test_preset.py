"""Preset — the saved scan: device group + plan call (GEECS-Plugins#807, PR 2)."""

import pytest
from pydantic import ValidationError

from geecs_schemas import SCHEMA_REGISTRY, PlanCall, Preset, PresetDevice


def test_registry_has_preset_and_no_save_set():
    assert SCHEMA_REGISTRY["preset"] is Preset
    assert "save_set" not in SCHEMA_REGISTRY


def test_minimal_preset_is_a_device_group():
    preset = Preset(name="group", devices=[PresetDevice(device="UC_Cam")])
    assert preset.plan is None
    assert preset.devices[0].save_images is True
    assert preset.trigger_profile is None and preset.background is False


def test_full_preset_round_trips():
    document = {
        "schema_version": 1,
        "name": "emq1",
        "description": "emq1 sweep",
        "trigger_profile": "HTU-Normal",
        "background": False,
        "devices": [
            {"device": "UC_ALineEBeam3", "save_images": True},
            {"device": "U_BCaveICT", "save_images": False},
        ],
        "plan": {
            "name": "scan",
            "args": ["EMQ1 Current", 1.2, 1.7, 6],
            "kwargs": {"shots_per_step": 20},
        },
    }
    preset = Preset.model_validate(document)
    assert preset.plan == PlanCall(
        name="scan", args=["EMQ1 Current", 1.2, 1.7, 6], kwargs={"shots_per_step": 20}
    )
    assert preset.model_dump(mode="json") == document


def test_duplicate_devices_are_refused():
    with pytest.raises(ValidationError, match="more than once"):
        Preset(
            name="p",
            devices=[PresetDevice(device="UC_Cam"), PresetDevice(device="UC_Cam")],
        )


def test_unknown_keys_and_empty_names_are_refused():
    with pytest.raises(ValidationError):
        Preset.model_validate({"name": "p", "save_sets": ["x"]})
    with pytest.raises(ValidationError):
        Preset.model_validate({"name": "p", "devices": [{"device": ""}]})
    with pytest.raises(ValidationError):
        Preset.model_validate({"name": "p", "plan": {"name": ""}})


def test_plan_arguments_are_plain_json():
    preset = Preset.model_validate(
        {
            "name": "p",
            "plan": {"name": "list_scan", "args": ["U_S1H:Current", [0, 0.5, 1.0]]},
        }
    )
    assert preset.plan.args[1] == [0, 0.5, 1.0]
