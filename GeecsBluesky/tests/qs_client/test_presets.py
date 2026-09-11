"""expand_preset: a saved preset → the stock plan queue item (PR 2, #807)."""

from __future__ import annotations

import pytest

from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.qs_client.presets import expand_preset, scan_variable_reference
from geecs_bluesky.utils import device_reference
from geecs_schemas import Preset, ScanVariables

CATALOG = ScanVariables.model_validate(
    {
        "schema_version": 1,
        "variables": {
            "EMQ1 Current": {
                "target": "U_EMQTripletBipolar:Current_Limit.Ch1",
                "kind": "setpoint",
            },
            "JetZ_with_probe": {
                "kind": "pseudo",
                "targets": [{"target": "U_Jet:Z", "forward": "x"}],
                "mode": "absolute",
            },
        },
    }
).variables


def _preset(**overrides) -> Preset:
    base = {
        "name": "emq1",
        "description": "emq1 sweep",
        "trigger_profile": "HTU-Normal",
        "devices": [
            {"device": "UC_ALineEBeam3"},
            {"device": "UC_VisaEBeam1", "save_images": False},
            {"device": "U_BCaveICT", "save_images": False},
        ],
        "plan": {
            "name": "scan",
            "args": ["EMQ1 Current", 1.2, 1.7, 6],
            "kwargs": {"shots_per_step": 20},
        },
    }
    base.update(overrides)
    return Preset.model_validate(base)


def test_device_reference_spellings() -> None:
    assert device_reference("U_S1H") == "U_S1H"
    assert device_reference("U_S1H", "Current") == "U_S1H.current"
    assert (
        device_reference("U_ESP_JetXYZ", "Position.Axis 1")
        == "U_ESP_JetXYZ.position_axis_1"
    )
    assert device_reference("2BL-Shutter") == "2bl_shutter"


def test_scan_variable_reference_from_pair_and_catalog() -> None:
    assert scan_variable_reference("U_S1H:Current") == "U_S1H.current"
    assert scan_variable_reference("U_S1H") == "U_S1H"
    assert (
        scan_variable_reference("EMQ1 Current", CATALOG)
        == "U_EMQTripletBipolar.current_limit_ch1"
    )
    with pytest.raises(GeecsConfigurationError, match="pseudo"):
        scan_variable_reference("JetZ_with_probe", CATALOG)


def test_expand_builds_the_stock_plan_item() -> None:
    item = expand_preset(
        _preset(), catalog=CATALOG, md={"geecs": {"submission": {"client": "c"}}}
    )
    assert item.name == "scan"
    assert item.args == [
        ["UC_ALineEBeam3", "UC_VisaEBeam1.scalars", "U_BCaveICT.scalars"],
        "U_EMQTripletBipolar.current_limit_ch1",
        1.2,
        1.7,
        6,
    ]
    assert item.kwargs["shots_per_step"] == 20
    assert item.kwargs["trigger_profile"] == "HTU-Normal"
    assert item.kwargs["md"] == {
        "description": "emq1 sweep",
        "background": False,
        "geecs": {"submission": {"client": "c"}, "preset": "emq1"},
    }


def test_count_preset_and_pair_spelled_variables() -> None:
    preset = _preset(
        trigger_profile=None,
        background=True,
        plan={"name": "list_scan", "args": ["U_S1H:Current", [0, 0.5, 1.0]]},
    )
    item = expand_preset(preset)
    assert item.args[1] == "U_S1H.current" and item.args[2] == [0, 0.5, 1.0]
    assert "trigger_profile" not in item.kwargs
    assert item.kwargs["md"]["background"] is True


def test_expand_refuses_no_plan_unknown_plan_and_pseudo() -> None:
    with pytest.raises(GeecsConfigurationError, match="no plan call"):
        expand_preset(_preset(plan=None))
    with pytest.raises(GeecsConfigurationError, match="scan verb"):
        expand_preset(_preset(plan={"name": "tune_centroid"}))
    with pytest.raises(GeecsConfigurationError, match="pseudo"):
        expand_preset(
            _preset(plan={"name": "scan", "args": ["JetZ_with_probe", 1, 2, 3]}),
            catalog=CATALOG,
        )


def test_expansion_records_its_references_and_leaves_literal_strings_alone() -> None:
    preset = _preset(
        plan={"name": "list_scan", "args": ["U_S1H:Enable_Output", ["on", "off"]]},
    )
    item = expand_preset(preset)
    assert item.args == [
        ["UC_ALineEBeam3", "UC_VisaEBeam1.scalars", "U_BCaveICT.scalars"],
        "U_S1H.enable_output",
        ["on", "off"],
    ]
    assert item.references == [
        "UC_ALineEBeam3",
        "UC_VisaEBeam1.scalars",
        "U_BCaveICT.scalars",
        "U_S1H.enable_output",
    ]
    catalog_item = expand_preset(_preset(), catalog=CATALOG)
    assert catalog_item.references[-1] == "U_EMQTripletBipolar.current_limit_ch1"


def test_a_preset_cannot_name_mv() -> None:
    with pytest.raises(GeecsConfigurationError, match="submit_plan\\('mv'"):
        expand_preset(_preset(plan={"name": "mv", "args": ["U_S1H:Current", 0.0]}))
