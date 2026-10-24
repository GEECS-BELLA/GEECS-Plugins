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


def sweep_call(axis="EMQ1 Current", positions=None):
    spec = {
        "kind": "list",
        "axis": axis,
        "positions": positions or [1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
    }
    return {
        "name": "sweep",
        "kwargs": {
            "sweep": {"trajectory": {"kind": "axes", "axes": [spec]}},
            "shots_per_step": 20,
        },
    }


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
        "plan": sweep_call(),
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
    # the namespace's collision rule: a protocol-named variable binds with "_"
    assert scan_variable_reference("UC_Cam:trigger") == "UC_Cam.trigger_"
    assert (
        scan_variable_reference("EMQ1 Current", CATALOG)
        == "U_EMQTripletBipolar.current_limit_ch1"
    )
    # a pseudo is its own namespace noun under the catalog name
    assert scan_variable_reference("JetZ_with_probe", CATALOG) == "JetZ_with_probe"


def test_expand_builds_the_stock_plan_item() -> None:
    item = expand_preset(
        _preset(), catalog=CATALOG, md={"geecs": {"submission": {"client": "c"}}}
    )
    assert item.name == "sweep"
    assert item.args == [
        ["UC_ALineEBeam3", "UC_VisaEBeam1.scalars", "U_BCaveICT.scalars"],
    ]
    assert (
        item.kwargs["sweep"]["trajectory"]["axes"][0]["axis"]
        == "U_EMQTripletBipolar.current_limit_ch1"
    )
    assert item.kwargs["shots_per_step"] == 20
    assert item.kwargs["trigger_profile"] == "HTU-Normal"
    assert item.kwargs["md"] == {
        "description": "emq1 sweep",
        "background": False,
        "geecs": {"submission": {"client": "c"}, "preset": "emq1"},
    }


def test_native_image_save_rides_as_the_plans_keyword_only_when_set() -> None:
    """Unset defers to the experiment default the worker reads per run (#738)."""
    assert "native_image_save" not in expand_preset(_preset()).kwargs
    item = expand_preset(_preset(native_image_save=False))
    assert item.kwargs["native_image_save"] is False
    assert "native_image_save" not in item.kwargs["md"]  # a plan argument, not md


def test_background_telemetry_rides_as_the_plans_keyword_only_when_set() -> None:
    """Unset defers to the experiment default the worker reads per run (#1016)."""
    assert "background_telemetry" not in expand_preset(_preset()).kwargs
    item = expand_preset(_preset(background_telemetry=False))
    assert item.kwargs["background_telemetry"] is False
    assert "background_telemetry" not in item.kwargs["md"]


@pytest.mark.parametrize(
    ("field", "copy", "top_levels"),
    [
        ("native_image_save", True, (False, None)),
        ("trigger_profile", "HTU-Other", ("HTU-Normal", None)),
        ("background_telemetry", True, (False, None)),
    ],
)
def test_run_level_field_in_plan_kwargs_is_refused_not_merged(
    field, copy, top_levels
) -> None:
    """The preset field is the one source of truth: a kwargs copy cannot win (Codex, #944).

    Both run-level fields: the switch, and the trigger profile whose
    ``setdefault`` used to let a kwargs copy override the field.
    """
    from geecs_bluesky.qs_client.presets import RUN_LEVEL_FIELDS

    assert field in RUN_LEVEL_FIELDS
    call = sweep_call()
    call["kwargs"][field] = copy
    for top_level in top_levels:
        with pytest.raises(
            GeecsConfigurationError, match="preset field, not a plan keyword"
        ):
            expand_preset(_preset(**{field: top_level}, plan=call))


def test_count_preset_and_pair_spelled_variables() -> None:
    preset = _preset(
        trigger_profile=None,
        background=True,
        plan=sweep_call("U_S1H:Current", [0, 0.5, 1.0]),
    )
    item = expand_preset(preset)
    assert item.kwargs["sweep"]["trajectory"]["axes"][0] == {
        "kind": "list",
        "axis": "U_S1H.current",
        "positions": [0, 0.5, 1],
        "relative": False,
    }
    assert "trigger_profile" not in item.kwargs
    assert item.kwargs["md"]["background"] is True


def test_expand_refuses_no_plan_and_unknown_plan() -> None:
    with pytest.raises(GeecsConfigurationError, match="no plan call"):
        expand_preset(_preset(plan=None))
    with pytest.raises(GeecsConfigurationError, match="scan verb"):
        expand_preset(_preset(plan={"name": "tune_centroid"}))


def test_expand_a_pseudo_axis_to_its_namespace_noun() -> None:
    item = expand_preset(
        _preset(plan=sweep_call("JetZ_with_probe", [1, 2, 3])),
        catalog=CATALOG,
    )
    assert item.kwargs["sweep"]["trajectory"]["axes"][0]["axis"] == "JetZ_with_probe"
    assert "JetZ_with_probe" in item.references  # the preflight checks it exists


def test_expansion_records_axis_references_and_preserves_order():
    item = expand_preset(_preset(plan=sweep_call("U_S1H.current", [2, 1, 2])))
    axis = item.kwargs["sweep"]["trajectory"]["axes"][0]
    assert axis["positions"] == [2, 1, 2]
    assert item.references[-1] == axis["axis"] == "U_S1H.current"


@pytest.mark.parametrize(
    "name", ["scan", "rel_scan", "grid_scan", "list_scan", "spiral"]
)
def test_moving_stock_presets_are_retired(name):
    with pytest.raises(GeecsConfigurationError, match="scan verb"):
        expand_preset(_preset(plan={"name": name}))


def test_alias_collision_is_rejected():
    call = sweep_call()
    call["kwargs"]["sweep"]["trajectory"]["axes"].append(
        {
            "kind": "list",
            "axis": "U_EMQTripletBipolar:Current_Limit.Ch1",
            "positions": [1, 2, 3, 4, 5, 6],
        }
    )
    with pytest.raises(GeecsConfigurationError, match="only once"):
        expand_preset(_preset(plan=call), catalog=CATALOG)


def test_a_preset_cannot_name_mv() -> None:
    with pytest.raises(GeecsConfigurationError, match="submit_plan\\('mv'"):
        expand_preset(_preset(plan={"name": "mv", "args": ["U_S1H:Current", 0.0]}))


def test_a_preset_cannot_name_run_action() -> None:
    with pytest.raises(GeecsConfigurationError, match="submit_plan\\('run_action'"):
        expand_preset(_preset(plan={"name": "run_action", "args": ["Amp4_DUMP_HP"]}))


# ------------------------------------------------------------- phase 2b
def test_non_essential_devices_ride_as_the_plans_keyword() -> None:
    preset = _preset(
        devices=[
            {"device": "UC_ALineEBeam3"},
            {"device": "UC_SlowCam", "essential": False},
            {"device": "U_BCaveICT", "save_images": False},
        ],
        plan={"name": "count", "kwargs": {"num": 10, "acquisition": "gated"}},
    )
    item = expand_preset(preset)
    assert item.args == [["UC_ALineEBeam3", "U_BCaveICT.scalars"]]
    assert item.kwargs["non_essential"] == ["UC_SlowCam"]
    assert item.kwargs["acquisition"] == "gated"
    assert item.references == ["UC_ALineEBeam3", "U_BCaveICT.scalars", "UC_SlowCam"]


def test_all_essential_preset_carries_no_non_essential_keyword() -> None:
    item = expand_preset(_preset())
    assert "non_essential" not in item.kwargs and "acquisition" not in item.kwargs


def test_non_essential_scalars_only_streams_its_view() -> None:
    """2026-09-26: a non-essential with save_images off is its ``.scalars`` view.

    It streams its scalars by stamp (no files) — the refusal this replaces
    said a scalars-only device "cannot fly".
    """
    item = expand_preset(
        _preset(
            devices=[
                {"device": "UC_A"},
                {"device": "UC_Cam", "essential": False, "save_images": False},
            ]
        )
    )
    assert item.args == [["UC_A"]]
    assert item.kwargs["non_essential"] == ["UC_Cam.scalars"]
    assert item.references[:2] == ["UC_A", "UC_Cam.scalars"]


def test_bad_acquisition_is_refused() -> None:
    with pytest.raises(GeecsConfigurationError, match="acquisition='sloppy'"):
        expand_preset(
            _preset(plan={"name": "count", "kwargs": {"acquisition": "sloppy"}})
        )


def test_generic_reference_resolution_preserves_literal_strings():
    from geecs_bluesky.qs_client.presets import _resolve

    references = []
    result = [
        _resolve(value, CATALOG, references)
        for value in ["U_S1H:Enable_Output", ["on", "off"], "EMQ1 Current", "on"]
    ]
    assert result == [
        "U_S1H.enable_output",
        ["on", "off"],
        "U_EMQTripletBipolar.current_limit_ch1",
        "on",
    ]
    assert references == [
        "U_S1H.enable_output",
        "U_EMQTripletBipolar.current_limit_ch1",
    ]


# -- a native EPICS camera's refused run shapes, in the worker's words ---------

#: The worker's bind-time sentence (``devices/native.refuse_native_cameras``).
STRICT_ONLY = "ADTEST_CAM1: native EPICS cameras run in strict scans only for now — "


def _native_preset(device: dict, **kwargs) -> Preset:
    return Preset.model_validate(
        {
            "name": "native",
            "devices": [{"device": "UC_Other"}, device],
            "plan": {"name": "count", "kwargs": {"num": 2, **kwargs}},
        }
    )


@pytest.mark.parametrize(
    ("device", "kwargs", "why"),
    [
        ({"device": "ADTEST_CAM1"}, {"acquisition": "gated"}, "not in a gated batch"),
        (
            {"device": "ADTEST_CAM1", "essential": False},
            {},
            "not as a non-essential stream",
        ),
        (
            {"device": "ADTEST_CAM1", "essential": False},
            {"acquisition": "gated"},
            "not as a non-essential stream",
        ),
    ],
    ids=["gated-essential", "strict-non-essential", "gated-non-essential"],
)
def test_a_native_camera_in_a_refused_shape_is_refused_at_expansion(
    device: dict, kwargs: dict, why: str
) -> None:
    """The worker's own sentence, before the preflight's device rules can misread it.

    Those rules read a native camera as a device with no ``acq_timestamp``:
    a gated run of one was refused as "no essential triggered device".
    """
    with pytest.raises(GeecsConfigurationError) as info:
        expand_preset(_native_preset(device, **kwargs), native_devices={"adtest_cam1"})
    assert str(info.value) == STRICT_ONLY + why


def test_the_strict_essential_shape_of_a_native_camera_expands() -> None:
    item = expand_preset(
        _native_preset({"device": "ADTEST_CAM1"}), native_devices={"ADTEST_CAM1"}
    )
    assert item.args[0] == ["UC_Other", "ADTEST_CAM1"]
    # unknown to the registry: a gated run is the preflight's to judge
    gated = _native_preset({"device": "ADTEST_CAM1"}, acquisition="gated")
    assert expand_preset(gated).kwargs["acquisition"] == "gated"


def test_a_native_positioner_takes_none_of_the_cameras_rules() -> None:
    """The registry judged by kind: a motor is a scan axis, never a stackless camera."""
    from geecs_schemas import NativeDevices

    registry = NativeDevices.model_validate(
        {
            "devices": {
                "ADTEST_CAM1": {"kind": "aravis_camera", "trigger_source": "software"},
                "SimMotor1": {"kind": "epics_motor", "axes": ["m1"]},
            }
        }
    ).devices
    # A per-row scalar device, with or without its scalars view, in any shape.
    item = expand_preset(
        _native_preset(
            {"device": "SimMotor1", "save_images": False}, acquisition="gated"
        ),
        native_devices=registry,
    )
    assert item.args[0] == ["UC_Other", "SimMotor1.scalars"]
    # And its axis is a scan variable spelled like a DB device's settable.
    assert scan_variable_reference("SimMotor1:m1") == "SimMotor1.m1"
    # The camera in the same registry keeps its refusals.
    with pytest.raises(GeecsConfigurationError, match="no scalars-only view"):
        expand_preset(
            _native_preset({"device": "ADTEST_CAM1", "save_images": False}),
            native_devices=registry,
        )


def test_an_optimize_preset_with_a_native_camera_is_refused_at_expansion() -> None:
    from types import SimpleNamespace

    from geecs_schemas import OptimizerConfig

    config = OptimizerConfig(
        vocs={
            "variables": {"Motor:Current": [-1, 1]},
            "objectives": {"score": "MINIMIZE"},
        },
        measurements={"score": {"signal": "Meter:Value"}},
        generator={"name": "random"},
        run={"max_iterations": 3, "shots_per_step": 2},
    )
    resolver = SimpleNamespace(
        resolve_optimizer_config=lambda _: config,
        resolve_native_devices=lambda: SimpleNamespace(devices={"ADTEST_CAM1": None}),
    )
    preset = Preset(
        name="opt",
        devices=[{"device": "ADTEST_CAM1"}],
        plan={"name": "optimize", "kwargs": {"optimizer_config": "test"}},
    )
    with pytest.raises(GeecsConfigurationError) as info:
        expand_preset(preset, resolver=resolver)
    assert str(info.value) == (
        STRICT_ONLY + "not in optimize (its live frames are matched on gateway PVs)"
    )
