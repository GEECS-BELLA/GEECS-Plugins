"""A native camera's unsupported run shapes are refused at bind (``NATIVE_DEVICES.md``).

Every shape outside "an essential detector of a strict run under a
box-fired profile, no gateway plugin camera beside it" is refused before
the box is driven and before ``open_run`` (no scan number claimed), each
naming the camera — the camera's own fly refusals come after the claim.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("aioca")

from bluesky import RunEngine

from geecs_bluesky.devices.native import NativeTriggerSource
from geecs_bluesky.devices.shot_control import ShotControl, ShotControlWrites
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.plans.registry import TriggerProfiles, bind_plans, refuse_native_run
from geecs_bluesky.qs_client.presets import expand_preset
from geecs_schemas import Preset
from tests.ca_mock_helpers import DocCollector, connect_mock
from tests.test_native_camera import _camera as _native_camera
from tests.test_strict_plans import WRITES, FakeBox, _camera, _plugin_camera

STRICT_ONLY = "ADTEST_CAM1: native EPICS cameras run in strict scans only for now"
#: The strict states plus the gated bracket's, so a gated run could start.
STATES = {
    **WRITES.states,
    "OFF": [("DG", "Trigger.Source", "off")],
    "SCAN": [("DG", "Trigger.Source", "edges")],
}


@pytest.fixture
def RE() -> RunEngine:
    return RunEngine()


@pytest.fixture
def box() -> FakeBox:
    return FakeBox()


def _profiles(RE: RunEngine, box: FakeBox, **kw) -> TriggerProfiles:
    writes = ShotControlWrites(name="P", states=kw.pop("states", STATES), **kw)
    sc = ShotControl(writes, experiment="TestExp", name="p", setter_factory=box)
    connect_mock(RE, sc)
    return TriggerProfiles({"P": sc}, default="P")


def _refused(RE: RunEngine, box: FakeBox, plan, reason: str) -> None:
    """*plan* is refused naming the camera and *reason*: nothing driven, nothing claimed."""
    col = DocCollector()
    RE.subscribe(col)
    with pytest.raises(GeecsConfigurationError, match=STRICT_ONLY) as info:
        RE(plan)
    assert reason in str(info.value)
    assert box.puts == [] and box.fires == 0
    assert col.docs["start"] == []


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"acquisition": "gated"}, "not in a gated batch"),
        ({"non_essential": "native"}, "not as a non-essential stream"),
        (
            {"non_essential": "native", "acquisition": "gated"},
            "not as a non-essential stream",
        ),
    ],
    ids=["gated-essential", "strict-non-essential", "gated-non-essential"],
)
def test_gated_runs_and_non_essential_streams_are_refused_before_the_claim(
    RE: RunEngine, box: FakeBox, tmp_path: Path, kwargs: dict, reason: str
) -> None:
    native, _ = _native_camera(RE, tmp_path)
    other = _camera(RE, box, "UC_Other")
    if kwargs.get("non_essential") == "native":
        detectors, kwargs = [other], {**kwargs, "non_essential": [native]}
    else:
        detectors = [native, other]
    count = bind_plans(_profiles(RE, box))["count"]
    _refused(RE, box, count(detectors, 2, **kwargs), reason)


@pytest.mark.parametrize("states", [STATES, {}], ids=["box", "box-less"])
def test_a_scan_mode_profile_is_refused_before_the_preflight(
    RE: RunEngine, box: FakeBox, tmp_path: Path, states: dict
) -> None:
    """Refused ahead of the scan-mode preflight (no control here, which it would refuse)."""
    native, _ = _native_camera(RE, tmp_path)
    count = bind_plans(_profiles(RE, box, states=states, use_scan_mode=True))["count"]
    _refused(RE, box, count([native], 2), "not under a scan-mode profile")


@pytest.mark.parametrize("listed", ["essential", "non_essential"])
def test_a_gateway_plugin_camera_beside_a_native_one_is_refused(
    RE: RunEngine, box: FakeBox, tmp_path: Path, listed: str
) -> None:
    native, _ = _native_camera(RE, tmp_path)
    plugin, _ = _plugin_camera(RE, box, "UC_Plugin", tmp_path)
    count = bind_plans(_profiles(RE, box))["count"]
    plan = (
        count([native, plugin], 2)
        if listed == "essential"
        else count([native], 2, non_essential=[plugin])
    )
    _refused(RE, box, plan, "not beside a gateway plugin camera (uc_plugin)")


@pytest.mark.parametrize("name", ["measure_shot_offsets", "check_shot_sync"])
def test_the_calibration_plans_refuse_a_native_camera(
    RE: RunEngine, box: FakeBox, tmp_path: Path, name: str
) -> None:
    native, _ = _native_camera(RE, tmp_path)
    other = _camera(RE, box, "UC_Other")
    plan = bind_plans(_profiles(RE, box))[name]
    _refused(RE, box, plan([native, other]), "not in a shot-offset calibration")


def test_optimize_refuses_a_native_camera_before_reading_its_config(
    RE: RunEngine, box: FakeBox, tmp_path: Path
) -> None:
    """Refused before the optimizer config is resolved (the stand-ins have no methods)."""
    pytest.importorskip("xopt")
    from geecs_bluesky.plans.optimize import optimize_plan

    native, _ = _native_camera(RE, tmp_path)
    plan = optimize_plan(_profiles(RE, box), object(), object())  # type: ignore[arg-type]
    _refused(RE, box, plan([native], optimizer_config="any"), "not in optimize")


def test_the_strict_essential_shape_is_admitted(
    RE: RunEngine, box: FakeBox, tmp_path: Path
) -> None:
    """The one v1 shape passes, a plugin camera's scalars view beside it included."""
    native, _ = _native_camera(RE, tmp_path, source=NativeTriggerSource.LINE1)
    plugin, _ = _plugin_camera(RE, box, "UC_Plugin", tmp_path)
    shot_control = _profiles(RE, box).resolve(None)
    refuse_native_run(
        [native, plugin.scalars, _camera(RE, box, "UC_Other")],
        [],
        acquisition="strict",
        shot_control=shot_control,
    )


def test_a_preset_asking_for_a_native_cameras_scalars_is_refused_at_expansion() -> None:
    preset = Preset.model_validate(
        {
            "name": "native",
            "devices": [
                {"device": "UC_Other", "save_images": False},
                {"device": "ADTEST_CAM1", "save_images": False},
            ],
            "plan": {"name": "count", "kwargs": {"num": 2}},
        }
    )
    with pytest.raises(GeecsConfigurationError, match="ADTEST_CAM1 is a native"):
        expand_preset(preset, native_devices={"adtest_cam1"})
    # unknown to the registry, or saving images: expanded as before
    assert expand_preset(preset).args[0] == ["UC_Other.scalars", "ADTEST_CAM1.scalars"]
    saved = preset.model_copy(
        update={
            "devices": [
                d.model_copy(update={"save_images": True}) for d in preset.devices
            ]
        }
    )
    assert (
        expand_preset(saved, native_devices={"ADTEST_CAM1"}).args[0][1] == "ADTEST_CAM1"
    )
