"""The bridge's devices: the registry joined with panel data; the bench file in both forms."""

from __future__ import annotations

from pathlib import Path

import pytest
from geecs_schemas.epics.family import load_families
from geecs_schemas.epics.panel_devices import PanelDevices

from geecs_panels.bridge.catalog import (
    build_catalog,
    load_bench_file,
    load_experiment,
    load_panel_devices,
    panel_devices_path,
)
from geecs_panels.registry import Device, RegistryError

FAMILIES = load_families()


def _panels(devices: dict) -> PanelDevices:
    return PanelDevices.model_validate({"devices": devices})


def test_a_registry_device_takes_its_kinds_family_and_its_own_prefix_and_axes() -> None:
    registry = [
        Device("Cam", "aravis_camera", "expt:cam:"),
        Device("Stage", "epics_motor", "expt:stage:", axes=("m1",)),
    ]
    devices, errors = build_catalog("expt", registry, PanelDevices(), FAMILIES)
    assert not errors
    cam, stage = devices
    assert (cam.family.template, cam.prefix, cam.source) == (
        "ad_camera_chain",
        "expt:cam:",
        "registry",
    )
    assert stage.axes == ("m1",)
    assert stage.actions == ("move", "tweak", "stop", "home", "velocity")


def test_panel_data_overrides_a_registry_device_but_not_its_prefix_or_axes() -> None:
    registry = [Device("Stage", "epics_motor", "expt:stage:", axes=("m1",))]
    panels = _panels(
        {
            "Stage": {
                "family": "motor_record",
                "prefix": "bench:other:",
                "axes": ["m9"],
                "allow": ["move", "stop"],
                "labels": {"rbv": "position"},
                "note": "mirror mount",
            }
        }
    )
    (stage,), errors = build_catalog("expt", registry, panels, FAMILIES)
    assert not errors
    assert stage.prefix == "expt:stage:" and stage.axes == ("m1",)
    assert stage.actions == ("move", "stop")
    assert stage.role("rbv").label == "position"
    assert stage.note == "mirror mount"


def test_a_device_in_panel_data_alone_is_a_bench_device() -> None:
    panels = _panels(
        {
            "PS1": {
                "family": "fastps",
                "prefix": "bench:ps1:",
                "omit": ["interlock_2"],
            },
            "Cam2": {"family": "ad_camera_chain"},
        }
    )
    devices, errors = build_catalog("Undulator", [], panels, FAMILIES)
    assert not errors
    ps1, cam2 = devices
    assert ps1.source == "panel_devices" and ps1.prefix == "bench:ps1:"
    assert not ps1.has_role("interlock_2")
    assert cam2.prefix == "undulator:cam2:"  # the naming rule's prefix


def test_a_family_that_does_not_serve_the_registry_kind_is_an_error() -> None:
    registry = [Device("Cam", "aravis_camera", "expt:cam:")]
    panels = _panels({"Cam": {"family": "motor_record", "axes": ["m1"]}})
    devices, errors = build_catalog("expt", registry, panels, FAMILIES)
    assert devices == []
    assert "registry lists a aravis_camera" in errors[0]


def test_an_old_form_bench_supply_is_named_not_dropped_silently() -> None:
    registry = [Device("PS", "power_supply", "bench:ps:", roles=(("current", "c"),))]
    devices, errors = build_catalog("expt", registry, PanelDevices(), FAMILIES)
    assert devices == []
    assert "names no family" in errors[0]


def test_a_kind_without_a_family_has_no_stream() -> None:
    devices, errors = build_catalog(
        "expt", [Device("X", "motor_stage", "expt:x:")], PanelDevices(), FAMILIES
    )
    assert devices == [] and errors == []


def test_a_per_axis_device_without_axes_is_an_error() -> None:
    panels = _panels({"Stage": {"family": "motor_record", "prefix": "b:s:"}})
    devices, errors = build_catalog("expt", [], panels, FAMILIES)
    assert devices == [] and "names no axes" in errors[0]


BENCH_NEW = """
schema_version: 1
devices:
  U_ChicaneInner:
    family: fastps
    prefix: "fpstest1:"
    allow: [current, output]
    note: bench supply
  U_Stage:
    family: motor_record
    prefix: "simmotor1:"
    axes: [m1]
"""

BENCH_OLD = """
devices:
  U_Stage:
    kind: epics_motor
    prefix: "simmotor1:"
    axes: [m1]
    allow: [move, stop]
"""


def test_a_bench_file_as_panel_devices_gives_the_pages_and_the_bridge(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bench.yaml"
    path.write_text(BENCH_NEW)
    pages, panels = load_bench_file(path, "expt", FAMILIES)
    assert panels is not None and set(panels.devices) == {"U_ChicaneInner", "U_Stage"}
    supply, stage = pages
    assert supply.kind == "power_supply" and supply.prefix == "fpstest1:"
    assert dict(supply.roles)["current_sp"] == "current_ramp:SP"
    assert [label for label, _ in supply.bits] == [
        "fault",
        "interlock_1",
        "interlock_2",
    ]
    assert supply.allow == frozenset({"current", "output"})
    assert stage.kind == "epics_motor" and stage.axes == ("m1",)
    devices, errors = build_catalog("expt", [], panels, FAMILIES)
    assert not errors and [d.name for d in devices] == ["U_ChicaneInner", "U_Stage"]


def test_a_bench_file_in_the_old_form_still_reads_for_the_pages(tmp_path: Path) -> None:
    path = tmp_path / "bench.yaml"
    path.write_text(BENCH_OLD)
    pages, panels = load_bench_file(path, "expt", FAMILIES)
    assert panels is None
    (stage,) = pages
    assert stage.allow == frozenset({"move", "stop"})
    (device,), errors = build_catalog("expt", pages, PanelDevices(), FAMILIES)
    assert device.actions == ("move", "stop")


def test_a_broken_bench_file_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "bench.yaml"
    path.write_text("devices:\n  X:\n    family: no_such_family\n")
    with pytest.raises(RegistryError, match="no_such_family"):
        load_bench_file(path, "expt", FAMILIES)


def test_panel_devices_come_from_the_experiments_configs_folder(tmp_path: Path) -> None:
    assert panel_devices_path("Undulator", tmp_path) == (
        tmp_path / "Undulator" / "panel_devices.yaml"
    )
    assert load_panel_devices(tmp_path / "absent.yaml").devices == {}
    broken = tmp_path / "broken.yaml"
    broken.write_text("devices:\n  X:\n    family: fastps\n    roles: {nope: x}\n")
    with pytest.raises(RegistryError):
        load_panel_devices(broken)


def test_the_experiment_loads_registry_panel_data_bench_and_scopes(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "Undulator"
    folder.mkdir()
    (folder / "native_devices.yaml").write_text(
        "devices:\n  UC_Cam:\n    kind: aravis_camera\n    trigger_source: line0\n"
    )
    (folder / "panel_devices.yaml").write_text(
        "devices:\n  U_Ez:\n    family: easydriver\n    prefix: 'bench:ez:'\n"
    )
    bench = tmp_path / "bench.yaml"
    bench.write_text(BENCH_NEW)
    loaded = load_experiment(
        "Undulator",
        FAMILIES,
        bench=bench,
        scopes=["scope1=bench:scope1:"],
        base=tmp_path,
    )
    assert loaded.errors == [] and loaded.unserved == []
    assert [d.name for d in loaded.served] == [
        "UC_Cam",
        "scope1",
        "U_Ez",
        "U_ChicaneInner",
        "U_Stage",
    ]
    assert {d.name for d in loaded.pages} == {d.name for d in loaded.served}
    with pytest.raises(RegistryError, match="--scope"):
        load_experiment("Undulator", FAMILIES, scopes=["nope"], base=tmp_path)
    (folder / "panel_devices.yaml").write_text(
        "devices:\n  U_Stage:\n    family: motor_record\n    prefix: 'b:s:'\n    axes: [m1]\n"
    )
    with pytest.raises(RegistryError, match="both panel_devices.yaml"):
        load_experiment("Undulator", FAMILIES, bench=bench, base=tmp_path)


def test_an_unreadable_panel_devices_file_is_an_error_not_no_devices(
    tmp_path: Path,
) -> None:
    (tmp_path / "Undulator").mkdir()
    (tmp_path / "Undulator" / "panel_devices.yaml").write_text("devices: [")
    loaded = load_experiment("Undulator", FAMILIES, base=tmp_path)
    assert "panel devices" in loaded.errors[0]
