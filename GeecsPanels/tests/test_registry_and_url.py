"""The registry read and the stream-URL rehosting."""

from __future__ import annotations

from pathlib import Path

import pytest

from geecs_panels.ca import with_ca_host
from geecs_panels.registry import RegistryError, load_devices, registry_path

REGISTRY = """\
schema_version: 1
devices:
  ADTEST_CAM1:
    kind: aravis_camera
    trigger_source: software
  UC_VisaEBeam6:
    kind: aravis_camera
    trigger_source: line0
  SimMotor1:
    kind: epics_motor
    axes: [m1, m2]
"""


def test_devices_come_in_file_order_with_derived_prefixes(tmp_path: Path) -> None:
    path = tmp_path / "native_devices.yaml"
    path.write_text(REGISTRY)
    devices = load_devices("Undulator", path)
    assert [(d.name, d.kind, d.prefix) for d in devices] == [
        ("ADTEST_CAM1", "aravis_camera", "undulator:adtest_cam1:"),
        ("UC_VisaEBeam6", "aravis_camera", "undulator:uc_visaebeam6:"),
        ("SimMotor1", "epics_motor", "undulator:simmotor1:"),
    ]


def test_a_positioner_carries_its_axes_and_has_a_panel(tmp_path: Path) -> None:
    """The registry's epics_motor entry reaches the panel: its axes, and the kind PANELS keys on."""
    from geecs_panels.app import PANELS

    path = tmp_path / "native_devices.yaml"
    path.write_text(REGISTRY)
    cam, _, stage = load_devices("Undulator", path)
    assert stage.axes == ("m1", "m2") and cam.axes == ()
    assert PANELS[stage.kind] == "motor"


def test_an_absent_registry_is_no_devices(tmp_path: Path) -> None:
    assert load_devices("Undulator", tmp_path / "native_devices.yaml") == []
    assert load_devices("Undulator", None) == []


def test_a_broken_registry_raises_rather_than_reading_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "native_devices.yaml"
    path.write_text("schema_version: 1\ndevices:\n  CAM:\n    kind: toaster\n")
    with pytest.raises(RegistryError, match="invalid"):
        load_devices("Undulator", path)


def test_the_registry_lives_in_the_experiment_folder(tmp_path: Path) -> None:
    assert registry_path("Undulator", base=tmp_path) == (
        tmp_path / "Undulator" / "native_devices.yaml"
    )


@pytest.mark.parametrize(
    ("url", "ca_host", "expected"),
    [
        (
            "http://camserv:8081/CAM1.MJPG.jpg",
            "192.168.7.166:5064",
            "http://192.168.7.166:8081/CAM1.MJPG.jpg",
        ),
        # An IOC on its own CA port: the CA port is dropped, the HTTP port kept.
        (
            "http://camserv:8082/CAM1.MJPG.jpg",
            "192.168.7.166:5066",
            "http://192.168.7.166:8082/CAM1.MJPG.jpg",
        ),
        ("http://camserv/x.jpg", "10.0.0.5:5064", "http://10.0.0.5/x.jpg"),
    ],
)
def test_the_stream_url_is_rehosted_to_the_ca_address(url, ca_host, expected) -> None:
    assert with_ca_host(url, ca_host) == expected


def test_an_unreachable_configs_tree_is_an_error_not_no_devices(tmp_path: Path) -> None:
    """An unmounted share must not read as "no native devices"."""
    path = tmp_path / "gone" / "Undulator" / "native_devices.yaml"
    with pytest.raises(RegistryError, match="not reachable"):
        load_devices("Undulator", path)


# ---- the bench file -------------------------------------------------------------

BENCH = """\
devices:
  U_ESP302_01:
    kind: epics_motor
    prefix: "undulator:esptest1:"
    axes: [m1, m2, m3]
    allow: [move, tweak, stop]
    note: Between moves the readback is the last reading.
  SimMotor2:
    kind: epics_motor
    prefix: "undulator:simmotor2:"
    axes: [m1]
"""


def test_a_bench_device_carries_its_prefix_axes_and_permitted_actions(
    tmp_path: Path,
) -> None:
    from geecs_panels.registry import load_bench

    path = tmp_path / "bench.yaml"
    path.write_text(BENCH)
    esp, sim = load_bench(path)
    assert (esp.name, esp.kind, esp.prefix, esp.axes) == (
        "U_ESP302_01", "epics_motor", "undulator:esptest1:", ("m1", "m2", "m3")
    )  # fmt: skip
    assert esp.allow == {"move", "tweak", "stop"} and esp.note.startswith("Between")
    assert sim.allow is None and sim.note == ""


@pytest.mark.parametrize(
    "entry, message",
    [
        (
            "kind: epics_motor\n    prefix: 'p:'\n    axes: [m1]\n    allow: [move]",
            "stop",
        ),
        (
            "kind: epics_motor\n    prefix: 'p:'\n    axes: [m1]\n    allow: [jog, stop]",
            "unknown",
        ),
        ("kind: epics_motor\n    prefix: 'p:'\n    axes: []", "axes"),
        ("kind: epics_motor\n    prefix: 'p'\n    axes: [m1]", "prefix"),
        (
            "kind: epics_motor\n    prefix: 'p:'\n    axes: [m1]\n    alow: [stop]",
            "alow",
        ),
        ("kind: toaster\n    prefix: 'p:'", "kind"),
    ],
)
def test_a_broken_bench_file_raises_naming_why(tmp_path: Path, entry, message) -> None:
    from geecs_panels.registry import load_bench

    path = tmp_path / "bench.yaml"
    path.write_text(f"devices:\n  D:\n    {entry}\n")
    with pytest.raises(RegistryError, match=message):
        load_bench(path)


@pytest.mark.parametrize(
    "name, axes, message",
    [
        ("ESP/mirrors", "[m1]", "devices"),  # a name is a URL path segment
        ("ESP", "[X, x]", "same axis after normalization"),  # the registry's own rule
    ],
)
def test_a_bench_entry_keeps_the_registrys_rules_and_routable_names(
    tmp_path: Path, name, axes, message
) -> None:
    from geecs_panels.registry import load_bench

    path = tmp_path / "bench.yaml"
    path.write_text(
        f"devices:\n  '{name}':\n    kind: epics_motor\n    prefix: 'p:'\n    axes: {axes}\n"
    )
    with pytest.raises(RegistryError, match=message):
        load_bench(path)


def test_a_registry_error_still_lists_the_bench_devices() -> None:
    from fastapi.testclient import TestClient

    from geecs_panels.app import create_app
    from geecs_panels.registry import Device

    bench = Device("U_ESP302_01", "epics_motor", "p:", axes=("m1",))
    page = TestClient(
        create_app(
            "Undulator",
            [bench],
            object(),
            registry_error="share not mounted",
            motors=object(),
        )
    ).get("/")
    assert "share not mounted" in page.text
    assert 'href="/motor/U_ESP302_01"' in page.text
