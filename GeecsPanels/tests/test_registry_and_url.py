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
"""


def test_devices_come_in_file_order_with_derived_prefixes(tmp_path: Path) -> None:
    path = tmp_path / "native_devices.yaml"
    path.write_text(REGISTRY)
    devices = load_devices("Undulator", path)
    assert [(d.name, d.kind, d.prefix) for d in devices] == [
        ("ADTEST_CAM1", "aravis_camera", "undulator:adtest_cam1:"),
        ("UC_VisaEBeam6", "aravis_camera", "undulator:uc_visaebeam6:"),
    ]


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
