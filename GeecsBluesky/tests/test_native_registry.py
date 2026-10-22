"""The native device registry: read from the configs repo, built into the namespace.

``native_devices.yaml`` names the devices served by native EPICS IOCs
(``NATIVE_DEVICES.md`` §1–2).  Pinned here: an absent file is no devices
and a broken one is loud; each entry becomes a ``NativeCamera`` at the
derived prefix with the run's POSIX path provider; a name the DB roster
also owns refuses to open; the background telemetry never reads one.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest
import yaml
from ophyd_async.core import PathInfo

from geecs_schemas import NativeDevices

from geecs_bluesky import config_resolver
from geecs_bluesky.config_resolver import ConfigsRepoResolver
from geecs_bluesky.exceptions import GeecsConfigurationError

ENTRY = {"kind": "aravis_camera", "trigger_source": "software"}


def _write(tmp_path: Path, document) -> ConfigsRepoResolver:
    (tmp_path / "TestExp").mkdir(exist_ok=True)
    (tmp_path / "TestExp" / "native_devices.yaml").write_text(yaml.safe_dump(document))
    return ConfigsRepoResolver("TestExp", experiments_root=tmp_path)


# ---------------------------------------------------------------- resolver
def test_no_registry_file_is_no_native_devices(tmp_path: Path) -> None:
    resolver = ConfigsRepoResolver("TestExp", experiments_root=tmp_path)
    assert resolver.resolve_native_devices().devices == {}


def test_an_unresolvable_configs_root_is_no_native_devices(monkeypatch) -> None:
    monkeypatch.setattr(config_resolver, "scanner_configs_base", lambda: None)
    assert ConfigsRepoResolver("TestExp").resolve_native_devices().devices == {}


def test_an_unreachable_configs_root_is_loud_not_empty(tmp_path: Path) -> None:
    # A configured root whose share is not mounted must not drop the cameras.
    resolver = ConfigsRepoResolver("TestExp", experiments_root=tmp_path / "unmounted")
    with pytest.raises(GeecsConfigurationError, match="not reachable"):
        resolver.resolve_native_devices()


def test_the_registry_is_read_and_validated(tmp_path: Path) -> None:
    resolver = _write(tmp_path, {"schema_version": 1, "devices": {"UC_Cam": ENTRY}})
    registry = resolver.resolve_native_devices()
    assert registry.devices["UC_Cam"].trigger_source == "software"


@pytest.mark.parametrize(
    "document",
    [
        {"devices": {"UC_Cam": {"kind": "pco_camera", "trigger_source": "line1"}}},
        {"schema_version": 2, "devices": {}},
        ["UC_Cam"],  # not a mapping
    ],
)
def test_a_broken_registry_is_loud_not_empty(tmp_path: Path, document) -> None:
    resolver = _write(tmp_path, document)
    with pytest.raises(GeecsConfigurationError, match="native_devices.yaml"):
        resolver.resolve_native_devices()


# --------------------------------------------------------------- namespace
@pytest.fixture
def ca():
    """The namespace builds CA devices: the ``ca`` extra (CI's pure-unit job lacks it)."""
    pytest.importorskip("aioca")


class _Shared:
    """The run's shared provider, pointed at an existing scan folder."""

    def __init__(self, scan: Path) -> None:
        self.scan = scan

    def __call__(self, stem: str | None = None) -> PathInfo:
        return PathInfo(
            directory_path=self.scan / str(stem),
            filename=str(stem),
            directory_uri=f"file://localhost{self.scan / str(stem)}/",
        )


def _registry(**devices) -> NativeDevices:
    return NativeDevices.model_validate({"devices": devices})


def _namespace(registry: NativeDevices, tmp_path: Path, **kwargs):
    from tests.test_namespace import ROSTER

    from geecs_bluesky.namespace import GeecsNamespace

    return GeecsNamespace(
        ROSTER,
        path_provider=_Shared(tmp_path),
        file_plugin_hosts=None,
        native_devices=registry,
        **kwargs,
    )


def test_each_entry_is_a_native_camera_at_the_derived_prefix(
    ca, tmp_path: Path, monkeypatch
) -> None:
    from tests.test_namespace import _offset

    from geecs_bluesky import data_paths
    from geecs_bluesky.data_paths import native_ioc_save_path
    from geecs_bluesky.devices.hdf_plugin import PluginPathProvider
    from geecs_bluesky.devices.native import NativeCamera, NativeTriggerSource

    registry = _registry(
        **{
            "UC_NativeCam": ENTRY,
            "UC Line Cam": {"kind": "aravis_camera", "trigger_source": "line1"},
        }
    )
    ns = _namespace(registry, tmp_path, drain_offsets={"uc_nativecam": 0.012})
    cam = ns["UC_NativeCam"]
    assert isinstance(cam, NativeCamera) and ns.devices["UC_NativeCam"] is cam
    assert cam.name == "uc_nativecam" and cam._geecs_device_name == "UC_NativeCam"
    assert cam._geecs_namespace_member
    assert cam.trigger_source == NativeTriggerSource.SOFTWARE
    assert cam.driver.acquire.source == "ca://testexp:uc_nativecam:cam1:Acquire_RBV"
    assert cam.hdf.capture.source == "ca://testexp:uc_nativecam:HDF1:Capture_RBV"
    assert _offset(cam) == pytest.approx(0.012)
    # A GEECS name that is not an identifier binds by safe_name, like a DB device.
    line = ns.devices["uc_line_cam"]
    assert (
        line is ns["UC Line Cam"] and line.trigger_source == NativeTriggerSource.LINE1
    )
    # The writer's path: the run's folder, POSIX, through the native IOC mapping.
    (provider,) = {
        logic.path_provider
        for logic in cam._data_logics
        if hasattr(logic, "path_provider")
    }
    assert isinstance(provider, PluginPathProvider)
    assert provider._plugin_path is native_ioc_save_path
    monkeypatch.setattr(data_paths, "read_native_ioc_data_base_path", lambda: None)
    info = provider()
    assert isinstance(info.directory_path, PurePosixPath)
    assert info.directory_path == PurePosixPath(tmp_path / "UC_NativeCam")
    assert info.filename == "UC_NativeCam"
    assert (tmp_path / "UC_NativeCam").is_dir()
    # No scalar children: a variable lookup says so instead of a bare KeyError.
    with pytest.raises(KeyError, match="no served scalar"):
        ns.variable("UC_NativeCam", "exposure")


@pytest.mark.parametrize("spelling", ["UC_TestCam", "uc_testcam", "UC TestCam"])
def test_a_registry_name_the_db_owns_refuses_to_open(ca, tmp_path, spelling) -> None:
    with pytest.raises(GeecsConfigurationError) as caught:
        _namespace(_registry(**{spelling: ENTRY}), tmp_path)
    message = str(caught.value)
    assert repr(spelling) in message and "'UC_TestCam'" in message
    assert "one owner" in message


def test_a_db_device_without_served_scalars_still_owns_its_name(ca, tmp_path) -> None:
    """``U_ImagesOnly`` is not built (no scalars, not triggered) but is still the DB's."""
    with pytest.raises(GeecsConfigurationError, match="U_ImagesOnly"):
        _namespace(_registry(U_ImagesOnly=ENTRY), tmp_path)


def test_two_registry_names_normalising_alike_are_refused(ca, tmp_path) -> None:
    with pytest.raises(GeecsConfigurationError, match="both normalise"):
        _namespace(_registry(UC_Nat=ENTRY, uc_nat=ENTRY), tmp_path)


def test_a_native_camera_needs_the_runs_path_provider(ca) -> None:
    from tests.test_namespace import ROSTER

    from geecs_bluesky.namespace import GeecsNamespace

    with pytest.raises(GeecsConfigurationError, match="path provider"):
        GeecsNamespace(
            ROSTER, file_plugin_hosts=None, native_devices=_registry(UC_Nat=ENTRY)
        )


def test_telemetry_never_reads_a_native_camera(ca, tmp_path) -> None:
    """A native camera is Triggerable with no scalars: a telemetry read would wait for a shot."""
    without = _namespace(NativeDevices(), tmp_path)
    ns = _namespace(_registry(UC_Nat=ENTRY), tmp_path)
    assert ns["UC_Nat"] not in ns.telemetry()
    assert len(ns.telemetry()) == len(without.telemetry())
