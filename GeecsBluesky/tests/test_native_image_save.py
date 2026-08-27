"""The native_image_save toggle: selection, application, resolution, md.

One devicetype-scoped decision (Planning/data_capture/ Phase 3 redesign):
capture-eligible cameras either write native per-shot files or the capture
daemon owns their images. Everything here is pre-claim, pure, and
fail-open — a DB failure means nothing is capture-eligible and native
saving stays on.
"""

from __future__ import annotations

from types import SimpleNamespace

from geecs_bluesky.db_runtime import GeecsDbDeviceTypes
from geecs_bluesky.scan_request_runner import (
    apply_native_image_save_off,
    build_step_scan_spec,
    resolve_native_image_save,
    select_capture_devices,
)
from geecs_schemas import ScanRequest


class _FakeTypesProvider:
    def __init__(self, types: dict[str, str]) -> None:
        self._types = types

    def by_device(self) -> dict[str, str]:
        return self._types


def _config(**devices):
    return {name: dict(cfg) for name, cfg in devices.items()}


def test_select_capture_devices_by_devicetype_and_save_flag() -> None:
    """Only image-saving devices of registry devicetypes are selected."""
    cfg = _config(
        UC_Cam={"save_nonscalar_data": True, "synchronous": True},
        U_Haso={"save_nonscalar_data": True, "synchronous": True},
        UC_NoImages={"save_nonscalar_data": False, "synchronous": True},
    )
    provider = _FakeTypesProvider(
        {
            "UC_Cam": "Point Grey Camera",
            "U_Haso": "HASO WFS",
            "UC_NoImages": "Point Grey Camera",
        }
    )
    assert select_capture_devices("Undulator", cfg, provider=provider) == ["UC_Cam"]


def test_select_capture_devices_fail_open_on_db_failure() -> None:
    """A DB failure yields no eligible devices — native saving untouched."""

    class _BoomDb:
        @staticmethod
        def get_experiment_device_types(experiment, *, enabled_only=True):
            raise RuntimeError("db down")

    provider = GeecsDbDeviceTypes("Undulator", db=_BoomDb)
    cfg = _config(UC_Cam={"save_nonscalar_data": True})
    assert select_capture_devices("Undulator", cfg, provider=provider) == []


def test_apply_native_image_save_off_copies_and_flips() -> None:
    """The off-switch flips save_nonscalar_data on a copy, nothing else."""
    cfg = _config(
        UC_Cam={"save_nonscalar_data": True, "synchronous": True},
        U_Haso={"save_nonscalar_data": True, "synchronous": True},
    )
    updated = apply_native_image_save_off(cfg, ["UC_Cam"])
    assert updated["UC_Cam"]["save_nonscalar_data"] is False
    assert updated["UC_Cam"]["synchronous"] is True
    assert updated["U_Haso"]["save_nonscalar_data"] is True
    assert cfg["UC_Cam"]["save_nonscalar_data"] is True  # original untouched


def test_resolve_native_image_save_tristate() -> None:
    """Request override wins; unset inherits the experiment default."""
    defaults_off = SimpleNamespace(native_image_save=False)
    base = {
        "mode": "noscan",
        "shots_per_step": 5,
        "save_sets": ["Amp4In"],
    }
    unset = ScanRequest.model_validate(base)
    forced_on = ScanRequest.model_validate({**base, "native_image_save": True})
    forced_off = ScanRequest.model_validate({**base, "native_image_save": False})
    assert resolve_native_image_save(unset, defaults_off) is False
    assert resolve_native_image_save(forced_on, defaults_off) is True
    assert resolve_native_image_save(forced_off, None) is False
    assert resolve_native_image_save(unset, None) is True  # default of defaults


def test_spec_md_records_capture_devices_and_toggle() -> None:
    """The run picture carries the daemon's device list + effective flag."""
    request = ScanRequest.model_validate(
        {"mode": "noscan", "shots_per_step": 3, "save_sets": ["Amp4In"]}
    )
    spec = build_step_scan_spec(
        request,
        [],
        applied_defaults={},
        slots={},
        dropped_unserved={},
        dropped_unserved_devices=[],
        disconnected_devices=[],
        telemetry_selected={},
        capture_devices=["UC_Cam"],
        native_image_save=False,
    )
    assert spec.md["capture_devices"] == ["UC_Cam"]
    assert spec.md["native_image_save"] is False

    spec_none = build_step_scan_spec(
        request,
        [],
        applied_defaults={},
        slots={},
        dropped_unserved={},
        dropped_unserved_devices=[],
        disconnected_devices=[],
        telemetry_selected={},
    )
    assert "capture_devices" not in spec_none.md
    assert "native_image_save" not in spec_none.md
