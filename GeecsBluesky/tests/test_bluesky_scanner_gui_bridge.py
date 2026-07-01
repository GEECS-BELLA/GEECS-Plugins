"""Tests for Scanner GUI compatibility helpers in BlueskyScanner."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.models.shot_control import ShotControlConfig
from geecs_bluesky.scanner_bridge import bluesky_scanner
from geecs_bluesky.scanner_bridge.bluesky_scanner import (
    BlueskyScanner,
    _prepare_descriptor_for_tiled,
    _SafeDocumentCallback,
    _ensure_timestamp_variable,
    _translate_save_path_for_device_server,
)


@dataclass
class _FakeLifecycleEvent:
    """Minimal replacement for geecs_scanner.engine ScanLifecycleEvent."""

    state: str
    total_shots: int = 0


def test_set_state_emits_gui_lifecycle_event(monkeypatch) -> None:
    """BlueskyScanner should emit GUI lifecycle events when callback is present."""
    events: list[_FakeLifecycleEvent] = []
    scanner = BlueskyScanner.__new__(BlueskyScanner)
    scanner._on_event = events.append
    scanner._current_state = None

    monkeypatch.setattr(
        bluesky_scanner,
        "ScanState",
        SimpleNamespace(DONE="done"),
    )
    monkeypatch.setattr(
        bluesky_scanner,
        "ScanLifecycleEvent",
        _FakeLifecycleEvent,
    )

    scanner._set_state("DONE", total_shots=12)

    assert scanner.current_state == "done"
    assert events == [_FakeLifecycleEvent(state="done", total_shots=12)]


def test_prepare_descriptor_for_tiled_internalizes_geecs_assets() -> None:
    """Tiled should store GEECS asset datum ids without registering readers."""
    doc = {
        "data_keys": {
            "uc_topview-image": {
                "source": "geecs://UC_TopView/image",
                "external": "OLD:",
                "dtype": "array",
            },
            "external-hdf5": {
                "source": "AD_HDF5",
                "external": "STREAM:",
                "dtype": "array",
            },
            "scalar": {"source": "sim://scalar", "dtype": "number"},
        }
    }

    patched = _prepare_descriptor_for_tiled(doc)

    geecs_key = patched["data_keys"]["uc_topview-image"]
    assert "external" not in geecs_key
    assert geecs_key["geecs_external_asset"] is True
    assert patched["data_keys"]["external-hdf5"]["external"] == "STREAM:"
    assert "external" not in patched["data_keys"]["scalar"]


def test_safe_document_callback_warns_and_disables(caplog) -> None:
    """Persistence callbacks should not be able to abort acquisition."""
    calls: list[str] = []

    def callback(name: str, _doc: dict) -> None:
        calls.append(name)
        if name == "event":
            raise RuntimeError("storage failed")

    safe_callback = _SafeDocumentCallback(callback, label="Storage")

    with caplog.at_level(logging.WARNING):
        safe_callback("start", {})
        safe_callback("event", {})
        safe_callback("stop", {})

    assert calls == ["start", "event"]
    assert "Storage failed while handling event document" in caplog.text


def test_translate_save_path_for_device_server() -> None:
    """Mac/Linux scan paths should translate to the configured device path."""
    path = _translate_save_path_for_device_server(
        "/Volumes/hdna2/data/Undulator/Y2026/06-Jun/26_0623/scans/Scan011/UC_Cam",
        local_base_path="/Volumes/hdna2/data",
        device_server_base_path="Z:/data",
    )

    assert path == r"Z:\data\Undulator\Y2026\06-Jun\26_0623\scans\Scan011\UC_Cam"


def test_timestamp_variable_selection_for_registered_diagnostics() -> None:
    """Registered diagnostics currently use the live ``acq_timestamp`` field."""
    assert BlueskyScanner._timestamp_variable_for_device_type("MagSpecCamera") == (
        "acq_timestamp"
    )
    assert BlueskyScanner._timestamp_variable_for_device_type("PicoscopeV2") == (
        "acq_timestamp"
    )
    assert BlueskyScanner._timestamp_variable_for_device_type("Point Grey Camera") == (
        "acq_timestamp"
    )


def test_ensure_timestamp_variable_replaces_default_timestamp_name() -> None:
    """Detector configs using the default timestamp name should adapt by device."""
    assert _ensure_timestamp_variable(
        ["acq_timestamp"],
        timestamp_variable="timestamp",
    ) == ["timestamp"]
    assert _ensure_timestamp_variable(
        ["Signal", "timestamp"],
        timestamp_variable="timestamp",
    ) == ["Signal", "timestamp"]
    assert _ensure_timestamp_variable(
        ["Signal"],
        timestamp_variable="acq_timestamp",
    ) == ["Signal", "acq_timestamp"]


# ---------------------------------------------------------------------------
# Acquisition-mode resolution
# ---------------------------------------------------------------------------


def test_resolve_acquisition_mode_defaults_to_strict() -> None:
    options = SimpleNamespace(rep_rate_hz=1.0)  # no acquisition_mode attr
    mode = BlueskyScanner._resolve_acquisition_mode(options, env={})
    assert mode == "strict_shot_control"


def test_resolve_acquisition_mode_from_options() -> None:
    options = SimpleNamespace(acquisition_mode="free_run_time_sync")
    mode = BlueskyScanner._resolve_acquisition_mode(options, env={})
    assert mode == "free_run_time_sync"


def test_resolve_acquisition_mode_env_overrides_options() -> None:
    options = SimpleNamespace(acquisition_mode="strict_shot_control")
    mode = BlueskyScanner._resolve_acquisition_mode(
        options, env={"GEECS_BLUESKY_ACQUISITION_MODE": "free_run_time_sync"}
    )
    assert mode == "free_run_time_sync"


def test_resolve_acquisition_mode_unknown_raises() -> None:
    options = SimpleNamespace(acquisition_mode="nonsense")
    with pytest.raises(GeecsConfigurationError, match="Unknown acquisition_mode"):
        BlueskyScanner._resolve_acquisition_mode(options, env={})


# ---------------------------------------------------------------------------
# Strict shot-control validation
# ---------------------------------------------------------------------------


def _scanner_with_shot_control(
    information: dict | None, setters: dict | None = None
) -> BlueskyScanner:
    scanner = BlueskyScanner.__new__(BlueskyScanner)
    scanner._shot_control = ShotControlConfig.from_information(information)
    scanner._shot_control_setters = setters or {}
    return scanner


def test_strict_single_shot_requires_shot_control_config() -> None:
    scanner = _scanner_with_shot_control(None)

    with pytest.raises(GeecsConfigurationError, match="shot_control_information"):
        scanner._require_strict_single_shot()


def test_strict_single_shot_requires_nonempty_armed_state() -> None:
    scanner = _scanner_with_shot_control(
        {
            "device": "U_DG645_ShotControl",
            "variables": {
                "Trigger.Source": {
                    "SCAN": "External rising edges",
                    "ARMED": "",
                }
            },
        },
        setters={"Trigger.Source": object()},
    )

    with pytest.raises(GeecsConfigurationError, match="non-empty ARMED"):
        scanner._require_strict_single_shot()


def test_strict_single_shot_requires_reachable_setters() -> None:
    scanner = _scanner_with_shot_control(
        {
            "device": "U_DG645_ShotControl",
            "variables": {
                "Trigger.Source": {
                    "ARMED": "Single shot external rising edges",
                }
            },
        }
    )

    with pytest.raises(GeecsConfigurationError, match="reachable shot-control"):
        scanner._require_strict_single_shot()


def test_strict_single_shot_accepts_armed_state_and_setters() -> None:
    scanner = _scanner_with_shot_control(
        {
            "device": "U_DG645_ShotControl",
            "variables": {
                "Trigger.Source": {
                    "ARMED": "Single shot external rising edges",
                }
            },
        },
        setters={"Trigger.Source": object()},
    )

    scanner._require_strict_single_shot()


# ---------------------------------------------------------------------------
# Device role classification
# ---------------------------------------------------------------------------

_DEVICES = {
    "U_CamA": {"synchronous": True},
    "U_CamB": {"synchronous": True},
    "U_Stage": {"synchronous": False},
}


def test_classify_roles_strict_mode() -> None:
    roles = dict(BlueskyScanner._classify_device_roles(_DEVICES, "strict_shot_control"))
    assert roles == {
        "U_CamA": "triggered",
        "U_CamB": "triggered",
        "U_Stage": "snapshot",
    }


def test_classify_roles_free_run_first_sync_is_reference() -> None:
    roles = dict(BlueskyScanner._classify_device_roles(_DEVICES, "free_run_time_sync"))
    assert roles == {
        "U_CamA": "reference",
        "U_CamB": "contributor",
        "U_Stage": "snapshot",
    }


def test_classify_roles_free_run_all_async_has_no_reference() -> None:
    devices = {"U_Stage": {"synchronous": False}}
    roles = dict(BlueskyScanner._classify_device_roles(devices, "free_run_time_sync"))
    assert "reference" not in roles.values()
    assert roles == {"U_Stage": "snapshot"}
