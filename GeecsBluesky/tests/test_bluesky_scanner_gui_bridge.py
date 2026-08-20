"""Tests for Scanner GUI compatibility helpers in BlueskyScanner."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.models.shot_control import ShotControlConfig
from geecs_bluesky.plans.orchestration import build_step_scan_plan
from geecs_bluesky.shot_controller import ShotController
from geecs_bluesky.scanner_bridge import bluesky_scanner
from geecs_bluesky.data_paths import (
    translate_save_path_for_device_server as _translate_save_path_for_device_server,
)
from geecs_bluesky.scanner_bridge.bluesky_scanner import BlueskyScanner
from geecs_bluesky.tiled_integration import (
    SafeDocumentCallback as _SafeDocumentCallback,
)
from geecs_bluesky.tiled_integration import (
    prepare_descriptor_for_tiled as _prepare_descriptor_for_tiled,
)


@dataclass
class _FakeLifecycleEvent:
    """Minimal replacement for geecs_scanner.engine ScanLifecycleEvent."""

    state: str
    total_shots: int = 0
    scan_number: int | None = None


def test_set_state_emits_gui_lifecycle_event(monkeypatch) -> None:
    """BlueskyScanner should emit GUI lifecycle events when callback is present."""
    events: list[_FakeLifecycleEvent] = []
    scanner = BlueskyScanner.__new__(BlueskyScanner)
    scanner._on_event = events.append
    scanner._current_state = None
    scanner._scan_number = None

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
    assert events == [
        _FakeLifecycleEvent(state="done", total_shots=12, scan_number=None)
    ]


def test_set_state_carries_claimed_scan_number(monkeypatch) -> None:
    """Once a scan number is claimed, every lifecycle emission carries it."""
    events: list[_FakeLifecycleEvent] = []
    scanner = BlueskyScanner.__new__(BlueskyScanner)
    scanner._on_event = events.append
    scanner._current_state = None
    scanner._scan_number = 41

    monkeypatch.setattr(
        bluesky_scanner,
        "ScanState",
        SimpleNamespace(RUNNING="running", DONE="done"),
    )
    monkeypatch.setattr(bluesky_scanner, "ScanLifecycleEvent", _FakeLifecycleEvent)

    scanner._set_state("RUNNING")
    scanner._set_state("DONE")

    assert [e.scan_number for e in events] == [41, 41]


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


# ---------------------------------------------------------------------------
# Strict shot-control validation
# ---------------------------------------------------------------------------


def test_strict_plan_requires_a_shot_controller() -> None:
    """Building a strict plan without shot control fails loudly."""
    with pytest.raises(GeecsConfigurationError, match="reachable shot-control"):
        build_step_scan_plan(
            strict=True,
            motor=None,
            positions=[None],
            reference=None,
            detectors=[object()],
            shots_per_step=1,
            controller=None,
            experiment="Test",
            scan_number=None,
            scan_folder=None,
            saving_detectors=[],
        )


def test_strict_single_shot_requires_nonempty_armed_state() -> None:
    config = ShotControlConfig.from_information(
        {
            "device": "U_DG645_ShotControl",
            "variables": {
                "Trigger.Source": {
                    "SCAN": "External rising edges",
                    "ARMED": "",
                }
            },
        }
    )
    controller = ShotController(config, {"Trigger.Source": object()})
    with pytest.raises(GeecsConfigurationError, match="non-empty ARMED"):
        controller.require_strict_single_shot()


def test_strict_single_shot_requires_nonempty_singleshot_state() -> None:
    """All-empty SINGLESHOT entries would make fire_shot a silent no-op."""
    config = ShotControlConfig.from_information(
        {
            "device": "U_DG645_ShotControl",
            "variables": {
                "Trigger.Source": {
                    "ARMED": "Single shot external rising edges",
                    "SINGLESHOT": "",
                },
                "Trigger.ExecuteSingleShot": {"SINGLESHOT": ""},
            },
        }
    )
    controller = ShotController(config, {"Trigger.Source": object()})
    with pytest.raises(GeecsConfigurationError, match="non-empty SINGLESHOT"):
        controller.require_strict_single_shot()


def test_strict_single_shot_requires_reachable_setters() -> None:
    config = ShotControlConfig.from_information(
        {
            "device": "U_DG645_ShotControl",
            "variables": {
                "Trigger.Source": {
                    "ARMED": "Single shot external rising edges",
                },
                "Trigger.ExecuteSingleShot": {"SINGLESHOT": "on"},
            },
        }
    )
    controller = ShotController(config, {})
    with pytest.raises(GeecsConfigurationError, match="reachable shot-control"):
        controller.require_strict_single_shot()


def test_strict_single_shot_accepts_armed_state_and_setters() -> None:
    config = ShotControlConfig.from_information(
        {
            "device": "U_DG645_ShotControl",
            "variables": {
                "Trigger.Source": {
                    "ARMED": "Single shot external rising edges",
                },
                "Trigger.ExecuteSingleShot": {"SINGLESHOT": "on"},
            },
        }
    )
    ShotController(config, {"Trigger.Source": object()}).require_strict_single_shot()
