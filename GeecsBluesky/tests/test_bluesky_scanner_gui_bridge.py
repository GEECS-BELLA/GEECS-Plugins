"""Tests for Scanner GUI compatibility helpers in BlueskyScanner."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from geecs_bluesky.scanner_bridge import bluesky_scanner
from geecs_bluesky.scanner_bridge.bluesky_scanner import BlueskyScanner


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


def test_resolve_acquisition_mode_unknown_falls_back_to_strict() -> None:
    options = SimpleNamespace(acquisition_mode="nonsense")
    mode = BlueskyScanner._resolve_acquisition_mode(options, env={})
    assert mode == "strict_shot_control"


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
