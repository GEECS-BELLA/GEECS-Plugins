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


def test_filter_available_variables_skips_missing_entries(caplog) -> None:
    """Unavailable save variables should be logged and removed before readout."""

    class FakeDb:
        """Minimal DB facade returning available variables."""

        @staticmethod
        def get_device_variables(_device_name: str) -> list[dict]:
            """Return one valid variable."""
            return [{"name": "valid"}, {"name": "also_valid"}]

    with caplog.at_level("WARNING"):
        valid = BlueskyScanner._filter_available_variables(
            "U_Test",
            ["valid", "missing", "also_valid"],
            FakeDb,
        )

    assert valid == ["valid", "also_valid"]
    assert "Ignoring unavailable save variables for U_Test" in caplog.text
    assert "missing" in caplog.text


def test_filter_available_variables_keeps_config_when_db_validation_fails(
    caplog,
) -> None:
    """DB validation failures should not block detector construction."""

    class BrokenDb:
        """Minimal DB facade that raises during validation."""

        @staticmethod
        def get_device_variables(_device_name: str) -> list[dict]:
            """Raise like an unavailable DB lookup."""
            raise RuntimeError("db offline")

    configured = ["a", "b"]
    with caplog.at_level("WARNING"):
        valid = BlueskyScanner._filter_available_variables(
            "U_Test",
            configured,
            BrokenDb,
        )

    assert valid == configured
    assert "Could not validate save variables for U_Test" in caplog.text
