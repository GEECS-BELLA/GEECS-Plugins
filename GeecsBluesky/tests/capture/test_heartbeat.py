"""Heartbeat module + the engine's fail-closed toggle-off liveness preflight."""

from __future__ import annotations

import json
import time

import pytest

from geecs_bluesky.capture.heartbeat import (
    STALE_AFTER_S,
    daemon_looks_alive,
    heartbeat_age,
    write_heartbeat,
)
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.scan_request_runner import preflight_capture_liveness


def test_write_and_read_roundtrip(tmp_path) -> None:
    """A fresh heartbeat reads back young and alive."""
    hb = tmp_path / "state" / "heartbeat.json"
    write_heartbeat(3, path=hb)
    payload = json.loads(hb.read_text())
    assert payload["targets"] == 3
    age = heartbeat_age(path=hb)
    assert age is not None and age < 5.0
    assert daemon_looks_alive(path=hb)


def test_missing_and_stale_and_garbage(tmp_path) -> None:
    """Absent, stale, and corrupt heartbeats all read as not-alive."""
    hb = tmp_path / "heartbeat.json"
    assert heartbeat_age(path=hb) is None
    assert not daemon_looks_alive(path=hb)

    hb.write_text(json.dumps({"time": time.time() - STALE_AFTER_S - 5}))
    assert not daemon_looks_alive(path=hb)

    hb.write_text("not json")
    assert heartbeat_age(path=hb) is None


def test_preflight_refuses_toggle_off_without_daemon(monkeypatch) -> None:
    """Toggle-off + no heartbeat = refusal naming the orphaned devices."""
    import geecs_bluesky.capture.heartbeat as hb_mod

    monkeypatch.setattr(hb_mod, "daemon_looks_alive", lambda **kw: False)
    monkeypatch.setattr(hb_mod, "heartbeat_age", lambda **kw: None)
    with pytest.raises(GeecsConfigurationError, match="UC_Cam.*NOWHERE"):
        preflight_capture_liveness(["UC_Cam"], native_image_save=False)


def test_preflight_passes_with_live_daemon(monkeypatch) -> None:
    """Fresh heartbeat: toggle-off proceeds."""
    import geecs_bluesky.capture.heartbeat as hb_mod

    monkeypatch.setattr(hb_mod, "daemon_looks_alive", lambda **kw: True)
    preflight_capture_liveness(["UC_Cam"], native_image_save=False)


def test_preflight_inert_on_default_path() -> None:
    """Toggle on (or no capture devices): never consulted, never refuses."""
    preflight_capture_liveness(["UC_Cam"], native_image_save=True)
    preflight_capture_liveness([], native_image_save=False)
