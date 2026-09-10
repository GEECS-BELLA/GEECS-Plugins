"""Heartbeat module (the capture daemon's liveness file)."""

from __future__ import annotations

import json
import time

import pytest

from geecs_bluesky.capture.heartbeat import (
    STALE_AFTER_S,
    clear_heartbeat,
    daemon_looks_alive,
    heartbeat_age,
    read_heartbeat,
    write_heartbeat,
)


def test_write_and_read_roundtrip(tmp_path) -> None:
    """A fresh heartbeat reads back young, alive, and carrying the roster."""
    hb = tmp_path / "state" / "heartbeat.json"
    write_heartbeat(["UC_CamB", "UC_CamA"], path=hb)
    payload = json.loads(hb.read_text())
    assert payload["targets"] == ["UC_CamA", "UC_CamB"]
    age = heartbeat_age(path=hb)
    assert age is not None and age < 5.0
    assert daemon_looks_alive(path=hb)


def test_clear_heartbeat_makes_daemon_look_down(tmp_path) -> None:
    """The clean-stop tombstone: removal reads as not-alive immediately."""
    hb = tmp_path / "heartbeat.json"
    write_heartbeat(["UC_Cam"], path=hb)
    assert daemon_looks_alive(path=hb)
    clear_heartbeat(path=hb)
    assert not hb.exists()
    assert not daemon_looks_alive(path=hb)
    clear_heartbeat(path=hb)  # idempotent on an already-missing file


def test_missing_and_stale_and_garbage(tmp_path) -> None:
    """Absent, stale, and corrupt heartbeats all read as not-alive."""
    hb = tmp_path / "heartbeat.json"
    assert heartbeat_age(path=hb) is None
    assert not daemon_looks_alive(path=hb)

    hb.write_text(json.dumps({"time": time.time() - STALE_AFTER_S - 5}))
    assert not daemon_looks_alive(path=hb)

    hb.write_text("not json")
    assert heartbeat_age(path=hb) is None
    assert read_heartbeat(path=hb) is None


def _point_preflight_at(monkeypatch, payload: dict | None) -> None:
    import geecs_bluesky.capture.heartbeat as hb_mod

    monkeypatch.setattr(hb_mod, "read_heartbeat", lambda **kw: payload)


def test_sigterm_runs_finally_cleanup(tmp_path) -> None:
    """SIGTERM must reach the finally block — the systemctl-stop tombstone.

    Python's default SIGTERM action kills the process WITHOUT running
    ``finally`` (codex gate P1); the daemon installs a handler converting
    it to ``SystemExit``. Simulate the daemon's structure: install, block,
    receive a real SIGTERM, verify the cleanup ran.
    """
    import os
    import signal

    from geecs_bluesky.capture.__main__ import _install_sigterm_handler

    previous = signal.getsignal(signal.SIGTERM)
    hb = tmp_path / "heartbeat.json"
    write_heartbeat(["UC_Cam"], path=hb)
    try:
        _install_sigterm_handler()
        with pytest.raises(SystemExit):
            try:
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(5)  # interrupted by the handler's SystemExit
            finally:
                clear_heartbeat(path=hb)
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert not hb.exists()
