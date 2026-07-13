"""Pins for the two scan QoL fixes (0.33.0).

1. ``is_scanning_active()`` must already be ``False`` when the terminal
   DONE/ABORTED lifecycle event is delivered — an event-driven GUI that
   re-checks it from its terminal-state handler must never race the scan
   thread's last instructions (observed live: the console's Start button
   stayed disabled after every scan until the operator clicked Stop).
2. Telemetry devices connect **concurrently** (``telemetry_batch``) — the
   sequential per-device connects cost ~9 s of start-to-execution latency
   at ~87 devices (measured live 2026-07-13).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import geecs_bluesky.session as session_module
from geecs_bluesky.scan_request_runner import build_telemetry_readables
from geecs_bluesky.scanner_bridge.bluesky_scanner import BlueskyScanner


class TestDoneImpliesInactive:
    """Terminal lifecycle events must observe an inactive scanner."""

    def _bare_scanner(self, events: list) -> BlueskyScanner:
        scanner = BlueskyScanner.__new__(BlueskyScanner)
        scanner._on_event = lambda ev: events.append(
            (getattr(ev, "state", None), scanner.is_scanning_active())
        )
        scanner._current_state = None
        scanner._scan_number = None
        scanner._scan_thread = None
        scanner._scan_request = None
        scanner._total_shots = 0
        scanner._disconnect_devices_sync = lambda: None  # type: ignore[method-assign]
        return scanner

    def test_done_event_observes_inactive_scanner(self) -> None:
        events: list[tuple] = []
        scanner = self._bare_scanner(events)
        # An unsupported mode runs the thread body straight to the finally
        # block — the shortest real path through _run_scan's cleanup.
        scanner._scan_config = SimpleNamespace(scan_mode="unsupported-mode")

        scanner.start_scan_thread()
        scanner._scan_thread.join(timeout=5.0)
        assert not scanner._scan_thread.is_alive()

        done = [
            active
            for state, active in events
            if str(getattr(state, "value", state)).lower() == "done"
        ]
        assert done == [False], (
            "the DONE lifecycle event was emitted while is_scanning_active() "
            "still reported True — the GUI Start/Stop race is back"
        )

    def test_aborted_event_observes_inactive_scanner(self) -> None:
        events: list[tuple] = []
        scanner = self._bare_scanner(events)
        scanner._scan_config = SimpleNamespace(scan_mode="unsupported-mode")

        scanner.start_scan_thread()
        scanner._abort_requested = True  # simulate a stop arriving mid-run
        scanner._scan_thread.join(timeout=5.0)

        terminal = [
            (str(getattr(state, "value", state)).lower(), active)
            for state, active in events
            if str(getattr(state, "value", state)).lower() in ("done", "aborted")
        ]
        assert terminal and all(active is False for _, active in terminal)

    def test_restart_resets_the_finished_flag(self) -> None:
        events: list[tuple] = []
        scanner = self._bare_scanner(events)
        scanner._scan_config = SimpleNamespace(scan_mode="unsupported-mode")
        scanner.start_scan_thread()
        scanner._scan_thread.join(timeout=5.0)
        assert scanner.is_scanning_active() is False

        # A second scan must report active again while its thread runs.
        import threading

        release = threading.Event()

        def blocking_run(_config):
            release.wait(timeout=5.0)

        scanner._run_scan = blocking_run  # type: ignore[method-assign]
        scanner.start_scan_thread()
        try:
            assert scanner.is_scanning_active() is True
        finally:
            release.set()
            scanner._scan_thread.join(timeout=5.0)


class TestTelemetryBatchConnect:
    """Telemetry connects run concurrently, drop-on-failure per device."""

    def test_batch_connects_all_and_drops_failures(self, monkeypatch) -> None:
        pytest.importorskip("aioca")
        real = session_module.CaTelemetryReadable

        class Flaky(real):  # type: ignore[misc,valid-type]
            async def connect(self, **kwargs):
                if self._geecs_device_name == "bad":
                    raise RuntimeError("dead at scan start")
                return await super().connect(**kwargs)

        monkeypatch.setattr(session_module, "CaTelemetryReadable", Flaky)
        session = session_module.GeecsSession("Test", tiled=False, mock=True)
        connected = session.telemetry_batch({"a": ["X"], "bad": ["Y"], "c": ["Z"]})
        assert [d._geecs_device_name for d in connected] == ["a", "c"]

    def test_runner_uses_batch_and_records_only_connected(self) -> None:
        class BatchSession:
            def __init__(self) -> None:
                self.batch_calls: list[dict] = []

            def telemetry_batch(self, selected):
                self.batch_calls.append(dict(selected))
                return [
                    SimpleNamespace(_geecs_device_name=device)
                    for device in selected
                    if device != "dropped"
                ]

            def telemetry(self, *args, **kwargs):  # pragma: no cover
                raise AssertionError(
                    "sequential telemetry factory used despite batch support"
                )

        policy = SimpleNamespace(
            subscribed_by_device=lambda: {
                "d1": ["v1"],
                "dropped": ["v2"],
                "d3": ["v3"],
            }
        )
        session = BatchSession()
        readables, recorded = build_telemetry_readables(session, None, policy)
        assert session.batch_calls  # the batch path was taken
        assert [r._geecs_device_name for r in readables] == ["d1", "d3"]
        assert recorded == {"d1": ["v1"], "d3": ["v3"]}

    def test_runner_falls_back_without_batch(self) -> None:
        class LegacySession:
            def telemetry(self, device, variables, **kwargs):
                if device == "dead":
                    return None
                return SimpleNamespace(_geecs_device_name=device)

        policy = SimpleNamespace(
            subscribed_by_device=lambda: {"d1": ["v1"], "dead": ["v2"]}
        )
        readables, recorded = build_telemetry_readables(LegacySession(), None, policy)
        assert [r._geecs_device_name for r in readables] == ["d1"]
        assert recorded == {"d1": ["v1"]}
