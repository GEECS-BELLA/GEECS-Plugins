"""Hermetic tests for the scan monitor controller (app/scan_monitor.py).

No sockets: the stream workers' parsing and lifecycle are tested without
their daemon threads (the parse helpers are called directly; ``stop()``
gates emission), and the controller runs against the stub queue client.
"""

from __future__ import annotations

from geecs_console.app.scan_monitor import (
    ConsoleStreamWorker,
    DocumentStreamWorker,
    ScanMonitorController,
    _FAILED_MOVE_PREFIX_FALLBACK,
    _failed_move_prefix,
)
from geecs_console.services.queue_client import StubQueueClient


class TestFailedMovePrefix:
    def test_matches_the_engine_constant_when_importable(self):
        # The fallback exists for hermetic installs; when the engine is
        # importable (it is, in this env) the two must agree — this is the
        # keep-in-sync pin.
        try:
            from geecs_bluesky.plans.pause_semantics import FAILED_MOVE_LOG_PREFIX
        except ImportError:
            return
        assert _failed_move_prefix() == FAILED_MOVE_LOG_PREFIX
        assert _FAILED_MOVE_PREFIX_FALLBACK == FAILED_MOVE_LOG_PREFIX


class TestConsoleStreamWorker:
    def test_lines_and_pause_reason_extraction(self, qtbot):
        worker = ConsoleStreamWorker("tcp://x:1")
        lines: list[str] = []
        reasons: list[str] = []
        worker.line.connect(lines.append)
        worker.pause_reason.connect(reasons.append)
        prefix = _failed_move_prefix()
        worker._handle_text(
            f"some noise\n{prefix}: commanded u_s1h -> 1.05, one axis "
            "failed - see cause\n\ntrailing",
            prefix,
        )
        assert lines[0] == "some noise"
        assert len(lines) == 3  # blank line dropped
        assert len(reasons) == 1
        assert reasons[0].startswith("commanded u_s1h")

    def test_stop_gates_emission(self, qtbot):
        worker = ConsoleStreamWorker("tcp://x:1")
        seen: list[str] = []
        worker.line.connect(seen.append)
        worker.stop()
        worker._handle_text("a line", "PREFIX")
        assert seen == []


class TestDocumentStreamWorker:
    def test_stop_is_idempotent_without_a_thread(self, qtbot):
        worker = DocumentStreamWorker("x:5568")
        worker.stop()
        worker.stop()
        assert worker._stopped


class TestScanMonitorController:
    def test_stub_client_disables_the_streams(self, qtbot):
        controller = ScanMonitorController(StubQueueClient())
        assert controller.documents is None
        assert controller.console is None
        controller.dispose()

    def test_poller_delivers_the_stub_status(self, qtbot):
        controller = ScanMonitorController(StubQueueClient())
        reports: list = []
        controller.status_ready.connect(reports.append)
        controller._poller.poll_async()
        qtbot.waitUntil(lambda: bool(reports), timeout=2000)
        assert not reports[0].connected
        controller.dispose()

    def test_start_and_dispose_are_idempotent(self, qtbot):
        from PySide6.QtWidgets import QWidget

        parent = QWidget()
        qtbot.addWidget(parent)
        controller = ScanMonitorController(
            StubQueueClient(), info_addr=None, doc_addr=None
        )
        controller.start(parent)
        controller.start(parent)  # second start is a no-op
        controller.dispose()
        controller.dispose()
        # After dispose, start must not resurrect the timer.
        controller.start(parent)
        assert controller._timer is None
