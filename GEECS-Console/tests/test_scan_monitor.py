"""Hermetic tests for the scan monitor controller (app/scan_monitor.py).

No sockets: the stream workers' parsing and lifecycle are tested without
their daemon threads (the parse helpers are called directly, or a payload
is posted from a plain thread), and the controller runs against the stub
queue client.  Since 0.30.0 (#787) every stream signal is emitted on the
GUI thread through the ``_GuiHopWorker`` hop — the tests pin where the
slot runs, that ``stop()`` gates a payload already in flight, and that
``dispose()`` severs the public signals without cutting the hop.
"""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Slot

from geecs_console.app.scan_monitor import (
    ConsoleStreamWorker,
    DocumentStreamWorker,
    ScanMonitorController,
    _FAILED_MOVE_PREFIX_FALLBACK,
    _failed_move_prefix,
)
from geecs_console.services import background
from geecs_bluesky.qs_client import StubQueueClient


class Sink(QObject):
    """Records payloads and the thread each slot ran on."""

    def __init__(self):
        super().__init__()
        self.items: list = []
        self.threads: list[int] = []

    @Slot(str)
    def take_line(self, text):
        self.items.append(text)
        self.threads.append(threading.get_ident())

    @Slot(str, object)
    def take_doc(self, name, doc):
        self.items.append((name, doc))
        self.threads.append(threading.get_ident())


def _settled(qtbot, worker):
    qtbot.waitUntil(lambda: worker not in background._INFLIGHT, timeout=3000)


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
        qtbot.waitUntil(lambda: len(lines) == 3, timeout=3000)  # blank line dropped
        assert lines[0] == "some noise"
        assert len(reasons) == 1
        assert reasons[0].startswith("commanded u_s1h")

    def test_lines_are_delivered_on_the_gui_thread(self, qtbot):
        worker = ConsoleStreamWorker("tcp://x:1")
        sink = Sink()
        worker.line.connect(sink.take_line)
        thread = threading.Thread(
            target=worker._handle_text, args=("from the zmq thread", "PREFIX")
        )
        thread.start()
        thread.join()
        qtbot.waitUntil(lambda: sink.items == ["from the zmq thread"], timeout=3000)
        assert sink.threads == [threading.get_ident()]
        _settled(qtbot, worker)

    def test_stop_gates_emission(self, qtbot):
        worker = ConsoleStreamWorker("tcp://x:1")
        seen: list[str] = []
        worker.line.connect(seen.append)
        worker.stop()
        worker._handle_text("a line", "PREFIX")
        qtbot.wait(50)
        assert seen == []
        assert worker not in background._INFLIGHT  # nothing was even posted

    def test_stop_drops_a_payload_already_in_flight(self, qtbot):
        # Posted before stop(), landing after it: the GUI-side gate wins,
        # so dispose() means "nothing more reaches the window" — not
        # "nothing more is *posted*".
        worker = ConsoleStreamWorker("tcp://x:1")
        seen: list[str] = []
        worker.line.connect(seen.append)
        worker._handle_text("in flight", "PREFIX")
        worker.stop()
        _settled(qtbot, worker)
        assert seen == []

    def test_sever_detaches_consumers_but_keeps_the_hop(self, qtbot):
        worker = ConsoleStreamWorker("tcp://x:1")
        seen: list[str] = []
        worker.line.connect(seen.append)
        worker.sever()
        worker._handle_text("after sever", "PREFIX")
        # The hop still lands (hold released) — a whole-object disconnect
        # would have cut _landed → _forward and leaked the hold forever.
        _settled(qtbot, worker)
        assert seen == []


class TestDocumentStreamWorker:
    def test_stop_is_idempotent_without_a_thread(self, qtbot):
        worker = DocumentStreamWorker("x:5568")
        worker.stop()
        worker.stop()
        assert worker._stopped

    def test_documents_are_delivered_on_the_gui_thread(self, qtbot):
        worker = DocumentStreamWorker("x:5568")
        sink = Sink()
        worker.document.connect(sink.take_doc)
        thread = threading.Thread(
            target=worker._post, args=(("document", "start", {"scan_number": 7}),)
        )
        thread.start()
        thread.join()
        qtbot.waitUntil(lambda: bool(sink.items), timeout=3000)
        assert sink.items == [("start", {"scan_number": 7})]
        assert sink.threads == [threading.get_ident()]
        _settled(qtbot, worker)

    def test_setup_failure_reaches_stream_failed_on_the_gui_thread(self, qtbot):
        worker = DocumentStreamWorker("x:5568")
        failures: list[str] = []
        worker.stream_failed.connect(failures.append)
        threading.Thread(target=worker._fail, args=("boom",)).start()
        qtbot.waitUntil(lambda: failures == ["boom"], timeout=3000)
        _settled(qtbot, worker)


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

    def test_dispose_on_a_never_connected_poller_is_silent(self, qtbot, recwarn):
        controller = ScanMonitorController(StubQueueClient())
        controller.dispose()
        assert not [w for w in recwarn if "Failed to disconnect" in str(w.message)]

    def test_dispose_severs_stream_consumers_without_cutting_the_hop(self, qtbot):
        # Streams exist (addresses given) but their threads are never
        # started — dispose() must sever the window-facing signals and
        # leave each worker able to land a straggling payload.
        controller = ScanMonitorController(
            StubQueueClient(), info_addr="tcp://x:1", doc_addr="tcp://x:2"
        )
        seen: list = []
        controller.console.line.connect(seen.append)
        controller.documents.document.connect(lambda n, d: seen.append(n))
        controller.dispose()
        controller.console._handle_text("late", "PREFIX")
        controller.documents._post(("document", "late", {}))
        _settled(qtbot, controller.console)
        _settled(qtbot, controller.documents)
        assert seen == []
