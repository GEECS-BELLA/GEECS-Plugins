"""The interlock windows, offscreen: what they show for each frame and for silence."""

from __future__ import annotations

import json
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)  # also: no libEGL

from geecs_bluesky.interlocks.detail import detail_document
from geecs_bluesky.interlocks.gui.client import NO_SIGNAL_S, ClientWindow, judge
from geecs_bluesky.interlocks.gui.server import ServerWindow
from geecs_bluesky.interlocks.install import InterlockStatus
from geecs_bluesky.interlocks.status_server import InterlockStatusServer
from geecs_bluesky.interlocks.wire import (
    ALL_SAFE,
    STATE_NO_SIGNAL,
    TRIPPED,
    parse_status,
)
from PyQt6.QtWidgets import QApplication

from tests.test_interlock_detail import _RE, _registry


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _pump(app: QApplication, until, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if until():
            return True
        time.sleep(0.01)
    return False


def test_judge_never_reads_silence_as_safe() -> None:
    safe = parse_status("SAFE | OK: a | OK: b")
    assert judge(safe, 0.1).kind == ALL_SAFE
    assert judge(safe, NO_SIGNAL_S + 0.1).kind == STATE_NO_SIGNAL
    assert judge(safe, None).kind == STATE_NO_SIGNAL
    assert judge(None, 0.0).kind == STATE_NO_SIGNAL
    tripped = judge(parse_status("WARNING! TRIPPED | TRIPPED: a - x | OK: b"), 0.1)
    assert tripped.kind == TRIPPED and tripped.lines == ("TRIPPED a",)
    assert tripped.names == ("a",)


def test_the_client_shows_each_frame_then_no_signal(app: QApplication) -> None:
    window = ClientWindow("localhost", 1)
    window.sound.setChecked(False)
    try:
        window.connected_at = time.monotonic()
        window.on_message("SAFE | OK: vacuum | OK: cam")
        assert window.panel.kind.text() == ALL_SAFE
        assert window.table.rowCount() == 2
        window.on_message(
            "WARNING! TRIPPED | TRIPPED: vacuum - Pressure=0.001 | OK: cam"
        )
        assert window.panel.kind.text() == TRIPPED
        assert window.table.item(0, 2).text() == "Pressure=0.001"
        assert "vacuum" in window.compact.lines.text()
        assert window.history[-1][1:] == (TRIPPED, "vacuum")
        window.last_frame_at = time.monotonic() - NO_SIGNAL_S - 0.5
        window.tick()
        assert window.panel.kind.text() == STATE_NO_SIGNAL
        assert not window.table.isEnabled()
        assert [what for _, what, _ in window.history] == [
            STATE_NO_SIGNAL,
            ALL_SAFE,
            TRIPPED,
            STATE_NO_SIGNAL,
        ]
    finally:
        window.close()


def test_the_client_reads_a_live_status_stream(app: QApplication) -> None:
    statuses = [InterlockStatus("vacuum", unsafe=False)]
    server = InterlockStatusServer(
        lambda: list(statuses), "127.0.0.1", 0, interval=0.02
    )
    server.start()
    window = ClientWindow("127.0.0.1", server.address[1])
    window.sound.setChecked(False)
    try:
        window.connect_stream()
        assert _pump(app, lambda: window.panel.kind.text() == ALL_SAFE)
        statuses[0] = InterlockStatus("vacuum", True, "Pressure=0.001")
        assert _pump(app, lambda: window.panel.kind.text() == TRIPPED)
        window.disconnect_stream()
        window.tick()
        assert window.panel.kind.text() == STATE_NO_SIGNAL
    finally:
        window.close()
        server.stop()


def test_the_server_window_shows_the_detail_document(app: QApplication) -> None:
    window = ServerWindow("localhost", 1)
    try:
        doc = detail_document(_registry(), run_engine=_RE(), opened_at=50.0)
        window.on_message(json.dumps(doc))
        window.tick()
        assert window.panel.kind.text() == TRIPPED
        assert window.table.rowCount() == 3
        assert window.table.item(0, 1).text() == "cam"
        assert window.table.item(0, 3).text() == "4095"
        assert window.re_state.text() == "PAUSED"
        assert window.re_suspended_by.text() == "cam"
        assert window.detail_box.title() == "Selected: cam"
        assert window.trend.points and window.trend.limit == 4094
        assert "TRIPPED" in window.events.toPlainText()
        window.last_frame_at = time.monotonic() - 10
        window.tick()
        assert window.panel.kind.text() == STATE_NO_SIGNAL
    finally:
        window.close()


def test_the_server_window_refuses_the_status_port(app: QApplication) -> None:
    window = ServerWindow("localhost", 1)
    try:
        window.on_message("SAFE | OK: a")
        assert window.document is None
        assert "not JSON" in window.stream_text.text()
    finally:
        window.close()
