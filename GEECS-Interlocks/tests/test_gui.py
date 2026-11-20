"""The server window, offscreen: both tabs draw, the builder edits, saves and reloads."""

import logging
import sys

import pytest
import yaml
from conftest import FakeSource, threshold

from geecs_interlocks.catalog import StaticCatalog
from geecs_interlocks.config import Settings
from geecs_interlocks.engine import Engine
from geecs_interlocks.logs import LogBuffer

QtCore = pytest.importorskip("PyQt6.QtCore")
QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
Qt = QtCore.Qt

pytestmark = pytest.mark.gui

CAM = {
    "kind": "threshold",
    "device": "Cam",
    "variable": "Max",
    "operator": ">=",
    "limit": 4000,
}
DRIFT = {
    "kind": "alignment",
    "device": "Cam",
    "variable": "X",
    "target_variable": "X0",
    "tolerance": 5,
}


@pytest.fixture(scope="module")
def app():
    from geecs_interlocks.gui import log_uncaught

    sys.excepthook = log_uncaught  # as run_window does
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app, tmp_path):
    from geecs_interlocks.gui.window import ServerWindow

    path = tmp_path / "interlocks" / "main.yaml"
    path.parent.mkdir()
    # The fake source stamps its own clock, not the monitor's: no implicit staleness here.
    doc = {
        "schema_version": 1,
        "stale_after_s": None,
        "interlocks": [threshold(), {"name": "cam", "conditions": [CAM, DRIFT]}],
    }
    path.write_text(yaml.safe_dump(doc))
    sources: list[FakeSource] = []

    def make(_doc):
        sources.append(FakeSource())
        sources[-1].push("Gauge", P=1.0)
        sources[-1].push("Cam", Max=4095, X=1, X0=0)
        return sources[-1]

    engine = Engine(path, make, Settings(status_port=0))
    engine.start()
    buffer = LogBuffer()
    logging.getLogger().addHandler(buffer)
    catalog = StaticCatalog({"Gauge": ["P", "T"], "Cam": ["Max", "X", "X0", "Mean"]})
    win = ServerWindow(engine, buffer, catalog, None, experiment="Demo")
    win.engine_sources = sources
    yield win
    win.close()
    logging.getLogger().removeHandler(buffer)
    engine.stop()


def settle(app, window):
    engine = window.engine
    engine.monitor.step()
    window.monitor.refresh()
    app.processEvents()


class TestMonitorTab:
    def test_rows_worst_first_with_the_group_expanded_into_checks(self, app, window):
        settle(app, window)
        table = window.monitor.table
        first = table.topLevelItem(0)
        assert (first.text(0), first.text(1)) == ("tripped", "cam")
        assert first.childCount() == 2
        assert first.child(0).text(3).startswith("Max=4095")
        assert first.child(1).text(3) == "ok"
        assert window.monitor.headline.text() == "Tripped"

    def test_the_log_pane_shows_new_records(self, app, window):
        logging.getLogger("geecs_interlocks.test").warning("a line for the pane")
        app.processEvents()
        assert "a line for the pane" in window.monitor.log.text.toPlainText()

    def test_the_level_filter(self, app, window):
        logging.getLogger("geecs_interlocks.test").info("quiet line")
        app.processEvents()
        window.monitor.log.level.setCurrentIndex(1)  # warnings only
        assert "quiet line" not in window.monitor.log.text.toPlainText()

    def test_always_on_top(self, app, window):
        window.monitor.on_top.setChecked(True)
        assert window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
        window.monitor.on_top.setChecked(False)

    def test_arm_with_nothing_to_arm_says_why(self, app, window):
        notes = []
        window.monitor._notify = notes.append
        window.monitor.arm_selected()
        assert notes and notes[0].startswith("Not armed")


def catalog_loaded(app, builder):
    """Wait for the worker thread's catalog to reach the tab."""
    import time

    deadline = time.time() + 3
    while (
        "in the experiment" not in builder.devices_note.text()
        and time.time() < deadline
    ):
        app.processEvents()
        time.sleep(0.01)
    assert "in the experiment" in builder.devices_note.text()


class TestBuilderTab:
    @pytest.fixture(autouse=True)
    def _loaded(self, app, window):
        catalog_loaded(app, window.builder)

    def pick(self, tree, text):
        for i in range(tree.topLevelItemCount()):
            if tree.topLevelItem(i).text(0) == text:
                tree.setCurrentItem(tree.topLevelItem(i))
                return
        raise AssertionError(f"{text!r} not listed")

    def test_devices_and_watched_markers(self, app, window):
        builder = window.builder
        app.processEvents()
        rows = {
            builder.device_list.topLevelItem(i).text(
                0
            ): builder.device_list.topLevelItem(i).text(1)
            for i in range(builder.device_list.topLevelItemCount())
        }
        assert rows["Cam"] == "watched" and rows["Gauge"] == "watched"

    def test_add_a_check_to_a_group_then_save_and_reload(self, app, window):
        builder = window.builder
        app.processEvents()
        self.pick(builder.device_list, "Cam")
        self.pick(builder.variable_list, "Mean")
        builder.operator.setCurrentText(">")
        builder.limit.setText("500")
        builder.add_to.setCurrentText("cam")
        builder.add_check()
        assert builder.draft.dirty
        assert "unsaved changes" in window.windowTitle()
        builder._write(window.engine.path)
        saved = yaml.safe_load(window.engine.path.read_text())
        cam = next(i for i in saved["interlocks"] if i["name"] == "cam")
        assert cam["conditions"][-1]["variable"] == "Mean"
        assert len(window.engine.interlocks.interlocks[1].conditions) == 3
        assert "unsaved changes" not in window.windowTitle()

    def test_an_incomplete_check_is_not_added(self, app, window):
        builder = window.builder
        notes = []
        builder._notify = notes.append
        self.pick(builder.device_list, "Cam")
        self.pick(builder.variable_list, "Mean")
        builder.limit.setText("")
        builder.add_check()
        assert not builder.draft.dirty and notes[0].startswith("Not added")

    def test_selecting_a_check_loads_it_for_editing(self, app, window):
        builder = window.builder
        cam = next(
            builder.tree.topLevelItem(i)
            for i in range(builder.tree.topLevelItemCount())
            if builder.tree.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole) == "cam"
        )
        builder.tree.setCurrentItem(cam.child(1))
        assert builder.kind.currentText() == "alignment"
        assert builder.tolerance.text() == "5"
        builder.tolerance.setText("8")
        builder.update_check()
        assert builder.draft.check("cam", 1)["tolerance"] == 8

    def test_regroup_through_a_drop(self, app, window):
        builder = window.builder
        builder.tree.moved.emit("cam", 0, "vac")
        assert len(builder.draft.check("vac", 1)) and builder.draft.names() == [
            "vac",
            "cam",
        ]


def test_an_error_in_a_slot_is_logged_not_fatal(app, window, caplog):
    def broken():
        raise RuntimeError("drawing bug")

    QtCore.QTimer.singleShot(0, broken)
    with caplog.at_level(logging.ERROR):
        app.processEvents()
    assert "the server keeps running" in caplog.text
    assert window.engine.snapshot() is not None
