"""Behavioral tests for the ScanBrowserWindow (hermetic, offscreen).

Everything runs against the FakeCatalog — no Tiled, no network, no real
data root.  The open-folder handler gets an injected resolver so
``geecs_data_utils`` is never imported here.
"""

from __future__ import annotations

import time

import pytest
from PySide6.QtCore import Qt

from geecs_console.browser.browser_window import ScanBrowserWindow
from geecs_console.services.settings import ConsoleSettings
from fake_catalog import TEST_DAY, FakeCatalog, make_detail


def _make_window(qtbot, catalog, **kwargs):
    window = ScanBrowserWindow(
        catalog=catalog,
        settings=ConsoleSettings(),
        today=TEST_DAY,
        scan_folder_resolver=kwargs.pop("scan_folder_resolver", lambda d, day: None),
        **kwargs,
    )
    qtbot.addWidget(window)
    return window


def _two_run_catalog(**kwargs):
    return FakeCatalog(
        [
            make_detail(1, hour=9, minute=11, description="warmup"),
            make_detail(2, hour=9, minute=27, description="alignment"),
            make_detail(3, hour=17, minute=42, exit_status="abort", description="died"),
        ],
        **kwargs,
    )


def _wait_runs(qtbot, window, count):
    qtbot.waitUntil(lambda: window.b2_run_list.count() == count, timeout=3000)


def _select_run(qtbot, window, row):
    window.b2_run_list.setCurrentRow(row)
    qtbot.waitUntil(lambda: window._detail is not None, timeout=3000)


class TestB1Session:
    def test_offline_stub_default_opens_empty(self, qtbot):
        window = ScanBrowserWindow(settings=ConsoleSettings(), today=TEST_DAY)
        qtbot.addWidget(window)
        window.show()
        qtbot.waitUntil(
            lambda: "not connected" in window.b1_connection_chip.text(),
            timeout=3000,
        )
        assert window.b2_run_list.count() == 0

    def test_probe_updates_connection_chip(self, qtbot):
        window = _make_window(qtbot, _two_run_catalog())
        qtbot.waitUntil(
            lambda: window.b1_connection_chip.text() == "tiled: fake:8000",
            timeout=3000,
        )

    def test_experiment_commit_is_remembered(self, qtbot):
        settings = ConsoleSettings()
        window = ScanBrowserWindow(
            catalog=_two_run_catalog(),
            settings=settings,
            today=TEST_DAY,
            scan_folder_resolver=lambda d, day: None,
        )
        qtbot.addWidget(window)
        window.b1_experiment_combo.setEditText("Undulator")
        window._on_experiment_committed()
        assert settings.last_experiment == "Undulator"

    def test_filter_hides_non_matching_rows(self, qtbot):
        window = _make_window(qtbot, _two_run_catalog())
        _wait_runs(qtbot, window, 3)
        window.b1_filter_edit.setText("alignment")
        hidden = [
            window.b2_run_list.item(i).isHidden()
            for i in range(window.b2_run_list.count())
        ]
        assert hidden.count(False) == 1
        window.b1_filter_edit.setText("")
        assert all(
            not window.b2_run_list.item(i).isHidden()
            for i in range(window.b2_run_list.count())
        )


class TestB2RunList:
    def test_day_listing_newest_first(self, qtbot):
        window = _make_window(qtbot, _two_run_catalog())
        _wait_runs(qtbot, window, 3)
        texts = [
            window.b2_run_list.item(i).text() for i in range(window.b2_run_list.count())
        ]
        assert "Scan 003" in texts[0]  # 17:42, newest
        assert "Scan 002" in texts[1]
        assert "Scan 001" in texts[2]
        assert "NOSCAN" in texts[0]

    def test_listing_is_metadata_only(self, qtbot):
        catalog = _two_run_catalog()
        window = _make_window(qtbot, catalog)
        _wait_runs(qtbot, window, 3)
        assert catalog.load_calls == []  # nothing loads until selected


class TestB3Identity:
    def test_selection_populates_identity_strip(self, qtbot):
        window = _make_window(qtbot, _two_run_catalog())
        _wait_runs(qtbot, window, 3)
        _select_run(qtbot, window, 1)  # Scan 002
        assert "Scan 002" in window.b3_title.text()
        assert "noscan" in window.b3_title.text()
        assert "10 shots" in window.b3_title.text()
        assert "74 s" in window.b3_title.text()
        pills = window.b3_pills.text()
        assert "success" in pills
        assert "Amp4In" in pills
        assert "uid-002"[:8] in pills
        assert window.b3_copy_uid_button.isEnabled()
        assert window.b3_open_folder_button.isEnabled()

    def test_copy_uid_puts_full_uid_on_clipboard(self, qtbot):
        from PySide6.QtGui import QGuiApplication

        window = _make_window(qtbot, _two_run_catalog())
        _wait_runs(qtbot, window, 3)
        _select_run(qtbot, window, 1)
        window._on_copy_uid()
        assert QGuiApplication.clipboard().text() == "uid-002"

    def test_open_folder_missing_reports_and_never_creates(self, qtbot, tmp_path):
        calls = []

        def resolver(detail, day):
            calls.append((detail.summary.uid, day))
            return None

        window = _make_window(qtbot, _two_run_catalog(), scan_folder_resolver=resolver)
        _wait_runs(qtbot, window, 3)
        _select_run(qtbot, window, 1)
        before = sorted(tmp_path.rglob("*"))
        window._on_open_scan_folder()
        assert calls == [("uid-002", TEST_DAY)]  # resolved for the SELECTED date
        assert sorted(tmp_path.rglob("*")) == before
        assert "not created" in window.statusBar().currentMessage()


class TestB4Plot:
    def test_add_and_remove_series(self, qtbot):
        window = _make_window(qtbot, _two_run_catalog())
        _wait_runs(qtbot, window, 3)
        _select_run(qtbot, window, 1)
        assert window.add_series("cam-counts") is True
        assert window.plotted_columns() == ["cam-counts"]
        assert len(window.b4_plot.getPlotItem().listDataItems()) == 1
        assert window.add_series("telemetry_mag-current") is True
        assert len(window.b4_plot.getPlotItem().listDataItems()) == 2
        assert window.b4_series_list.count() == 2
        # Legend/series list carries mean ± σ.
        assert "±" in window.b4_series_list.item(0).text()

        window.remove_series("cam-counts")
        assert window.plotted_columns() == ["telemetry_mag-current"]
        assert len(window.b4_plot.getPlotItem().listDataItems()) == 1

    def test_duplicate_and_unknown_columns_refused(self, qtbot):
        window = _make_window(qtbot, _two_run_catalog())
        _wait_runs(qtbot, window, 3)
        _select_run(qtbot, window, 1)
        assert window.add_series("cam-counts") is True
        assert window.add_series("cam-counts") is False
        assert window.add_series("no-such-column") is False
        assert "Unknown column" in window.statusBar().currentMessage()

    def test_non_numeric_column_guarded_with_status_message(self, qtbot):
        window = _make_window(qtbot, _two_run_catalog())
        _wait_runs(qtbot, window, 3)
        _select_run(qtbot, window, 1)
        assert window.add_series("telemetry_mode-label") is False
        assert "not numeric" in window.statusBar().currentMessage()
        assert window.plotted_columns() == []

    def test_y_line_edit_commit_adds_series(self, qtbot):
        window = _make_window(qtbot, _two_run_catalog())
        _wait_runs(qtbot, window, 3)
        _select_run(qtbot, window, 1)
        window.b4_y_edit.setText("cam-counts")
        window._on_y_committed()
        assert window.plotted_columns() == ["cam-counts"]
        assert window.b4_y_edit.text() == ""  # cleared after a successful add

    def test_stepped_scan_offers_scan_variable_x_with_error_bars(self, qtbot):
        catalog = FakeCatalog(
            [make_detail(5, motor="mono", num_points=4, shots_per_step=5)]
        )
        window = _make_window(qtbot, catalog)
        _wait_runs(qtbot, window, 1)
        _select_run(qtbot, window, 0)
        # Stepped scan: X defaults to the scan variable.
        assert window.b4_x_combo.currentText() == "mono"
        assert window.add_series("cam-counts") is True
        # Per-step rendering: error-bar item + mean curve.
        assert len(window._series["cam-counts"]) == 2
        # Switching X back to shot # replots as raw points.
        window.b4_x_combo.setCurrentText("shot #")
        assert len(window._series["cam-counts"]) == 1

    def test_new_selection_clears_series(self, qtbot):
        window = _make_window(qtbot, _two_run_catalog())
        _wait_runs(qtbot, window, 3)
        _select_run(qtbot, window, 1)
        window.add_series("cam-counts")
        detail_before = window._detail
        window.b2_run_list.setCurrentRow(0)
        qtbot.waitUntil(
            lambda: window._detail is not None and window._detail is not detail_before,
            timeout=3000,
        )
        assert window.plotted_columns() == []
        assert len(window.b4_plot.getPlotItem().listDataItems()) == 0


class TestB5Table:
    def test_table_shows_pinned_plus_plotted(self, qtbot):
        window = _make_window(qtbot, _two_run_catalog())
        _wait_runs(qtbot, window, 3)
        _select_run(qtbot, window, 1)
        assert window.visible_columns() == [
            "scan_event_index",
            "cam-acq_timestamp",
        ]
        window.add_series("cam-counts")
        assert window.visible_columns() == [
            "scan_event_index",
            "cam-acq_timestamp",
            "cam-counts",
        ]
        assert window.b5_table.columnCount() == 3
        assert window.b5_table.rowCount() == 10
        # Prettified header from geecs_scalar_headers.
        headers = [
            window.b5_table.horizontalHeaderItem(i).text()
            for i in range(window.b5_table.columnCount())
        ]
        assert "UC_Cam Counts" in headers
        assert "10 shots" in window.b5_footer.text()

    def test_export_csv_writes_visible_selection(self, qtbot, tmp_path):
        window = _make_window(qtbot, _two_run_catalog())
        _wait_runs(qtbot, window, 3)
        _select_run(qtbot, window, 1)
        window.add_series("cam-counts")
        target = tmp_path / "scan_002.csv"
        assert window.export_csv(target) is True
        text = target.read_text()
        header = text.splitlines()[0]
        assert header == "scan_event_index,cam-acq_timestamp,cam-counts"
        assert len(text.splitlines()) == 11  # header + 10 shots

    def test_export_dialog_path(self, qtbot, tmp_path, monkeypatch):
        from geecs_console.browser import browser_window as bw

        target = tmp_path / "picked.csv"
        monkeypatch.setattr(
            bw.QFileDialog,
            "getSaveFileName",
            staticmethod(lambda *a, **k: (str(target), "CSV files (*.csv)")),
        )
        window = _make_window(qtbot, _two_run_catalog())
        _wait_runs(qtbot, window, 3)
        _select_run(qtbot, window, 1)
        window._on_export_csv()
        assert target.exists()


class TestB6Drift:
    def test_drifting_telemetry_listed_steady_summarized(self, qtbot):
        window = _make_window(qtbot, _two_run_catalog())
        _wait_runs(qtbot, window, 3)
        _select_run(qtbot, window, 1)
        # telemetry_mag-current drifts; quiet is steady; label is string;
        # gap is all-NaN (not judged).
        items = [
            window.b6_drift_list.item(i) for i in range(window.b6_drift_list.count())
        ]
        columns = [i.data(Qt.ItemDataRole.UserRole) for i in items]
        assert columns == ["telemetry_mag-current"]
        assert "1 of 2 columns steady" in window.b6_summary.text()
        assert "3σ" in window.b6_summary.text()

    def test_click_adds_column_to_plot(self, qtbot):
        window = _make_window(qtbot, _two_run_catalog())
        _wait_runs(qtbot, window, 3)
        _select_run(qtbot, window, 1)
        item = window.b6_drift_list.item(0)
        window._on_drift_clicked(item)
        assert window.plotted_columns() == ["telemetry_mag-current"]


class TestTeardown:
    def test_close_during_slow_catalog_read_returns_fast(self, qtbot):
        catalog = _two_run_catalog(load_delay_s=0.5)
        window = _make_window(qtbot, catalog)
        window.show()
        _wait_runs(qtbot, window, 3)
        window.b2_run_list.setCurrentRow(1)  # kicks the slow load
        started = time.monotonic()
        window.close()
        elapsed = time.monotonic() - started
        assert elapsed < 0.4  # close never waits for the catalog
        # Let the straggler daemon finish and flush posted events — the
        # disconnected worker's late emission must go nowhere, no crash.
        qtbot.wait(700)

    def test_close_with_no_activity(self, qtbot):
        window = _make_window(qtbot, FakeCatalog([]))
        window.show()
        window.close()
        qtbot.wait(50)


class TestListErrorPath:
    def test_catalog_failure_surfaces_in_status_bar(self, qtbot):
        class BrokenCatalog:
            def probe(self):
                from geecs_console.browser.catalog import CatalogStatus

                return CatalogStatus(ok=False, label="tiled: down")

            def list_runs(self, experiment, day):
                raise RuntimeError("VPN fell over")

            def load_run(self, uid):
                raise RuntimeError("VPN fell over")

        window = _make_window(qtbot, BrokenCatalog())
        qtbot.waitUntil(
            lambda: "Run listing failed" in window.statusBar().currentMessage(),
            timeout=3000,
        )
        assert "VPN fell over" in window.statusBar().currentMessage()


class TestStaleResults:
    def test_superseded_list_result_is_dropped(self, qtbot):
        window = _make_window(qtbot, _two_run_catalog())
        _wait_runs(qtbot, window, 3)
        # Simulate a straggler from a retired generation.
        stale = (window._list_generation - 1, [])
        window._on_list_result((stale, None))
        assert window.b2_run_list.count() == 3  # unchanged


@pytest.fixture(autouse=True)
def _flush_events(qtbot):
    """Flush posted events after each test before teardown."""
    yield
    qtbot.wait(10)


class TestStaleDetail:
    """A loaded detail must never outlive its query context (review P2).

    Open-scan-folder resolves against the *currently selected* day, so a
    stale ``RunDetail`` surviving a reload could target the wrong ScanNNN.
    """

    def _assert_detail_cleared(self, window):
        assert window._detail is None
        assert not window.b3_copy_uid_button.isEnabled()
        assert not window.b3_open_folder_button.isEnabled()
        assert window.b3_title.text() == "No scan selected"
        assert window.b6_drift_list.count() == 0
        assert window.b5_table.rowCount() == 0

    def test_reload_clears_previous_detail(self, qtbot):
        window = _make_window(qtbot, _two_run_catalog())
        _wait_runs(qtbot, window, 3)
        _select_run(qtbot, window, 0)
        window.catalog = FakeCatalog([])  # the new day has no runs
        window.reload_runs()
        self._assert_detail_cleared(window)
        qtbot.waitUntil(lambda: window.b2_run_list.count() == 0, timeout=3000)
        self._assert_detail_cleared(window)

    def test_emptied_selection_clears_detail(self, qtbot):
        window = _make_window(qtbot, _two_run_catalog())
        _wait_runs(qtbot, window, 3)
        _select_run(qtbot, window, 0)
        window.b2_run_list.setCurrentRow(-1)
        self._assert_detail_cleared(window)

    def test_load_error_clears_previous_detail(self, qtbot):
        catalog = _two_run_catalog()
        window = _make_window(qtbot, catalog)
        _wait_runs(qtbot, window, 3)
        _select_run(qtbot, window, 0)

        def _boom(uid):
            raise RuntimeError("share went away")

        catalog.load_run = _boom
        window.b2_run_list.setCurrentRow(1)
        qtbot.waitUntil(
            lambda: "Run load failed" in window.statusBar().currentMessage(),
            timeout=3000,
        )
        self._assert_detail_cleared(window)

    def test_late_error_from_superseded_load_is_ignored(self, qtbot):
        window = _make_window(qtbot, _two_run_catalog())
        _wait_runs(qtbot, window, 3)
        _select_run(qtbot, window, 0)
        detail_before = window._detail
        stale = (window._detail_generation - 1, None, RuntimeError("old query"))
        window._on_detail_result((stale, None))
        assert window._detail is detail_before
        assert window.b3_copy_uid_button.isEnabled()
