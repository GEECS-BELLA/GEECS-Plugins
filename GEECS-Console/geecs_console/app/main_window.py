"""The GEECS-Console main window: loads the .ui, wires the controller seams.

The window owns no engine logic.  It reads widgets into a
:class:`~geecs_console.request_builder.ConsoleFormState`, builds requests via
:func:`~geecs_console.request_builder.build_scan_request`, submits through
the :class:`~geecs_console.submission.Submitter` protocol, lists configs
through :class:`~geecs_console.services.configs.ConsoleConfigs`, and shows
health via :class:`~geecs_console.services.health.HealthProbe`.  All four are
constructor-injectable, so tests drive the window with fakes and zero
network.
"""

from __future__ import annotations

import importlib.metadata
import os
import threading
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QFile, QObject, Qt, QTimer, Signal, Slot
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QWidget,
)
from pydantic import ValidationError

from geecs_console.events_adapter import ScanEventsAdapter
from geecs_console.request_builder import (
    MAXIMUM_SCAN_SIZE,
    ConsoleFormError,
    ConsoleFormState,
    ConsoleMode,
    FormAxis,
    build_scan_request,
    estimate_total_shots,
)
from geecs_console.services.configs import ConsoleConfigs
from geecs_console.services.health import (
    HealthProbe,
    HealthReport,
    HealthStatus,
    StubHealth,
)
from geecs_console.submission import Submitter, make_bluesky_submitter

_UI_PATH = Path(__file__).parent / "ui" / "main_window.ui"
_QSS_PATH = Path(__file__).parent / "style.qss"

_SCAN_NUMBER_EXPIRY_MS = 10_000

#: How often the background poller re-checks the health probe.
_HEALTH_POLL_INTERVAL_MS = 5_000

#: Screen-map semantic colors (see style.qss header for the full palette).
_COLOR_DIM = "#6b7681"
_COLOR_GREY = "#b9c0c7"
_COLOR_GREEN = "#2f9e63"
_COLOR_AMBER = "#d9a21b"
_COLOR_RED = "#c4453a"

_HEALTH_DOT_COLORS = {
    HealthStatus.OK: _COLOR_GREEN,
    HealthStatus.WARN: _COLOR_AMBER,
    HealthStatus.DOWN: _COLOR_RED,
    HealthStatus.UNKNOWN: _COLOR_GREY,
}

#: ScanState value → state-pill dot color (grey covers idle/unknown).
_STATE_DOT_COLORS = {
    "running": _COLOR_GREEN,
    "done": _COLOR_GREEN,
    "complete": _COLOR_GREEN,
    "completed": _COLOR_GREEN,
    "initializing": _COLOR_AMBER,
    "aborted": _COLOR_RED,
    "error": _COLOR_RED,
    "failed": _COLOR_RED,
}


def _package_version() -> str:
    """Return the installed geecs-console version, or a dev placeholder."""
    try:
        return importlib.metadata.version("geecs-console")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


def load_stylesheet() -> str:
    """Read the packaged QSS, resolving the ui-directory asset token.

    Returns
    -------
    str
        The stylesheet text with ``@UI_DIR@`` replaced by the absolute
        path of the packaged ``ui/`` directory (combo/spin arrow SVGs).
    """
    qss = _QSS_PATH.read_text(encoding="utf-8")
    ui_dir = (Path(__file__).parent / "ui").as_posix()
    return qss.replace("@UI_DIR@", ui_dir)


class HealthPoller(QObject):
    """Runs ``probe.poll()`` off the GUI thread and reports the result.

    The poller itself lives on the GUI thread; each :meth:`poll_async` call
    spawns a short-lived daemon thread that runs the (possibly slow) blocking
    ``poll()`` and emits :attr:`report_ready` with the result.  Qt marshals the
    emit back to the GUI-thread slot as a queued delivery, so the chips update
    without ever blocking the event loop — and there is no worker Qt event loop
    or cross-thread QTimer to manage.

    Parameters
    ----------
    probe : HealthProbe
        The probe to poll; only its ``poll()`` method is used, so it works
        with the real probe, the stub, or a test fake.
    """

    report_ready = Signal(object)
    """Carries one :class:`HealthReport` per completed poll."""

    def __init__(self, probe: HealthProbe) -> None:
        super().__init__()
        self._probe = probe
        self._busy = False

    @Slot()
    def poll_async(self) -> None:
        """Kick off one poll in a daemon thread (skipped if one is in flight).

        Called on the GUI thread from the interval timer; returns immediately.
        """
        if self._busy:
            return
        self._busy = True
        threading.Thread(
            target=self._run, name="console-health-poll", daemon=True
        ).start()

    def _run(self) -> None:
        """Poll the probe (on the daemon thread) and emit the report."""
        report = None
        try:
            report = self._probe.poll()
        except Exception:  # noqa: BLE001 — a probe fault must not kill the poller
            report = None
        finally:
            self._busy = False
        if report is not None:
            self.report_ready.emit(report)


class MainWindow(QMainWindow):
    """The operator console main window (screen map regions R1-R7).

    Parameters
    ----------
    experiment : str, optional
        Experiment to open with; empty selects nothing (offline default).
    configs : ConsoleConfigs, optional
        Configs-repo service; tests inject a fake, default reads the repo.
    health : HealthProbe, optional
        Session-bar chip source; default is the all-unknown stub.
    submitter : Submitter, optional
        Scan engine; tests inject a fake.  When ``None`` one is built
        lazily by *submitter_factory* on the first Start click.
    submitter_factory : callable, optional
        ``(experiment, on_event) -> Submitter``; defaults to
        :func:`~geecs_console.submission.make_bluesky_submitter`.
    """

    def __init__(
        self,
        experiment: str = "",
        configs: Optional[ConsoleConfigs] = None,
        health: Optional[HealthProbe] = None,
        submitter: Optional[Submitter] = None,
        submitter_factory: Optional[Callable[..., Submitter]] = None,
    ) -> None:
        super().__init__()
        self._configs = configs if configs is not None else ConsoleConfigs(experiment)
        self._health = health if health is not None else StubHealth()
        self._submitter = submitter
        self._submitter_factory = (
            submitter_factory
            if submitter_factory is not None
            else make_bluesky_submitter
        )
        self.events = ScanEventsAdapter(self)
        self._total_shots = 0
        self._shot_count_valid = False

        self._apply_stylesheet()
        self._load_ui()
        self._bind_widgets()
        self._build_menus()
        self._build_status_bar()
        self._wire_signals()

        self.setWindowTitle("GEECS Console")
        self._scan_number_timer = QTimer(self)
        self._scan_number_timer.setSingleShot(True)
        self._scan_number_timer.setInterval(_SCAN_NUMBER_EXPIRY_MS)
        self._scan_number_timer.timeout.connect(self._expire_scan_number)

        self._populate_from_configs()
        # Chips read UNKNOWN until the first background poll returns; seed the
        # markup synchronously (no probe call — never touches the network).
        self._apply_health_report(HealthReport())
        self._push_experiment_to_probe(self._configs.experiment)
        self._start_health_poller()
        self._set_state_pill("idle")
        self._on_mode_changed()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _apply_stylesheet(self) -> None:
        """Apply the packaged QSS application-wide (once per process)."""
        app = QApplication.instance()
        if app is not None and not app.styleSheet():
            app.setStyleSheet(load_stylesheet())

    def _load_ui(self) -> None:
        """Load the Designer .ui as the central widget."""
        loader = QUiLoader()
        ui_file = QFile(str(_UI_PATH))
        ui_file.open(QFile.OpenModeFlag.ReadOnly)
        try:
            self._ui: QWidget = loader.load(ui_file, self)
        finally:
            ui_file.close()
        if self._ui is None:
            raise RuntimeError(f"Failed to load {_UI_PATH}: {loader.errorString()}")
        self.setCentralWidget(self._ui)
        # Screen-map column proportions (R2 26% | center 46% | right 28%).
        # QUiLoader ignores the .ui stretch attribute, so set it here.
        from PySide6.QtWidgets import QHBoxLayout

        columns = self._ui.findChild(QHBoxLayout, "columns_layout")
        if columns is not None:
            for index, stretch in enumerate((26, 46, 28)):
                columns.setStretch(index, stretch)

    def _child(self, cls: type, name: str):
        """Return the named child widget, failing loudly when missing."""
        widget = self._ui.findChild(cls, name)
        if widget is None:
            raise LookupError(f"{name!r} ({cls.__name__}) not found in {_UI_PATH}")
        return widget

    def _bind_widgets(self) -> None:
        """Resolve every wired widget from the loaded .ui once."""
        # R1 session bar
        self.experiment_combo: QComboBox = self._child(QComboBox, "r1_experiment_combo")
        self.rep_rate: QDoubleSpinBox = self._child(QDoubleSpinBox, "r1_rep_rate")
        self.trigger_profile_combo: QComboBox = self._child(
            QComboBox, "r1_trigger_profile_combo"
        )
        self.trigger_variant_combo: QComboBox = self._child(
            QComboBox, "r1_trigger_variant_combo"
        )
        self.gateway_chip: QLabel = self._child(QLabel, "r1_gateway_chip")
        self.tiled_chip: QLabel = self._child(QLabel, "r1_tiled_chip")
        self.db_chip: QLabel = self._child(QLabel, "r1_db_chip")
        # R2 save sets
        self.available_list: QListWidget = self._child(QListWidget, "r2_available_list")
        self.selected_list: QListWidget = self._child(QListWidget, "r2_selected_list")
        self.add_button: QPushButton = self._child(QPushButton, "r2_add_button")
        self.remove_button: QPushButton = self._child(QPushButton, "r2_remove_button")
        self.union_label: QLabel = self._child(QLabel, "r2_union_label")
        self.hint_label: QLabel = self._child(QLabel, "r2_hint_label")
        # R3 scan form
        self.radio_noscan: QRadioButton = self._child(QRadioButton, "r3_radio_noscan")
        self.radio_1d: QRadioButton = self._child(QRadioButton, "r3_radio_1d")
        self.radio_grid: QRadioButton = self._child(QRadioButton, "r3_radio_grid")
        self.radio_optimization: QRadioButton = self._child(
            QRadioButton, "r3_radio_optimization"
        )
        self.radio_background: QRadioButton = self._child(
            QRadioButton, "r3_radio_background"
        )
        self.variable_combo: QComboBox = self._child(QComboBox, "r3_variable_combo")
        self.start_spin: QDoubleSpinBox = self._child(QDoubleSpinBox, "r3_start")
        self.stop_spin: QDoubleSpinBox = self._child(QDoubleSpinBox, "r3_stop")
        self.step_spin: QDoubleSpinBox = self._child(QDoubleSpinBox, "r3_step")
        self.variable2_combo: QComboBox = self._child(QComboBox, "r3_variable2_combo")
        self.start2_spin: QDoubleSpinBox = self._child(QDoubleSpinBox, "r3_start2")
        self.stop2_spin: QDoubleSpinBox = self._child(QDoubleSpinBox, "r3_stop2")
        self.step2_spin: QDoubleSpinBox = self._child(QDoubleSpinBox, "r3_step2")
        self.shots_per_step: QSpinBox = self._child(QSpinBox, "r3_shots_per_step")
        self.acquisition_combo: QComboBox = self._child(
            QComboBox, "r3_acquisition_combo"
        )
        self.shot_count_label: QLabel = self._child(QLabel, "r3_shot_count_label")
        self.description_edit: QLineEdit = self._child(QLineEdit, "r3_description")
        # R4 presets
        self.preset_combo: QComboBox = self._child(QComboBox, "r4_preset_combo")
        self.apply_button: QPushButton = self._child(QPushButton, "r4_apply_button")
        self.save_as_button: QPushButton = self._child(QPushButton, "r4_save_as_button")
        # R5 submit row
        self.stop_button: QPushButton = self._child(QPushButton, "r5_stop_button")
        self.start_button: QPushButton = self._child(QPushButton, "r5_start_button")
        # R6 now panel
        self.state_pill: QLabel = self._child(QLabel, "r6_state_pill")
        self.progress_bar: QProgressBar = self._child(QProgressBar, "r6_progress")
        self.scan_number_label: QLabel = self._child(QLabel, "r6_scan_number_label")
        self.log_tail: QPlainTextEdit = self._child(QPlainTextEdit, "r6_log_tail")
        # R7 device panel
        self.device_combo: QComboBox = self._child(QComboBox, "r7_device_combo")
        self.readback_label: QLabel = self._child(QLabel, "r7_readback_label")
        self.set_field: QLineEdit = self._child(QLineEdit, "r7_set_field")
        self.set_button: QPushButton = self._child(QPushButton, "r7_set_button")

        from geecs_schemas import AcquisitionMode

        for mode in (AcquisitionMode.FREE_RUN, AcquisitionMode.STRICT):
            self.acquisition_combo.addItem(mode.value)

    def _build_menus(self) -> None:
        """Create the menu bar (Ops / Actions / Editors / Preferences / Help)."""
        for title in ("Ops", "Actions", "Editors", "Preferences"):
            menu = self.menuBar().addMenu(title)
            placeholder = menu.addAction("(not wired yet)")
            placeholder.setEnabled(False)
        help_menu = self.menuBar().addMenu("Help")
        about = help_menu.addAction(f"GEECS Console {_package_version()}")
        about.setEnabled(False)

    def _build_status_bar(self) -> None:
        """Create the status bar: gateway addr, configs path, version."""
        gateway = os.environ.get("EPICS_CA_ADDR_LIST", "unset")
        self._status_gateway = QLabel(f"gateway: {gateway}")
        self._status_configs = QLabel("configs: —")
        self._status_version = QLabel(f"v{_package_version()}")
        self.statusBar().addWidget(self._status_gateway)
        self.statusBar().addWidget(self._status_configs)
        self.statusBar().addPermanentWidget(self._status_version)

    def _wire_signals(self) -> None:
        """Connect widget and adapter signals to the handlers."""
        for radio in (
            self.radio_noscan,
            self.radio_1d,
            self.radio_grid,
            self.radio_optimization,
            self.radio_background,
        ):
            radio.toggled.connect(self._on_mode_changed)
        for spin in (
            self.start_spin,
            self.stop_spin,
            self.step_spin,
            self.start2_spin,
            self.stop2_spin,
            self.step2_spin,
        ):
            spin.valueChanged.connect(self._refresh_shot_count)
        self.shots_per_step.valueChanged.connect(self._refresh_shot_count)
        self.add_button.clicked.connect(self._on_add_save_set)
        self.remove_button.clicked.connect(self._on_remove_save_set)
        self.experiment_combo.currentTextChanged.connect(self._on_experiment_changed)
        self.trigger_profile_combo.currentTextChanged.connect(
            self._on_trigger_profile_changed
        )
        self.start_button.clicked.connect(self._on_start_clicked)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.apply_button.clicked.connect(self._on_preset_stub)
        self.save_as_button.clicked.connect(self._on_preset_stub)
        self.set_button.clicked.connect(self._on_device_set_stub)

        self.events.state_changed.connect(self._on_scan_state)
        self.events.totals_known.connect(self._on_totals_known)
        self.events.progress.connect(self._on_progress)
        self.events.error.connect(self._on_scan_error)
        self.events.log_line.connect(self.append_log)
        self.events.dialog_requested.connect(self._on_operator_dialog)

    # ------------------------------------------------------------------
    # Configs / health population
    # ------------------------------------------------------------------

    def _populate_from_configs(self) -> None:
        """Fill the combos and lists from the configs service (offline-safe)."""
        listing = self._configs.listing()
        self.experiment_combo.blockSignals(True)
        self.experiment_combo.clear()
        self.experiment_combo.addItems(listing.experiments)
        if self._configs.experiment:
            self.experiment_combo.setCurrentText(self._configs.experiment)
        else:
            # Populated but nothing selected: show a placeholder rather
            # than rendering blank (the combo is editable, so its line
            # edit carries the hint until the operator picks one).
            self.experiment_combo.setCurrentIndex(-1)
            line_edit = self.experiment_combo.lineEdit()
            if line_edit is not None:
                line_edit.setPlaceholderText("select experiment…")
        self.experiment_combo.blockSignals(False)

        self.available_list.clear()
        self.available_list.addItems(listing.save_sets)
        self.selected_list.clear()
        self.trigger_profile_combo.blockSignals(True)
        self.trigger_profile_combo.clear()
        self.trigger_profile_combo.addItem("")
        self.trigger_profile_combo.addItems(listing.trigger_profiles)
        self.trigger_profile_combo.blockSignals(False)
        self.trigger_variant_combo.clear()
        self.variable_combo.clear()
        self.variable_combo.addItems(listing.scan_variables)
        self.variable2_combo.clear()
        self.variable2_combo.addItems(listing.scan_variables)

        root = str(listing.configs_root) if listing.configs_root else "not found"
        self._status_configs.setText(f"configs: {root}")
        if listing.message:
            self.statusBar().showMessage(listing.message, 10_000)
            self.append_log(listing.message)
        self._refresh_union_preview()
        self._refresh_shot_count()

    @staticmethod
    def _chip_markup(name: str, status: HealthStatus) -> str:
        """Rich-text pill body for one R1 health chip: colored dot + text.

        Parameters
        ----------
        name : str
            The chip's service name (``gateway`` / ``tiled`` / ``db``).
        status : HealthStatus
            The polled status (drives the dot color).

        Returns
        -------
        str
            QLabel rich text — the pill border/background come from QSS.
        """
        color = _HEALTH_DOT_COLORS.get(status, _COLOR_GREY)
        return f'<span style="color:{color};">●</span> {name}: {status.value}'

    @Slot(object)
    def _apply_health_report(self, report: HealthReport) -> None:
        """Render a health report into the R1 chips (GUI-thread slot).

        Parameters
        ----------
        report : HealthReport
            The polled chip states (delivered queued from the background
            :class:`HealthPoller`, or passed directly to seed the initial
            all-unknown markup).
        """
        self.gateway_chip.setText(self._chip_markup("gateway", report.gateway))
        self.tiled_chip.setText(self._chip_markup("tiled", report.tiled))
        self.db_chip.setText(self._chip_markup("db", report.db))

    def _start_health_poller(self) -> None:
        """Start the background health poller and its GUI-thread interval timer.

        A GUI-thread :class:`~PySide6.QtCore.QTimer` fires every
        :data:`_HEALTH_POLL_INTERVAL_MS`; each tick dispatches the blocking
        ``poll()`` to a daemon thread inside :class:`HealthPoller`, whose
        ``report_ready`` signal is delivered queued back to
        :meth:`_apply_health_report` on the GUI thread.  Works with any probe
        (stub or real).  One immediate poll runs so the chips leave ``UNKNOWN``
        as soon as the first result lands.
        """
        self._health_poller = HealthPoller(self._health)
        # Force a queued connection so the chip update always runs on the GUI
        # thread — an undecorated bound method can otherwise be wired direct,
        # which would paint QLabels from the daemon thread (a hard crash).
        self._health_poller.report_ready.connect(
            self._apply_health_report, Qt.ConnectionType.QueuedConnection
        )
        self._health_timer = QTimer(self)
        self._health_timer.setInterval(_HEALTH_POLL_INTERVAL_MS)
        self._health_timer.timeout.connect(self._health_poller.poll_async)
        self._health_timer.start()
        self._health_poller.poll_async()

    def _push_experiment_to_probe(self, experiment: str) -> None:
        """Point the probe at *experiment*'s gateway PV, if it supports it.

        StubHealth has no ``experiment`` attribute, so this is a guarded no-op
        for the offline default; the real probe picks up the new prefix on its
        next poll.

        Parameters
        ----------
        experiment : str
            The selected experiment name ("" for none).
        """
        if hasattr(self._health, "experiment"):
            setattr(self._health, "experiment", experiment or None)

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt override
        """Stop the background health poller cleanly before closing.

        Stops the GUI-thread interval timer (no further polls) and disconnects
        the report signal so a still-running daemon poll can't paint a chip on
        a window being torn down.  In-flight daemon threads are daemonic and
        finish on their own without blocking shutdown.
        """
        timer = getattr(self, "_health_timer", None)
        if timer is not None:
            timer.stop()
        poller = getattr(self, "_health_poller", None)
        if poller is not None:
            try:
                poller.report_ready.disconnect(self._apply_health_report)
            except (RuntimeError, TypeError):
                pass
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Form state (the round-trip surface tests exercise)
    # ------------------------------------------------------------------

    def current_mode(self) -> ConsoleMode:
        """Return the mode the R3 radios currently select.

        Returns
        -------
        ConsoleMode
            The checked radio's mode (1D when nothing is checked yet).
        """
        if self.radio_noscan.isChecked():
            return ConsoleMode.NOSCAN
        if self.radio_grid.isChecked():
            return ConsoleMode.GRID
        if self.radio_optimization.isChecked():
            return ConsoleMode.OPTIMIZATION
        if self.radio_background.isChecked():
            return ConsoleMode.BACKGROUND
        return ConsoleMode.ONE_D

    def selected_save_sets(self) -> list[str]:
        """Return the R2 selected save-set names, in list order.

        Returns
        -------
        list of str
            One name per row of the selected list.
        """
        return [
            self.selected_list.item(row).text()
            for row in range(self.selected_list.count())
        ]

    def form_state(self) -> ConsoleFormState:
        """Snapshot the widgets into a :class:`ConsoleFormState`.

        Returns
        -------
        ConsoleFormState
            The validated form model :func:`build_scan_request` consumes.

        Raises
        ------
        pydantic.ValidationError
            When the widgets hold an invalid combination (e.g. an empty
            scan-variable name in a step mode).
        """
        mode = self.current_mode()
        axes: list[FormAxis] = []
        if mode in (ConsoleMode.ONE_D, ConsoleMode.GRID):
            axes.append(
                FormAxis(
                    variable=self.variable_combo.currentText(),
                    start=self.start_spin.value(),
                    stop=self.stop_spin.value(),
                    step=self.step_spin.value(),
                )
            )
        if mode is ConsoleMode.GRID:
            axes.append(
                FormAxis(
                    variable=self.variable2_combo.currentText(),
                    start=self.start2_spin.value(),
                    stop=self.stop2_spin.value(),
                    step=self.step2_spin.value(),
                )
            )
        profile = self.trigger_profile_combo.currentText() or None
        variant = self.trigger_variant_combo.currentText() or None
        from geecs_schemas import AcquisitionMode

        return ConsoleFormState(
            mode=mode,
            axes=axes,
            shots_per_step=self.shots_per_step.value(),
            save_sets=self.selected_save_sets(),
            trigger_profile=profile,
            trigger_variant=variant if profile else None,
            acquisition=AcquisitionMode(self.acquisition_combo.currentText()),
            description=self.description_edit.text(),
        )

    # ------------------------------------------------------------------
    # R3 handlers
    # ------------------------------------------------------------------

    def _on_mode_changed(self) -> None:
        """Apply the mode's enable states to the axis rows, then refresh."""
        mode = self.current_mode()
        axis1 = mode in (ConsoleMode.ONE_D, ConsoleMode.GRID)
        axis2 = mode is ConsoleMode.GRID
        for widget in (
            self.variable_combo,
            self.start_spin,
            self.stop_spin,
            self.step_spin,
        ):
            widget.setEnabled(axis1)
        for widget in (
            self.variable2_combo,
            self.start2_spin,
            self.stop2_spin,
            self.step2_spin,
        ):
            widget.setEnabled(axis2)
        self._refresh_shot_count()

    def _estimation_form(self) -> ConsoleFormState:
        """Form state for shot counting only (placeholder variable names)."""
        mode = self.current_mode()
        axes: list[FormAxis] = []
        if mode in (ConsoleMode.ONE_D, ConsoleMode.GRID):
            axes.append(
                FormAxis(
                    variable="axis1",
                    start=self.start_spin.value(),
                    stop=self.stop_spin.value(),
                    step=self.step_spin.value(),
                )
            )
        if mode is ConsoleMode.GRID:
            axes.append(
                FormAxis(
                    variable="axis2",
                    start=self.start2_spin.value(),
                    stop=self.stop2_spin.value(),
                    step=self.step2_spin.value(),
                )
            )
        return ConsoleFormState(
            mode=mode, axes=axes, shots_per_step=self.shots_per_step.value()
        )

    def _refresh_shot_count(self) -> None:
        """Recompute the live shot-count label and the runaway guard."""
        try:
            total = estimate_total_shots(self._estimation_form())
        except (ConsoleFormError, ValidationError):
            self._shot_count_valid = False
            self.shot_count_label.setText("total shots: —")
            self._refresh_submit_enabled()
            return
        if total > MAXIMUM_SCAN_SIZE:
            self._shot_count_valid = False
            self.shot_count_label.setText(
                f"total shots: {total} — exceeds {MAXIMUM_SCAN_SIZE:.0e} limit"
            )
        else:
            self._shot_count_valid = True
            self.shot_count_label.setText(f"total shots: {total}")
        self._refresh_submit_enabled()

    # ------------------------------------------------------------------
    # R2 handlers
    # ------------------------------------------------------------------

    def _move_items(self, source: QListWidget, target: QListWidget) -> None:
        """Move the selected rows from *source* to *target*."""
        for item in source.selectedItems():
            source.takeItem(source.row(item))
            target.addItem(item.text())
        self._refresh_union_preview()
        self._refresh_submit_enabled()

    def _on_add_save_set(self) -> None:
        """R2 Add: available → selected."""
        self._move_items(self.available_list, self.selected_list)

    def _on_remove_save_set(self) -> None:
        """R2 Remove: selected → available."""
        self._move_items(self.selected_list, self.available_list)

    def _refresh_union_preview(self) -> None:
        """Update the R2 union line and role-conflict/reference hint."""
        preview = self._configs.union_preview(self.selected_save_sets())
        if preview.device_count is None:
            self.union_label.setText("union: —")
        else:
            self.union_label.setText(f"union: {preview.device_count} devices")
        self.hint_label.setText(preview.hint)

    # ------------------------------------------------------------------
    # R1 handlers
    # ------------------------------------------------------------------

    def _on_experiment_changed(self, experiment: str) -> None:
        """Repopulate everything for the newly selected experiment."""
        self._configs.set_experiment(experiment)
        self._push_experiment_to_probe(experiment)
        self._populate_from_configs()

    def _on_trigger_profile_changed(self, profile: str) -> None:
        """Repopulate the variant combo for the selected trigger profile."""
        self.trigger_variant_combo.clear()
        if profile:
            self.trigger_variant_combo.addItem("")
            self.trigger_variant_combo.addItems(self._configs.trigger_variants(profile))

    # ------------------------------------------------------------------
    # R5 submit row
    # ------------------------------------------------------------------

    def _scanning(self) -> bool:
        """Whether the submitter reports an active scan."""
        return self._submitter is not None and self._submitter.is_scanning_active()

    def _refresh_submit_enabled(self) -> None:
        """Recompute Start/Stop enabled state from form + engine."""
        scanning = self._scanning()
        ready = (
            not scanning
            and self._shot_count_valid
            and bool(self.selected_save_sets())
            and self.current_mode() is not ConsoleMode.OPTIMIZATION
        )
        self.start_button.setEnabled(ready)
        self.stop_button.setEnabled(scanning)

    def _ensure_submitter(self) -> Optional[Submitter]:
        """Return the injected submitter, or lazily build the real engine."""
        if self._submitter is not None:
            return self._submitter
        try:
            self._submitter = self._submitter_factory(
                self.experiment_combo.currentText(), self.events.handle
            )
        except Exception as exc:
            message = f"Scan engine unavailable: {exc}"
            self.statusBar().showMessage(message, 10_000)
            self.append_log(message)
            return None
        return self._submitter

    def _on_start_clicked(self) -> None:
        """Build the request from the form and submit it."""
        try:
            request = build_scan_request(self.form_state())
        except (ConsoleFormError, ValidationError) as exc:
            message = f"Cannot submit: {exc}"
            self.statusBar().showMessage(message, 10_000)
            self.append_log(message)
            return
        submitter = self._ensure_submitter()
        if submitter is None:
            return
        try:
            accepted = submitter.reinitialize(request)
            if not accepted:
                # The Submitter protocol returns False on a refused request
                # (e.g. unresolvable names) — starting anyway would run the
                # scanner on stale state from a previous reinitialize.
                message = "Submission refused: the scanner did not accept the request"
                self.statusBar().showMessage(message, 10_000)
                self.append_log(message)
                return
            submitter.start_scan_thread()
        except Exception as exc:
            message = f"Submission failed: {exc}"
            self.statusBar().showMessage(message, 10_000)
            self.append_log(message)
        self._refresh_submit_enabled()

    def _on_stop_clicked(self) -> None:
        """Stop the running scan."""
        if self._submitter is not None and self._submitter.is_scanning_active():
            self._submitter.stop_scanning_thread()
        self._refresh_submit_enabled()

    # ------------------------------------------------------------------
    # R4 / R7 stubs
    # ------------------------------------------------------------------

    def _on_preset_stub(self) -> None:
        """R4 placeholder: presets are scan requests, not wired yet."""
        self.statusBar().showMessage(
            "Presets are not wired yet (a preset is a saved ScanRequest).", 10_000
        )

    def _on_device_set_stub(self) -> None:
        """R7 placeholder: gateway-PV set/readback backend arrives later."""
        message = "Device panel backend not wired yet (gateway PV set/readback)."
        self.statusBar().showMessage(message, 10_000)
        self.append_log(message)

    # ------------------------------------------------------------------
    # R6 now panel
    # ------------------------------------------------------------------

    def append_log(self, line: str) -> None:
        """Append one line to the compact log tail.

        Parameters
        ----------
        line : str
            The text to append (one log-tail row).
        """
        self.log_tail.appendPlainText(line)

    def set_scan_number(self, number: int) -> None:
        """Show the current scan number, expiring to 'previous' after 10 s.

        Parameters
        ----------
        number : int
            The claimed scan number.
        """
        self.scan_number_label.setText(f"Scan {number:03d}")
        self._scan_number_timer.start()

    def _expire_scan_number(self) -> None:
        """Mark the displayed scan number as previous once the timer fires."""
        text = self.scan_number_label.text()
        if text.startswith("Scan "):
            self.scan_number_label.setText(f"{text} (previous)")

    def _set_state_pill(self, state: str) -> None:
        """Render the R6 state pill: colored dot + uppercase state word.

        Parameters
        ----------
        state : str
            A ``ScanState`` value (e.g. ``"running"``); dot color is green
            for running/done, amber for initializing, red for aborted or
            error, grey otherwise (idle).
        """
        word = (state or "idle").strip()
        color = _STATE_DOT_COLORS.get(word.lower(), _COLOR_GREY)
        self.state_pill.setText(f'<span style="color:{color};">●</span> {word.upper()}')

    def _on_scan_state(self, state: str) -> None:
        """Update the state pill and button gating on lifecycle events."""
        self._set_state_pill(state)
        self._refresh_submit_enabled()

    def _on_totals_known(self, total_shots: int) -> None:
        """Size the progress bar once the scan announces its totals."""
        self._total_shots = total_shots
        self.progress_bar.setMaximum(max(1, total_shots))
        self.progress_bar.setValue(0)

    def _on_progress(
        self, step_index: int, total_steps: int, shots_completed: int
    ) -> None:
        """Advance the progress bar from step events."""
        if self._total_shots:
            self.progress_bar.setValue(min(shots_completed, self._total_shots))
        elif total_steps:
            self.progress_bar.setMaximum(total_steps)
            self.progress_bar.setValue(min(step_index + 1, total_steps))

    def _on_scan_error(self, message: str) -> None:
        """Show scan errors in the status bar."""
        self.statusBar().showMessage(message, 10_000)

    # ------------------------------------------------------------------
    # Operator / pre-flight dialogs
    # ------------------------------------------------------------------

    def _on_operator_dialog(self, request: object) -> None:
        """Render an operator question modally and unblock the engine thread.

        Delivered queued from :meth:`ScanEventsAdapter.handle` (which runs on
        the engine/scan thread, blocked on ``request.response_event``), so this
        slot runs on the GUI thread — where a modal must live.  The engine
        resumes with the operator's choice: Abort sets ``request.abort[0]``
        before the response event is set.

        Parameters
        ----------
        request : object
            A ``DialogRequest`` (duck-typed): ``exc``, optional ``title`` /
            ``continue_label`` / ``abort_label``, a mutable one-element
            ``abort`` list, and a ``response_event`` (:class:`threading.Event`).
        """
        exc = getattr(request, "exc", None)
        title = getattr(request, "title", None) or "Operator confirmation"
        continue_label = getattr(request, "continue_label", None) or "Continue"
        abort_label = getattr(request, "abort_label", None) or "Abort"

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(str(title))
        box.setText(str(exc) if exc is not None else str(title))
        continue_button = box.addButton(
            str(continue_label), QMessageBox.ButtonRole.AcceptRole
        )
        box.addButton(str(abort_label), QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(continue_button)
        box.exec()

        aborted = box.clickedButton() is not continue_button
        abort_flag = getattr(request, "abort", None)
        if aborted and isinstance(abort_flag, list) and abort_flag:
            abort_flag[0] = True
        self.append_log(f"operator: {'abort' if aborted else 'continue'}")

        response_event = getattr(request, "response_event", None)
        if response_event is not None and hasattr(response_event, "set"):
            response_event.set()
