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

import logging
import os
import random
import threading
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QEvent, QFile, QObject, Qt, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QInputDialog,
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

from geecs_console.app.action_dialog import ActionRunDialog
from geecs_console.editors.action_library_editor import open_action_library_editor
from geecs_console.editors.save_set_editor import open_save_set_editor
from geecs_console.editors.scan_variable_editor import open_scan_variable_editor
from geecs_console.editors.shot_control_editor import open_shot_control_editor
from geecs_console.events_adapter import ScanEventsAdapter
from geecs_console.services import ops_paths
from geecs_console.services.action_library_store import ActionLibraryStore
from geecs_console.services.background import BackgroundResult
from geecs_console.services.device_completions import (
    CompletionsProvider,
    EmptyCompletions,
    GeecsDbCompletions,
)
from geecs_console.version import console_version
from geecs_console.request_builder import (
    MAXIMUM_SCAN_SIZE,
    ConsoleFormError,
    ConsoleFormState,
    ConsoleMode,
    FormAxis,
    build_scan_request,
    estimate_total_shots,
    form_state_from_request,
)
from geecs_console.services.configs import ConsoleConfigs
from geecs_console.services.presets import PresetStore, PresetStoreError
from geecs_console.services.settings import ConsoleSettings
from geecs_console.services.device_panel import (
    DevicePanelBackend,
    StubDevicePanel,
    format_readback,
    parse_device_variable,
    parse_set_value,
)
from geecs_console.services.health import (
    HealthProbe,
    HealthReport,
    HealthStatus,
    StubHealth,
)
from geecs_console.submission import Submitter, make_bluesky_submitter

logger = logging.getLogger(__name__)

_UI_PATH = Path(__file__).parent / "ui" / "main_window.ui"
_QSS_PATH = Path(__file__).parent / "style.qss"

_SCAN_NUMBER_EXPIRY_MS = 10_000

#: With "Randomized beeps" on, the fraction of shots that actually beep.
_RANDOM_BEEP_PROBABILITY = 0.25

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


def _default_completions_factory(experiment: str) -> CompletionsProvider:
    """Build the production R7 completions provider for *experiment*."""
    return GeecsDbCompletions(experiment)


def _idle_scan_lookup(experiment: str) -> Optional[int]:
    """Highest existing ``ScanNNN`` in today's daily folder — read-only.

    The production R6 idle-scan lookup: resolves today's ``scans/`` path
    via :func:`ops_paths.todays_scan_folder` and lists it with
    :func:`ops_paths.highest_scan_number`.  Strictly read-only (repo
    scan-folder invariant) and possibly slow (network data root), so the
    window only ever calls it on a daemon thread.

    Parameters
    ----------
    experiment : str
        The selected experiment ("" falls back to the config default).

    Returns
    -------
    int or None
        The highest scan number, or ``None`` when unresolvable/absent.
    """
    return ops_paths.highest_scan_number(ops_paths.todays_scan_folder(experiment))


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


class ToolTipSuppressor(QObject):
    """Application-level event filter that swallows every tooltip event.

    Installed on the ``QApplication`` **only while tooltips are turned
    off** — presence means suppression, so one switch covers every console
    widget (the main window *and* the editor dialogs, whose schema-derived
    tooltips are applied unconditionally) and the default-on path pays no
    per-event Python filter cost at all.  Parented to the window: Qt
    removes a destroyed filter from the application automatically, so a
    closed window can never leave a dangling suppressor behind.

    Parameters
    ----------
    parent : QObject
        The owning window (keeps the wrapper referenced — the usual
        PySide6 GC hazard — and bounds the filter's lifetime).
    """

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 — Qt override
        """Swallow ``QEvent.ToolTip``; pass everything else through.

        Parameters
        ----------
        obj : QObject
            The event's target (unused — suppression is global).
        event : QEvent
            The event under consideration.

        Returns
        -------
        bool
            ``True`` (consume the event) for tooltip events.
        """
        if event.type() == QEvent.Type.ToolTip:
            return True
        return super().eventFilter(obj, event)


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
    device_panel : DevicePanelBackend, optional
        R7 readback/set backend; default is the no-op stub (readback never
        updates, sets report unwired).  ``main.py`` injects the real
        :class:`~geecs_console.services.device_panel.GatewayDevicePanel`.
    presets : PresetStore, optional
        R4 preset persistence (a preset IS a saved ``ScanRequest``); tests
        inject a fake or a tmp-dir-backed store, default reads/writes the
        experiment's ``presets/`` dir in the configs repo.
    action_store : ActionLibraryStore, optional
        The Actions-menu name source (only ``list_names`` /
        ``set_experiment`` are used — listing runs on a daemon thread);
        tests inject a fake, default reads the experiment's
        ``action_library/actions.yaml`` in the configs repo.
    settings : ConsoleSettings, optional
        Persisted GUI state (last selected experiment); tests inject one
        backed by a tmp INI file, default is the user-scope QSettings store.
    submitter : Submitter, optional
        Scan engine; tests inject a fake.  When ``None`` one is built
        lazily by *submitter_factory* on the first Start click.
    submitter_factory : callable, optional
        ``(experiment, on_event) -> Submitter``; defaults to
        :func:`~geecs_console.submission.make_bluesky_submitter`.
    rng : random.Random, optional
        The source of randomness for the "Randomized beeps" option; tests
        inject a seeded instance.  Defaults to a fresh ``random.Random()``.
    completions_factory : callable, optional
        ``(experiment) -> CompletionsProvider`` for the R7 device combo's
        ``device:variable`` items; tests inject a fake.  Defaults to the
        DB-backed provider (daemon-thread fetch, empty offline).
    scan_number_lookup : callable, optional
        ``(experiment) -> int | None`` returning today's highest existing
        scan number for the R6 idle display; tests inject a fake.  Defaults
        to the read-only ``ops_paths`` lookup (daemon-thread call — the data
        root may be a slow network mount).
    """

    #: One readback value from the device-panel backend (emitted from the CA
    #: monitor thread; delivered queued to :meth:`_apply_device_value`).
    device_value_ready = Signal(object)

    def __init__(
        self,
        experiment: str = "",
        configs: Optional[ConsoleConfigs] = None,
        health: Optional[HealthProbe] = None,
        device_panel: Optional[DevicePanelBackend] = None,
        presets: Optional[PresetStore] = None,
        action_store: Optional[ActionLibraryStore] = None,
        settings: Optional[ConsoleSettings] = None,
        submitter: Optional[Submitter] = None,
        submitter_factory: Optional[Callable[..., Submitter]] = None,
        rng: Optional[random.Random] = None,
        completions_factory: Optional[Callable[[str], CompletionsProvider]] = None,
        scan_number_lookup: Optional[Callable[[str], Optional[int]]] = None,
    ) -> None:
        super().__init__()
        self._configs = configs if configs is not None else ConsoleConfigs(experiment)
        self._health = health if health is not None else StubHealth()
        self._device_panel = (
            device_panel if device_panel is not None else StubDevicePanel()
        )
        self._presets = (
            presets if presets is not None else PresetStore(self._configs.experiment)
        )
        self._action_store = (
            action_store
            if action_store is not None
            else ActionLibraryStore(self._configs.experiment)
        )
        self._settings = settings if settings is not None else ConsoleSettings()
        self._device_set_in_flight = False
        self._submitter = submitter
        self._submitter_factory = (
            submitter_factory
            if submitter_factory is not None
            else make_bluesky_submitter
        )
        self.events = ScanEventsAdapter(self)
        self._total_shots = 0
        self._shot_count_valid = False
        self._beep_rng = rng if rng is not None else random.Random()
        self._last_beep_shots = 0
        self._completions_factory = completions_factory
        self._scan_number_lookup = scan_number_lookup
        #: Non-modal editor dialogs opened from the Editors menu.  PySide6
        #: garbage-collects an unreferenced dialog wrapper and tears down the
        #: C++ dialog with it, so every opened editor is kept here.
        self._open_editors: list = []
        #: Non-modal ActionRunDialogs opened from the Actions menu — same
        #: PySide6 GC hazard, same keep-a-reference cure.
        self._open_action_dialogs: list = []

        self._apply_stylesheet()
        self._load_ui()
        self._bind_widgets()
        self._apply_operator_tooltips()
        self._build_menus()
        self._build_status_bar()
        self._wire_signals()

        self.setWindowTitle("GEECS Console")
        self._scan_number_timer = QTimer(self)
        self._scan_number_timer.setSingleShot(True)
        self._scan_number_timer.setInterval(_SCAN_NUMBER_EXPIRY_MS)
        self._scan_number_timer.timeout.connect(self._expire_scan_number)

        self._populate_from_configs()
        self._restore_last_experiment()
        # Chips read UNKNOWN until the first background poll returns; seed the
        # markup synchronously (no probe call — never touches the network).
        self._apply_health_report(HealthReport())
        self._push_experiment_to_probe(self._configs.experiment)
        self._start_health_poller()
        self._set_state_pill("idle")
        self._on_mode_changed()
        self._refresh_device_set_enabled()
        # Startup fetches for the selected experiment (no-ops when none):
        # R7 device:variable completions, the R6 idle scan-number peek, and
        # the Actions-menu plan names.  Restoring the last experiment already
        # fired the experiment-changed path (which starts all three); these
        # cover the explicit-experiment and no-experiment startups.  Stale
        # results are dropped by experiment tag, so a duplicate fetch is
        # harmless.
        self._start_device_completions_fetch()
        self._start_idle_scan_probe()
        self._start_actions_fetch()

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
        self.optimization_label: QLabel = self._child(QLabel, "r3_optimization_label")
        self.optimization_combo: QComboBox = self._child(
            QComboBox, "r3_optimization_combo"
        )
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
        self.delete_button: QPushButton = self._child(QPushButton, "r4_delete_button")
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
        device_line_edit = self.device_combo.lineEdit()
        if device_line_edit is not None:
            device_line_edit.setPlaceholderText("device:variable")

        from geecs_schemas import AcquisitionMode

        for mode in (AcquisitionMode.FREE_RUN, AcquisitionMode.STRICT):
            self.acquisition_combo.addItem(mode.value)

    def _apply_operator_tooltips(self) -> None:
        """Set the operator-language tooltips on the main-window controls.

        These are hand-written (what it does / what happens), not copies of
        the widget labels.  Editor form fields are different: their tooltips
        come from the schema field descriptions via
        :mod:`~geecs_console.services.schema_tooltips` — single source of
        truth (issue #497 phase 1).
        """
        tooltips = {
            # R1 session bar
            self.experiment_combo: (
                "Which experiment's configs, presets, devices, and gateway "
                "PVs the console works with. Changing it repopulates "
                "everything below."
            ),
            self.rep_rate: (
                "The machine repetition rate in Hz — informational for "
                "free-run pacing; it does not command any hardware."
            ),
            self.trigger_profile_combo: (
                "Trigger profile the scan drives the machine with (OFF / "
                "STANDBY / SCAN / ... device writes). Empty means the scan "
                "leaves the trigger alone."
            ),
            self.trigger_variant_combo: (
                "Named operating condition of the trigger profile (e.g. "
                "laser_off) — overlays a few writes on the base profile. "
                "Empty runs the base behaviour."
            ),
            self.gateway_chip: (
                "CA gateway health: reads the experiment's heartbeat PV "
                "every few seconds. WARN means the gateway runs but reports "
                "zero connected devices."
            ),
            self.tiled_chip: (
                "Tiled data-server health: an HTTP check of the configured "
                "Tiled URI every few seconds."
            ),
            self.db_chip: (
                "GEECS experiment database health: a cheap MySQL query "
                "every few seconds."
            ),
            # R2 save sets
            self.available_list: (
                "The experiment's save sets (named device groups) not yet "
                "picked for this scan. Select and Add to require them."
            ),
            self.selected_list: (
                "Save sets this scan records. Their devices are unioned; "
                "each required device gets guarantees (completeness, "
                "dialogs, images). At least one set is needed to start."
            ),
            self.add_button: "Require the selected save sets for this scan.",
            self.remove_button: (
                "Drop the selected save sets from this scan (their devices "
                "may still be logged as background telemetry)."
            ),
            self.union_label: (
                "How many distinct devices the selected save sets add up to "
                "after merging duplicates."
            ),
            # R3 scan form
            self.radio_noscan: (
                "Collect shots without moving anything — statistics at the "
                "current machine settings."
            ),
            self.radio_1d: (
                "Sweep one scan variable through start → stop in step-sized "
                "moves, taking a batch of shots at each position."
            ),
            self.radio_grid: (
                "Sweep two variables as a grid visiting every combination — "
                "axis 1 is the outer (slow) loop, axis 2 the inner (fast) "
                "one."
            ),
            self.radio_optimization: (
                "Let an optimizer pick the settings each iteration, per the "
                "optimizer config chosen below. Submission is accepted only "
                "when the scan engine supports optimize scans."
            ),
            self.radio_background: (
                "A no-scan whose data is marked as background/calibration "
                "shots so analysis can find them later."
            ),
            self.optimization_combo: (
                "Which optimizer config to run — the YAML files in this "
                "experiment's optimizer_configs folder. Empty when offline "
                "or the experiment has none (Start stays disabled)."
            ),
            self.shots_per_step: (
                "How many shots to take at each scan position / grid point "
                "(or in total for a no-scan or background run)."
            ),
            self.acquisition_combo: (
                "free_run: the trigger runs at the machine rate and device "
                "rows are matched by timestamp afterwards. strict: the scan "
                "fires each shot itself and waits for every device — "
                "slower, but nothing is ever missing."
            ),
            self.shot_count_label: (
                "Total shots this form implies (positions × shots per "
                "step). Above the runaway-scan limit, Start is disabled."
            ),
            self.description_edit: (
                "Free-text note about this scan — it ends up in the scan's "
                "metadata and the experiment log."
            ),
            # R4 presets
            self.preset_combo: (
                "Saved scan requests for this experiment (one YAML each in "
                "the configs repo's presets folder)."
            ),
            self.apply_button: (
                "Load the selected preset into the form. Anything the form "
                "cannot express (action bindings, position lists) is "
                "refused, leaving the form untouched."
            ),
            self.save_as_button: (
                "Save the current form as a named preset — the exact scan "
                "request it would submit."
            ),
            self.delete_button: "Delete the selected preset's YAML file.",
            # R5 submit row
            self.start_button: (
                "Build the scan request from this form and hand it to the "
                "scan engine. Needs a valid shot count and at least one "
                "selected save set; engine refusals show in the status bar."
            ),
            self.stop_button: (
                "Ask the engine to stop the running scan at the next safe "
                "point (closeout actions still run)."
            ),
            # R6 now panel
            self.state_pill: "What the scan engine is doing right now.",
            self.progress_bar: (
                "Shots completed out of the running scan's announced total."
            ),
            self.scan_number_label: (
                "The scan folder claimed by the running scan; '(previous)' "
                "is the last scan found in today's data folder."
            ),
            self.log_tail: "The most recent scan-engine and console messages.",
            # R7 device panel
            self.device_combo: (
                "Type or pick 'DeviceName:Variable Name' to watch its live "
                "readback from the gateway (updates on commit, not per "
                "keystroke)."
            ),
            self.readback_label: (
                "Live readback of the selected device variable, streamed "
                "from the gateway."
            ),
            self.set_field: (
                "Value to write to the selected device variable — a number, "
                "or a word the device understands (e.g. 'on')."
            ),
            self.set_button: (
                "Write the value via the gateway setpoint and report the "
                "outcome in the status bar. Disabled while a write is in "
                "flight."
            ),
        }
        for widget, text in tooltips.items():
            widget.setToolTip(text)

    def _build_menus(self) -> None:
        """Create the menu bar (Ops / Actions / Editors / Preferences / Help).

        Every created ``QMenu`` is kept in ``self._menus`` — PySide6 can
        garbage-collect the Python wrapper returned by ``addMenu`` and take
        the C++ menu (and its actions) down with it.
        """
        self._menus: list = []
        ops = self.menuBar().addMenu("Ops")
        self._menus.append(ops)
        for text, handler in (
            ("Open experiment config folder", self._on_open_experiment_configs),
            ("Open user config (config.ini)", self._on_open_user_config),
            ("Open today's scan folder", self._on_open_todays_scans),
        ):
            action = ops.addAction(text)
            action.triggered.connect(handler)
        ops.addSeparator()
        github = ops.addAction("GEECS-Plugins on GitHub")
        github.triggered.connect(self._on_open_github)

        # Actions menu: the arming switch on top, then one entry per action
        # plan in the current experiment's library (filled asynchronously by
        # _apply_action_names).  The arming state is deliberately NOT
        # persisted — a fresh session must never start armed, so every
        # launch begins with execution off and preview-only dialogs.
        actions_menu = self.menuBar().addMenu("Actions")
        self._menus.append(actions_menu)
        self._actions_menu = actions_menu
        self.enable_actions_action = actions_menu.addAction("Enable action execution")
        self.enable_actions_action.setCheckable(True)
        self.enable_actions_action.setChecked(False)
        self.enable_actions_action.toggled.connect(self._on_enable_actions_toggled)
        actions_menu.addSeparator()
        self._action_plan_actions: list = []
        self._set_action_names([])

        editors = self.menuBar().addMenu("Editors")
        self._menus.append(editors)
        self._editor_actions = []
        for text, handler in (
            ("Save Elements…", self._on_edit_save_sets),
            ("Scan Variables…", self._on_edit_scan_variables),
            ("Shot Control…", self._on_edit_shot_control),
            ("Action Library…", self._on_edit_action_library),
        ):
            action = editors.addAction(text)
            action.triggered.connect(handler)
            self._editor_actions.append(action)

        prefs = self.menuBar().addMenu("Preferences")
        self._menus.append(prefs)
        self.beep_action = prefs.addAction("Per-shot beep")
        self.beep_action.setCheckable(True)
        self.beep_action.setChecked(bool(self._settings.per_shot_beep))
        self.beep_action.toggled.connect(self._on_per_shot_beep_toggled)
        self.random_beep_action = prefs.addAction("Randomized beeps")
        self.random_beep_action.setCheckable(True)
        self.random_beep_action.setChecked(bool(self._settings.randomized_beeps))
        self.random_beep_action.toggled.connect(self._on_randomized_beeps_toggled)
        prefs.addSeparator()
        # Tooltips default on (discoverability); an experienced operator
        # turns them off here.  One application-level suppressor covers the
        # main window and every editor dialog, installed only while off.
        self.show_tooltips_action = prefs.addAction("Show tooltips")
        self.show_tooltips_action.setCheckable(True)
        self.show_tooltips_action.setChecked(bool(self._settings.show_tooltips))
        self.show_tooltips_action.toggled.connect(self._on_show_tooltips_toggled)
        self._tooltip_suppressor = ToolTipSuppressor(self)
        self._tooltip_suppressor_installed = False
        if not self._settings.show_tooltips:
            self._set_tooltips_shown(False)

        help_menu = self.menuBar().addMenu("Help")
        self._menus.append(help_menu)
        about = help_menu.addAction(f"GEECS Console {console_version()}")
        about.setEnabled(False)

    def _build_status_bar(self) -> None:
        """Create the status bar: gateway addr, configs path, version."""
        gateway = os.environ.get("EPICS_CA_ADDR_LIST", "unset")
        self._status_gateway = QLabel(f"gateway: {gateway}")
        self._status_configs = QLabel("configs: —")
        self._status_version = QLabel(f"v{console_version()}")
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
        self.optimization_combo.currentTextChanged.connect(self._refresh_submit_enabled)
        self.add_button.clicked.connect(self._on_add_save_set)
        self.remove_button.clicked.connect(self._on_remove_save_set)
        self.experiment_combo.currentTextChanged.connect(self._on_experiment_changed)
        self.trigger_profile_combo.currentTextChanged.connect(
            self._on_trigger_profile_changed
        )
        self.start_button.clicked.connect(self._on_start_clicked)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.apply_button.clicked.connect(self._on_preset_apply)
        self.save_as_button.clicked.connect(self._on_preset_save_as)
        self.delete_button.clicked.connect(self._on_preset_delete)

        # R7 device panel.  Selection commits (dropdown pick / Enter / focus
        # leave) resubscribe the readback; per-keystroke edits only regate the
        # Set button — never churn CA monitors while the operator types.
        self.set_button.clicked.connect(self._on_device_set_clicked)
        self.set_field.textChanged.connect(self._refresh_device_set_enabled)
        self.set_field.returnPressed.connect(self._on_device_set_clicked)
        self.device_combo.editTextChanged.connect(self._refresh_device_set_enabled)
        self.device_combo.activated.connect(self._resubscribe_device)
        device_line_edit = self.device_combo.lineEdit()
        if device_line_edit is not None:
            device_line_edit.editingFinished.connect(self._resubscribe_device)
        # Force a queued connection: the signal is emitted off the GUI thread
        # (the CA monitor thread), and an undecorated direct delivery would
        # paint widgets there (hard crash).
        self.device_value_ready.connect(
            self._apply_device_value, Qt.ConnectionType.QueuedConnection
        )
        # Device-set completion, the R7 completions, and the R6 idle
        # scan-number peek: each result is emitted by a BackgroundResult
        # worker (never by the window itself — a daemon-thread emit on a
        # window-owned signal races teardown, issue #510) and delivered
        # queued to its GUI-thread apply slot.
        self._device_set_worker = BackgroundResult()
        self._device_set_worker.result_ready.connect(
            self._apply_device_set_result, Qt.ConnectionType.QueuedConnection
        )
        self._completions_worker = BackgroundResult()
        self._completions_worker.result_ready.connect(
            self._apply_device_completions, Qt.ConnectionType.QueuedConnection
        )
        self._idle_scan_worker = BackgroundResult()
        self._idle_scan_worker.result_ready.connect(
            self._apply_idle_scan_number, Qt.ConnectionType.QueuedConnection
        )
        self._actions_worker = BackgroundResult()
        self._actions_worker.result_ready.connect(
            self._apply_action_names, Qt.ConnectionType.QueuedConnection
        )

        self.events.state_changed.connect(self._on_scan_state)
        self.events.totals_known.connect(self._on_totals_known)
        self.events.scan_number_known.connect(self.set_scan_number)
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
        # Optimizer configs (R3): keep the selection when the name survives
        # the repopulation; offline the listing is empty, which leaves the
        # combo empty and Start disabled in optimization mode.
        current_optimization = self.optimization_combo.currentText()
        self.optimization_combo.blockSignals(True)
        self.optimization_combo.clear()
        self.optimization_combo.addItems(listing.optimization_configs)
        self.optimization_combo.setCurrentIndex(
            self.optimization_combo.findText(current_optimization)
        )
        self.optimization_combo.blockSignals(False)
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
        self._refresh_presets()
        self._refresh_union_preview()
        self._refresh_shot_count()
        self._refresh_editor_actions()

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
        """Stop background I/O cleanly before closing — never joins a thread.

        Stops the GUI-thread health interval timer (no further polls),
        unsubscribes the device-panel readback monitor (non-blocking), and
        disconnects every cross-thread signal so a still-running daemon
        poll/put/monitor can't paint a widget on a window being torn down.
        In-flight daemon threads finish on their own without blocking
        shutdown.
        """
        timer = getattr(self, "_health_timer", None)
        if timer is not None:
            timer.stop()
        # A closed window must not keep suppressing application tooltips
        # (Qt would also drop the filter when the window is destroyed, but
        # close-without-destroy is the common test teardown shape).
        if getattr(self, "_tooltip_suppressor_installed", False):
            self._set_tooltips_shown(True)
        poller = getattr(self, "_health_poller", None)
        if poller is not None:
            try:
                poller.report_ready.disconnect(self._apply_health_report)
            except (RuntimeError, TypeError):
                pass
        backend = getattr(self, "_device_panel", None)
        if backend is not None:
            try:
                backend.unsubscribe()
            except Exception:  # noqa: BLE001 — teardown must not raise
                pass
        for worker, slot in (
            (
                getattr(self, "_completions_worker", None),
                self._apply_device_completions,
            ),
            (getattr(self, "_idle_scan_worker", None), self._apply_idle_scan_number),
            (getattr(self, "_device_set_worker", None), self._apply_device_set_result),
            (getattr(self, "_actions_worker", None), self._apply_action_names),
        ):
            if worker is None:
                continue
            try:
                worker.result_ready.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        try:
            self.device_value_ready.disconnect(self._apply_device_value)
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
            In optimization mode this resolves the selected optimizer
            config's name into its loaded spec (the one place the form
            snapshot reads a file), keeping the request builder pure.

        Raises
        ------
        pydantic.ValidationError
            When the widgets hold an invalid combination (e.g. an empty
            scan-variable name in a step mode).
        ConsoleFormError
            When the selected optimizer config cannot be loaded.
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
        optimization = None
        if mode is ConsoleMode.OPTIMIZATION:
            optimization = self._load_selected_optimization()
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
            optimization=optimization,
        )

    def _load_selected_optimization(self):
        """Resolve the R3 optimizer-config selection into its loaded spec.

        Returns
        -------
        geecs_schemas.OptimizationSpec or None
            The selected config's spec; ``None`` with nothing selected
            (:func:`build_scan_request` then refuses with a clear message).

        Raises
        ------
        ConsoleFormError
            When the selected config exists in the combo but cannot be
            loaded (missing file, bad YAML, schema rejection).
        """
        name = self.optimization_combo.currentText()
        if not name:
            return None
        try:
            return self._configs.optimization_spec(name)
        except Exception as exc:  # ConsoleConfigsError, or a fake's failure
            raise ConsoleFormError(
                f"Cannot load optimizer config {name!r}: {exc}"
            ) from exc

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
        # The optimizer-config row only exists for optimization mode.
        optimize = mode is ConsoleMode.OPTIMIZATION
        self.optimization_label.setVisible(optimize)
        self.optimization_combo.setVisible(optimize)
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
        self._presets.set_experiment(experiment)
        self._action_store.set_experiment(experiment)
        self._settings.last_experiment = experiment
        self._push_experiment_to_probe(experiment)
        self._populate_from_configs()
        # The readback PV is experiment-prefixed — re-point the monitor.
        self._resubscribe_device()
        # New experiment: new device list, new daily scans folder, and a
        # new action-plan library.
        self._start_device_completions_fetch()
        self._start_idle_scan_probe()
        self._start_actions_fetch()

    def _restore_last_experiment(self) -> None:
        """Select the remembered experiment at startup (explicit choice wins).

        Runs once from the constructor, after the combo is populated.  A
        remembered experiment is restored only when nothing was selected
        explicitly and the name is still in the combo's list; selecting it
        fires :meth:`_on_experiment_changed`, so configs, presets, the
        health probe, and the device panel all follow.
        """
        if self._configs.experiment:
            return
        remembered = self._settings.last_experiment
        if not remembered or self.experiment_combo.findText(remembered) < 0:
            return
        self.experiment_combo.setCurrentText(remembered)

    def _on_trigger_profile_changed(self, profile: str) -> None:
        """Repopulate the variant combo for the selected trigger profile."""
        self.trigger_variant_combo.clear()
        if profile:
            self.trigger_variant_combo.addItem("")
            self.trigger_variant_combo.addItems(self._configs.trigger_variants(profile))

    # ------------------------------------------------------------------
    # Ops menu (path resolution lives in services/ops_paths — pure & tested)
    # ------------------------------------------------------------------

    def _open_local_path(self, path: Path) -> None:
        """Open *path* in the platform file browser (Finder / Explorer).

        Parameters
        ----------
        path : Path
            An existing file or directory.
        """
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _on_open_experiment_configs(self) -> None:
        """Ops: open the current experiment's configs-repo directory."""
        folder = ops_paths.experiment_configs_folder(
            self.experiment_combo.currentText()
        )
        if folder is None:
            self._report("Experiment config folder not found — no configs repo.")
            return
        self._open_local_path(folder)

    def _on_open_user_config(self) -> None:
        """Ops: open the shared user ``config.ini`` (or its folder if absent)."""
        target = ops_paths.user_config_target()
        if target is None:
            self._report(f"User config not found: {ops_paths.USER_CONFIG_PATH}")
            return
        if target.is_dir():
            self._report("config.ini not found — opening its folder instead.")
        self._open_local_path(target)

    def _on_open_todays_scans(self) -> None:
        """Ops: open today's daily ``scans/`` folder — strictly read-only.

        The scanner side is the only producer of scan folders; when today's
        folder does not exist yet this reports "no scans today" and creates
        nothing (repo scan-folder invariant).
        """
        folder = ops_paths.todays_scan_folder(self.experiment_combo.currentText())
        if folder is None:
            self._report(
                "Cannot resolve today's scan folder — no data root or experiment."
            )
            return
        if not folder.is_dir():
            self._report("No scans today — the daily folder does not exist yet.")
            return
        self._open_local_path(folder)

    def _on_open_github(self) -> None:
        """Ops: open the GEECS-Plugins GitHub page in the browser."""
        QDesktopServices.openUrl(QUrl(ops_paths.GITHUB_URL))

    # ------------------------------------------------------------------
    # Editors menu (each entry point shows a non-modal dialog and returns it)
    # ------------------------------------------------------------------

    def _refresh_editor_actions(self) -> None:
        """Enable the Editors-menu actions only when an experiment is selected."""
        enabled = bool(self.experiment_combo.currentText())
        for action in self._editor_actions:
            action.setEnabled(enabled)

    def _open_editor(self, opener: Callable[..., object]) -> None:
        """Open one editor for the current experiment, holding a reference.

        The ``open_*_editor`` entry points show their dialog non-modally
        (``show()``, not ``exec()``) and return it; an unreferenced PySide6
        wrapper would be garbage-collected — taking the C++ dialog down with
        it — so every opened editor is kept in ``self._open_editors``
        (closed ones are pruned on the next open).

        Parameters
        ----------
        opener : callable
            One of the four ``open_*_editor`` entry points, called as
            ``opener(self, experiment=<current>)``.
        """
        experiment = self.experiment_combo.currentText()
        if not experiment:
            self._report("Select an experiment before opening an editor.")
            return
        dialog = opener(self, experiment=experiment)
        self._open_editors = [d for d in self._open_editors if d.isVisible()]
        self._open_editors.append(dialog)

    def _on_edit_save_sets(self) -> None:
        """Editors: open the save-set editor for the current experiment."""
        self._open_editor(open_save_set_editor)

    def _on_edit_scan_variables(self) -> None:
        """Editors: open the scan-variable editor for the current experiment."""
        self._open_editor(open_scan_variable_editor)

    def _on_edit_shot_control(self) -> None:
        """Editors: open the trigger-profile editor for the current experiment."""
        self._open_editor(open_shot_control_editor)

    def _on_edit_action_library(self) -> None:
        """Editors: open the action-library editor for the current experiment."""
        self._open_editor(open_action_library_editor)

    # ------------------------------------------------------------------
    # Actions menu (G-actions v1: arming switch + per-plan Run/Preview)
    # ------------------------------------------------------------------

    def _start_actions_fetch(self) -> None:
        """Fetch the experiment's action-plan names off the GUI thread.

        ``list_names`` parses the library YAML on a possibly slow configs
        mount, so it runs on a short-lived daemon thread (via the
        :class:`BackgroundResult` worker), tagged with the experiment and
        delivered queued to :meth:`_apply_action_names`.  No experiment
        answers inline with an empty list, spawning no thread.
        """
        experiment = self.experiment_combo.currentText()
        if not experiment:
            self._apply_action_names((experiment, []))
            return
        store = self._action_store

        def fetch() -> tuple[str, list[str]]:
            return (experiment, store.list_names())

        self._actions_worker.run_async(fetch, name="console-action-names")

    @Slot(object)
    def _apply_action_names(self, payload: object) -> None:
        """Rebuild the Actions-menu entries (GUI-thread slot, delivered queued).

        Parameters
        ----------
        payload : tuple
            ``(experiment, [plan_name, ...])``; a result tagged with an
            experiment that is no longer selected is dropped (a stale fetch
            racing an experiment change).
        """
        experiment, names = payload
        if experiment != self.experiment_combo.currentText():
            return
        self._set_action_names(list(names))

    def _set_action_names(self, names: list[str]) -> None:
        """Replace the per-plan menu entries below the arming switch.

        Parameters
        ----------
        names : list of str
            The experiment's action-plan names; empty (or offline) renders
            one disabled ``(no actions)`` entry.
        """
        for action in self._action_plan_actions:
            self._actions_menu.removeAction(action)
            action.deleteLater()
        self._action_plan_actions = []
        if not names:
            placeholder = self._actions_menu.addAction("(no actions)")
            placeholder.setEnabled(False)
            self._action_plan_actions.append(placeholder)
            return
        for name in names:
            action = self._actions_menu.addAction(name)
            action.triggered.connect(
                lambda checked=False, plan=name: self._on_action_plan(plan)
            )
            self._action_plan_actions.append(action)

    def _on_enable_actions_toggled(self, checked: bool) -> None:
        """Push the arming state into every open action dialog.

        Deliberately **not** persisted (unlike the Preferences toggles): a
        fresh console session must never start with action execution armed.

        Parameters
        ----------
        checked : bool
            The "Enable action execution" state.
        """
        self._open_action_dialogs = [
            d for d in self._open_action_dialogs if d.isVisible()
        ]
        for dialog in self._open_action_dialogs:
            dialog.set_execution_enabled(checked)

    def _on_action_plan(self, name: str) -> None:
        """Open the Run/Preview dialog for action plan *name*.

        The dialog is shown non-modally and kept in
        ``self._open_action_dialogs`` (the PySide6 GC hazard).  Preview is
        always available; Run is gated by the arming switch.

        Parameters
        ----------
        name : str
            The action-plan name (a menu entry).
        """
        submitter = self._ensure_submitter()
        if submitter is None:
            return
        dialog = ActionRunDialog(
            self,
            name,
            describe=submitter.describe_action,
            run=submitter.run_action,
            execution_enabled=self.enable_actions_action.isChecked(),
            report=self._report,
        )
        self._open_action_dialogs = [
            d for d in self._open_action_dialogs if d.isVisible()
        ]
        self._open_action_dialogs.append(dialog)
        dialog.show()

    # ------------------------------------------------------------------
    # Preferences (beeps)
    # ------------------------------------------------------------------

    def _on_per_shot_beep_toggled(self, checked: bool) -> None:
        """Persist the per-shot beep preference.

        Parameters
        ----------
        checked : bool
            The action's new checked state.
        """
        self._settings.per_shot_beep = checked

    def _on_randomized_beeps_toggled(self, checked: bool) -> None:
        """Persist the randomized-beeps preference.

        Parameters
        ----------
        checked : bool
            The action's new checked state.
        """
        self._settings.randomized_beeps = checked

    def _on_show_tooltips_toggled(self, checked: bool) -> None:
        """Persist the tooltip preference and apply it.

        Parameters
        ----------
        checked : bool
            The action's new checked state (``True`` shows tooltips).
        """
        self._settings.show_tooltips = checked
        self._set_tooltips_shown(checked)

    def _set_tooltips_shown(self, shown: bool) -> None:
        """Install or remove the application-wide tooltip suppressor.

        The suppressor is present on the ``QApplication`` only while
        tooltips are off, so the default-on path adds no per-event filter
        overhead.  Idempotent via the installed flag.

        Parameters
        ----------
        shown : bool
            ``True`` shows tooltips (suppressor removed).
        """
        app = QApplication.instance()
        if app is None:
            return
        if shown and self._tooltip_suppressor_installed:
            app.removeEventFilter(self._tooltip_suppressor)
            self._tooltip_suppressor_installed = False
        elif not shown and not self._tooltip_suppressor_installed:
            app.installEventFilter(self._tooltip_suppressor)
            self._tooltip_suppressor_installed = True

    def _maybe_beep(self) -> None:
        """Sound one per-shot beep, honoring the Preferences options.

        Silent when "Per-shot beep" is off; with "Randomized beeps" on, only
        ~1 in 4 shots beep (:data:`_RANDOM_BEEP_PROBABILITY`, drawn from the
        injectable ``rng``).  ``QApplication.beep()`` — no sound assets, no
        multimedia dependency.
        """
        if not self.beep_action.isChecked():
            return
        if (
            self.random_beep_action.isChecked()
            and self._beep_rng.random() >= _RANDOM_BEEP_PROBABILITY
        ):
            return
        QApplication.beep()

    # ------------------------------------------------------------------
    # R5 submit row
    # ------------------------------------------------------------------

    def _scanning(self) -> bool:
        """Whether the submitter reports an active scan."""
        return self._submitter is not None and self._submitter.is_scanning_active()

    def _refresh_submit_enabled(self) -> None:
        """Recompute Start/Stop enabled state from form + engine.

        Optimization mode additionally needs a selected optimizer config —
        that is the only optimize-specific gate; whether the engine accepts
        an optimize submission is the engine's call, surfaced from
        :meth:`_on_start_clicked` rather than pre-blocked here.
        """
        scanning = self._scanning()
        ready = (
            not scanning
            and self._shot_count_valid
            and bool(self.selected_save_sets())
            and (
                self.current_mode() is not ConsoleMode.OPTIMIZATION
                or bool(self.optimization_combo.currentText())
            )
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
    # R4 presets (a preset IS a saved ScanRequest)
    # ------------------------------------------------------------------

    def _report(self, message: str) -> None:
        """Show *message* in the status bar and append it to the log tail.

        Parameters
        ----------
        message : str
            The operator-facing line.
        """
        self.statusBar().showMessage(message, 10_000)
        self.append_log(message)

    def _refresh_presets(self) -> None:
        """Repopulate the R4 combo from the store, keeping the selection."""
        current = self.preset_combo.currentText()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItems(self._presets.list_names())
        # findText returns -1 when the previous selection is gone, which
        # setCurrentIndex(-1) renders as "nothing selected" — exactly right.
        self.preset_combo.setCurrentIndex(self.preset_combo.findText(current))
        self.preset_combo.blockSignals(False)

    def _on_preset_save_as(self) -> None:
        """R4 Save-as: current form → ScanRequest → named YAML in the store."""
        try:
            request = build_scan_request(self.form_state())
        except (ConsoleFormError, ValidationError) as exc:
            self._report(f"Cannot save preset: {exc}")
            return
        name, accepted = QInputDialog.getText(
            self,
            "Save preset",
            "Preset name:",
            text=self.preset_combo.currentText(),
        )
        name = name.strip()
        if not accepted or not name:
            return
        try:
            self._presets.save(name, request)
        except PresetStoreError as exc:
            self._report(f"Cannot save preset: {exc}")
            return
        self._refresh_presets()
        self.preset_combo.setCurrentText(name)
        self._report(f"Saved preset {name!r}.")

    def _on_preset_apply(self) -> None:
        """R4 Apply: load the selected preset and populate the form from it.

        Anything unloadable or inexpressible on the form (a missing or
        invalid file, an optimize preset, action bindings, explicit position
        lists, more than two axes) reports a status-bar error and leaves the
        form untouched.
        """
        name = self.preset_combo.currentText()
        if not name:
            self._report("No preset selected.")
            return
        try:
            form = form_state_from_request(self._presets.load(name))
            self._apply_form_state(form)
        except (PresetStoreError, ConsoleFormError) as exc:
            self._report(f"Cannot apply preset {name!r}: {exc}")
            return
        self._report(f"Applied preset {name!r}.")

    def _on_preset_delete(self) -> None:
        """R4 Delete: remove the selected preset and refresh the combo."""
        name = self.preset_combo.currentText()
        if not name:
            self._report("No preset selected.")
            return
        try:
            self._presets.delete(name)
        except PresetStoreError as exc:
            self._report(f"Cannot delete preset {name!r}: {exc}")
            return
        self._refresh_presets()
        self._report(f"Deleted preset {name!r}.")

    def _apply_form_state(self, form: ConsoleFormState) -> None:
        """Populate the R1–R3 form widgets from *form* (the Apply inverse).

        Validates everything the widgets cannot express **before** touching
        any of them, so a failed apply leaves the form exactly as it was.

        Parameters
        ----------
        form : ConsoleFormState
            The form snapshot to render (usually from
            :func:`form_state_from_request`).

        Raises
        ------
        ConsoleFormError
            More than two axes (the form has two axis rows), an axis with
            an explicit values list (the form only shows start/stop/step),
            or an optimization spec matching none of the experiment's
            optimizer configs (the form shows a config *name*, not a spec).
        """
        if len(form.axes) > 2:
            raise ConsoleFormError(
                f"it sweeps {len(form.axes)} axes — the form has two axis rows."
            )
        for axis in form.axes:
            if axis.values is not None:
                raise ConsoleFormError(
                    f"axis {axis.variable!r} uses an explicit position list, "
                    "which the form cannot show (start/stop/step only)."
                )
        optimization_name = ""
        if form.mode is ConsoleMode.OPTIMIZATION and form.optimization is not None:
            optimization_name = self._match_optimization_config(form.optimization)

        radio = {
            ConsoleMode.NOSCAN: self.radio_noscan,
            ConsoleMode.ONE_D: self.radio_1d,
            ConsoleMode.GRID: self.radio_grid,
            ConsoleMode.OPTIMIZATION: self.radio_optimization,
            ConsoleMode.BACKGROUND: self.radio_background,
        }[form.mode]
        radio.setChecked(True)

        axis_rows = (
            (self.variable_combo, self.start_spin, self.stop_spin, self.step_spin),
            (self.variable2_combo, self.start2_spin, self.stop2_spin, self.step2_spin),
        )
        for axis, (combo, start, stop, step) in zip(form.axes, axis_rows):
            combo.setCurrentText(axis.variable)
            start.setValue(axis.start)
            stop.setValue(axis.stop)
            step.setValue(axis.step)

        if optimization_name:
            self.optimization_combo.setCurrentText(optimization_name)
        self.shots_per_step.setValue(form.shots_per_step)
        self.acquisition_combo.setCurrentText(form.acquisition.value)
        self.description_edit.setText(form.description)
        self.trigger_profile_combo.setCurrentText(form.trigger_profile or "")
        # Changing the profile text repopulated the variant combo (the
        # currentTextChanged handler); now pick the preset's variant.
        self.trigger_variant_combo.setCurrentText(form.trigger_variant or "")
        self._apply_save_sets(form.save_sets)
        self._refresh_shot_count()

    def _match_optimization_config(self, spec: object) -> str:
        """Find the listed optimizer config whose loaded spec equals *spec*.

        The form expresses optimization as a config *name* (the R3 combo),
        so applying a preset that carries an inline spec means finding the
        experiment's config with identical content — pydantic equality over
        the loaded documents.  Unloadable configs are skipped.

        Parameters
        ----------
        spec : OptimizationSpec
            The preset's optimization block.

        Returns
        -------
        str
            The matching config name from the combo.

        Raises
        ------
        ConsoleFormError
            When no listed config matches — selecting anything else would
            silently change what the preset submits.
        """
        for index in range(self.optimization_combo.count()):
            name = self.optimization_combo.itemText(index)
            try:
                if self._configs.optimization_spec(name) == spec:
                    return name
            except Exception as exc:  # noqa: BLE001 — an unloadable config just can't match
                logger.info(
                    "optimizer config %r unloadable while matching: %s", name, exc
                )
        raise ConsoleFormError(
            "its optimization spec matches none of this experiment's "
            "optimizer configs — the form can only show a named config."
        )

    def _apply_save_sets(self, names: list[str]) -> None:
        """Make the R2 selected list exactly *names* (known ones, in order).

        Save-set names the current experiment's configs don't list are
        skipped with a status-bar warning — putting an unresolvable name in
        the selected list would only fail later, at submission.

        Parameters
        ----------
        names : list of str
            The preset's save-set names, in list order.
        """
        known = set(self.selected_save_sets()) | {
            self.available_list.item(row).text()
            for row in range(self.available_list.count())
        }
        selected = [name for name in names if name in known]
        missing = [name for name in names if name not in known]
        self.selected_list.clear()
        self.selected_list.addItems(selected)
        self.available_list.clear()
        self.available_list.addItems(sorted(known - set(selected)))
        if missing:
            self._report(
                "Preset save sets not in this experiment's configs "
                f"(skipped): {', '.join(missing)}"
            )
        self._refresh_union_preview()
        self._refresh_submit_enabled()

    # ------------------------------------------------------------------
    # R7 device panel
    # ------------------------------------------------------------------

    def _refresh_device_set_enabled(self) -> None:
        """Gate the Set button: valid ``device:variable``, a value, no put in flight."""
        ready = (
            not self._device_set_in_flight
            and parse_device_variable(self.device_combo.currentText()) is not None
            and bool(self.set_field.text().strip())
        )
        self.set_button.setEnabled(ready)

    def _warn_device_format(self) -> None:
        """Status-bar hint for an unparsable R7 device selection."""
        self.statusBar().showMessage("Device format: DeviceName:Variable Name", 10_000)

    def _start_device_completions_fetch(self) -> None:
        """Fetch R7 ``device:variable`` completions off the GUI thread.

        Mirrors the editors' completions pattern: one blocking provider call
        on a short-lived daemon thread (via the :class:`BackgroundResult`
        worker), marshaled back queued to :meth:`_apply_device_completions`
        — never on the GUI thread.  No experiment (or the
        :class:`EmptyCompletions` default in tests) answers inline with an
        empty list, spawning no thread.
        """
        experiment = self.experiment_combo.currentText()
        factory = (
            self._completions_factory
            if self._completions_factory is not None
            else _default_completions_factory
        )
        provider = factory(experiment) if experiment else EmptyCompletions()
        if isinstance(provider, EmptyCompletions):
            self._apply_device_completions((experiment, []))
            return

        def fetch() -> tuple[str, list[str]]:
            try:
                mapping = provider.device_variables()
            except Exception as exc:  # noqa: BLE001 — completions are best-effort
                logger.info("device completions failed: %s", exc)
                mapping = {}
            words = sorted(
                f"{device}:{variable}"
                for device, variables in (mapping or {}).items()
                for variable in variables
            )
            return (experiment, words)

        self._completions_worker.run_async(fetch, name="console-device-completions")

    @Slot(object)
    def _apply_device_completions(self, payload: object) -> None:
        """Populate the R7 combo's dropdown (GUI-thread slot, delivered queued).

        Parameters
        ----------
        payload : tuple
            ``(experiment, ["device:variable", ...])``; a result tagged with
            an experiment that is no longer selected is dropped (a stale
            fetch racing an experiment change).
        """
        experiment, words = payload
        if experiment != self.experiment_combo.currentText():
            return
        current = self.device_combo.currentText()
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        self.device_combo.addItems(list(words))
        self.device_combo.setCurrentIndex(-1)
        line_edit = self.device_combo.lineEdit()
        if line_edit is not None:
            line_edit.setText(current)
        self.device_combo.blockSignals(False)

    def _resubscribe_device(self, *_args: object) -> None:
        """Re-point the readback monitor at the combo's current selection.

        Closes the previous monitor first (the backend guarantees straggler
        callbacks from it are dropped) and resets the label to the em-dash
        until the new monitor's first value lands.  An unparsable selection
        leaves the panel unsubscribed.
        """
        try:
            self._device_panel.unsubscribe()
        except Exception as exc:  # noqa: BLE001 — teardown must not break the GUI
            self.append_log(f"Readback unsubscribe failed: {exc}")
        self.readback_label.setText("—")
        self._refresh_device_set_enabled()
        parsed = parse_device_variable(self.device_combo.currentText())
        if parsed is None:
            # Committed text that doesn't parse gets a visible hint — a
            # silent no-op looked like a dead panel (user report).
            if self.device_combo.currentText().strip():
                self._warn_device_format()
            return
        device, variable = parsed
        try:
            self._device_panel.subscribe(
                self.experiment_combo.currentText(),
                device,
                variable,
                self.device_value_ready.emit,
            )
        except Exception as exc:  # noqa: BLE001 — surface, don't crash
            message = f"Readback subscribe failed for {device}:{variable}: {exc}"
            self.statusBar().showMessage(message, 10_000)
            self.append_log(message)

    @Slot(object)
    def _apply_device_value(self, value: object) -> None:
        """Render one readback value (GUI-thread slot, delivered queued).

        Parameters
        ----------
        value : object
            The monitor value (float / int / string) from the backend.
        """
        self.readback_label.setText(format_readback(value))

    def _on_device_set_clicked(self) -> None:
        """Dispatch the blocking backend set through the set worker.

        The blocking ``backend.set`` runs on a short-lived daemon thread via
        the :class:`BackgroundResult` set worker, whose ``result_ready``
        signal delivers the ``(ok, message)`` outcome queued back to
        :meth:`_apply_device_set_result` — the worker owns the cross-thread
        emission, never the window (issue #510).  The dispatched callable
        catches every backend failure and returns it as a result, so the
        in-flight flag is always re-armed.
        """
        parsed = parse_device_variable(self.device_combo.currentText())
        text = self.set_field.text().strip()
        if parsed is None:
            if self.device_combo.currentText().strip():
                self._warn_device_format()
            return
        if not text or self._device_set_in_flight:
            return
        device, variable = parsed
        value = parse_set_value(text)
        experiment = self.experiment_combo.currentText()
        backend = self._device_panel

        def run_set() -> tuple[bool, str]:
            try:
                backend.set(experiment, device, variable, value)
            except Exception as exc:  # noqa: BLE001 — any failure is a status report
                return (False, f"Set {device}:{variable} failed: {exc}")
            return (True, f"Set {device}:{variable} = {value}")

        self._device_set_in_flight = True
        self._refresh_device_set_enabled()
        self._device_set_worker.run_async(run_set, name="console-device-set")

    @Slot(object)
    def _apply_device_set_result(self, payload: object) -> None:
        """Report one finished set and re-arm the button (GUI-thread slot).

        Parameters
        ----------
        payload : tuple
            ``(ok, message)`` from the set worker; ``ok`` is unused beyond
            the message — failures already carry the exception text.
        """
        _ok, message = payload
        self._device_set_in_flight = False
        self.statusBar().showMessage(message, 10_000)
        self.append_log(message)
        self._refresh_device_set_enabled()

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
        if text.startswith("Scan ") and not text.endswith("(previous)"):
            self.scan_number_label.setText(f"{text} (previous)")

    def _start_idle_scan_probe(self) -> None:
        """Peek at today's scans dir for the R6 idle display — read-only.

        The lookup resolves today's daily ``scans/`` folder and lists it for
        the highest existing ``ScanNNN`` — resolution and listdir only,
        never creating anything on that path (repo scan-folder invariant).
        The data root is typically a network mount, so the blocking lookup
        runs on a short-lived daemon thread (via the
        :class:`BackgroundResult` worker) and reports back queued to
        :meth:`_apply_idle_scan_number`.
        """
        experiment = self.experiment_combo.currentText()
        lookup = (
            self._scan_number_lookup
            if self._scan_number_lookup is not None
            else _idle_scan_lookup
        )

        def probe() -> tuple[str, Optional[int]]:
            try:
                number = lookup(experiment)
            except Exception as exc:  # noqa: BLE001 — a flaky mount is not a crash
                logger.info("idle scan-number lookup failed: %s", exc)
                number = None
            return (experiment, number)

        self._idle_scan_worker.run_async(probe, name="console-idle-scan-probe")

    @Slot(object)
    def _apply_idle_scan_number(self, payload: object) -> None:
        """Render the idle scan-number peek (GUI-thread slot, delivered queued).

        A live scan number on display (the 10 s expiry timer still running)
        is never clobbered, and a result tagged with an experiment that is
        no longer selected is dropped.

        Parameters
        ----------
        payload : tuple
            ``(experiment, int | None)`` from the daemon-thread probe.
        """
        experiment, number = payload
        if experiment != self.experiment_combo.currentText():
            return
        if self._scan_number_timer.isActive():
            return
        if number is None:
            self.scan_number_label.setText("No scans today")
        else:
            self.scan_number_label.setText(f"Scan {number:03d} (previous)")

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
        self._last_beep_shots = 0  # new scan: re-arm the per-shot beep

    def _on_progress(
        self, step_index: int, total_steps: int, shots_completed: int
    ) -> None:
        """Advance the progress bar from step events; beep on shot increments."""
        if self._total_shots:
            self.progress_bar.setValue(min(shots_completed, self._total_shots))
        elif total_steps:
            self.progress_bar.setMaximum(total_steps)
            self.progress_bar.setValue(min(step_index + 1, total_steps))
        if shots_completed > self._last_beep_shots:
            self._last_beep_shots = shots_completed
            self._maybe_beep()
        elif shots_completed < self._last_beep_shots:
            # A scan that never announced totals restarted the count.
            self._last_beep_shots = shots_completed

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
