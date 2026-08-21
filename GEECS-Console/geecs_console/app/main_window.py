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
import time
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QFile, Qt, QTimer, QUrl, Slot
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

from geecs_console.app.actions_menu import ActionsMenuController
from geecs_console.app.movable_panel import MovablePanelController
from geecs_console.app.now_panel import NowPanelController
from geecs_console.app.scan_monitor import ScanMonitorController
from geecs_console.editors.action_library_editor import open_action_library_editor
from geecs_console.editors.save_set_editor import open_save_set_editor
from geecs_console.editors.scan_variable_editor import open_scan_variable_editor
from geecs_console.editors.shot_control_editor import open_shot_control_editor
from geecs_console.services import ops_paths
from geecs_console.services.action_library_store import ActionLibraryStore
from geecs_console.app.tooltips import ToolTipSuppressor, apply_operator_tooltips
from geecs_console.services.background import BackgroundResult, HealthPoller
from geecs_console.services.device_completions import (
    CompletionsProvider,
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
)
from geecs_console.services.health import (
    HealthProbe,
    HealthReport,
    HealthStatus,
    StubHealth,
)
from geecs_console.services.queue_client import QueueStatus, SubmitResult
from geecs_console.services.submit_preflight import (
    PreflightReport,
    run_submit_preflight,
    stamp_submission,
)
from geecs_console.submission import Submitter, make_queue_submitter

logger = logging.getLogger(__name__)

_UI_PATH = Path(__file__).parent / "ui" / "main_window.ui"
_QSS_PATH = Path(__file__).parent / "style.qss"

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

# States that settle an in-flight Stop request, releasing the Stop button's
# "Stopping…" hold.  "done"/"aborted" come from the document stream's stop
# document (exit_status success / anything else); "idle" is the manager
# status poll's fallback for when the document stream is down — the worker
# RE returning to idle means the scan is over either way.  Deliberately NOT
# "paused" — that state is resumable, the scan is not over.
_TERMINAL_SCAN_STATES = frozenset({"aborted", "done", "idle"})

#: After a terminal document renders, live-state asserts from the status
#: poll are suppressed this long — a snapshot *taken* pre-stop but
#: *delivered* post-stop must not narrate the transition backwards (the
#: poll cadence is 1 s; two periods + slack).
_TERMINAL_GRACE_S = 2.5


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
        The scan service (a queueserver-manager client since #648); tests
        inject a fake.  When ``None`` one is built by *submitter_factory*
        at construction (cheap — no sockets until the first call).
    submitter_factory : callable, optional
        ``(experiment) -> Submitter``; defaults to
        :func:`~geecs_console.submission.make_queue_submitter`.
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
        self._submitter = submitter
        self._submitter_factory = (
            submitter_factory if submitter_factory is not None else make_queue_submitter
        )
        #: Latest manager status snapshot (the scan-monitor poll); the
        #: disconnected default until the first poll lands.
        self._queue_status = QueueStatus()
        #: The engine's recorded failed-move reason for the current pause
        #: ("" when the pause was operator-requested or none is known).
        self._pause_reason = ""
        #: Document-stream bookkeeping: descriptor uid → stream name, so
        #: per-shot progress counts only the primary stream's events.
        self._descriptor_names: dict = {}
        self._shot_count_valid = False
        self._beep_rng = rng if rng is not None else random.Random()
        self._last_beep_shots = 0
        self._completions_factory = completions_factory
        self._scan_number_lookup = scan_number_lookup
        #: Non-modal editor dialogs opened from the Editors menu.  PySide6
        #: garbage-collects an unreferenced dialog wrapper and tears down the
        #: C++ dialog with it, so every opened editor is kept here.
        self._open_editors: list = []

        self._apply_stylesheet()
        self._load_ui()
        self._bind_widgets()
        apply_operator_tooltips(self)
        self._build_menus()
        self._build_status_bar()
        self._wire_signals()

        self.setWindowTitle("GEECS Console")
        # R6 rendering lives on NowPanelController (app/now_panel.py); the
        # widgets stay window attributes (tests, tooltips), the lookup is
        # resolved at probe time so the module-level default stays
        # test-patchable.
        self._now = NowPanelController(
            state_pill=self.state_pill,
            progress_bar=self.progress_bar,
            scan_number_label=self.scan_number_label,
            log_tail=self.log_tail,
            current_experiment=lambda: self.experiment_combo.currentText(),
            resolve_lookup=lambda: (
                self._scan_number_lookup
                if self._scan_number_lookup is not None
                else _idle_scan_lookup
            ),
        )

        # First populate quietly: a remembered experiment restored on the
        # next line makes its "No experiment selected." a lie (it used to
        # flash for 10 s and stick in the log tail — user report).
        self._populate_from_configs(announce=False)
        if not self._restore_last_experiment() and self._listing_message:
            self._report(self._listing_message)
        # Chips read UNKNOWN until the first background poll returns; seed the
        # markup synchronously (no probe call — never touches the network).
        self._apply_health_report(HealthReport())
        self._push_experiment_to_probe(self._configs.experiment)
        self._start_health_poller()
        self._now.set_state_pill("idle")
        self._on_mode_changed()
        # Startup fetches for the selected experiment (no-ops when none):
        # R7 device:variable completions, the R6 idle scan-number peek, and
        # the Actions-menu plan names.  Restoring the last experiment already
        # fired the experiment-changed path (which starts all three); these
        # cover the explicit-experiment and no-experiment startups.  A
        # duplicate fetch is *result*-safe (stale results are dropped by
        # experiment tag) but NOT thread-safe in general: two concurrent
        # fetch threads can race the lazy first import of a native chain
        # (demonstrated in the actions fetch's geecs_bluesky chain, which
        # aborted the process — hence the one-in-flight dedupe in the
        # actions and now-panel controllers; the idle probe's own lazy
        # chain is geecs_data_utils/pandas).  The completions pair still
        # double-spawns and shares the hazard shape (mysql-connector
        # chain); dedupe it the same way if this ever bites.
        self._movable.start_completions_fetch()
        self._now.start_idle_probe()
        self._actions.start_fetch()
        self._start_scan_monitor()

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
        self.iterations_label: QLabel = self._child(QLabel, "r3_iterations_label")
        self.iterations_spin: QSpinBox = self._child(QSpinBox, "r3_iterations_spin")
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
        self.pause_button: QPushButton = self._child(QPushButton, "r5_pause_button")
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

        # Actions menu: contents and dialogs live on ActionsMenuController
        # (app/actions_menu.py); the window keeps the QMenu reference and
        # the controller composition.
        actions_menu = self.menuBar().addMenu("Actions")
        self._menus.append(actions_menu)
        self._actions = ActionsMenuController(
            actions_menu,
            window=self,
            store=self._action_store,
            current_experiment=lambda: self.experiment_combo.currentText(),
            ensure_submitter=self._ensure_submitter,
            report=self._report,
        )

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
        self.iterations_spin.valueChanged.connect(self._refresh_shot_count)
        self.optimization_combo.currentTextChanged.connect(
            self._on_optimization_config_changed
        )
        self.add_button.clicked.connect(self._on_add_save_set)
        self.remove_button.clicked.connect(self._on_remove_save_set)
        self.experiment_combo.currentTextChanged.connect(self._on_experiment_changed)
        self.trigger_profile_combo.currentTextChanged.connect(
            self._on_trigger_profile_changed
        )
        self.start_button.clicked.connect(self._on_start_clicked)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.pause_button.clicked.connect(self._on_pause_clicked)
        self.apply_button.clicked.connect(self._on_preset_apply)
        self.save_as_button.clicked.connect(self._on_preset_save_as)
        self.delete_button.clicked.connect(self._on_preset_delete)

        # R7 movable panel: selection, readback(s), completions, and manual
        # sets/moves all live on MovablePanelController (app/movable_panel.py)
        # — the window keeps the widget attributes and the composition (the
        # #534 controller shape; its workers and queued value signal are
        # controller-owned, per issue #510).
        self._movable = MovablePanelController(
            device_combo=self.device_combo,
            readback_label=self.readback_label,
            set_field=self.set_field,
            set_button=self.set_button,
            backend=self._device_panel,
            current_experiment=lambda: self.experiment_combo.currentText(),
            ensure_submitter=self._ensure_submitter,
            # getattr-guarded: the configs seam is duck-typed and test fakes
            # (or older injected configs) may not expose the catalog.
            catalog_specs=lambda: getattr(self._configs, "scan_variable_specs", dict)(),
            completions_provider=lambda experiment: (
                self._completions_factory
                if self._completions_factory is not None
                else _default_completions_factory
            )(experiment),
            report=self._report,
        )
        # R3 → R7 auto-select (the legacy scanner behavior): picking a scan
        # variable re-points the movable panel at it, composites included.
        # Commit-only (textActivated: dropdown pick / Enter) — NEVER
        # currentTextChanged: both R3 combos are editable, and a
        # per-keystroke connection would churn CA monitors and hijack the
        # panel while the operator types an axis name (review, PR #598).
        self.variable_combo.textActivated.connect(self._movable.select_from_scan_combo)
        self.variable2_combo.textActivated.connect(self._movable.select_from_scan_combo)
        # (The actions-menu fetch worker lives on ActionsMenuController;
        # the idle scan-number probe worker on NowPanelController.)
        # Stop dispatch (issue #571 shape, queue edition): stop_scan blocks —
        # from a running scan it sequences deferred-pause → stop, waiting
        # out an in-flight blocking move — so it runs on a worker, never
        # the GUI thread.  The terminal state (stop document, or the status
        # poll's idle) clears the in-flight hold in _on_scan_state.
        self._stop_worker = BackgroundResult()
        self._stop_worker.result_ready.connect(
            self._on_stop_result, Qt.ConnectionType.QueuedConnection
        )
        self._stop_in_flight = False
        self._stop_button_label = self.stop_button.text()
        # Submission worker: the pre-submit preflight (config + DB + CA
        # reads) and the queue submission (0MQ round trips) both block, so
        # each phase runs here; the phases hand off through queued results
        # (_on_submit_phase_done).
        self._submit_worker = BackgroundResult()
        self._submit_worker.result_ready.connect(
            self._on_submit_phase_done, Qt.ConnectionType.QueuedConnection
        )
        self._submit_in_flight = False
        #: The stamped request held across the pending-items question (a
        #: clear-and-retry resubmits it verbatim, no re-stamp).
        self._pending_submission = None
        self._start_button_label = self.start_button.text()
        #: Last scan state word (lowercase) — drives the Pause/Resume
        #: button and the stop hold.
        self._scan_state_text = "idle"
        #: When a terminal document last rendered (monotonic; arms the
        #: status poll's backwards-narration grace window).
        self._terminal_state_at = 0.0
        self._pause_button_label = self.pause_button.text()
        self._refresh_pause_button()

    # ------------------------------------------------------------------
    # Configs / health population
    # ------------------------------------------------------------------

    def _populate_from_configs(self, announce: bool = True) -> None:
        """Fill the combos and lists from the configs service (offline-safe).

        Parameters
        ----------
        announce : bool, optional
            Show the listing's message (or clear a stale one) in the status
            bar.  The constructor's *first* populate passes ``False``: it
            runs before the last-experiment restore, so its transient
            "No experiment selected." would flash — and sit in the log tail
            right above whatever the operator does next — even though an
            experiment is selected one line later (the constructor
            re-announces afterwards only if the restore selected nothing).
        """
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
        # Repopulation is programmatic — block signals so the R7 auto-select
        # only follows *operator* picks (and preset applies), never the
        # populate churn itself.
        for combo in (self.variable_combo, self.variable2_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(listing.scan_variables)
            combo.blockSignals(False)

        root = str(listing.configs_root) if listing.configs_root else "not found"
        self._status_configs.setText(f"configs: {root}")
        self._listing_message = listing.message
        if announce:
            if listing.message:
                self._report(listing.message)
            else:
                # A clean listing supersedes any earlier listing complaint
                # still sitting in the status bar (e.g. the no-experiment
                # message after the operator picks one).
                self.statusBar().clearMessage()
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

    def _start_scan_monitor(self) -> None:
        """Start the queueserver scan monitor (status poll + streams).

        Built from the submitter: its ``status()`` is the poll probe, and a
        real :class:`~geecs_console.submission.QueueSubmitter` carries the
        stream addresses (fakes without them get a poll-only monitor).  A
        submitter that cannot be built leaves the monitor off — the pill
        stays wherever the last state put it and submission reports the
        reason on Start.
        """
        submitter = self._ensure_submitter()
        if submitter is None:
            self._monitor = None
            return
        self._monitor = ScanMonitorController(
            submitter,
            info_addr=getattr(submitter, "info_addr", None),
            doc_addr=getattr(submitter, "doc_addr", None),
        )
        # Queued connections throughout — poll results and stream documents
        # arrive from daemon threads and must never paint widgets directly.
        self._monitor.status_ready.connect(
            self._on_queue_status, Qt.ConnectionType.QueuedConnection
        )
        if self._monitor.documents is not None:
            self._monitor.documents.document.connect(
                self._on_scan_document, Qt.ConnectionType.QueuedConnection
            )
        if self._monitor.console is not None:
            self._monitor.console.line.connect(
                self.append_log, Qt.ConnectionType.QueuedConnection
            )
            self._monitor.console.pause_reason.connect(
                self._on_pause_reason, Qt.ConnectionType.QueuedConnection
            )
        # Degraded mode must be visible (#654 review finding 2): a stream
        # that cannot set up says so in the status bar + log tail instead
        # of leaving progress/log silently empty.
        for worker in (self._monitor.documents, self._monitor.console):
            if worker is not None:
                worker.stream_failed.connect(
                    self._report, Qt.ConnectionType.QueuedConnection
                )
        self._monitor.start(self)

    @Slot(object)
    def _on_queue_status(self, status: QueueStatus) -> None:
        """Render one manager status snapshot (GUI-thread slot).

        The manager poll is the state pill's *fallback* narrator: the
        document stream announces the interesting transitions (start / stop
        documents → running / done / aborted), so this slot asserts the
        worker RE's live states (running/paused and the transitional
        pausing/stopping/… — rendered as-is), falls an active pill back to
        idle when the RE is idle (stream down, or another client's stop),
        and reads "unknown" when the manager is unreachable **or the
        worker environment is gone mid-scan** (``re_state`` ``None`` —
        the crash case must never leave a RUNNING pill lying; #654 review
        finding 1).  A snapshot taken before a stop document but delivered
        after it must not narrate the transition backwards, so live-state
        asserts are suppressed for a short grace window after a terminal
        document (#654 review finding 3).
        """
        self._queue_status = status
        pill = self._scan_state_text
        active_pill = pill not in ("idle", "done", "aborted", "unknown")
        if not status.connected or status.re_state is None:
            if pill != "unknown":
                self._on_scan_state("unknown")
                if status.connected and active_pill:
                    self._report(
                        "worker environment is down — scan state lost "
                        "(restart the worker; check its journal)"
                    )
            return
        re_state = status.re_state
        if re_state == "idle":
            if active_pill or pill == "unknown":
                self._on_scan_state("idle")
            else:
                # No state change — but Start/Stop gating may still depend
                # on the fresh snapshot (e.g. connectivity returning).
                self._refresh_submit_enabled()
            return
        # A live RE state (running/paused/pausing/stopping/…): assert it,
        # unless a terminal document just rendered — a pre-stop snapshot
        # arriving late would flip DONE/ABORTED back to RUNNING for a poll.
        if (
            pill in ("done", "aborted")
            and time.monotonic() - self._terminal_state_at < _TERMINAL_GRACE_S
        ):
            return
        if pill != re_state:
            self._on_scan_state(re_state)

    @Slot(str, object)
    def _on_scan_document(self, name: str, doc: dict) -> None:
        """Consume one bluesky document from the worker's stream (GUI thread).

        Start documents open the run (scan number, totals, narration),
        primary-stream events drive per-shot progress and the beep, and the
        stop document ends it (done/aborted per ``exit_status``).
        """
        if name == "start":
            self._descriptor_names = {}
            number = doc.get("scan_number")
            if number is not None:
                self.set_scan_number(int(number))
                self.append_log(f"scan running (Scan {int(number):03d})")
            else:
                self.append_log("scan running")
            num_points = doc.get("num_points")
            shots_per_step = doc.get("shots_per_step")
            if num_points and shots_per_step:
                self._on_totals_known(int(num_points) * int(shots_per_step))
            self._on_scan_state("running")
        elif name == "descriptor":
            self._descriptor_names[doc.get("uid")] = doc.get("name")
        elif name == "event":
            if self._descriptor_names.get(doc.get("descriptor")) == "primary":
                shots = int(doc.get("seq_num") or 0)
                self._on_progress(0, 0, shots)
        elif name == "stop":
            exit_status = str(doc.get("exit_status") or "")
            word = "done" if exit_status == "success" else "aborted"
            reason = str(doc.get("reason") or "").strip()
            line = f"scan {word}" + (f" — {reason}" if reason else "")
            self.append_log(line)
            self._on_scan_state(word)

    @Slot(str)
    def _on_pause_reason(self, reason: str) -> None:
        """Record and announce the engine's failed-move pause reason."""
        self._pause_reason = reason
        self._report(f"paused: {reason}")

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
        monitor = getattr(self, "_monitor", None)
        if monitor is not None:
            monitor.dispose()
        for worker_name in ("_stop_worker", "_submit_worker"):
            worker = getattr(self, worker_name, None)
            if worker is not None:
                try:
                    worker.result_ready.disconnect()
                except (RuntimeError, TypeError):
                    pass
        movable = getattr(self, "_movable", None)
        if movable is not None:
            movable.dispose()
        actions = getattr(self, "_actions", None)
        if actions is not None:
            actions.dispose()
        now = getattr(self, "_now", None)
        if now is not None:
            now.dispose()
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
        max_iterations = None
        if mode is ConsoleMode.OPTIMIZATION:
            optimization = self._load_selected_optimization()
            # The spinner's special value 0 renders as "auto" = no limit.
            max_iterations = self.iterations_spin.value() or None
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
            max_iterations=max_iterations,
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
        self.iterations_label.setVisible(optimize)
        self.iterations_spin.setVisible(optimize)
        # The union line is mode-aware (optimization notes the optimizer's
        # own provisioned diagnostics), so a mode flip must repaint it.
        self._refresh_union_preview()
        self._refresh_shot_count()

    def _on_optimization_config_changed(self, name: str) -> None:
        """Seed the Iterations spinner from the newly selected config.

        The spinner owns the submitted iteration count (see
        :class:`~geecs_console.request_builder.ConsoleFormState`), so a
        config's own ``max_iterations`` must surface *here* — otherwise it
        would be silently overridden at submission.  Best-effort: an
        unloadable config leaves the spinner alone (submission reports the
        load failure properly).

        Parameters
        ----------
        name : str
            The R3 optimizer-config combo's new text ("" for none).
        """
        if name:
            try:
                spec = self._configs.optimization_spec(name)
            except Exception as exc:  # noqa: BLE001 — seeding is best-effort
                logger.info(
                    "optimizer config %r unloadable while seeding: %s", name, exc
                )
            else:
                self.iterations_spin.setValue(
                    getattr(spec, "max_iterations", None) or 0
                )
        self._refresh_submit_enabled()

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
            mode=mode,
            axes=axes,
            shots_per_step=self.shots_per_step.value(),
            max_iterations=(
                self.iterations_spin.value() or None
                if mode is ConsoleMode.OPTIMIZATION
                else None
            ),
        )

    def _refresh_shot_count(self) -> None:
        """Recompute the live shot-count label and the runaway guard.

        Optimization mode shows ``iterations × shots per step`` when the
        Iterations spinner is set (the engine's announced upper bound —
        the suggester may stop early), or ``auto`` when it isn't; the
        runaway guard applies to the product like any other mode.
        """
        form = self._estimation_form()
        if form.mode is ConsoleMode.OPTIMIZATION and form.max_iterations is None:
            self._shot_count_valid = True
            self.shot_count_label.setText("total shots: auto")
            self._refresh_submit_enabled()
            return
        try:
            total = estimate_total_shots(form)
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
        """Update the R2 union line and role-conflict/reference hint.

        Optimization mode notes the optimizer's own contribution — the
        engine merges the optimizer config's ``device_requirements`` into
        the effective device set, so the save-set union alone undercounts
        what the scan will record (and zero selected sets still records
        the optimizer's diagnostics).
        """
        preview = self._configs.union_preview(self.selected_save_sets())
        optimize = self.current_mode() is ConsoleMode.OPTIMIZATION
        if preview.device_count is None:
            self.union_label.setText("union: —")
        elif optimize and not preview.device_count:
            self.union_label.setText("union: diagnostics from optimizer config")
        elif optimize:
            self.union_label.setText(
                f"union: {preview.device_count} devices + optimizer diagnostics"
            )
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
        # The readback PVs are experiment-prefixed — re-point the movable
        # panel's monitors and refresh its completions; then the new daily
        # scans folder and the new action-plan library.
        self._movable.on_experiment_changed()
        self._now.start_idle_probe()
        self._actions.start_fetch()

    def _restore_last_experiment(self) -> bool:
        """Select the remembered experiment at startup (explicit choice wins).

        Runs once from the constructor, after the combo is populated.  A
        remembered experiment is restored only when nothing was selected
        explicitly and the name is still in the combo's list; selecting it
        fires :meth:`_on_experiment_changed`, so configs, presets, the
        health probe, and the device panel all follow.

        Returns
        -------
        bool
            Whether the restore selected an experiment (and so repopulated
            with a fresh listing message) — ``False`` tells the constructor
            the quiet first populate's message still stands.
        """
        if self._configs.experiment:
            return False
        remembered = self._settings.last_experiment
        if not remembered or self.experiment_combo.findText(remembered) < 0:
            return False
        self.experiment_combo.setCurrentText(remembered)
        return True

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
    # Actions menu (contents on ActionsMenuController; thin test surface)
    # ------------------------------------------------------------------

    @property
    def enable_actions_action(self):
        """The arming switch QAction (owned by the actions-menu controller)."""
        return self._actions.enable_action

    @property
    def _open_action_dialogs(self) -> list:
        """The controller's open ActionRunDialogs (read-only view)."""
        return self._actions.open_dialogs

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
        """Whether the manager reports an active plan (any live RE state).

        From the polled status snapshot — under the queue this covers any
        client's scan, not just this console's, which is exactly right for
        Start gating (one worker, one machine).  Transitional states
        (pausing/stopping/…) count as active; ``None`` (worker environment
        gone) does not — there is nothing to stop.
        """
        return self._queue_status.connected and self._queue_status.re_state not in (
            None,
            "idle",
        )

    def _refresh_submit_enabled(self) -> None:
        """Recompute Start/Stop enabled state from form + engine.

        Optimization mode needs a selected optimizer config but — unlike
        every other mode — no selected save sets: the engine
        auto-provisions the optimizer's ``device_requirements`` (the
        evaluator's diagnostics) into the effective device set
        (GeecsBluesky ≥ 0.38.0), so an optimize run records something even
        with an empty R2 selection.  Whether the engine accepts an
        optimize submission remains the engine's call, surfaced from
        :meth:`_on_start_clicked` rather than pre-blocked here.
        """
        scanning = self._scanning()
        optimize = self.current_mode() is ConsoleMode.OPTIMIZATION
        ready = (
            not scanning
            and not self._submit_in_flight
            and self._shot_count_valid
            and (bool(self.selected_save_sets()) or optimize)
            and (not optimize or bool(self.optimization_combo.currentText()))
        )
        self.start_button.setEnabled(ready)
        # While a stop is in flight the button stays disabled ("Stopping…")
        # until the terminal lifecycle event lands (_on_scan_state).
        self.stop_button.setEnabled(scanning and not self._stop_in_flight)
        self._refresh_pause_button()

    def _refresh_pause_button(self) -> None:
        """Set the Pause/Resume button from the current lifecycle state.

        Pause while RUNNING, Resume while PAUSED; disabled otherwise
        (transitional PAUSING/STOPPING, idle, or terminal).  The engine's
        operator-pause holds the scan non-modally, so the GUI stays usable
        while paused.
        """
        state = self._scan_state_text
        if state == "paused":
            self.pause_button.setText("▶ Resume")
            self.pause_button.setEnabled(not self._stop_in_flight)
            self.pause_button.setToolTip("Resume the paused scan.")
        elif state == "running":
            self.pause_button.setText(self._pause_button_label)
            self.pause_button.setEnabled(not self._stop_in_flight)
            self.pause_button.setToolTip(
                "Pause the scan at its next safe point (the machine goes to "
                "its quiescent state; Resume or Stop from here)."
            )
        else:
            self.pause_button.setText(self._pause_button_label)
            self.pause_button.setEnabled(False)

    def _ensure_submitter(self) -> Optional[Submitter]:
        """Return the injected submitter, or lazily build the queue client."""
        if self._submitter is not None:
            return self._submitter
        try:
            self._submitter = self._submitter_factory(
                self.experiment_combo.currentText()
            )
        except Exception as exc:
            message = f"Scan service unavailable: {exc}"
            self.statusBar().showMessage(message, 10_000)
            self.append_log(message)
            return None
        return self._submitter

    def _on_start_clicked(self) -> None:
        """Build the request and run the pre-submit → stamp → queue pipeline.

        Decision 3 of the queueserver migration: the preflight checks run
        client-side *before* queueing (a typo must fail at submit, not at
        queue-front) with their questions as ordinary modals; the answers
        are stamped into the request's ``submission`` record.  Both
        blocking phases (checks: config/DB/CA reads; submission: 0MQ round
        trips) run on the submit worker — the GUI thread only builds,
        asks, and reports.  The worker re-validates authoritatively at
        execution; the duplication is by design.
        """
        if self._submit_in_flight:
            return
        try:
            request = build_scan_request(self.form_state())
        except (ConsoleFormError, ValidationError) as exc:
            self._report(f"Cannot submit: {exc}")
            return
        submitter = self._ensure_submitter()
        if submitter is None:
            return
        experiment = self.experiment_combo.currentText()
        self._submit_in_flight = True
        self.start_button.setText("Checking…")
        self._refresh_submit_enabled()

        # Both worker callables must capture their own exceptions:
        # BackgroundResult swallows a raise without emitting, which would
        # strand the pipeline in-flight (Start disabled) forever.
        def check() -> tuple:
            try:
                return ("preflight", request, run_submit_preflight(request, experiment))
            except Exception as exc:  # noqa: BLE001 — deliver as a refusal
                return ("preflight", request, PreflightReport(refusal=str(exc)))

        self._submit_worker.run_async(check, "submit-preflight")

    def _queue_submission(self, stamped, *, clear_pending: bool, name: str) -> None:
        """Run one submit call on the worker (exception-capturing)."""
        submitter = self._submitter

        def call() -> tuple:
            try:
                return (
                    "submit",
                    submitter.submit(
                        stamped.model_dump(mode="json"), clear_pending=clear_pending
                    ),
                )
            except Exception as exc:  # noqa: BLE001 — deliver as a failure
                return ("submit", SubmitResult(ok=False, message=str(exc)))

        self._submit_worker.run_async(call, name)

    def _finish_submission(self, message: str) -> None:
        """End the submission pipeline, restoring the Start button."""
        self._submit_in_flight = False
        self.start_button.setText(self._start_button_label)
        self._report(message)
        self._refresh_submit_enabled()

    @Slot(object)
    def _on_submit_phase_done(self, payload: object) -> None:
        """Advance the submission pipeline as each worker phase completes."""
        if not self._submit_in_flight or not isinstance(payload, tuple):
            return
        phase = payload[0]
        if phase == "preflight":
            _, request, report = payload
            self._continue_after_preflight(request, report)
        elif phase == "submit":
            _, result = payload
            self._continue_after_submit(result)

    def _continue_after_preflight(self, request, report) -> None:
        """Ask the preflight questions and queue the stamped request."""
        if report.refusal is not None:
            self._finish_submission(f"Cannot submit: {report.refusal}")
            return
        outcomes = list(report.outcomes)
        for question in report.questions:
            if self._ask_binary(
                question.title,
                question.message,
                continue_label=question.continue_label,
                abort_label=question.abort_label,
            ):
                # The detail travels into run metadata in the operator's
                # own words (the question they actually saw).
                outcomes.append((question.check, "continued", question.message))
                self.append_log(f"preflight {question.check}: operator continued")
            else:
                self._finish_submission(
                    f"Submission aborted at the {question.check} check"
                )
                return
        stamped = stamp_submission(
            request, outcomes, client=f"geecs-console {console_version()}"
        )
        self.start_button.setText("Submitting…")
        # Keep the stamped payload for a clear-and-retry after the
        # pending-items question (no re-stamp: same outcomes, same click).
        self._pending_submission = stamped
        self._queue_submission(stamped, clear_pending=False, name="submit-queue")

    def _continue_after_submit(self, result) -> None:
        """Handle the queue's answer, including the failed-item-front trap."""
        if result.ok:
            self._finish_submission("Scan queued — starting")
            return
        if result.pending_items:
            count = len(result.pending_items)
            if self._ask_binary(
                "Queue not empty",
                (
                    f"The queue already holds {count} item(s) — usually a "
                    "failed scan returned to the queue front, which would "
                    "re-run before yours. Remove the pending item(s) and "
                    "submit this scan?"
                ),
                continue_label="Remove && submit",
                abort_label="Cancel",
            ):
                self._queue_submission(
                    self._pending_submission,
                    clear_pending=True,
                    name="submit-queue-clear",
                )
                return
            self._finish_submission("Submission cancelled (queue left as-is)")
            return
        self._finish_submission(f"Submission failed: {result.message}")

    def _ask_binary(
        self,
        title: str,
        message: str,
        *,
        continue_label: str = "Continue",
        abort_label: str = "Abort",
    ) -> bool:
        """One modal continue/abort question; ``True`` means continue.

        A render failure reads as abort — a warning the operator could not
        see must never wave a submission through.
        """
        try:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle(str(title))
            box.setText(str(message))
            continue_button = box.addButton(
                str(continue_label), QMessageBox.ButtonRole.AcceptRole
            )
            box.addButton(str(abort_label), QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(continue_button)
            box.exec()
            return box.clickedButton() is continue_button
        except Exception:  # noqa: BLE001 — unrenderable question = abort
            logger.exception("question dialog render failed; treating as abort")
            return False

    def _on_stop_clicked(self) -> None:
        """Gracefully stop the current scan — never blocking the GUI thread.

        ``stop_scan`` blocks (from a running scan it sequences deferred
        pause → stop, waiting out an in-flight blocking move — the #571
        rule with a longer worst case), so it runs on the stop worker.
        The button shows "Stopping…" until a terminal state releases the
        hold (:meth:`_on_scan_state`); the sequencing outcome itself lands
        in :meth:`_on_stop_result`.
        """
        submitter = self._submitter
        if submitter is None or not self._scanning():
            self._refresh_submit_enabled()
            return
        self._stop_in_flight = True
        self.stop_button.setText("Stopping…")

        # Same rule as the submit pipeline: a raise inside the worker is
        # swallowed by BackgroundResult without emitting, which would leave
        # the "Stopping…" hold stuck forever — capture into a failure tuple.
        def call() -> tuple[bool, str]:
            try:
                return submitter.stop_scan()
            except Exception as exc:  # noqa: BLE001 — deliver as a failure
                return (False, str(exc))

        self._stop_worker.run_async(call, "scan-stop")
        self.append_log("stop requested (partial data is preserved)")
        self._refresh_submit_enabled()

    @Slot(object)
    def _on_stop_result(self, result: object) -> None:
        """Report the stop sequencing's outcome (GUI-thread slot).

        A failed sequencing (e.g. the pause never landed) releases the
        hold so the operator can retry or escalate; a successful one keeps
        it — the terminal state arrives via the document stream / status
        poll and clears it in :meth:`_on_scan_state`.
        """
        if not isinstance(result, tuple) or len(result) != 2:
            return
        ok, message = result
        self.append_log(f"stop: {message}")
        if not ok:
            self._stop_in_flight = False
            self.stop_button.setText(self._stop_button_label)
            self._refresh_submit_enabled()

    def _on_pause_clicked(self) -> None:
        """Pause the running scan, or resume it if already paused.

        Both manager calls are single short-timeout requests, so they run
        on the GUI thread (the old prompt-returning pause semantics); the
        actual pause lands at the plan's next checkpoint and is announced
        back by the status poll (the PAUSED/RUNNING states flip this
        button via :meth:`_refresh_pause_button`).
        """
        submitter = self._submitter
        if submitter is None or not self._scanning():
            return
        if self._scan_state_text == "paused":
            ok, message = submitter.request_resume()
            self._pause_reason = ""
            self.append_log(message if ok else f"resume refused: {message}")
        else:
            ok, message = submitter.request_pause()
            self.append_log(message if ok else f"pause refused: {message}")

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
        if form.axes:
            # setCurrentText is programmatic (no textActivated), so follow
            # the preset explicitly — axis 1 owns the panel.
            self._movable.select_from_scan_combo(form.axes[0].variable)

        if optimization_name:
            self.optimization_combo.setCurrentText(optimization_name)
        # After the combo (whose change handler seeds the spinner from the
        # config): the preset's own iteration count wins.
        self.iterations_spin.setValue(form.max_iterations or 0)
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
        the loaded documents, with ``max_iterations`` neutralized on both
        sides: that field belongs to the Iterations spinner (restored
        separately from the preset), so a preset saved with an overridden
        count must still match its source config.  Unloadable configs are
        skipped.

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
        neutral = {"max_iterations": None}
        wanted = spec.model_copy(update=neutral)
        for index in range(self.optimization_combo.count()):
            name = self.optimization_combo.itemText(index)
            try:
                if (
                    self._configs.optimization_spec(name).model_copy(update=neutral)
                    == wanted
                ):
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

    # ------------------------------------------------------------------
    # R6 now panel
    # ------------------------------------------------------------------

    def append_log(self, line: str) -> None:
        """Append one line to the compact log tail (delegates to the panel).

        Parameters
        ----------
        line : str
            The text to append (one log-tail row).
        """
        self._now.append_log(line)

    def set_scan_number(self, number: int) -> None:
        """Show the current scan number (delegates to the panel).

        Parameters
        ----------
        number : int
            The claimed scan number.
        """
        self._now.set_scan_number(number)

    def _on_scan_state(self, state: str) -> None:
        """Update the state pill and button gating on lifecycle events.

        A terminal state (aborted/done/error) also releases an in-flight
        Stop request: the button label restores and normal gating resumes
        (:meth:`_on_stop_clicked` set the hold).
        """
        lowered = (state or "").lower()
        self._scan_state_text = lowered or "idle"
        if lowered in ("done", "aborted"):
            # Arms the status poll's post-terminal grace window (a stale
            # pre-stop snapshot must not repaint RUNNING — see
            # _on_queue_status).
            self._terminal_state_at = time.monotonic()
        if self._stop_in_flight and lowered in _TERMINAL_SCAN_STATES:
            self._stop_in_flight = False
            self.stop_button.setText(self._stop_button_label)
        self._now.set_state_pill(state)
        self._refresh_submit_enabled()

        # Push the live scan state into open action dialogs so their Run
        # button disables while a scan is active (actions are idle-only
        # queue items since decision 2 dropped the pause-window flow).
        self._actions.set_scanning(lowered in ("running", "paused"))

    def _on_totals_known(self, total_shots: int) -> None:
        """Size the progress bar once the scan announces its totals."""
        self._now.set_totals(total_shots)
        self._last_beep_shots = 0  # new scan: re-arm the per-shot beep

    def _on_progress(
        self, step_index: int, total_steps: int, shots_completed: int
    ) -> None:
        """Advance the progress bar from step events; beep on shot increments."""
        self._now.update_progress(step_index, total_steps, shots_completed)
        if shots_completed > self._last_beep_shots:
            self._last_beep_shots = shots_completed
            self._maybe_beep()
        elif shots_completed < self._last_beep_shots:
            # A scan that never announced totals restarted the count.
            self._last_beep_shots = shots_completed
