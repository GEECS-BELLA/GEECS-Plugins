"""The ActionLibrary editor dialog — **editing only, never execution**.

Edits the named :class:`~geecs_schemas.ActionPlan`s of one experiment's
``action_library/actions.yaml`` through
:class:`~geecs_console.services.action_library_store.ActionLibraryStore`.
A plan is an ordered list of steps; the step kinds and their fields come
straight from the schema's discriminated union (``geecs_schemas.action_plan``):
``set`` (device / variable / value / wait_for_execution), ``wait``
(seconds), ``check`` (device / variable / expected), and ``run`` (the name
of another plan in the same library).

Design notes
------------
- **No execution surface.**  This dialog cannot run a plan: there is no run
  button, no Channel Access import, and nothing that reaches a device.
  Executing plans from the console is separate future work (G-actions).
- **Offline-first**: opens with zero network and zero configs (empty plan
  list, saving reports the missing configs repo inline).
- The ``.ui`` is hand-authored XML (``app/ui/action_library_editor.ui``,
  object names prefixed ``ale_``) loaded at runtime via ``QUiLoader``.
- Device/variable fields carry a ``QCompleter`` fed by an injectable
  :class:`CompletionsProvider`; the production
  :class:`GeecsDbCompletions` queries ``GeecsDb`` lazily and degrades to
  empty on any failure, tests inject fakes, and the default is
  :class:`EmptyCompletions` (offline).
- Enter never accepts the dialog (every button is non-default and the
  key is swallowed) — pressing Return in a field must not close the editor.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Protocol

from pydantic import ValidationError
from PySide6.QtCore import QFile, QStringListModel, Qt
from PySide6.QtGui import QCloseEvent, QKeyEvent
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDialog,
    QDoubleSpinBox,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from geecs_console.services.action_library_store import (
    ActionLibraryStore,
    ActionLibraryStoreError,
)
from geecs_schemas import ActionPlan

logger = logging.getLogger(__name__)

_UI_PATH = Path(__file__).parent.parent / "app" / "ui" / "action_library_editor.ui"

#: Step kinds in kind-combo / stacked-page order — mirrors the schema's
#: discriminated union (``geecs_schemas.action_plan.ActionStep``).
STEP_KINDS = ("set", "wait", "check", "run")


# ---------------------------------------------------------------------------
# Completions seam
# ---------------------------------------------------------------------------


class CompletionsProvider(Protocol):
    """Device/variable name completions for the step editor fields."""

    def devices(self) -> list[str]:
        """Return the known device names (may be empty)."""
        ...

    def variables(self, device: str) -> list[str]:
        """Return the known variable names of *device* (may be empty)."""
        ...


class EmptyCompletions:
    """The offline/test default: no completions at all."""

    def devices(self) -> list[str]:
        """Return no device names."""
        return []

    def variables(self, device: str) -> list[str]:
        """Return no variable names.

        Parameters
        ----------
        device : str
            Ignored.
        """
        return []


class GeecsDbCompletions:
    """DB-backed completions: devices and variables of one experiment.

    Queries ``GeecsDb.get_experiment_device_variables`` **lazily on first
    use** and caches the result; any failure (no lab network, missing
    ``mysql-connector``, missing config) degrades to empty completions with
    a log line — never an exception, never a retry loop.

    Parameters
    ----------
    experiment : str
        GEECS experiment name to enumerate.
    """

    def __init__(self, experiment: str) -> None:
        self._experiment = experiment
        self._cache: Optional[dict[str, list[str]]] = None

    def _catalog(self) -> dict[str, list[str]]:
        """Fetch (once) the ``{device: [variable, ...]}`` catalog."""
        if self._cache is not None:
            return self._cache
        try:
            from geecs_ca_gateway.db.geecs_db import GeecsDb

            raw = GeecsDb.get_experiment_device_variables(self._experiment)
            self._cache = {
                device: sorted({meta["name"] for meta in metas})
                for device, metas in raw.items()
            }
        except Exception as exc:  # no DB / no network / no driver — degrade
            logger.info("device/variable completions unavailable: %s", exc)
            self._cache = {}
        return self._cache

    def devices(self) -> list[str]:
        """Return the experiment's device names (empty offline)."""
        return sorted(self._catalog())

    def variables(self, device: str) -> list[str]:
        """Return *device*'s variable names (empty offline/unknown).

        Parameters
        ----------
        device : str
            The device to enumerate variables for.
        """
        return self._catalog().get(device, [])


# ---------------------------------------------------------------------------
# Step-dict helpers (the editor's working representation)
# ---------------------------------------------------------------------------


def default_step(kind: str) -> dict:
    """Return a fresh working step dict of *kind* with schema-default fields.

    Parameters
    ----------
    kind : str
        One of :data:`STEP_KINDS`.

    Returns
    -------
    dict
        The working step (plain dict — validated into the schema on save).
    """
    if kind == "set":
        return {
            "do": "set",
            "device": "",
            "variable": "",
            "value": "",
            "wait_for_execution": True,
        }
    if kind == "wait":
        return {"do": "wait", "seconds": 1.0}
    if kind == "check":
        return {"do": "check", "device": "", "variable": "", "expected": ""}
    if kind == "run":
        return {"do": "run", "plan": ""}
    raise ValueError(f"Unknown step kind {kind!r}. Expected one of {STEP_KINDS}.")


def convert_step_kind(step: dict, new_kind: str) -> dict:
    """Convert a working step to *new_kind*, carrying shared fields over.

    ``device``/``variable`` survive a ``set`` ↔ ``check`` switch, and the
    written ``value`` maps onto the expected reading (and back).

    Parameters
    ----------
    step : dict
        The current working step.
    new_kind : str
        The target kind (one of :data:`STEP_KINDS`).

    Returns
    -------
    dict
        A new working step of *new_kind*.
    """
    converted = default_step(new_kind)
    for key in ("device", "variable"):
        if key in converted and step.get(key):
            converted[key] = step[key]
    if new_kind == "check" and step.get("do") == "set":
        converted["expected"] = step.get("value", "")
    if new_kind == "set" and step.get("do") == "check":
        converted["value"] = step.get("expected", "")
    return converted


def parse_action_value(text: str, original: object = None) -> object:
    """Parse a value field's text into the schema's ``str | float | int``.

    When the text still renders the *original* value unchanged, the original
    object is returned as-is so an untouched field never changes type (a
    string ``"0"`` loaded from YAML stays a string through a save).

    Parameters
    ----------
    text : str
        The field text.
    original : str or float or int, optional
        The value the field was populated from.

    Returns
    -------
    str or float or int
        ``int`` when the text parses as one, else ``float``, else the text.
    """
    if original is not None and str(original) == text:
        return original
    stripped = text.strip()
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    return text


def step_summary(step: dict) -> str:
    """One-line summary of a working step for the steps table.

    Parameters
    ----------
    step : dict
        The working step.

    Returns
    -------
    str
        A compact human-readable description.
    """
    kind = step.get("do", "?")
    if kind == "set":
        suffix = "" if step.get("wait_for_execution", True) else "  (no wait)"
        return (
            f"{step.get('device', '')}:{step.get('variable', '')} "
            f"← {step.get('value', '')}{suffix}"
        )
    if kind == "wait":
        return f"{step.get('seconds', '')} s"
    if kind == "check":
        return (
            f"{step.get('device', '')}:{step.get('variable', '')} "
            f"== {step.get('expected', '')}"
        )
    if kind == "run":
        return f"→ {step.get('plan', '')}"
    return ""


# ---------------------------------------------------------------------------
# The dialog
# ---------------------------------------------------------------------------


class ActionLibraryEditor(QDialog):
    """Edit the named action plans of one experiment (no execution).

    Parameters
    ----------
    store : ActionLibraryStore, optional
        The persistence seam; defaults to an :class:`ActionLibraryStore`
        with no experiment (offline: empty list, saving reports why).
    completions : CompletionsProvider, optional
        Device/variable completions for the step fields; defaults to
        :class:`EmptyCompletions`.
    parent : QWidget, optional
        Standard Qt parent.
    """

    def __init__(
        self,
        store: ActionLibraryStore | None = None,
        completions: CompletionsProvider | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._store = store if store is not None else ActionLibraryStore()
        self._completions: CompletionsProvider = (
            completions if completions is not None else EmptyCompletions()
        )

        self._loading = False
        self._dirty = False
        self._current: Optional[str] = None
        #: Name of a new/duplicated plan that exists only in memory until saved.
        self._phantom: Optional[str] = None
        self._working_steps: list[dict] = []
        self._working_version = 1
        #: Snapshot for Revert (steps deep-ish copy + description).
        self._baseline_steps: list[dict] = []
        self._baseline_description = ""

        self._apply_stylesheet()
        self._load_ui()
        self._bind_widgets()
        self._configure_widgets()
        self._wire_signals()
        self._setup_completers()

        self._populate_plan_list()
        self._clear_plan_editor()
        self._update_title()
        self._refresh_enabled()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _apply_stylesheet(self) -> None:
        """Apply the console QSS application-wide when not already set."""
        app = QApplication.instance()
        if app is not None and not app.styleSheet():
            from geecs_console.app.main_window import load_stylesheet

            app.setStyleSheet(load_stylesheet())

    def _load_ui(self) -> None:
        """Load the hand-authored ``.ui`` as this dialog's content."""
        loader = QUiLoader()
        ui_file = QFile(str(_UI_PATH))
        ui_file.open(QFile.OpenModeFlag.ReadOnly)
        try:
            self._ui: QWidget = loader.load(ui_file, self)
        finally:
            ui_file.close()
        if self._ui is None:
            raise RuntimeError(f"Failed to load {_UI_PATH}: {loader.errorString()}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._ui)

    def _child(self, cls: type, name: str):
        """Return the named child widget, failing loudly when missing."""
        widget = self._ui.findChild(cls, name)
        if widget is None:
            raise LookupError(f"{name!r} ({cls.__name__}) not found in {_UI_PATH}")
        return widget

    def _bind_widgets(self) -> None:
        """Resolve every wired widget from the loaded ``.ui`` once."""
        self.plan_list: QListWidget = self._child(QListWidget, "ale_plan_list")
        self.new_button: QPushButton = self._child(QPushButton, "ale_new_button")
        self.duplicate_button: QPushButton = self._child(
            QPushButton, "ale_duplicate_button"
        )
        self.rename_button: QPushButton = self._child(QPushButton, "ale_rename_button")
        self.delete_button: QPushButton = self._child(QPushButton, "ale_delete_button")
        self.description_edit: QLineEdit = self._child(
            QLineEdit, "ale_description_edit"
        )
        self.steps_table: QTableWidget = self._child(QTableWidget, "ale_steps_table")
        self.add_step_button: QPushButton = self._child(
            QPushButton, "ale_add_step_button"
        )
        self.remove_step_button: QPushButton = self._child(
            QPushButton, "ale_remove_step_button"
        )
        self.up_button: QPushButton = self._child(QPushButton, "ale_up_button")
        self.down_button: QPushButton = self._child(QPushButton, "ale_down_button")
        self.kind_combo: QComboBox = self._child(QComboBox, "ale_kind_combo")
        self.step_stack: QStackedWidget = self._child(QStackedWidget, "ale_step_stack")
        self.set_device_edit: QLineEdit = self._child(QLineEdit, "ale_set_device_edit")
        self.set_variable_edit: QLineEdit = self._child(
            QLineEdit, "ale_set_variable_edit"
        )
        self.set_value_edit: QLineEdit = self._child(QLineEdit, "ale_set_value_edit")
        self.set_wait_check: QCheckBox = self._child(QCheckBox, "ale_set_wait_check")
        self.wait_seconds_spin: QDoubleSpinBox = self._child(
            QDoubleSpinBox, "ale_wait_seconds_spin"
        )
        self.check_device_edit: QLineEdit = self._child(
            QLineEdit, "ale_check_device_edit"
        )
        self.check_variable_edit: QLineEdit = self._child(
            QLineEdit, "ale_check_variable_edit"
        )
        self.check_expected_edit: QLineEdit = self._child(
            QLineEdit, "ale_check_expected_edit"
        )
        self.run_plan_combo: QComboBox = self._child(QComboBox, "ale_run_plan_combo")
        self.run_warning_label: QLabel = self._child(QLabel, "ale_run_warning_label")
        self.error_label: QLabel = self._child(QLabel, "ale_error_label")
        self.revert_button: QPushButton = self._child(QPushButton, "ale_revert_button")
        self.save_button: QPushButton = self._child(QPushButton, "ale_save_button")

    def _configure_widgets(self) -> None:
        """Post-load widget configuration ``.ui`` XML cannot express well."""
        table = self.steps_table
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.run_warning_label.setStyleSheet("color: #d9a21b;")  # --warn amber
        self.error_label.setStyleSheet("color: #c4453a;")  # --abort red
        self.run_plan_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)

    def _wire_signals(self) -> None:
        """Connect every widget signal to its handler."""
        self.plan_list.currentItemChanged.connect(self._on_plan_selected)
        self.new_button.clicked.connect(self._on_new)
        self.duplicate_button.clicked.connect(self._on_duplicate)
        self.rename_button.clicked.connect(self._on_rename)
        self.delete_button.clicked.connect(self._on_delete)
        self.description_edit.textEdited.connect(self._on_description_edited)
        self.steps_table.currentCellChanged.connect(self._on_step_row_changed)
        self.add_step_button.clicked.connect(self._on_add_step)
        self.remove_step_button.clicked.connect(self._on_remove_step)
        self.up_button.clicked.connect(lambda: self._on_move_step(-1))
        self.down_button.clicked.connect(lambda: self._on_move_step(+1))
        self.kind_combo.currentIndexChanged.connect(self._on_kind_changed)
        self.set_device_edit.textEdited.connect(self._on_step_field_edited)
        self.set_variable_edit.textEdited.connect(self._on_step_field_edited)
        self.set_value_edit.textEdited.connect(self._on_step_field_edited)
        self.set_wait_check.toggled.connect(self._on_step_field_edited)
        self.wait_seconds_spin.valueChanged.connect(self._on_step_field_edited)
        self.check_device_edit.textEdited.connect(self._on_step_field_edited)
        self.check_variable_edit.textEdited.connect(self._on_step_field_edited)
        self.check_expected_edit.textEdited.connect(self._on_step_field_edited)
        self.run_plan_combo.editTextChanged.connect(self._on_run_plan_changed)
        self.revert_button.clicked.connect(self._on_revert)
        self.save_button.clicked.connect(self._on_save)
        # Variable completions follow the device the operator typed.
        self.set_device_edit.textChanged.connect(
            lambda text: self._refresh_variable_completer(
                self._set_variable_completer_model, text
            )
        )
        self.check_device_edit.textChanged.connect(
            lambda text: self._refresh_variable_completer(
                self._check_variable_completer_model, text
            )
        )

    def _setup_completers(self) -> None:
        """Attach device/variable ``QCompleter``s fed by the provider."""
        try:
            devices = list(self._completions.devices())
        except Exception as exc:  # a provider must never break the editor
            logger.info("completions provider failed: %s", exc)
            devices = []
        self._device_completer_model = QStringListModel(devices, self)
        self._set_variable_completer_model = QStringListModel([], self)
        self._check_variable_completer_model = QStringListModel([], self)
        for edit, model in (
            (self.set_device_edit, self._device_completer_model),
            (self.check_device_edit, self._device_completer_model),
            (self.set_variable_edit, self._set_variable_completer_model),
            (self.check_variable_edit, self._check_variable_completer_model),
        ):
            completer = QCompleter(model, edit)
            completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            completer.setFilterMode(Qt.MatchFlag.MatchContains)
            edit.setCompleter(completer)

    def _refresh_variable_completer(self, model: QStringListModel, device: str) -> None:
        """Repopulate a variable completer for the typed *device*."""
        try:
            model.setStringList(list(self._completions.variables(device)))
        except Exception as exc:  # a provider must never break the editor
            logger.info("completions provider failed: %s", exc)
            model.setStringList([])

    # ------------------------------------------------------------------
    # Dialog-level behaviour
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 (Qt API)
        """Swallow Return/Enter so they never accept (close) the dialog."""
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.accept()
            return
        super().keyPressEvent(event)

    def reject(self) -> None:
        """Prompt for unsaved changes before closing via Escape/close."""
        if self._dirty and not self._confirm_discard():
            return
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt API)
        """Prompt for unsaved changes before the window closes."""
        if self._dirty and not self._confirm_discard():
            event.ignore()
            return
        super().closeEvent(event)

    def _update_title(self) -> None:
        """Render the window title (experiment + unsaved marker)."""
        experiment = self._store.experiment
        title = "Action Library" + (f" — {experiment}" if experiment else "")
        if self._dirty:
            title += " *"
        self.setWindowTitle(title)

    # ------------------------------------------------------------------
    # Prompt seams (tests monkeypatch these)
    # ------------------------------------------------------------------

    def _prompt_name(self, title: str, initial: str = "") -> Optional[str]:
        """Ask the operator for a plan name; ``None`` on cancel."""
        name, accepted = QInputDialog.getText(self, title, "Plan name:", text=initial)
        return name if accepted else None

    def _confirm_discard(self) -> bool:
        """Ask whether unsaved changes may be discarded."""
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            "The current plan has unsaved changes. Discard them?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Discard

    def _confirm_delete(self, name: str) -> bool:
        """Ask whether plan *name* should really be deleted."""
        answer = QMessageBox.question(
            self,
            "Delete plan",
            f"Delete action plan {name!r}? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Yes

    # ------------------------------------------------------------------
    # Inline messages
    # ------------------------------------------------------------------

    def _show_error(self, message: str) -> None:
        """Show *message* in the inline error label."""
        self.error_label.setText(message)

    def _clear_error(self) -> None:
        """Clear the inline error label."""
        self.error_label.setText("")

    def _set_dirty(self, dirty: bool) -> None:
        """Set the dirty flag and re-render the title."""
        if self._dirty != dirty:
            self._dirty = dirty
            self._update_title()

    # ------------------------------------------------------------------
    # Plan list
    # ------------------------------------------------------------------

    def _known_names(self) -> list[str]:
        """All referenceable plan names: the library plus the unsaved one."""
        names = self._store.list_names()
        if self._phantom is not None and self._phantom not in names:
            names.append(self._phantom)
        return sorted(names)

    def _populate_plan_list(self, select: Optional[str] = None) -> None:
        """Repopulate the plan list from the store (plus the phantom)."""
        self._loading = True
        try:
            self.plan_list.clear()
            for name in self._known_names():
                self.plan_list.addItem(name)
            if select is not None:
                self._select_name(select)
        finally:
            self._loading = False

    def _select_name(self, name: Optional[str]) -> None:
        """Move the list selection to *name* (no-op when absent)."""
        if name is None:
            self.plan_list.setCurrentItem(None)
            return
        for row in range(self.plan_list.count()):
            if self.plan_list.item(row).text() == name:
                self.plan_list.setCurrentRow(row)
                return

    def _on_plan_selected(self, current, _previous) -> None:
        """Load the newly selected plan (prompting on unsaved changes)."""
        if self._loading:
            return
        name = current.text() if current is not None else None
        if name == self._current:
            return
        if self._dirty and not self._confirm_discard():
            self._loading = True
            try:
                self._select_name(self._current)
            finally:
                self._loading = False
            return
        discarded_phantom = self._phantom is not None and self._current == self._phantom
        self._phantom = None if discarded_phantom else self._phantom
        self._set_dirty(False)
        if discarded_phantom:
            # The discarded new plan disappears with its unsaved state.
            self._populate_plan_list(select=name)
        self._load_plan(name)

    def _load_plan(self, name: Optional[str]) -> None:
        """Populate the editor from plan *name* (or clear it for ``None``)."""
        self._clear_error()
        if name is None:
            self._current = None
            self._clear_plan_editor()
            self._refresh_enabled()
            return
        try:
            plan = self._store.load(name)
        except ActionLibraryStoreError as exc:
            self._current = None
            self._clear_plan_editor()
            self._show_error(str(exc))
            self._refresh_enabled()
            return
        dumped = plan.model_dump(mode="python")
        self._current = name
        self._working_version = dumped.get("schema_version", 1)
        self._working_steps = [dict(step) for step in dumped["steps"]]
        self._baseline_steps = [dict(step) for step in self._working_steps]
        self._baseline_description = dumped.get("description", "")
        self._loading = True
        try:
            self.description_edit.setText(self._baseline_description)
        finally:
            self._loading = False
        self._refresh_table(select_row=0 if self._working_steps else None)
        self._set_dirty(False)
        self._refresh_enabled()

    def _clear_plan_editor(self) -> None:
        """Blank the right-hand plan editor (no plan selected)."""
        self._working_steps = []
        self._baseline_steps = []
        self._baseline_description = ""
        self._loading = True
        try:
            self.description_edit.setText("")
            self.steps_table.setRowCount(0)
        finally:
            self._loading = False

    def _refresh_enabled(self) -> None:
        """Enable/disable buttons and fields for the current selection."""
        has_plan = self._current is not None
        row = self.steps_table.currentRow()
        has_step = has_plan and 0 <= row < len(self._working_steps)
        for widget in (
            self.duplicate_button,
            self.rename_button,
            self.delete_button,
            self.description_edit,
            self.add_step_button,
            self.save_button,
            self.revert_button,
        ):
            widget.setEnabled(has_plan)
        self.remove_step_button.setEnabled(has_step and len(self._working_steps) > 1)
        self.up_button.setEnabled(has_step and row > 0)
        self.down_button.setEnabled(has_step and row < len(self._working_steps) - 1)
        self.kind_combo.setEnabled(has_step)
        self.step_stack.setEnabled(has_step)

    # ------------------------------------------------------------------
    # Plan-level actions
    # ------------------------------------------------------------------

    def _start_phantom(self, name: str, steps: list[dict], description: str) -> None:
        """Begin editing a new in-memory plan *name* (unsaved until Save)."""
        self._phantom = name
        self._current = name
        self._working_version = 1
        self._working_steps = [dict(step) for step in steps]
        self._baseline_steps = [dict(step) for step in steps]
        self._baseline_description = description
        self._populate_plan_list(select=name)
        self._loading = True
        try:
            self.description_edit.setText(description)
        finally:
            self._loading = False
        self._refresh_table(select_row=0 if self._working_steps else None)
        self._set_dirty(True)
        self._clear_error()
        self._refresh_enabled()

    def _on_new(self) -> None:
        """Create a new plan (in memory until saved)."""
        if self._dirty and not self._confirm_discard():
            return
        name = self._prompt_name("New action plan")
        if not name or not name.strip():
            return
        name = name.strip()
        if name in self._store.list_names():
            self._show_error(f"Plan {name!r} already exists.")
            return
        self._set_dirty(False)
        self._start_phantom(name, [default_step("wait")], "")

    def _on_duplicate(self) -> None:
        """Duplicate the selected plan under a new name (in memory)."""
        if self._current is None:
            return
        if self._dirty and not self._confirm_discard():
            return
        name = self._prompt_name("Duplicate plan", initial=f"{self._current}_copy")
        if not name or not name.strip():
            return
        name = name.strip()
        if name in self._store.list_names() or name == self._phantom:
            self._show_error(f"Plan {name!r} already exists.")
            return
        steps = [dict(step) for step in self._working_steps]
        description = self.description_edit.text()
        self._phantom = None
        self._set_dirty(False)
        self._start_phantom(name, steps, description)

    def _on_rename(self) -> None:
        """Rename the selected plan (updating every ``run`` reference)."""
        if self._current is None:
            return
        new = self._prompt_name("Rename plan", initial=self._current)
        if not new or not new.strip() or new.strip() == self._current:
            return
        new = new.strip()
        old = self._current
        if self._phantom == old:
            if new in self._store.list_names():
                self._show_error(f"Plan {new!r} already exists.")
                return
            self._phantom = new
            self._current = new
            self._populate_plan_list(select=new)
            self._update_title()
            return
        try:
            self._store.rename(old, new)
        except ActionLibraryStoreError as exc:
            self._show_error(str(exc))
            return
        # Working run-references to the old name follow the rename too.
        for step in self._working_steps:
            if step.get("do") == "run" and step.get("plan") == old:
                step["plan"] = new
        self._current = new
        self._populate_plan_list(select=new)
        self._refresh_table(select_row=self.steps_table.currentRow())
        self._clear_error()

    def _on_delete(self) -> None:
        """Delete the selected plan (store-checked for ``run`` referrers)."""
        if self._current is None:
            return
        name = self._current
        if not self._confirm_delete(name):
            return
        if self._phantom == name:
            self._phantom = None
            self._current = None
            self._set_dirty(False)
            self._populate_plan_list()
            self._clear_plan_editor()
            self._clear_error()
            self._refresh_enabled()
            return
        try:
            self._store.delete(name)
        except ActionLibraryStoreError as exc:
            self._show_error(str(exc))
            return
        self._current = None
        self._set_dirty(False)
        self._populate_plan_list()
        self._clear_plan_editor()
        self._clear_error()
        self._refresh_enabled()

    def _on_description_edited(self, _text: str) -> None:
        """Mark the plan dirty when its description is edited."""
        if self._loading:
            return
        self._set_dirty(True)

    # ------------------------------------------------------------------
    # Steps table
    # ------------------------------------------------------------------

    def _refresh_table(self, select_row: Optional[int] = None) -> None:
        """Rebuild the steps table from the working steps."""
        self._loading = True
        try:
            table = self.steps_table
            table.setRowCount(len(self._working_steps))
            for row, step in enumerate(self._working_steps):
                table.setItem(row, 0, QTableWidgetItem(step.get("do", "?")))
                table.setItem(row, 1, QTableWidgetItem(step_summary(step)))
            if select_row is not None and 0 <= select_row < len(self._working_steps):
                table.setCurrentCell(select_row, 0)
        finally:
            self._loading = False
        self._load_step_detail(self.steps_table.currentRow())
        self._refresh_enabled()

    def _update_summary_cell(self, row: int) -> None:
        """Refresh one row's kind + summary cells in place."""
        if not (0 <= row < len(self._working_steps)):
            return
        step = self._working_steps[row]
        self._loading = True
        try:
            self.steps_table.setItem(row, 0, QTableWidgetItem(step.get("do", "?")))
            self.steps_table.setItem(row, 1, QTableWidgetItem(step_summary(step)))
        finally:
            self._loading = False

    def _on_step_row_changed(
        self, row: int, _col: int, _prev_row: int, _prev_col: int
    ) -> None:
        """Load the newly selected step into the detail editor."""
        if self._loading:
            return
        self._load_step_detail(row)
        self._refresh_enabled()

    def _load_step_detail(self, row: int) -> None:
        """Populate the kind combo + stacked fields from step *row*."""
        if not (0 <= row < len(self._working_steps)):
            return
        step = self._working_steps[row]
        kind = step.get("do", "set")
        index = STEP_KINDS.index(kind) if kind in STEP_KINDS else 0
        self._loading = True
        try:
            self.kind_combo.setCurrentIndex(index)
            self.step_stack.setCurrentIndex(index)
            if kind == "set":
                self.set_device_edit.setText(str(step.get("device", "")))
                self.set_variable_edit.setText(str(step.get("variable", "")))
                self.set_value_edit.setText(str(step.get("value", "")))
                self.set_wait_check.setChecked(
                    bool(step.get("wait_for_execution", True))
                )
            elif kind == "wait":
                try:
                    self.wait_seconds_spin.setValue(float(step.get("seconds", 1.0)))
                except (TypeError, ValueError):
                    self.wait_seconds_spin.setValue(0.0)
            elif kind == "check":
                self.check_device_edit.setText(str(step.get("device", "")))
                self.check_variable_edit.setText(str(step.get("variable", "")))
                self.check_expected_edit.setText(str(step.get("expected", "")))
            elif kind == "run":
                self.run_plan_combo.clear()
                self.run_plan_combo.addItems(self._known_names())
                self.run_plan_combo.setEditText(str(step.get("plan", "")))
        finally:
            self._loading = False
        if kind == "run":
            self._update_run_warning(str(step.get("plan", "")))

    def _on_add_step(self) -> None:
        """Append a fresh step and select it."""
        if self._current is None:
            return
        self._working_steps.append(default_step("set"))
        self._refresh_table(select_row=len(self._working_steps) - 1)
        self._set_dirty(True)

    def _on_remove_step(self) -> None:
        """Remove the selected step (a plan keeps at least one step)."""
        row = self.steps_table.currentRow()
        if not (0 <= row < len(self._working_steps)) or len(self._working_steps) <= 1:
            return
        del self._working_steps[row]
        self._refresh_table(select_row=min(row, len(self._working_steps) - 1))
        self._set_dirty(True)

    def _on_move_step(self, delta: int) -> None:
        """Move the selected step up (``-1``) or down (``+1``)."""
        row = self.steps_table.currentRow()
        target = row + delta
        if not (0 <= row < len(self._working_steps)):
            return
        if not (0 <= target < len(self._working_steps)):
            return
        steps = self._working_steps
        steps[row], steps[target] = steps[target], steps[row]
        self._refresh_table(select_row=target)
        self._set_dirty(True)

    # ------------------------------------------------------------------
    # Step detail editing
    # ------------------------------------------------------------------

    def _current_step(self) -> Optional[dict]:
        """Return the working step of the selected row, or ``None``."""
        row = self.steps_table.currentRow()
        if 0 <= row < len(self._working_steps):
            return self._working_steps[row]
        return None

    def _on_kind_changed(self, index: int) -> None:
        """Switch the selected step to the newly picked kind."""
        if self._loading:
            return
        row = self.steps_table.currentRow()
        step = self._current_step()
        if step is None or not (0 <= index < len(STEP_KINDS)):
            return
        new_kind = STEP_KINDS[index]
        if step.get("do") == new_kind:
            return
        self._working_steps[row] = convert_step_kind(step, new_kind)
        self.step_stack.setCurrentIndex(index)
        self._update_summary_cell(row)
        self._load_step_detail(row)
        self._set_dirty(True)

    def _on_step_field_edited(self, *_args) -> None:
        """Fold the visible per-kind fields back into the working step."""
        if self._loading:
            return
        row = self.steps_table.currentRow()
        step = self._current_step()
        if step is None:
            return
        kind = step.get("do")
        if kind == "set":
            step["device"] = self.set_device_edit.text()
            step["variable"] = self.set_variable_edit.text()
            step["value"] = parse_action_value(
                self.set_value_edit.text(), step.get("value")
            )
            step["wait_for_execution"] = self.set_wait_check.isChecked()
        elif kind == "wait":
            step["seconds"] = self.wait_seconds_spin.value()
        elif kind == "check":
            step["device"] = self.check_device_edit.text()
            step["variable"] = self.check_variable_edit.text()
            step["expected"] = parse_action_value(
                self.check_expected_edit.text(), step.get("expected")
            )
        self._update_summary_cell(row)
        self._set_dirty(True)

    def _on_run_plan_changed(self, text: str) -> None:
        """Track the ``run`` step's target plan and warn on dangling names."""
        if self._loading:
            return
        row = self.steps_table.currentRow()
        step = self._current_step()
        if step is None or step.get("do") != "run":
            return
        step["plan"] = text
        self._update_summary_cell(row)
        self._update_run_warning(text)
        self._set_dirty(True)

    def _update_run_warning(self, plan_name: str) -> None:
        """Show the inline dangling-reference warning when needed."""
        if plan_name and plan_name not in self._known_names():
            self.run_warning_label.setText(
                f"⚠ plan {plan_name!r} is not in the library"
            )
        else:
            self.run_warning_label.setText("")

    # ------------------------------------------------------------------
    # Save / Revert
    # ------------------------------------------------------------------

    def _build_plan(self) -> ActionPlan:
        """Validate the working state into an :class:`ActionPlan`.

        Raises
        ------
        pydantic.ValidationError
            When the working steps violate the schema (surfaced inline).
        """
        return ActionPlan.model_validate(
            {
                "schema_version": self._working_version,
                "steps": self._working_steps,
                "description": self.description_edit.text(),
            }
        )

    def _on_save(self) -> None:
        """Validate and persist the current plan through the store."""
        if self._current is None:
            return
        self._clear_error()
        try:
            plan = self._build_plan()
        except ValidationError as exc:
            self._show_error(f"Invalid plan: {exc}")
            return
        try:
            self._store.save(self._current, plan)
        except ActionLibraryStoreError as exc:
            self._show_error(str(exc))
            return
        self._phantom = None
        self._baseline_steps = [dict(step) for step in self._working_steps]
        self._baseline_description = self.description_edit.text()
        self._set_dirty(False)
        self._populate_plan_list(select=self._current)

    def _on_revert(self) -> None:
        """Restore the plan to its last saved (or freshly created) state."""
        if self._current is None:
            return
        self._working_steps = [dict(step) for step in self._baseline_steps]
        self._loading = True
        try:
            self.description_edit.setText(self._baseline_description)
        finally:
            self._loading = False
        self._refresh_table(select_row=0 if self._working_steps else None)
        # A brand-new unsaved plan stays unsaved (and therefore dirty).
        self._set_dirty(self._phantom == self._current)
        self._clear_error()


def open_action_library_editor(
    parent: QWidget | None,
    experiment: str,
    configs_base: str | Path | None = None,
    completions: CompletionsProvider | None = None,
) -> QDialog:
    """Open the ActionLibrary editor for *experiment* (editing only).

    The integration entry point the main window's Editors menu will call.
    Opens offline with zero configs: the plan list is simply empty and
    saving reports the missing configs repo inline.

    Parameters
    ----------
    parent : QWidget or None
        Parent widget for the dialog.
    experiment : str
        Experiment folder under ``scanner_configs/experiments``.
    configs_base : str or Path, optional
        Override for the experiments root (defaults to the production
        resolution, lazily — offline stays fine).
    completions : CompletionsProvider, optional
        Device/variable completions; e.g. ``GeecsDbCompletions(experiment)``
        on the lab network. Defaults to none.

    Returns
    -------
    QDialog
        The shown (non-modal) editor dialog.
    """
    store = ActionLibraryStore(experiment, experiments_root=configs_base)
    dialog = ActionLibraryEditor(store=store, completions=completions, parent=parent)
    dialog.show()
    return dialog
