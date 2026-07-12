"""The trigger-profile (shot control) editor — the M5 editor dialog.

Edits the versioned :class:`geecs_schemas.TriggerProfile` documents in the
per-experiment ``shot_control_configurations/`` folder through
:class:`~geecs_console.services.trigger_profile_store.TriggerProfileStore`.
Trigger profiles gate real hardware at scan time, so correctness of the
saved YAML outranks features here: the working document is a plain
``model_dump(mode="json")`` dict, every edit re-validates it against the
real schema, and Save refuses anything pydantic rejects — the error is
shown inline, never raised at the operator.

Layout: a profile list (New / Duplicate / Rename / Delete) on the left; the
document on the right — a variant selector (the table edits the base
profile or one variant's overlay), a per-layer description, and the
state × (device, variable, value) write tree with Add/Remove/Move up/Move
down (order matters within a transition: writes are sent top to bottom).
Dirty tracking with Save/Revert and an unsaved-changes prompt on profile
switch and close.

Offline-first: with no configs repo the list is empty and the dialog still
opens.  Device/variable cells get QCompleter suggestions from an injectable
:class:`~geecs_console.services.device_completions.CompletionsProvider` —
its one blocking ``device_variables()`` call runs on a short-lived daemon
thread (the package's no-QThread rule) with the result marshaled back
through a queued signal, so a slow or unreachable DB never blocks the GUI.
Enter never accepts the dialog — it commits the active cell edit at most.
"""

from __future__ import annotations

import copy
import logging
import threading
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QFile, Qt, Signal, Slot
from PySide6.QtGui import QKeyEvent
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QCompleter,
    QDialog,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QStyledItemDelegate,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pydantic import ValidationError

from geecs_console.services.device_completions import (
    CompletionsProvider,
    EmptyCompletions,
)
from geecs_console.services.trigger_profile_store import (
    TriggerProfileStore,
    TriggerProfileStoreError,
)
from geecs_schemas import TriggerProfile, TriggerState

logger = logging.getLogger(__name__)

_UI_PATH = Path(__file__).parent.parent / "app" / "ui" / "shot_control_editor.ui"

#: The variant combo's entry for editing the profile's base states.
BASE_LAYER = "(base)"

#: Semantic colors matching the main window's palette (style.qss header).
_COLOR_RED = "#c4453a"
_COLOR_AMBER = "#d9a21b"

#: Tree roles: which state a top-level item represents.
_STATE_ROLE = Qt.ItemDataRole.UserRole


def _format_validation_error(exc: ValidationError) -> str:
    """Render a pydantic error compactly for the inline validation label.

    Parameters
    ----------
    exc : ValidationError
        The schema rejection to summarize.

    Returns
    -------
    str
        Up to three ``location: message`` lines plus an overflow count.
    """
    errors = exc.errors()
    lines = [
        f"{'.'.join(str(part) for part in error['loc']) or 'profile'}: {error['msg']}"
        for error in errors[:3]
    ]
    if len(errors) > 3:
        lines.append(f"… and {len(errors) - 3} more")
    return "; ".join(lines)


class _WriteCellDelegate(QStyledItemDelegate):
    """Line-edit cells for write rows, with device/variable completion.

    The completer is rebuilt each time a cell editor opens, so suggestions
    that arrive asynchronously (the real provider fetches on a daemon
    thread) appear on the next edit without any signal plumbing.

    Parameters
    ----------
    editor : ShotControlEditor
        The owning dialog — source of the experiment name and the
        completions provider.
    """

    def __init__(self, editor: "ShotControlEditor") -> None:
        super().__init__(editor)
        self._editor = editor

    def createEditor(self, parent, option, index):  # noqa: N802 — Qt override
        """Create the cell's line edit, wiring a completer for columns 0/1.

        Parameters
        ----------
        parent : QWidget
            The editor's parent (the tree viewport).
        option : QStyleOptionViewItem
            Qt's style options (unused beyond the base call).
        index : QModelIndex
            The cell being edited; column selects the completion source.

        Returns
        -------
        QWidget
            A ``QLineEdit`` (completer-armed for device/variable columns).
        """
        line_edit = QLineEdit(parent)
        words: list[str] = []
        device_vars = self._editor.device_vars
        if index.column() == 0:
            words = sorted(device_vars)
        elif index.column() == 1:
            device = str(index.siblingAtColumn(0).data() or "").strip()
            if device:
                words = list(device_vars.get(device, []))
        if words:
            completer = QCompleter(words, line_edit)
            completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            completer.setFilterMode(Qt.MatchFlag.MatchContains)
            line_edit.setCompleter(completer)
        return line_edit


class ShotControlEditor(QDialog):
    """Modeless editor dialog for one experiment's trigger profiles.

    Parameters
    ----------
    parent : QWidget, optional
        The owning window (the console main window in production).
    experiment : str, optional
        Experiment whose ``shot_control_configurations/`` folder to edit;
        empty opens the dialog with no profiles (offline default).
    store : TriggerProfileStore, optional
        Persistence seam; tests inject one rooted at a tmp dir, default
        reads/writes the configs repo.
    completions : CompletionsProvider, optional
        Device/variable cell suggestions; defaults to
        :class:`~geecs_console.services.device_completions.EmptyCompletions`.
        The provider's one blocking ``device_variables()`` call runs on a
        daemon thread, never the GUI thread.
    """

    #: One ``{device: [variable, ...]}`` mapping (emitted from the fetch
    #: daemon thread; delivered queued to :meth:`_apply_completions`).
    completions_ready = Signal(object)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        experiment: str = "",
        store: Optional[TriggerProfileStore] = None,
        completions: Optional[CompletionsProvider] = None,
    ) -> None:
        super().__init__(parent)
        self._experiment = experiment
        self._store = store if store is not None else TriggerProfileStore(experiment)
        self._completions = (
            completions if completions is not None else EmptyCompletions()
        )
        #: The fetched completion word lists (empty until the fetch lands).
        self._device_vars: dict[str, list[str]] = {}
        #: True once the (async) completions fetch has been delivered on the
        #: GUI thread — tests wait on this instead of sleeping.
        self.completions_applied = False
        #: Guards the one-shot signal disconnect in :meth:`closeEvent`.
        self._completions_disconnected = False

        #: The working document: a TriggerProfile ``model_dump(mode="json")``
        #: dict, kept current on every edit.  ``None`` = no profile loaded.
        self._doc: Optional[dict] = None
        #: The last loaded/saved document — dirty means ``_doc != _snapshot``.
        self._snapshot: Optional[dict] = None
        #: The profile (file stem) the document belongs to.
        self._current_name: str = ""
        #: The layer the tree edits: ``BASE_LAYER`` or a variant name.
        self._current_layer: str = BASE_LAYER
        #: True while code (not the operator) is populating widgets.
        self._loading = False

        self._load_ui()
        self._bind_widgets()
        self._wire_signals()
        self.setWindowTitle(f"Trigger profiles — {experiment or 'no experiment'}")
        # Force queued: the fetch emits from a daemon thread, and a direct
        # delivery would touch state on the wrong thread.
        self.completions_ready.connect(
            self._apply_completions, Qt.ConnectionType.QueuedConnection
        )
        self._start_completions_fetch()
        self._refresh_profile_list()
        self._show_document(None)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _load_ui(self) -> None:
        """Load the hand-authored .ui into the dialog's layout."""
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
        self.resize(920, 560)

    def _child(self, cls: type, name: str):
        """Return the named child widget, failing loudly when missing."""
        widget = self._ui.findChild(cls, name)
        if widget is None:
            raise LookupError(f"{name!r} ({cls.__name__}) not found in {_UI_PATH}")
        return widget

    def _bind_widgets(self) -> None:
        """Resolve every wired widget from the loaded .ui once."""
        self.profile_list: QListWidget = self._child(QListWidget, "sce_profile_list")
        self.new_button: QPushButton = self._child(QPushButton, "sce_new_button")
        self.duplicate_button: QPushButton = self._child(
            QPushButton, "sce_duplicate_button"
        )
        self.rename_button: QPushButton = self._child(QPushButton, "sce_rename_button")
        self.delete_button: QPushButton = self._child(QPushButton, "sce_delete_button")
        self.profile_name_label: QLabel = self._child(QLabel, "sce_profile_name_label")
        self.legacy_label: QLabel = self._child(QLabel, "sce_legacy_label")
        self.variant_combo: QComboBox = self._child(QComboBox, "sce_variant_combo")
        self.add_variant_button: QPushButton = self._child(
            QPushButton, "sce_add_variant_button"
        )
        self.remove_variant_button: QPushButton = self._child(
            QPushButton, "sce_remove_variant_button"
        )
        self.description_edit: QLineEdit = self._child(
            QLineEdit, "sce_description_edit"
        )
        self.states_tree: QTreeWidget = self._child(QTreeWidget, "sce_states_tree")
        self.add_write_button: QPushButton = self._child(
            QPushButton, "sce_add_write_button"
        )
        self.remove_write_button: QPushButton = self._child(
            QPushButton, "sce_remove_write_button"
        )
        self.move_up_button: QPushButton = self._child(
            QPushButton, "sce_move_up_button"
        )
        self.move_down_button: QPushButton = self._child(
            QPushButton, "sce_move_down_button"
        )
        self.validation_label: QLabel = self._child(QLabel, "sce_validation_label")
        self.dirty_label: QLabel = self._child(QLabel, "sce_dirty_label")
        self.revert_button: QPushButton = self._child(QPushButton, "sce_revert_button")
        self.save_button: QPushButton = self._child(QPushButton, "sce_save_button")

        # Enter must never accept the dialog: no default/auto-default
        # buttons anywhere (keyPressEvent below swallows the rest).
        for button in self._ui.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)

        # Keep a Python reference for the delegate's full lifetime —
        # setItemDelegate does not take ownership, and a GC'd wrapper with
        # queued events is a segfault.
        self._delegate = _WriteCellDelegate(self)
        self.states_tree.setItemDelegate(self._delegate)
        self.states_tree.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.states_tree.setColumnWidth(0, 280)
        self.states_tree.setColumnWidth(1, 220)

    def _wire_signals(self) -> None:
        """Connect widget signals to the handlers."""
        self.profile_list.currentItemChanged.connect(self._on_profile_selected)
        self.new_button.clicked.connect(self._on_new)
        self.duplicate_button.clicked.connect(self._on_duplicate)
        self.rename_button.clicked.connect(self._on_rename)
        self.delete_button.clicked.connect(self._on_delete)
        self.variant_combo.currentTextChanged.connect(self._on_layer_changed)
        self.add_variant_button.clicked.connect(self._on_add_variant)
        self.remove_variant_button.clicked.connect(self._on_remove_variant)
        self.description_edit.textChanged.connect(self._on_description_edited)
        self.states_tree.itemChanged.connect(self._on_tree_edited)
        self.add_write_button.clicked.connect(self._on_add_write)
        self.remove_write_button.clicked.connect(self._on_remove_write)
        self.move_up_button.clicked.connect(lambda: self._on_move_write(-1))
        self.move_down_button.clicked.connect(lambda: self._on_move_write(+1))
        self.revert_button.clicked.connect(self._on_revert)
        self.save_button.clicked.connect(self._on_save)

    # ------------------------------------------------------------------
    # Small injectable surfaces (tests monkeypatch these)
    # ------------------------------------------------------------------

    @property
    def experiment(self) -> str:
        """The experiment whose profiles this dialog edits."""
        return self._experiment

    @property
    def device_vars(self) -> dict[str, list[str]]:
        """The fetched ``{device: [variable, ...]}`` completion word lists."""
        return self._device_vars

    def _start_completions_fetch(self) -> None:
        """Fetch completions on a short-lived daemon thread (queued delivery)."""
        provider = self._completions

        def fetch() -> None:
            """Run the provider's one blocking call off the GUI thread."""
            try:
                mapping = provider.device_variables()
            except Exception as exc:  # noqa: BLE001 — providers should not raise
                logger.info("completions fetch failed: %s", exc)
                mapping = {}
            self.completions_ready.emit(mapping)

        threading.Thread(
            target=fetch, name="sce-completions-fetch", daemon=True
        ).start()

    @Slot(object)
    def _apply_completions(self, mapping: object) -> None:
        """Store the fetched word lists (GUI-thread slot, delivered queued).

        Cell editors read :attr:`device_vars` when they open, so suggestions
        that land after the dialog is up appear on the next edit — no
        completer replumbing needed.

        Parameters
        ----------
        mapping : dict
            The provider result delivered by the queued signal.
        """
        self.completions_applied = True
        if not isinstance(mapping, dict):
            return
        self._device_vars = {
            str(device): [str(variable) for variable in variables]
            for device, variables in mapping.items()
        }

    def is_dirty(self) -> bool:
        """Whether the working document differs from the loaded snapshot.

        Returns
        -------
        bool
            ``True`` when there are unsaved edits.
        """
        return self._doc is not None and self._doc != self._snapshot

    def _prompt_text(self, title: str, label: str, default: str = "") -> str:
        """Ask the operator for one line of text ("" = cancelled).

        Parameters
        ----------
        title : str
            Dialog title.
        label : str
            Prompt label.
        default : str, optional
            Pre-filled text.

        Returns
        -------
        str
            The stripped answer, or "" when cancelled/empty.
        """
        text, accepted = QInputDialog.getText(self, title, label, text=default)
        return text.strip() if accepted else ""

    def _confirm(self, title: str, text: str) -> bool:
        """Ask a yes/no question (used for deletes).

        Parameters
        ----------
        title : str
            Dialog title.
        text : str
            The question.

        Returns
        -------
        bool
            ``True`` on Yes.
        """
        answer = QMessageBox.question(
            self,
            title,
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _prompt_unsaved(self) -> str:
        """Ask what to do with unsaved changes.

        Returns
        -------
        str
            ``"save"``, ``"discard"``, or ``"cancel"``.
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Unsaved changes")
        box.setText(f"Trigger profile {self._current_name!r} has unsaved changes.")
        save = box.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
        discard = box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(save)
        box.exec()
        clicked = box.clickedButton()
        if clicked is save:
            return "save"
        if clicked is discard:
            return "discard"
        return "cancel"

    def _resolve_unsaved(self) -> bool:
        """Settle unsaved edits before leaving the current profile.

        Returns
        -------
        bool
            ``True`` when it is safe to proceed (nothing dirty, saved, or
            discarded); ``False`` on cancel or a failed save.
        """
        if not self.is_dirty():
            return True
        choice = self._prompt_unsaved()
        if choice == "cancel":
            return False
        if choice == "save":
            return self._save_current()
        return True  # discard

    # ------------------------------------------------------------------
    # Document handling (the working dict IS model_dump(mode="json"))
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(doc: dict) -> dict:
        """Drop empty state lists (a state with no writes is "not defined").

        Keeps dirty tracking honest: the tree omits empty states when it
        rebuilds its layer, so the loaded document must not carry them
        either.  Semantically identical per the schema's ``defines_state``.

        Parameters
        ----------
        doc : dict
            A ``model_dump(mode="json")`` document (modified in place).

        Returns
        -------
        dict
            The same document, with empty state lists removed from the base
            profile and every variant.
        """
        doc["states"] = {
            state: writes for state, writes in doc.get("states", {}).items() if writes
        }
        for variant in doc.get("variants", {}).values():
            variant["states"] = {
                state: writes
                for state, writes in variant.get("states", {}).items()
                if writes
            }
        return doc

    def _layer_states(self) -> Optional[dict]:
        """Return the states dict of the layer the tree currently edits."""
        if self._doc is None:
            return None
        if self._current_layer == BASE_LAYER:
            return self._doc["states"]
        variant = self._doc["variants"].get(self._current_layer)
        return None if variant is None else variant["states"]

    def _set_layer_states(self, states: dict) -> None:
        """Replace the current layer's states dict with *states*."""
        if self._doc is None:
            return
        if self._current_layer == BASE_LAYER:
            self._doc["states"] = states
        elif self._current_layer in self._doc["variants"]:
            self._doc["variants"][self._current_layer]["states"] = states

    def _show_document(self, doc: Optional[dict]) -> None:
        """Render *doc* (or the empty no-selection state) into the widgets.

        Parameters
        ----------
        doc : dict or None
            The working document; ``None`` clears and disables the editor.
        """
        self._doc = doc
        self._current_layer = BASE_LAYER
        self._loading = True
        try:
            enabled = doc is not None
            for widget in (
                self.variant_combo,
                self.add_variant_button,
                self.remove_variant_button,
                self.description_edit,
                self.states_tree,
                self.add_write_button,
                self.remove_write_button,
                self.move_up_button,
                self.move_down_button,
            ):
                widget.setEnabled(enabled)
            self._populate_variant_combo()
            self._populate_tree()
            self._refresh_description()
            if doc is None:
                self.profile_name_label.setText("(no profile selected)")
                self.legacy_label.setText("")
                self.validation_label.setText("")
            else:
                self.profile_name_label.setText(
                    f"<b>{doc.get('name', self._current_name)}</b>"
                )
        finally:
            self._loading = False
        self._refresh_validation()

    def _populate_variant_combo(self) -> None:
        """Fill the layer selector: (base) plus the document's variants."""
        self.variant_combo.blockSignals(True)
        self.variant_combo.clear()
        if self._doc is not None:
            self.variant_combo.addItem(BASE_LAYER)
            self.variant_combo.addItems(sorted(self._doc.get("variants", {})))
            self.variant_combo.setCurrentText(self._current_layer)
        self.variant_combo.blockSignals(False)
        self.remove_variant_button.setEnabled(
            self._doc is not None and self._current_layer != BASE_LAYER
        )

    def _populate_tree(self) -> None:
        """Rebuild the state tree from the current layer's states dict."""
        self.states_tree.blockSignals(True)
        self.states_tree.clear()
        states = self._layer_states()
        if states is not None:
            for state in TriggerState:
                state_item = QTreeWidgetItem(self.states_tree)
                state_item.setText(0, state.value)
                state_item.setData(0, _STATE_ROLE, state.value)
                state_item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                )
                state_item.setFirstColumnSpanned(True)
                for write in states.get(state.value, []):
                    self._make_write_item(state_item, write)
                state_item.setExpanded(True)
        self.states_tree.blockSignals(False)

    @staticmethod
    def _make_write_item(parent: QTreeWidgetItem, write: dict) -> QTreeWidgetItem:
        """Append one editable write row under *parent*.

        Parameters
        ----------
        parent : QTreeWidgetItem
            The state item to append to.
        write : dict
            ``{"device": ..., "variable": ..., "value": ...}``.

        Returns
        -------
        QTreeWidgetItem
            The created row.
        """
        item = QTreeWidgetItem(parent)
        item.setText(0, str(write.get("device", "")))
        item.setText(1, str(write.get("variable", "")))
        item.setText(2, str(write.get("value", "")))
        item.setFlags(
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsEditable
        )
        return item

    def _rebuild_layer_from_tree(self) -> None:
        """Read the tree back into the current layer's states dict."""
        states: dict[str, list[dict]] = {}
        for row in range(self.states_tree.topLevelItemCount()):
            state_item = self.states_tree.topLevelItem(row)
            state = str(state_item.data(0, _STATE_ROLE))
            writes = [
                {
                    "device": state_item.child(child).text(0).strip(),
                    "variable": state_item.child(child).text(1).strip(),
                    "value": state_item.child(child).text(2),
                }
                for child in range(state_item.childCount())
            ]
            if writes:
                states[state] = writes
        self._set_layer_states(states)

    def _refresh_description(self) -> None:
        """Bind the description edit to the current layer's description."""
        self.description_edit.blockSignals(True)
        if self._doc is None:
            self.description_edit.setText("")
        elif self._current_layer == BASE_LAYER:
            self.description_edit.setText(self._doc.get("description", ""))
        else:
            variant = self._doc["variants"].get(self._current_layer, {})
            self.description_edit.setText(variant.get("description", ""))
        self.description_edit.blockSignals(False)

    def _validated_profile(self) -> Optional[TriggerProfile]:
        """Validate the working document, rendering any rejection inline.

        Returns
        -------
        TriggerProfile or None
            The validated model, or ``None`` (with the validation label set)
            when the document is empty or the schema rejects it.
        """
        if self._doc is None:
            return None
        try:
            profile = TriggerProfile.model_validate(self._doc)
        except ValidationError as exc:
            self.validation_label.setText(
                f'<span style="color:{_COLOR_RED};">'
                f"{_format_validation_error(exc)}</span>"
            )
            return None
        self.validation_label.setText("")
        return profile

    def _refresh_validation(self) -> None:
        """Re-validate the document; gate Save/Revert and the dirty marker."""
        profile = self._validated_profile()
        dirty = self.is_dirty()
        self.dirty_label.setText("● unsaved changes" if dirty else "")
        self.save_button.setEnabled(dirty and profile is not None)
        self.revert_button.setEnabled(dirty)

    def _report_store_error(self, exc: TriggerProfileStoreError) -> None:
        """Show a store failure inline (never a crash).

        Parameters
        ----------
        exc : TriggerProfileStoreError
            The failed operation's message.
        """
        self.validation_label.setText(f'<span style="color:{_COLOR_RED};">{exc}</span>')
        logger.warning("trigger-profile store: %s", exc)

    # ------------------------------------------------------------------
    # Profile list handling
    # ------------------------------------------------------------------

    def _refresh_profile_list(self, select: str = "") -> None:
        """Repopulate the profile list from the store (offline ⇒ empty).

        Parameters
        ----------
        select : str, optional
            Name to select after repopulating (no selection when absent).
        """
        self.profile_list.blockSignals(True)
        self.profile_list.clear()
        self.profile_list.addItems(self._store.list_names())
        if select:
            matches = self.profile_list.findItems(select, Qt.MatchFlag.MatchExactly)
            if matches:
                self.profile_list.setCurrentItem(matches[0])
        self.profile_list.blockSignals(False)
        if select:
            self._load_profile(select)

    def _on_profile_selected(self, current, previous) -> None:
        """Load the newly selected profile, honoring unsaved changes."""
        if self._loading:
            return
        name = current.text() if current is not None else ""
        if name == self._current_name:
            return
        if not self._resolve_unsaved():
            # Cancelled: put the selection back without re-firing.
            self.profile_list.blockSignals(True)
            self.profile_list.setCurrentItem(previous)
            self.profile_list.blockSignals(False)
            return
        self._load_profile(name)

    def _load_profile(self, name: str) -> None:
        """Load *name* from disk into the editor (inline error on failure).

        Parameters
        ----------
        name : str
            The profile (file stem) to load; "" clears the editor.
        """
        self._current_name = name
        if not name:
            self._snapshot = None
            self._show_document(None)
            return
        try:
            profile = self._store.load(name)
            legacy = self._store.is_legacy(name)
        except TriggerProfileStoreError as exc:
            self._snapshot = None
            self._show_document(None)
            self.profile_name_label.setText(f"<b>{name}</b>")
            self._report_store_error(exc)
            return
        doc = self._normalize(profile.model_dump(mode="json"))
        self._snapshot = copy.deepcopy(doc)
        self._show_document(doc)
        self.legacy_label.setText(
            f'<span style="color:{_COLOR_AMBER};">legacy file — '
            "Save migrates it to the versioned schema</span>"
            if legacy
            else ""
        )

    def _selected_name(self) -> str:
        """Return the profile list's current name ("" when none)."""
        item = self.profile_list.currentItem()
        return item.text() if item is not None else ""

    def _on_new(self) -> None:
        """Create (and immediately save) an empty profile."""
        if not self._resolve_unsaved():
            return
        name = self._prompt_text("New trigger profile", "Profile name:")
        if not name:
            return
        if self._store.exists(name):
            self._report_store_error(
                TriggerProfileStoreError(f"Trigger profile {name!r} already exists.")
            )
            return
        try:
            self._store.save(name, TriggerProfile(name=name))
        except TriggerProfileStoreError as exc:
            self._report_store_error(exc)
            return
        self._refresh_profile_list(select=name)

    def _on_duplicate(self) -> None:
        """Copy the selected profile's file under a new name."""
        source = self._selected_name()
        if not source:
            return
        if not self._resolve_unsaved():
            return
        name = self._prompt_text(
            "Duplicate trigger profile", "New profile name:", default=f"{source}-copy"
        )
        if not name:
            return
        if self._store.exists(name):
            self._report_store_error(
                TriggerProfileStoreError(f"Trigger profile {name!r} already exists.")
            )
            return
        try:
            profile = self._store.load(source)
            self._store.save(name, profile.model_copy(update={"name": name}))
        except TriggerProfileStoreError as exc:
            self._report_store_error(exc)
            return
        self._refresh_profile_list(select=name)

    def _on_rename(self) -> None:
        """Rename the selected profile (file stem + stored name field)."""
        source = self._selected_name()
        if not source:
            return
        if not self._resolve_unsaved():
            return
        name = self._prompt_text(
            "Rename trigger profile", "New profile name:", default=source
        )
        if not name or name == source:
            return
        try:
            self._store.rename(source, name)
        except TriggerProfileStoreError as exc:
            self._report_store_error(exc)
            return
        self._refresh_profile_list(select=name)

    def _on_delete(self) -> None:
        """Delete the selected profile file after confirmation."""
        name = self._selected_name()
        if not name:
            return
        if not self._confirm(
            "Delete trigger profile",
            f"Delete trigger profile {name!r}? The YAML file is removed "
            "from the configs repo.",
        ):
            return
        try:
            self._store.delete(name)
        except TriggerProfileStoreError as exc:
            self._report_store_error(exc)
            return
        self._current_name = ""
        self._snapshot = None
        self._show_document(None)
        self._refresh_profile_list()

    # ------------------------------------------------------------------
    # Layer (base/variant) handling
    # ------------------------------------------------------------------

    def _on_layer_changed(self, layer: str) -> None:
        """Point the tree and description at the newly selected layer."""
        if self._loading or self._doc is None or not layer:
            return
        self._current_layer = layer
        self._loading = True
        try:
            self._populate_tree()
            self._refresh_description()
        finally:
            self._loading = False
        self.remove_variant_button.setEnabled(layer != BASE_LAYER)

    def _on_add_variant(self) -> None:
        """Add an empty variant and switch the tree to it."""
        if self._doc is None:
            return
        name = self._prompt_text("Add variant", "Variant name:")
        if not name:
            return
        if name == BASE_LAYER or name in self._doc["variants"]:
            self._report_store_error(
                TriggerProfileStoreError(f"Variant {name!r} is taken.")
            )
            return
        self._doc["variants"][name] = {"states": {}, "description": ""}
        self._current_layer = name
        self._loading = True
        try:
            self._populate_variant_combo()
            self._populate_tree()
            self._refresh_description()
        finally:
            self._loading = False
        self._refresh_validation()

    def _on_remove_variant(self) -> None:
        """Remove the selected variant (with confirmation) — Save persists."""
        if self._doc is None or self._current_layer == BASE_LAYER:
            return
        name = self._current_layer
        if not self._confirm(
            "Remove variant", f"Remove variant {name!r} from this profile?"
        ):
            return
        self._doc["variants"].pop(name, None)
        self._current_layer = BASE_LAYER
        self._loading = True
        try:
            self._populate_variant_combo()
            self._populate_tree()
            self._refresh_description()
        finally:
            self._loading = False
        self._refresh_validation()

    def _on_description_edited(self, text: str) -> None:
        """Write the description edit into the current layer."""
        if self._loading or self._doc is None:
            return
        if self._current_layer == BASE_LAYER:
            self._doc["description"] = text
        elif self._current_layer in self._doc["variants"]:
            self._doc["variants"][self._current_layer]["description"] = text
        self._refresh_validation()

    # ------------------------------------------------------------------
    # Write-row handling
    # ------------------------------------------------------------------

    def _on_tree_edited(self, item: QTreeWidgetItem, column: int) -> None:
        """Fold a cell edit back into the document and re-validate."""
        if self._loading or self._doc is None:
            return
        self._rebuild_layer_from_tree()
        self._refresh_validation()

    def _target_state_item(self) -> Optional[QTreeWidgetItem]:
        """The state item Add-write appends to (selection or its parent)."""
        item = self.states_tree.currentItem()
        if item is None:
            return None
        parent = item.parent()
        return item if parent is None else parent

    def _on_add_write(self) -> None:
        """Append an empty write row to the selected state and edit it."""
        if self._doc is None:
            return
        state_item = self._target_state_item()
        if state_item is None:
            self.validation_label.setText(
                f'<span style="color:{_COLOR_AMBER};">Select a state '
                "(or one of its writes) to add a write to.</span>"
            )
            return
        self.states_tree.blockSignals(True)
        row = self._make_write_item(
            state_item, {"device": "", "variable": "", "value": ""}
        )
        state_item.setExpanded(True)
        self.states_tree.blockSignals(False)
        self._rebuild_layer_from_tree()
        self._refresh_validation()
        self.states_tree.setCurrentItem(row)
        if self.states_tree.isVisible():
            # Drop the operator straight into the device cell.  Never on a
            # hidden tree: an open cell editor on an unshown dialog leaves
            # queued events that break programmatic teardown.
            self.states_tree.editItem(row, 0)

    def _on_remove_write(self) -> None:
        """Remove the selected write row."""
        if self._doc is None:
            return
        item = self.states_tree.currentItem()
        if item is None or item.parent() is None:
            return
        item.parent().removeChild(item)
        self._rebuild_layer_from_tree()
        self._refresh_validation()

    def _on_move_write(self, delta: int) -> None:
        """Move the selected write up or down within its state.

        Order matters: a state's writes are sent top to bottom at scan time.

        Parameters
        ----------
        delta : int
            ``-1`` for up, ``+1`` for down.
        """
        if self._doc is None:
            return
        item = self.states_tree.currentItem()
        if item is None:
            return
        parent = item.parent()
        if parent is None:
            return
        row = parent.indexOfChild(item)
        target = row + delta
        if target < 0 or target >= parent.childCount():
            return
        self.states_tree.blockSignals(True)
        parent.takeChild(row)
        parent.insertChild(target, item)
        self.states_tree.blockSignals(False)
        self.states_tree.setCurrentItem(item)
        self._rebuild_layer_from_tree()
        self._refresh_validation()

    # ------------------------------------------------------------------
    # Save / Revert / close
    # ------------------------------------------------------------------

    def _save_current(self) -> bool:
        """Validate and persist the working document.

        Returns
        -------
        bool
            ``True`` on success; ``False`` with the error shown inline when
            the schema rejects the document or the store cannot write.
        """
        profile = self._validated_profile()
        if profile is None or not self._current_name:
            return False
        try:
            self._store.save(self._current_name, profile)
        except TriggerProfileStoreError as exc:
            self._report_store_error(exc)
            return False
        # Snapshot the normalized dump so dirty tracking restarts clean.
        doc = self._normalize(profile.model_dump(mode="json"))
        self._doc = doc
        self._snapshot = copy.deepcopy(doc)
        self.legacy_label.setText("")
        self._refresh_validation()
        return True

    def _on_save(self) -> None:
        """Save button: persist the working document."""
        self._save_current()

    def _on_revert(self) -> None:
        """Throw away unsaved edits and reload the profile from disk."""
        if self._current_name:
            self._load_profile(self._current_name)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 — Qt override
        """Swallow Return/Enter so it never accepts (closes) the dialog."""
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt override
        """Prompt for unsaved changes before closing.

        Only a *visible* dialog prompts — there is an operator to ask.  A
        programmatic close of a hidden dialog (test teardown, app shutdown)
        must never block on a modal nobody can answer.
        """
        if self.isVisible() and not self._resolve_unsaved():
            event.ignore()
            return
        # A still-running completions fetch must not queue an event onto a
        # dialog being torn down.  Once only: closeEvent can run twice
        # (explicit close + owner teardown) and a second disconnect warns.
        if not self._completions_disconnected:
            self._completions_disconnected = True
            try:
                self.completions_ready.disconnect(self._apply_completions)
            except (RuntimeError, TypeError):
                pass
        super().closeEvent(event)


def open_shot_control_editor(
    parent: Optional[QWidget],
    experiment: str,
    configs_base: str | Path | None = None,
    completions: Optional[CompletionsProvider] = None,
) -> QDialog:
    """Open the trigger-profile editor (the Editors-menu entry point).

    Opens offline with zero configs: a missing configs repo just lists no
    profiles.  The default completions provider fetches device/variable
    names from ``GeecsDb`` (one query, on a daemon thread — never blocking
    the GUI, degrading to no suggestions offline).

    Parameters
    ----------
    parent : QWidget or None
        The owning window.
    experiment : str
        Experiment whose ``shot_control_configurations/`` folder to edit.
    configs_base : str or Path, optional
        Override for the experiments root (defaults to the configs repo
        resolved lazily, exactly like the preset store).
    completions : CompletionsProvider, optional
        Device/variable cell suggestions; defaults to the DB-backed
        provider for a named experiment and to no completions otherwise.

    Returns
    -------
    QDialog
        The shown (modeless) editor dialog.
    """
    if completions is None:
        from geecs_console.services.device_completions import GeecsDbCompletions

        completions = (
            GeecsDbCompletions(experiment) if experiment else EmptyCompletions()
        )
    store = TriggerProfileStore(experiment, experiments_root=configs_base)
    dialog = ShotControlEditor(
        parent, experiment=experiment, store=store, completions=completions
    )
    dialog.show()
    return dialog
