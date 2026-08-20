"""The SaveElementEditor: edit an experiment's save sets (M5).

The editor's document IS the :class:`geecs_schemas.SaveSet` pydantic model —
the form is derived from the model's fields, and every save round-trips
through ``SaveSet.model_validate`` / ``model_dump(mode="json")`` via
:class:`~geecs_console.services.save_set_store.SaveSetStore`.  A validation
error is rendered inline in the message label, never a crash.

Offline-first: the dialog opens with zero network and zero configs (empty
list, Save disabled until the document validates).  Device and variable name
completions come from the shared
:class:`~geecs_console.services.device_completions.CompletionsProvider` seam
— one blocking ``device_variables()`` call returning ``{device: [variable,
...]}`` — fetched once on a daemon thread and delivered back through a
queued signal (the no-QThread rule; see ``services/health.py``).  The
production provider, ``GeecsDbCompletions``, imports ``GeecsDb`` lazily and
degrades to empty on any failure, so offline means an empty completer, not
an exception; ``EmptyCompletions`` is the no-fetch default.

The reserved ``at_scan_start`` / ``at_scan_end`` entry fields (not applied by
the engine in this version) have no widgets; the editor carries them through
the round-trip untouched.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QStringListModel, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QCompleter,
    QDialog,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QWidget,
)
from pydantic import ValidationError

from geecs_console.editors.base import ConfigEditorDialog
from geecs_console.services.device_completions import (
    CompletionsProvider,
    GeecsDbCompletions,
)
from geecs_console.services.save_set_store import SaveSetStore, SaveSetStoreError
from geecs_console.services.schema_tooltips import apply_schema_tooltips
from geecs_schemas import SaveRole, SaveSet, SaveSetEntry

logger = logging.getLogger(__name__)

_UI_PATH = Path(__file__).parent.parent / "app" / "ui" / "save_set_editor.ui"

#: Screen-map palette (see app/style.qss header).
_COLOR_DIM = "#6b7681"
_COLOR_RED = "#c4453a"

#: The role combo's "let the scanner decide" row (stores ``None``).
_ROLE_DERIVED_LABEL = "(derived)"


def _split_names(text: str) -> list[str]:
    """Parse a comma-separated line edit into a list of names.

    Parameters
    ----------
    text : str
        The raw line-edit text.

    Returns
    -------
    list of str
        The non-empty, stripped names in order.
    """
    return [part.strip() for part in text.split(",") if part.strip()]


class SaveSetEditor(ConfigEditorDialog):
    """Edit the save sets of one experiment (the M5 SaveElementEditor).

    Left: the experiment's save-set list (New / Duplicate / Rename /
    Delete).  Right: the selected set's document — description, device
    entries (add/remove), and the selected entry's fields, derived from
    :class:`geecs_schemas.SaveSetEntry` (images, ``db_scalars`` /
    ``all_scalars`` flags, role override, extra scalars, setup/closeout
    plan references).  Save/Revert with dirty tracking and an
    unsaved-changes prompt on switch/close.

    Parameters
    ----------
    parent : QWidget, optional
        The owning window (the main window, in production).
    experiment : str, optional
        Experiment whose save sets to edit; used when *store* is not given.
    store : SaveSetStore, optional
        The persistence seam; tests inject one backed by a tmp dir.
    configs_base : Path, optional
        Experiments-root override forwarded to the default store.
    completions : CompletionsProvider, optional
        Word-list source for the device/variable completers.  ``None`` (the
        default) means :class:`EmptyCompletions` — no fetch at all, so
        offline/tests get empty completers unless a provider is injected.
    """

    UI_PATH = _UI_PATH
    COMPLETIONS_THREAD_NAME = "sse-completions"

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        experiment: str = "",
        store: Optional[SaveSetStore] = None,
        configs_base: Optional[Path] = None,
        completions: Optional[CompletionsProvider] = None,
    ) -> None:
        super().__init__(parent)
        self._store = (
            store
            if store is not None
            else SaveSetStore(experiment, experiments_root=configs_base)
        )
        self._experiment = experiment or self._store.experiment
        self._init_completions(completions)

        #: The open document as a JSON-ish dict (``model_dump(mode="json")``
        #: shape) — ``None`` when nothing is open.
        self._document: Optional[dict] = None
        self._dirty = False
        #: Names present in the list but not yet written to the store.
        self._unsaved: set[str] = set()
        #: Row of the set the editor currently shows (-1 for none).
        self._selected_row = -1
        #: Reentrancy guards: populating widgets / programmatic selection.
        self._loading = False
        self._suppress_selection = False

        self._load_ui()
        self._bind_widgets()
        self._guard_enter_keys()
        self._build_completers()
        self._wire_signals()

        self.setWindowTitle(self._title())
        self._refresh_set_list()
        self._clear_document()
        self._start_completions_fetch()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _bind_widgets(self) -> None:
        """Resolve every wired widget from the loaded .ui once."""
        self.set_list: QListWidget = self._child(QListWidget, "sse_set_list")
        self.new_button: QPushButton = self._child(QPushButton, "sse_new_button")
        self.duplicate_button: QPushButton = self._child(
            QPushButton, "sse_duplicate_button"
        )
        self.rename_button: QPushButton = self._child(QPushButton, "sse_rename_button")
        self.delete_button: QPushButton = self._child(QPushButton, "sse_delete_button")

        self.name_label: QLabel = self._child(QLabel, "sse_name_label")
        self.description_edit: QLineEdit = self._child(
            QLineEdit, "sse_description_edit"
        )
        self.device_list: QListWidget = self._child(QListWidget, "sse_device_list")
        self.device_edit: QLineEdit = self._child(QLineEdit, "sse_device_edit")
        self.add_device_button: QPushButton = self._child(
            QPushButton, "sse_add_device_button"
        )
        self.remove_device_button: QPushButton = self._child(
            QPushButton, "sse_remove_device_button"
        )

        self.entry_group: QGroupBox = self._child(QGroupBox, "sse_entry_group")
        self.images_check: QCheckBox = self._child(QCheckBox, "sse_images_check")
        self.db_scalars_check: QCheckBox = self._child(
            QCheckBox, "sse_db_scalars_check"
        )
        self.all_scalars_check: QCheckBox = self._child(
            QCheckBox, "sse_all_scalars_check"
        )
        self.role_combo: QComboBox = self._child(QComboBox, "sse_role_combo")
        self.scalar_list: QListWidget = self._child(QListWidget, "sse_scalar_list")
        self.scalar_edit: QLineEdit = self._child(QLineEdit, "sse_scalar_edit")
        self.add_scalar_button: QPushButton = self._child(
            QPushButton, "sse_add_scalar_button"
        )
        self.remove_scalar_button: QPushButton = self._child(
            QPushButton, "sse_remove_scalar_button"
        )
        self.setup_edit: QLineEdit = self._child(QLineEdit, "sse_setup_edit")
        self.closeout_edit: QLineEdit = self._child(QLineEdit, "sse_closeout_edit")

        self.message_label: QLabel = self._child(QLabel, "sse_message_label")
        self.revert_button: QPushButton = self._child(QPushButton, "sse_revert_button")
        self.save_button: QPushButton = self._child(QPushButton, "sse_save_button")
        self.close_button: QPushButton = self._child(QPushButton, "sse_close_button")

        self.role_combo.addItem(_ROLE_DERIVED_LABEL, None)
        for role in SaveRole:
            self.role_combo.addItem(role.value, role.value)

        # Tooltips come from the schema field descriptions — one source of
        # truth for what each field means (issue #497 phase 1).
        apply_schema_tooltips(
            SaveSet,
            {
                "description": self.description_edit,
                "entries": self.device_list,
            },
        )
        apply_schema_tooltips(
            SaveSetEntry,
            {
                "device": self.device_edit,
                "images": self.images_check,
                "db_scalars": self.db_scalars_check,
                "all_scalars": self.all_scalars_check,
                "role": self.role_combo,
                "scalars": [self.scalar_list, self.scalar_edit],
                "setup": self.setup_edit,
                "closeout": self.closeout_edit,
            },
        )

    def _build_completers(self) -> None:
        """Attach the device/variable completers (empty until a fetch lands).

        Both completers (and their models) are kept as attributes —
        ``QLineEdit.setCompleter`` does not take ownership, so a
        local-variable completer would be garbage-collected and the next
        model update would touch a dangling C++ object (a hard crash).
        """
        self._device_model = QStringListModel(self)
        self._device_completer = QCompleter(self)
        self._device_completer.setModel(self._device_model)
        self._device_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.device_edit.setCompleter(self._device_completer)

        self._variable_model = QStringListModel(self)
        self._variable_completer = QCompleter(self)
        self._variable_completer.setModel(self._variable_model)
        self._variable_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.scalar_edit.setCompleter(self._variable_completer)

    def _wire_signals(self) -> None:
        """Connect widget signals to the handlers."""
        self.set_list.currentRowChanged.connect(self._on_set_row_changed)
        self.new_button.clicked.connect(self._on_new)
        self.duplicate_button.clicked.connect(self._on_duplicate)
        self.rename_button.clicked.connect(self._on_rename)
        self.delete_button.clicked.connect(self._on_delete)

        self.description_edit.textChanged.connect(self._on_description_changed)
        self.device_list.currentRowChanged.connect(self._on_device_row_changed)
        self.add_device_button.clicked.connect(self._on_add_device)
        self.device_edit.returnPressed.connect(self._on_add_device)
        self.remove_device_button.clicked.connect(self._on_remove_device)

        self.images_check.toggled.connect(self._on_images_toggled)
        self.db_scalars_check.toggled.connect(self._on_db_scalars_toggled)
        self.all_scalars_check.toggled.connect(self._on_all_scalars_toggled)
        self.role_combo.currentIndexChanged.connect(self._on_role_changed)
        self.add_scalar_button.clicked.connect(self._on_add_scalar)
        self.scalar_edit.returnPressed.connect(self._on_add_scalar)
        self.remove_scalar_button.clicked.connect(self._on_remove_scalar)
        self.setup_edit.textChanged.connect(self._on_setup_changed)
        self.closeout_edit.textChanged.connect(self._on_closeout_changed)

        self.save_button.clicked.connect(self._on_save)
        self.revert_button.clicked.connect(self._on_revert)
        self.close_button.clicked.connect(self.close)

    def _title(self) -> str:
        """Compose the window title from experiment, open set, and dirt."""
        parts = ["Save Set Editor"]
        if self._experiment:
            parts.append(f"— {self._experiment}")
        if self._document is not None:
            star = "*" if self._dirty else ""
            parts.append(f"— {self._document.get('name', '')}{star}")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Completions (scaffold on ConfigEditorDialog)
    # ------------------------------------------------------------------

    def _install_completions(self) -> None:
        """Feed the completer models from the fresh word lists."""
        self._device_model.setStringList(sorted(self._device_vars))
        self._refresh_variable_completer()

    def _refresh_variable_completer(self) -> None:
        """Point the scalar completer at the selected device's variables."""
        entry = self._current_entry()
        device = entry["device"] if entry is not None else ""
        self._variable_model.setStringList(self._device_vars.get(device, []))

    # ------------------------------------------------------------------
    # Document plumbing
    # ------------------------------------------------------------------

    def _show_message(self, text: str, error: bool = False) -> None:
        """Render *text* in the inline message label.

        Parameters
        ----------
        text : str
            The operator-facing line ("" clears the label).
        error : bool, optional
            Render in the danger color instead of the dim one.
        """
        self.message_label.setStyleSheet(
            f"color: {_COLOR_RED if error else _COLOR_DIM};"
        )
        self.message_label.setText(text)

    def _validate(self) -> Optional[SaveSet]:
        """Validate the open document, rendering any error inline.

        Returns
        -------
        SaveSet or None
            The validated model, or ``None`` when nothing is open or the
            document does not validate (the message label explains why).
        """
        if self._document is None:
            return None
        try:
            model = SaveSet.model_validate(self._document)
        except ValidationError as exc:
            errors = exc.errors()
            first = errors[0]
            location = ".".join(str(part) for part in first["loc"]) or "document"
            more = f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""
            self._show_message(
                f"Invalid save set — {location}: {first['msg']}{more}", error=True
            )
            return None
        self._show_message("")
        return model

    def _touch(self) -> None:
        """Mark the document dirty and refresh validation + button gating."""
        if self._loading:
            return
        self._dirty = True
        self._refresh_gating()

    def _refresh_gating(self) -> None:
        """Recompute Save/Revert enabled state, title, and validation label."""
        valid = self._validate() is not None
        self.save_button.setEnabled(self._dirty and valid)
        self.revert_button.setEnabled(self._dirty)
        self.setWindowTitle(self._title())

    def _current_name(self) -> str:
        """Return the open document's name ("" when nothing is open)."""
        if self._document is None:
            return ""
        return str(self._document.get("name", ""))

    def _current_entry(self) -> Optional[dict]:
        """Return the selected device's entry dict, or ``None``."""
        if self._document is None:
            return None
        row = self.device_list.currentRow()
        entries = self._document.get("entries", [])
        if 0 <= row < len(entries):
            return entries[row]
        return None

    # ------------------------------------------------------------------
    # List / selection management
    # ------------------------------------------------------------------

    def _refresh_set_list(self, select: str = "") -> None:
        """Rebuild the save-set list from the store plus unsaved names.

        Parameters
        ----------
        select : str, optional
            Name to select silently after the rebuild ("" selects nothing).
        """
        names = sorted(set(self._store.list_names()) | self._unsaved)
        self._suppress_selection = True
        try:
            self.set_list.clear()
            self.set_list.addItems(names)
            if select and select in names:
                row = names.index(select)
                self.set_list.setCurrentRow(row)
                self._selected_row = row
            else:
                self.set_list.setCurrentRow(-1)
                self._selected_row = -1
        finally:
            self._suppress_selection = False

    def _select_row_silently(self, row: int) -> None:
        """Move the list selection without triggering the change handler."""
        self._suppress_selection = True
        try:
            self.set_list.setCurrentRow(row)
        finally:
            self._suppress_selection = False

    def _on_set_row_changed(self, row: int) -> None:
        """Open the newly selected save set, honoring unsaved changes."""
        if self._suppress_selection or row == self._selected_row:
            return
        item = self.set_list.item(row)
        target = item.text() if item is not None else ""
        if not self._resolve_unsaved():
            self._select_row_silently(self._selected_row)
            return
        if not target:
            self._selected_row = row
            self._clear_document()
            return
        # Leaving may have dropped a discarded unsaved item — re-resolve by
        # name so row indices can't lie.
        names = [self.set_list.item(i).text() for i in range(self.set_list.count())]
        if target not in names:
            self._selected_row = -1
            self._clear_document()
            return
        new_row = names.index(target)
        self._select_row_silently(new_row)
        self._selected_row = new_row
        self._open_set(target)

    def _has_unsaved(self) -> bool:
        """Unsaved-changes hook: an open document with unsaved edits."""
        return self._dirty and self._document is not None

    def _save_unsaved(self) -> bool:
        """Unsaved-changes hook: run the normal save (staying put on failure)."""
        return self._save()

    def _discard_unsaved(self) -> None:
        """Unsaved-changes hook: a never-saved set evaporates with its edits."""
        name = self._current_name()
        if name in self._unsaved:
            self._unsaved.discard(name)
            self._drop_list_item(name)
        self._dirty = False

    def _unsaved_prompt_text(self) -> str:
        """Name the dirty save set in the unsaved-changes prompt."""
        return f"Save set {self._current_name()!r} has unsaved changes."

    def _drop_list_item(self, name: str) -> None:
        """Remove *name*'s row from the set list without side effects."""
        self._suppress_selection = True
        try:
            for row in range(self.set_list.count()):
                if self.set_list.item(row).text() == name:
                    self.set_list.takeItem(row)
                    break
        finally:
            self._suppress_selection = False

    # closeEvent / reject / keyPressEvent: ConfigEditorDialog (Esc now
    # routes through the unsaved-changes prompt instead of silently
    # hiding a dirty dialog).

    # ------------------------------------------------------------------
    # Opening / clearing documents
    # ------------------------------------------------------------------

    def _set_document_enabled(self, enabled: bool) -> None:
        """Enable/disable the whole right-hand document panel."""
        for widget in (
            self.description_edit,
            self.device_list,
            self.device_edit,
            self.add_device_button,
            self.remove_device_button,
            self.entry_group,
        ):
            widget.setEnabled(enabled)

    def _clear_document(self) -> None:
        """Show the no-document state (nothing selected / nothing open)."""
        self._document = None
        self._dirty = False
        self._loading = True
        try:
            self.name_label.setText("—")
            self.description_edit.clear()
            self.device_list.clear()
            self.device_edit.clear()
            self._populate_entry_panel()
        finally:
            self._loading = False
        self._set_document_enabled(False)
        self.save_button.setEnabled(False)
        self.revert_button.setEnabled(False)
        self.setWindowTitle(self._title())

    def _open_set(self, name: str) -> None:
        """Load save set *name* from the store into the form.

        A load failure (missing file, bad YAML, schema rejection, a lossy
        legacy element) is rendered inline and leaves the no-document state.

        Parameters
        ----------
        name : str
            The save-set name (list item text / file stem).
        """
        try:
            model = self._store.load(name)
        except SaveSetStoreError as exc:
            self._clear_document()
            self._show_message(str(exc), error=True)
            return
        document = model.model_dump(mode="json")
        if document.get("name") != name:
            logger.info(
                "save set %r carries internal name %r — normalizing to the file stem",
                name,
                document.get("name"),
            )
            document["name"] = name
        self._adopt_document(document, dirty=False)
        self._show_message("")

    def _adopt_document(self, document: dict, dirty: bool) -> None:
        """Render *document* in the form and make it the open document.

        Parameters
        ----------
        document : dict
            A ``SaveSet``-shaped mapping (``model_dump(mode="json")``).
        dirty : bool
            Whether the document already differs from the store (new /
            duplicated sets start dirty).
        """
        self._document = document
        self._dirty = dirty
        self._loading = True
        try:
            self.name_label.setText(document.get("name", "") or "—")
            self.description_edit.setText(document.get("description", ""))
            self.device_list.clear()
            for entry in document.get("entries", []):
                self.device_list.addItem(entry.get("device", ""))
            if self.device_list.count():
                self.device_list.setCurrentRow(0)
            self.device_edit.clear()
            self._populate_entry_panel()
        finally:
            self._loading = False
        self._set_document_enabled(True)
        self._refresh_gating()

    def _adopt_unsaved(self, document: dict) -> None:
        """Add *document* as a new, unsaved list entry and open it."""
        name = document["name"]
        self._unsaved.add(name)
        self._refresh_set_list(select=name)
        self._adopt_document(document, dirty=True)

    # ------------------------------------------------------------------
    # Left-column actions (New / Duplicate / Rename / Delete)
    # ------------------------------------------------------------------

    def _known_names(self) -> set[str]:
        """Every name the list shows (stored plus unsaved)."""
        return set(self._store.list_names()) | self._unsaved

    def _prompt_set_name(self, title: str, initial: str = "") -> str:
        """Ask for a save-set name; "" when cancelled, empty, or taken.

        A collision-checking wrapper over the base :meth:`_prompt_name`
        (the instance-monkeypatchable prompt seam).

        Parameters
        ----------
        title : str
            The input dialog's title.
        initial : str, optional
            Prefilled text.

        Returns
        -------
        str
            The accepted, stripped, collision-free name — or "".
        """
        name = self._prompt_name(title, "Save set name:", initial=initial)
        if not name:
            return ""
        if name in self._known_names():
            self._show_message(f"A save set named {name!r} already exists.", error=True)
            return ""
        return name

    def _on_new(self) -> None:
        """Create a new (empty, not-yet-valid) save set in memory."""
        if not self._resolve_unsaved():
            return
        name = self._prompt_set_name("New save set")
        if not name:
            return
        self._adopt_unsaved(
            {"schema_version": 1, "name": name, "entries": [], "description": ""}
        )

    def _on_duplicate(self) -> None:
        """Copy the open save set (as shown, edits included) under a new name."""
        if self._document is None:
            self._show_message("No save set selected.")
            return
        duplicate = copy.deepcopy(self._document)
        if not self._resolve_unsaved():
            return
        name = self._prompt_set_name(
            "Duplicate save set", initial=f"{duplicate.get('name', '')}-copy"
        )
        if not name:
            return
        duplicate["name"] = name
        self._adopt_unsaved(duplicate)

    def _on_rename(self) -> None:
        """Rename the open save set (file and document name stay in step)."""
        if self._document is None:
            self._show_message("No save set selected.")
            return
        old = self._current_name()
        new = self._prompt_set_name("Rename save set", initial=old)
        if not new:
            return
        if old in self._unsaved:
            self._unsaved.discard(old)
            self._unsaved.add(new)
        else:
            try:
                self._store.rename(old, new)
            except SaveSetStoreError as exc:
                self._show_message(str(exc), error=True)
                return
        self._document["name"] = new
        self.name_label.setText(new)
        self._refresh_set_list(select=new)
        self._refresh_gating()
        self._show_message(f"Renamed {old!r} to {new!r}.")

    def _on_delete(self) -> None:
        """Delete the open save set after a confirmation prompt."""
        if self._document is None:
            self._show_message("No save set selected.")
            return
        name = self._current_name()
        if not self._confirm("Delete save set", f"Delete save set {name!r}?"):
            return
        if name in self._unsaved:
            self._unsaved.discard(name)
        else:
            try:
                self._store.delete(name)
            except SaveSetStoreError as exc:
                self._show_message(str(exc), error=True)
                return
        self._dirty = False
        self._refresh_set_list()
        self._clear_document()
        self._show_message(f"Deleted {name!r}.")

    # ------------------------------------------------------------------
    # Document edits (write-through to the dict, then touch)
    # ------------------------------------------------------------------

    def _on_description_changed(self, text: str) -> None:
        """Write the description through to the document."""
        if self._loading or self._document is None:
            return
        self._document["description"] = text
        self._touch()

    def _on_add_device(self) -> None:
        """Append a new entry (schema defaults) for the typed device."""
        if self._document is None:
            return
        device = self.device_edit.text().strip()
        if not device:
            return
        existing = [entry.get("device") for entry in self._document.get("entries", [])]
        if device in existing:
            self._show_message(
                f"Device {device!r} is already in this save set.", error=True
            )
            return
        entry = SaveSetEntry(device=device).model_dump(mode="json")
        self._document.setdefault("entries", []).append(entry)
        self.device_list.addItem(device)
        self.device_list.setCurrentRow(self.device_list.count() - 1)
        self.device_edit.clear()
        self._touch()

    def _on_remove_device(self) -> None:
        """Remove the selected device's entry."""
        if self._document is None:
            return
        row = self.device_list.currentRow()
        entries = self._document.get("entries", [])
        if not (0 <= row < len(entries)):
            return
        del entries[row]
        self.device_list.takeItem(row)
        self._populate_entry_panel()
        self._touch()

    def _on_device_row_changed(self, _row: int) -> None:
        """Show the newly selected device's entry fields."""
        if self._loading:
            return
        self._populate_entry_panel()

    def _populate_entry_panel(self) -> None:
        """Render the selected entry into the detail widgets (or blank them)."""
        entry = self._current_entry()
        was_loading = self._loading
        self._loading = True
        try:
            if entry is None:
                self.entry_group.setEnabled(False)
                self.images_check.setChecked(False)
                self.db_scalars_check.setChecked(True)
                self.all_scalars_check.setChecked(False)
                self.role_combo.setCurrentIndex(0)
                self.scalar_list.clear()
                self.scalar_edit.clear()
                self.setup_edit.clear()
                self.closeout_edit.clear()
            else:
                self.entry_group.setEnabled(self._document is not None)
                self.images_check.setChecked(bool(entry.get("images", False)))
                self.db_scalars_check.setChecked(bool(entry.get("db_scalars", True)))
                self.all_scalars_check.setChecked(bool(entry.get("all_scalars", False)))
                index = self.role_combo.findData(entry.get("role"))
                self.role_combo.setCurrentIndex(max(index, 0))
                self.scalar_list.clear()
                self.scalar_list.addItems(entry.get("scalars", []))
                self.scalar_edit.clear()
                self.setup_edit.setText(", ".join(entry.get("setup", [])))
                self.closeout_edit.setText(", ".join(entry.get("closeout", [])))
        finally:
            self._loading = was_loading
        self._refresh_variable_completer()

    def _on_images_toggled(self, checked: bool) -> None:
        """Write the images flag through to the selected entry."""
        entry = self._current_entry()
        if self._loading or entry is None:
            return
        entry["images"] = checked
        self._touch()

    def _on_db_scalars_toggled(self, checked: bool) -> None:
        """Write the db_scalars flag through to the selected entry."""
        entry = self._current_entry()
        if self._loading or entry is None:
            return
        entry["db_scalars"] = checked
        self._touch()

    def _on_all_scalars_toggled(self, checked: bool) -> None:
        """Write the all_scalars flag through to the selected entry."""
        entry = self._current_entry()
        if self._loading or entry is None:
            return
        entry["all_scalars"] = checked
        self._touch()

    def _on_role_changed(self, index: int) -> None:
        """Write the role override through to the selected entry."""
        entry = self._current_entry()
        if self._loading or entry is None:
            return
        entry["role"] = self.role_combo.itemData(index)
        self._touch()

    def _on_add_scalar(self) -> None:
        """Append the typed variable to the selected entry's scalars."""
        entry = self._current_entry()
        if entry is None:
            return
        variable = self.scalar_edit.text().strip()
        if not variable:
            return
        if variable in entry.get("scalars", []):
            self._show_message(f"Variable {variable!r} is already listed.", error=True)
            return
        entry.setdefault("scalars", []).append(variable)
        self.scalar_list.addItem(variable)
        self.scalar_edit.clear()
        self._touch()

    def _on_remove_scalar(self) -> None:
        """Remove the selected variable from the entry's scalars."""
        entry = self._current_entry()
        if entry is None:
            return
        row = self.scalar_list.currentRow()
        scalars = entry.get("scalars", [])
        if not (0 <= row < len(scalars)):
            return
        del scalars[row]
        self.scalar_list.takeItem(row)
        self._touch()

    def _on_setup_changed(self, text: str) -> None:
        """Write the setup plan references through to the selected entry."""
        entry = self._current_entry()
        if self._loading or entry is None:
            return
        entry["setup"] = _split_names(text)
        self._touch()

    def _on_closeout_changed(self, text: str) -> None:
        """Write the closeout plan references through to the selected entry."""
        entry = self._current_entry()
        if self._loading or entry is None:
            return
        entry["closeout"] = _split_names(text)
        self._touch()

    # ------------------------------------------------------------------
    # Save / Revert
    # ------------------------------------------------------------------

    def _save(self) -> bool:
        """Validate and persist the open document.

        Returns
        -------
        bool
            ``True`` when the save landed (validation and write both OK).
        """
        model = self._validate()
        if model is None:
            self._refresh_gating()
            return False
        name = self._current_name()
        try:
            self._store.save(name, model)
        except SaveSetStoreError as exc:
            self._show_message(str(exc), error=True)
            return False
        self._dirty = False
        self._unsaved.discard(name)
        self._refresh_set_list(select=name)
        self._refresh_gating()
        self._show_message(f"Saved {name!r}.")
        return True

    def _on_save(self) -> None:
        """Save button handler."""
        self._save()

    def _on_revert(self) -> None:
        """Throw away unsaved edits, restoring the stored document.

        A never-saved set reverts to its empty starting point (there is
        nothing on disk to restore).
        """
        if self._document is None:
            return
        name = self._current_name()
        if name in self._unsaved:
            self._adopt_document(
                {"schema_version": 1, "name": name, "entries": [], "description": ""},
                dirty=True,
            )
            self._show_message(f"Reverted {name!r} to an empty new set.")
            return
        self._dirty = False
        self._open_set(name)
        self._show_message(f"Reverted {name!r}.")


def open_save_set_editor(
    parent: Optional[QWidget],
    experiment: str,
    configs_base: Optional[Path] = None,
    completions: Optional[CompletionsProvider] = None,
) -> QDialog:
    """Open the save-set editor for *experiment* (the Editors-menu entry point).

    Parameters
    ----------
    parent : QWidget or None
        The owning window.
    experiment : str
        Experiment whose save sets to edit ("" opens empty and offline-safe).
    configs_base : Path, optional
        Experiments-root override (tests); defaults to the production
        configs-repo resolution.
    completions : CompletionsProvider, optional
        Completer word-list override; defaults to
        :class:`~geecs_console.services.device_completions.GeecsDbCompletions`
        (lazy ``GeecsDb`` import, daemon-thread fetch, empty offline).

    Returns
    -------
    QDialog
        The shown (non-modal) editor dialog.
    """
    editor = SaveSetEditor(
        parent,
        experiment=experiment,
        configs_base=configs_base,
        completions=(
            completions if completions is not None else GeecsDbCompletions(experiment)
        ),
    )
    editor.show()
    return editor
