"""The scan-variable editor (M5): edit one experiment's catalog of scan variables.

One modeless :class:`ScanVariableEditor` dialog edits the whole
:class:`~geecs_schemas.ScanVariables` document —
``scan_devices/scan_variables.yaml`` — through
:class:`~geecs_console.services.scan_variable_store.ScanVariableStore`.
The left pane lists every variable with a *simple* / *composite* type tag;
the right pane is a form derived from the actual schema models:

- **Simple** (:class:`~geecs_schemas.ScanVariable`): device + variable (the
  ``target``), the ``kind`` (setpoint / motor), and the optional ``confirm``
  target for set-one-measure-another devices.
- **Composite** (:class:`~geecs_schemas.PseudoScanVariable`, ``kind:
  pseudo``): the ``mode`` (absolute / relative — the forward column header
  and hint re-label to match), an editable components table (device,
  variable, ``forward`` expression in terms of ``composite_var``), and the
  optional ``inverse`` readback formula.

Edits accumulate in an in-memory draft of plain dicts (so *invalid
intermediate* states are representable while typing); Save validates the
whole draft through ``ScanVariables.model_validate`` and writes via the
store — pydantic errors are shown inline in the error label, never a crash.
Revert reloads from disk; closing with unsaved changes prompts first.

Ergonomics: device and variable fields carry ``QCompleter`` popups fed by an
injectable :class:`~geecs_console.services.device_completions.
CompletionsProvider` — fetched once on a short-lived daemon thread (the
package's no-QThread rule) and marshaled back through a queued signal, so a
slow DB never blocks the GUI.  Enter never accepts the dialog (no default
buttons; Return/Enter are swallowed) — it must be safe to hit Enter after
typing in a field.

The later integration PR wires the Editors menu to
:func:`open_scan_variable_editor`; nothing here imports the main window.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import ValidationError
from PySide6.QtCore import QStringListModel, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QCompleter,
    QDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QWidget,
)

from geecs_console.editors.base import ConfigEditorDialog
from geecs_console.services.device_completions import (
    CompletionsProvider,
    EmptyCompletions,
    GeecsDbCompletions,
)
from geecs_console.services.scan_variable_store import (
    ScanVariableStore,
    ScanVariableStoreError,
)
from geecs_console.services.schema_tooltips import apply_schema_tooltips
from geecs_schemas import (
    CompositeMode,
    PseudoScanVariable,
    ScanVariable,
    ScanVariables,
)

logger = logging.getLogger(__name__)

_UI_PATH = Path(__file__).parent.parent / "app" / "ui" / "scan_variable_editor.ui"

#: The two simple-variable kinds, in combo order (schema Literal values).
_SIMPLE_KINDS = ("setpoint", "motor")

#: Composite modes in combo order (schema enum values).
_COMPOSITE_MODES = tuple(mode.value for mode in CompositeMode)

#: Per-mode forward-column header + hint (the abs/rel labeling the schema
#: semantics make relevant).
_MODE_LABELS = {
    "absolute": (
        "Forward → absolute value",
        "absolute: each device goes exactly where its formula says.",
    ),
    "relative": (
        "Forward → offset from scan start",
        "relative: each device is offset from where it was at scan start.",
    ),
}

#: Union branch tags in pydantic error locations, per draft shape — used to
#: drop the *other* branch's noise from smart-union validation errors.
_BRANCH_TAGS = {"ScanVariable", "PseudoScanVariable"}


def _split_target(target: str) -> tuple[str, str]:
    """Split a ``Device:Variable`` string on its first colon.

    Parameters
    ----------
    target : str
        The stored target string (possibly empty or malformed).

    Returns
    -------
    tuple of str
        ``(device, variable)`` — the variable part keeps any further colons.
    """
    device, _, variable = target.partition(":")
    return device, variable


def _join_target(device: str, variable: str) -> str:
    """Join device/variable form fields back into a ``Device:Variable`` string.

    Parameters
    ----------
    device : str
        The device field text (stripped).
    variable : str
        The variable field text (stripped).

    Returns
    -------
    str
        ``"Device:Variable"``, or ``""`` when both fields are empty (so an
        untouched pair reads as "unset" rather than a bare colon).
    """
    if not device and not variable:
        return ""
    return f"{device}:{variable}"


def _format_validation_error(exc: ValidationError, drafts: dict[str, dict]) -> str:
    """Render a pydantic error compactly for the inline error label.

    Parameters
    ----------
    exc : ValidationError
        The error ``ScanVariables.model_validate`` raised over the draft.
    drafts : dict
        The draft variables (name → field dict) — used to drop the smart
        union's *other-branch* noise: a composite draft only shows
        ``PseudoScanVariable`` branch errors and vice versa.

    Returns
    -------
    str
        One ``variables → name → field: message`` line per relevant error,
        capped at eight lines.
    """
    lines: list[str] = []
    for error in exc.errors():
        loc = [str(part) for part in error["loc"]]
        if len(loc) >= 2 and loc[0] == "variables":
            name = loc[1]
            is_pseudo = (drafts.get(name) or {}).get("kind") == "pseudo"
            wanted = "PseudoScanVariable" if is_pseudo else "ScanVariable"
            other = _BRANCH_TAGS - {wanted}
            if any(tag in loc for tag in other):
                continue
        display = [part for part in loc if part not in _BRANCH_TAGS]
        line = f"{' → '.join(display)}: {error['msg']}"
        if line not in lines:
            lines.append(line)
    if not lines:  # every branch filtered — fall back to the raw first error
        first = exc.errors()[0]
        lines = [f"{' → '.join(str(p) for p in first['loc'])}: {first['msg']}"]
    shown = lines[:8]
    if len(lines) > len(shown):
        shown.append(f"… and {len(lines) - len(shown)} more")
    return "\n".join(shown)


class ScanVariableEditor(ConfigEditorDialog):
    """Editor dialog for one experiment's scan-variable catalog.

    Parameters
    ----------
    store : ScanVariableStore
        Catalog persistence; tests inject one rooted at a tmp dir.
    completions : CompletionsProvider, optional
        Device/variable completion source; default is
        :class:`~geecs_console.services.device_completions.EmptyCompletions`
        (no suggestions — offline/test default).  The provider's one blocking
        call runs on a daemon thread, never the GUI thread.
    parent : QWidget, optional
        Parent widget (the main window in production).
    """

    UI_PATH = _UI_PATH
    INITIAL_SIZE = (900, 560)
    COMPLETIONS_THREAD_NAME = "sve-completions-fetch"

    def __init__(
        self,
        store: ScanVariableStore,
        completions: Optional[CompletionsProvider] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._init_completions(completions)
        #: name → plain field dict (a ScanVariableSpec dump; invalid
        #: intermediate states are representable while the operator types).
        self._variables: dict[str, dict] = {}
        self._snapshot: dict[str, dict] = {}
        self._schema_version = 1
        self._current_name: Optional[str] = None
        self._loading = False

        #: Test seam: confirm discarding the draft on Revert (the leave
        #: prompts go through the base `_prompt_unsaved` seam instead).
        self._confirm_discard: Callable[[], bool] = self._confirm_discard_modal

        self._load_ui()
        self._bind_widgets()
        self._populate_static_combos()
        self._build_completers()
        self._wire_signals()
        self._guard_enter_keys()

        self.setWindowTitle(
            f"Scan Variables — {store.experiment}"
            if store.experiment
            else "Scan Variables"
        )

        self._reload_from_store(initial=True)
        self._start_completions_fetch()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _bind_widgets(self) -> None:
        """Resolve every wired widget from the loaded .ui once."""
        self.variable_list: QListWidget = self._child(QListWidget, "sve_variable_list")
        self.new_simple_button: QPushButton = self._child(
            QPushButton, "sve_new_simple_button"
        )
        self.new_pseudo_button: QPushButton = self._child(
            QPushButton, "sve_new_pseudo_button"
        )
        self.duplicate_button: QPushButton = self._child(
            QPushButton, "sve_duplicate_button"
        )
        self.rename_button: QPushButton = self._child(QPushButton, "sve_rename_button")
        self.delete_button: QPushButton = self._child(QPushButton, "sve_delete_button")

        self.type_label: QLabel = self._child(QLabel, "sve_type_label")
        self.form_stack: QStackedWidget = self._child(QStackedWidget, "sve_form_stack")
        self.empty_page: QWidget = self._child(QWidget, "sve_empty_page")
        self.simple_page: QWidget = self._child(QWidget, "sve_simple_page")
        self.pseudo_page: QWidget = self._child(QWidget, "sve_pseudo_page")

        self.device_edit: QLineEdit = self._child(QLineEdit, "sve_device_edit")
        self.variable_edit: QLineEdit = self._child(QLineEdit, "sve_variable_edit")
        self.kind_combo: QComboBox = self._child(QComboBox, "sve_kind_combo")
        self.confirm_device_edit: QLineEdit = self._child(
            QLineEdit, "sve_confirm_device_edit"
        )
        self.confirm_variable_edit: QLineEdit = self._child(
            QLineEdit, "sve_confirm_variable_edit"
        )

        self.mode_combo: QComboBox = self._child(QComboBox, "sve_mode_combo")
        self.mode_hint_label: QLabel = self._child(QLabel, "sve_mode_hint_label")
        self.components_table: QTableWidget = self._child(
            QTableWidget, "sve_components_table"
        )
        self.add_component_button: QPushButton = self._child(
            QPushButton, "sve_add_component_button"
        )
        self.remove_component_button: QPushButton = self._child(
            QPushButton, "sve_remove_component_button"
        )
        self.inverse_edit: QLineEdit = self._child(QLineEdit, "sve_inverse_edit")

        self.error_label: QLabel = self._child(QLabel, "sve_error_label")
        self.dirty_label: QLabel = self._child(QLabel, "sve_dirty_label")
        self.revert_button: QPushButton = self._child(QPushButton, "sve_revert_button")
        self.save_button: QPushButton = self._child(QPushButton, "sve_save_button")
        self.close_button: QPushButton = self._child(QPushButton, "sve_close_button")

        # Tooltips come from the schema field descriptions — one source of
        # truth for what each field means (issue #497 phase 1).  The
        # Device/Variable edit pairs jointly form one 'Device:Variable'
        # schema field, so both halves carry that field's description.
        apply_schema_tooltips(
            ScanVariable,
            {
                "target": [self.device_edit, self.variable_edit],
                "kind": self.kind_combo,
                "confirm": [self.confirm_device_edit, self.confirm_variable_edit],
            },
        )
        apply_schema_tooltips(
            PseudoScanVariable,
            {
                "mode": self.mode_combo,
                "targets": self.components_table,
                "inverse": self.inverse_edit,
            },
        )

    def _populate_static_combos(self) -> None:
        """Fill the kind and mode combos from the schema vocabularies."""
        self.kind_combo.addItems(list(_SIMPLE_KINDS))
        self.mode_combo.addItems(list(_COMPOSITE_MODES))

    def _build_completers(self) -> None:
        """Attach device/variable completers (models filled when the fetch lands)."""
        #: One shared device-name model; each field gets its own QCompleter
        #: (a completer binds to a single widget) over this model.
        self._device_model = QStringListModel(self)
        for edit in (self.device_edit, self.confirm_device_edit):
            edit.setCompleter(self._make_completer(self._device_model))
        #: Variable models are per-field — the word list follows the paired
        #: device field's text.
        self._variable_model = QStringListModel(self)
        self.variable_edit.setCompleter(self._make_completer(self._variable_model))
        self._confirm_variable_model = QStringListModel(self)
        self.confirm_variable_edit.setCompleter(
            self._make_completer(self._confirm_variable_model)
        )

    def _make_completer(self, model: Any) -> QCompleter:
        """Build one case-insensitive, contains-matching completer over *model*.

        Parameters
        ----------
        model : QStringListModel
            The word-list model the completer reads.

        Returns
        -------
        QCompleter
            Ready to hand to ``QLineEdit.setCompleter``.
        """
        completer = QCompleter(model, self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        return completer

    def _wire_signals(self) -> None:
        """Connect widgets to handlers."""
        self.variable_list.currentItemChanged.connect(self._on_selection_changed)
        self.new_simple_button.clicked.connect(self._on_new_simple)
        self.new_pseudo_button.clicked.connect(self._on_new_pseudo)
        self.duplicate_button.clicked.connect(self._on_duplicate)
        self.rename_button.clicked.connect(self._on_rename)
        self.delete_button.clicked.connect(self._on_delete)

        for edit in (
            self.device_edit,
            self.variable_edit,
            self.confirm_device_edit,
            self.confirm_variable_edit,
            self.inverse_edit,
        ):
            edit.textChanged.connect(self._on_form_edited)
        self.kind_combo.currentIndexChanged.connect(self._on_form_edited)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self.add_component_button.clicked.connect(self._on_add_component)
        self.remove_component_button.clicked.connect(self._on_remove_component)

        self.device_edit.textChanged.connect(
            lambda text: self._refresh_variable_model(self._variable_model, text)
        )
        self.confirm_device_edit.textChanged.connect(
            lambda text: self._refresh_variable_model(
                self._confirm_variable_model, text
            )
        )

        self.revert_button.clicked.connect(self._on_revert)
        self.save_button.clicked.connect(self._on_save)
        self.close_button.clicked.connect(self.close)

    # ------------------------------------------------------------------
    # Modal seams (overridable in tests)
    # ------------------------------------------------------------------

    def _confirm_discard_modal(self) -> bool:
        """Ask whether to discard unsaved changes (modal).

        Returns
        -------
        bool
            True to discard and proceed, False to stay.
        """
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            "The scan-variable catalog has unsaved changes. Discard them?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Discard

    # ------------------------------------------------------------------
    # Catalog <-> draft
    # ------------------------------------------------------------------

    def _reload_from_store(self, initial: bool = False) -> None:
        """Load the catalog from disk into the draft and repopulate.

        Parameters
        ----------
        initial : bool, optional
            True on construction — a load failure then leaves an empty,
            usable editor with the error shown inline (never a crash).
        """
        try:
            catalog = self._store.load()
        except ScanVariableStoreError as exc:
            if not initial:
                raise
            logger.warning("scan-variable catalog failed to load: %s", exc)
            catalog = ScanVariables(variables={})
            self._show_error(str(exc))
        else:
            self._clear_error()
        self._schema_version = catalog.schema_version
        self._variables = {
            name: spec.model_dump(mode="json", exclude_none=True)
            for name, spec in catalog.variables.items()
        }
        self._snapshot = copy.deepcopy(self._variables)
        self._refresh_list(select=next(iter(self._variables), None))
        self._update_dirty()

    def _draft_catalog(self) -> ScanVariables:
        """Validate the current draft as a :class:`ScanVariables` document.

        Returns
        -------
        ScanVariables
            The validated catalog.

        Raises
        ------
        pydantic.ValidationError
            When any draft entry does not satisfy the schema.
        """
        return ScanVariables.model_validate(
            {"schema_version": self._schema_version, "variables": self._variables}
        )

    @property
    def dirty(self) -> bool:
        """Whether the draft differs from what was last loaded/saved."""
        return self._variables != self._snapshot

    # ------------------------------------------------------------------
    # List handling
    # ------------------------------------------------------------------

    def _refresh_list(self, select: Optional[str]) -> None:
        """Rebuild the variable list from the draft, selecting *select*.

        Parameters
        ----------
        select : str or None
            Variable name to select after the rebuild (None clears the form).
        """
        self._loading = True
        try:
            self.variable_list.clear()
            for name, draft in self._variables.items():
                kind = "composite" if draft.get("kind") == "pseudo" else "simple"
                item = QListWidgetItem(f"{name}    [{kind}]")
                item.setData(Qt.ItemDataRole.UserRole, name)
                self.variable_list.addItem(item)
        finally:
            self._loading = False
        if select is not None:
            for row in range(self.variable_list.count()):
                item = self.variable_list.item(row)
                if item.data(Qt.ItemDataRole.UserRole) == select:
                    self.variable_list.setCurrentItem(item)
                    return
        self.variable_list.setCurrentItem(None)
        self._populate_form(None)

    def _on_selection_changed(
        self, current: Optional[QListWidgetItem], _previous: Optional[QListWidgetItem]
    ) -> None:
        """Populate the form for the newly selected variable."""
        if self._loading:
            return
        name = current.data(Qt.ItemDataRole.UserRole) if current is not None else None
        self._populate_form(name)

    def _unique_name(self, base: str) -> str:
        """Return *base*, suffixed with a counter when the name is taken.

        Parameters
        ----------
        base : str
            The wanted name.

        Returns
        -------
        str
            ``base``, or ``base_2`` / ``base_3`` / … — first free.
        """
        if base not in self._variables:
            return base
        counter = 2
        while f"{base}_{counter}" in self._variables:
            counter += 1
        return f"{base}_{counter}"

    def _on_new_simple(self) -> None:
        """Add a fresh simple variable and select it."""
        name = self._unique_name("new_variable")
        self._variables[name] = {"target": "", "kind": "setpoint"}
        self._refresh_list(select=name)
        self._update_dirty()

    def _on_new_pseudo(self) -> None:
        """Add a fresh composite (pseudo) variable and select it."""
        name = self._unique_name("new_composite")
        self._variables[name] = {
            "kind": "pseudo",
            "targets": [{"target": "", "forward": ""}],
            "mode": "absolute",
        }
        self._refresh_list(select=name)
        self._update_dirty()

    def _on_duplicate(self) -> None:
        """Duplicate the selected variable under a fresh name."""
        if self._current_name is None:
            return
        name = self._unique_name(f"{self._current_name}_copy")
        self._variables[name] = copy.deepcopy(self._variables[self._current_name])
        self._refresh_list(select=name)
        self._update_dirty()

    def _on_rename(self) -> None:
        """Rename the selected variable (order-preserving), rejecting collisions."""
        old = self._current_name
        if old is None:
            return
        new = self._prompt_name("Rename scan variable", "New name:", old)
        if new is None:
            return
        new = new.strip()
        if not new or new == old:
            return
        if new in self._variables:
            self._show_error(f"A variable named {new!r} already exists.")
            return
        self._variables = {
            (new if name == old else name): draft
            for name, draft in self._variables.items()
        }
        self._refresh_list(select=new)
        self._update_dirty()
        self._clear_error()

    def _on_delete(self) -> None:
        """Delete the selected variable from the draft (persisted on Save)."""
        if self._current_name is None:
            return
        names = list(self._variables)
        index = names.index(self._current_name)
        del self._variables[self._current_name]
        remaining = list(self._variables)
        select = remaining[min(index, len(remaining) - 1)] if remaining else None
        self._refresh_list(select=select)
        self._update_dirty()

    # ------------------------------------------------------------------
    # Form handling
    # ------------------------------------------------------------------

    def _populate_form(self, name: Optional[str]) -> None:
        """Fill the form pane from the draft entry *name* (None clears it).

        Parameters
        ----------
        name : str or None
            The variable to show; None shows the empty page.
        """
        self._current_name = name
        has_selection = name is not None
        for button in (self.duplicate_button, self.rename_button, self.delete_button):
            button.setEnabled(has_selection)
        if name is None:
            self.form_stack.setCurrentWidget(self.empty_page)
            self.type_label.setText("")
            return
        draft = self._variables[name]
        self._loading = True
        try:
            if draft.get("kind") == "pseudo":
                self.form_stack.setCurrentWidget(self.pseudo_page)
                self.type_label.setText(
                    "Composite (pseudo) variable — one number moves several devices"
                )
                mode = str(draft.get("mode", "absolute"))
                index = self.mode_combo.findText(mode)
                self.mode_combo.setCurrentIndex(index if index >= 0 else 0)
                self.inverse_edit.setText(str(draft.get("inverse", "") or ""))
                self._set_component_rows(draft.get("targets", []))
                self._update_mode_labels()
            else:
                self.form_stack.setCurrentWidget(self.simple_page)
                self.type_label.setText("Simple variable — one device variable")
                device, variable = _split_target(str(draft.get("target", "")))
                self.device_edit.setText(device)
                self.variable_edit.setText(variable)
                kind = str(draft.get("kind", "setpoint"))
                index = self.kind_combo.findText(kind)
                self.kind_combo.setCurrentIndex(index if index >= 0 else 0)
                confirm_device, confirm_variable = _split_target(
                    str(draft.get("confirm", "") or "")
                )
                self.confirm_device_edit.setText(confirm_device)
                self.confirm_variable_edit.setText(confirm_variable)
        finally:
            self._loading = False

    def _set_component_rows(self, targets: list[dict]) -> None:
        """Rebuild the components table with one editable row per target.

        Parameters
        ----------
        targets : list of dict
            The draft's ``targets`` entries (``target`` + ``forward``).
        """
        self.components_table.setRowCount(0)
        for entry in targets:
            device, variable = _split_target(str(entry.get("target", "")))
            self._append_component_row(device, variable, str(entry.get("forward", "")))

    def _append_component_row(
        self, device: str = "", variable: str = "", forward: str = ""
    ) -> None:
        """Append one component row of line-edit cell widgets.

        Parameters
        ----------
        device : str, optional
            Device cell text.
        variable : str, optional
            Variable cell text.
        forward : str, optional
            Forward-expression cell text.
        """
        row = self.components_table.rowCount()
        self.components_table.insertRow(row)

        device_edit = QLineEdit(device)
        device_edit.setCompleter(self._make_completer(self._device_model))
        variable_edit = QLineEdit(variable)
        variable_model = QStringListModel(variable_edit)
        variable_edit.setCompleter(self._make_completer(variable_model))
        forward_edit = QLineEdit(forward)
        forward_edit.setPlaceholderText("e.g. composite_var * -2")

        self._refresh_variable_model(variable_model, device)
        device_edit.textChanged.connect(
            lambda text, model=variable_model: self._refresh_variable_model(model, text)
        )
        for edit in (device_edit, variable_edit, forward_edit):
            edit.textChanged.connect(self._on_form_edited)

        self.components_table.setCellWidget(row, 0, device_edit)
        self.components_table.setCellWidget(row, 1, variable_edit)
        self.components_table.setCellWidget(row, 2, forward_edit)

    def _component_rows(self) -> list[dict]:
        """Read the components table back into draft ``targets`` entries.

        Returns
        -------
        list of dict
            One ``{"target": ..., "forward": ...}`` per table row.
        """
        rows: list[dict] = []
        for row in range(self.components_table.rowCount()):
            device_edit = self.components_table.cellWidget(row, 0)
            variable_edit = self.components_table.cellWidget(row, 1)
            forward_edit = self.components_table.cellWidget(row, 2)
            if device_edit is None or variable_edit is None or forward_edit is None:
                continue
            rows.append(
                {
                    "target": _join_target(
                        device_edit.text().strip(), variable_edit.text().strip()
                    ),
                    "forward": forward_edit.text().strip(),
                }
            )
        return rows

    def _on_add_component(self) -> None:
        """Append an empty component row and commit."""
        if self._current_name is None:
            return
        self._append_component_row()
        self._on_form_edited()

    def _on_remove_component(self) -> None:
        """Remove the selected (else last) component row and commit."""
        if self._current_name is None:
            return
        count = self.components_table.rowCount()
        if count == 0:
            return
        row = self.components_table.currentRow()
        self.components_table.removeRow(row if 0 <= row < count else count - 1)
        self._on_form_edited()

    def _on_mode_changed(self) -> None:
        """Re-label the forward column/hint for the new mode, then commit."""
        self._update_mode_labels()
        self._on_form_edited()

    def _update_mode_labels(self) -> None:
        """Point the forward-column header and hint at the current mode."""
        header, hint = _MODE_LABELS.get(
            self.mode_combo.currentText(), _MODE_LABELS["absolute"]
        )
        item = self.components_table.horizontalHeaderItem(2)
        if item is not None:
            item.setText(header)
        self.mode_hint_label.setText(hint)

    def _on_form_edited(self, *_args: object) -> None:
        """Commit the visible form into the draft for the current variable."""
        if self._loading or self._current_name is None:
            return
        draft = self._variables[self._current_name]
        if draft.get("kind") == "pseudo":
            draft["mode"] = self.mode_combo.currentText()
            draft["targets"] = self._component_rows()
            inverse = self.inverse_edit.text().strip()
            if inverse:
                draft["inverse"] = inverse
            else:
                draft.pop("inverse", None)
        else:
            draft["target"] = _join_target(
                self.device_edit.text().strip(), self.variable_edit.text().strip()
            )
            draft["kind"] = self.kind_combo.currentText()
            confirm = _join_target(
                self.confirm_device_edit.text().strip(),
                self.confirm_variable_edit.text().strip(),
            )
            if confirm:
                draft["confirm"] = confirm
            else:
                draft.pop("confirm", None)
        self._update_dirty()

    # ------------------------------------------------------------------
    # Save / Revert / dirty
    # ------------------------------------------------------------------

    def _save_draft(self) -> bool:
        """Validate the draft and write it through the store (errors inline).

        Returns
        -------
        bool
            ``True`` when the save landed (validation and write both OK).
        """
        try:
            catalog = self._draft_catalog()
        except ValidationError as exc:
            self._show_error(_format_validation_error(exc, self._variables))
            return False
        try:
            path = self._store.save(catalog)
        except ScanVariableStoreError as exc:
            self._show_error(str(exc))
            return False
        self._snapshot = copy.deepcopy(self._variables)
        self._update_dirty()
        self._clear_error()
        self.dirty_label.setText(f"saved → {path}")
        return True

    def _on_save(self) -> None:
        """Save button handler."""
        self._save_draft()

    def _has_unsaved(self) -> bool:
        """Unsaved-changes hook: the draft differs from disk."""
        return self.dirty

    def _save_unsaved(self) -> bool:
        """Unsaved-changes hook: persist the whole draft catalog."""
        return self._save_draft()

    def _unsaved_prompt_text(self) -> str:
        """Describe the dirty draft in the unsaved-changes prompt."""
        return "The scan-variable catalog has unsaved changes."

    def _on_revert(self) -> None:
        """Discard the draft and reload the catalog from disk (prompted)."""
        if self.dirty and not self._confirm_discard():
            return
        selected = self._current_name
        self._reload_from_store()
        if selected in self._variables:
            self._refresh_list(select=selected)

    def _update_dirty(self) -> None:
        """Refresh the dirty indicator."""
        self.dirty_label.setText("● unsaved changes" if self.dirty else "")

    def _show_error(self, message: str) -> None:
        """Show *message* inline in the error label.

        Parameters
        ----------
        message : str
            Status-bar-fit error text (may be multi-line).
        """
        self.error_label.setStyleSheet("color: #d9534f;")
        self.error_label.setText(message)

    def _clear_error(self) -> None:
        """Clear the inline error label."""
        self.error_label.setText("")

    # ------------------------------------------------------------------
    # Completions (scaffold on ConfigEditorDialog)
    # ------------------------------------------------------------------

    def _install_completions(self) -> None:
        """Point every device/variable completer model at the fresh word lists."""
        self._device_model.setStringList(sorted(self._device_vars))
        self._refresh_variable_model(self._variable_model, self.device_edit.text())
        self._refresh_variable_model(
            self._confirm_variable_model, self.confirm_device_edit.text()
        )
        for row in range(self.components_table.rowCount()):
            device_edit = self.components_table.cellWidget(row, 0)
            variable_edit = self.components_table.cellWidget(row, 1)
            if device_edit is None or variable_edit is None:
                continue
            completer = variable_edit.completer()
            if completer is not None:
                self._refresh_variable_model(completer.model(), device_edit.text())

    def _refresh_variable_model(self, model: Any, device_text: str) -> None:
        """Point one variable word-list *model* at the variables of *device_text*.

        Parameters
        ----------
        model : QStringListModel
            The variable completer's model.
        device_text : str
            The paired device field's current text (exact match wins, else a
            unique case-insensitive match).
        """
        device = device_text.strip()
        variables = self._device_vars.get(device)
        if variables is None and device:
            matches = [
                known for known in self._device_vars if known.lower() == device.lower()
            ]
            if len(matches) == 1:
                variables = self._device_vars[matches[0]]
        model.setStringList(variables or [])

    # Close paths (keyPressEvent / reject / closeEvent): ConfigEditorDialog
    # — the unsaved-changes prompt now offers Save alongside Discard/Cancel.


def open_scan_variable_editor(
    parent: Optional[QWidget],
    experiment: str,
    configs_base: str | Path | None = None,
    completions: Optional[CompletionsProvider] = None,
) -> QDialog:
    """Open the scan-variable editor for *experiment* (the Editors-menu entry).

    Parameters
    ----------
    parent : QWidget or None
        Parent widget (the main window in production).
    experiment : str
        Experiment folder under ``scanner_configs/experiments`` ("" opens an
        empty, offline editor).
    configs_base : str or Path, optional
        Override for the experiments root (tests point this at a tmp dir);
        defaults to the production resolution.
    completions : CompletionsProvider, optional
        Device/variable completion source.  Defaults to the DB-backed
        provider for a named experiment (fetched on a daemon thread; offline
        it degrades to empty) and to no completions otherwise.

    Returns
    -------
    QDialog
        The shown (modeless) editor dialog.
    """
    store = ScanVariableStore(experiment, experiments_root=configs_base)
    if completions is None:
        completions = (
            GeecsDbCompletions(experiment) if experiment else EmptyCompletions()
        )
    dialog = ScanVariableEditor(store=store, completions=completions, parent=parent)
    dialog.show()
    return dialog
