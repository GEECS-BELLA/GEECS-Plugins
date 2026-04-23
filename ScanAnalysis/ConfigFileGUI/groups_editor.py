"""Groups editor panel for the Scan Configuration GUI.

Provides a purpose-built editor for ``groups.yaml`` — the file that maps
named groups to lists of analyzer IDs.  Unlike the library-analyzer editor
which uses a generic form builder, this panel has bespoke widgets for the
group/member workflow:

* **Group selector** — ``QComboBox`` with *New Group* / *Delete Group* buttons
* **Analyzer autocomplete bar** — ``QLineEdit`` with ``QCompleter`` plus an
  *Add* button (checkmark icon)
* **Group member list** — ``QListWidget`` with custom row widgets containing
  a checkbox (enable/disable), label (analyzer ID), and trash button (remove)

The data model mirrors what :func:`scan_config_io.load_groups_yaml` returns:
``Dict[str, List[dict]]`` where each entry is ``{"id": str, "enabled": bool}``.

Signals
-------
groups_changed
    Emitted whenever any group membership, enable/disable state, or
    group creation/deletion occurs (for YAML preview updates).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt, QSortFilterProxyModel, QStringListModel, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QCompleter,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: group member row widget
# ---------------------------------------------------------------------------


class _GroupMemberRow(QWidget):
    """Single row in the group member list.

    Displays a checkbox (enable/disable), an analyzer ID label, and a
    trash button for removal.

    Parameters
    ----------
    analyzer_id : str
        The analyzer identifier string.
    enabled : bool
        Whether the entry is enabled (unchecked = commented out / disabled).
    parent : QWidget, optional
        Parent Qt widget.

    Signals
    -------
    enable_toggled(str, bool)
        Emitted when the checkbox is toggled.  Arguments are the analyzer
        ID and the new enabled state.
    remove_clicked(str)
        Emitted when the trash button is clicked.  Argument is the
        analyzer ID.
    """

    enable_toggled = pyqtSignal(str, bool)
    remove_clicked = pyqtSignal(str)

    def __init__(
        self,
        analyzer_id: str,
        enabled: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._analyzer_id = analyzer_id

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        # -- Checkbox (left) ------------------------------------------------
        self._checkbox = QCheckBox()
        self._checkbox.setChecked(enabled)
        self._checkbox.setToolTip("Enable / disable this analyzer in the group")
        self._checkbox.toggled.connect(self._on_toggled)
        layout.addWidget(self._checkbox)

        # -- Label (center, stretches) --------------------------------------
        self._label = QLabel(analyzer_id)
        self._label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(self._label)

        # -- Trash button (right) -------------------------------------------
        self._trash_btn = QPushButton()
        self._trash_btn.setToolTip("Remove from group")
        self._trash_btn.setFixedSize(24, 24)
        self._trash_btn.setFlat(True)

        # Try SP_TrashIcon first; fall back to SP_DialogDiscardButton or text
        style = self.style()
        if style is not None:
            try:
                icon = style.standardIcon(QStyle.SP_TrashIcon)
                if not icon.isNull():
                    self._trash_btn.setIcon(icon)
                else:
                    raise AttributeError
            except AttributeError:
                try:
                    icon = style.standardIcon(QStyle.SP_DialogDiscardButton)
                    if not icon.isNull():
                        self._trash_btn.setIcon(icon)
                    else:
                        self._trash_btn.setText("\u2715")
                except AttributeError:
                    self._trash_btn.setText("\u2715")
        else:
            self._trash_btn.setText("\u2715")

        self._trash_btn.clicked.connect(self._on_remove)
        layout.addWidget(self._trash_btn)

    # -- properties ---------------------------------------------------------

    @property
    def analyzer_id(self) -> str:
        """Return the analyzer ID for this row."""
        return self._analyzer_id

    @property
    def is_enabled(self) -> bool:
        """Return whether the checkbox is currently checked."""
        return self._checkbox.isChecked()

    # -- internal slots -----------------------------------------------------

    def _on_toggled(self, checked: bool) -> None:
        """Forward checkbox toggle as a signal with the analyzer ID."""
        self.enable_toggled.emit(self._analyzer_id, checked)

    def _on_remove(self) -> None:
        """Forward trash-button click as a signal with the analyzer ID."""
        self.remove_clicked.emit(self._analyzer_id)


# ---------------------------------------------------------------------------
# Main editor panel
# ---------------------------------------------------------------------------


class GroupsEditorPanel(QWidget):
    """Editor panel for ``groups.yaml``.

    Provides a group selector, an autocomplete bar for adding analyzers,
    and a member list with enable/disable checkboxes and removal buttons.

    Parameters
    ----------
    parent : QWidget, optional
        Parent Qt widget.

    Signals
    -------
    groups_changed
        Emitted whenever any group membership, enable/disable state, or
        group creation/deletion occurs.
    """

    groups_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # -- data model -----------------------------------------------------
        self._groups: Dict[str, List[Dict[str, Any]]] = {}
        self._analyzer_ids: List[str] = []
        self._current_group: Optional[str] = None

        # -- build UI -------------------------------------------------------
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build all child widgets and lay them out."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)

        # -- 1. Group selector bar ------------------------------------------
        group_bar = QHBoxLayout()
        group_bar.setSpacing(6)

        group_label = QLabel("Group:")
        group_bar.addWidget(group_label)

        self._group_combo = QComboBox()
        self._group_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._group_combo.currentTextChanged.connect(self._on_group_selected)
        group_bar.addWidget(self._group_combo)

        self._new_group_btn = QPushButton("New Group")
        self._new_group_btn.clicked.connect(self._on_new_group)
        group_bar.addWidget(self._new_group_btn)

        self._delete_group_btn = QPushButton("Delete Group")
        self._delete_group_btn.clicked.connect(self._on_delete_group)
        group_bar.addWidget(self._delete_group_btn)

        outer.addLayout(group_bar)

        # -- 2. Analyzer autocomplete bar -----------------------------------
        add_bar = QHBoxLayout()
        add_bar.setSpacing(6)

        add_label = QLabel("Add analyzer:")
        add_bar.addWidget(add_label)

        self._add_edit = QLineEdit()
        self._add_edit.setPlaceholderText("Type analyzer ID…")
        self._add_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._add_edit.returnPressed.connect(self._on_add_analyzer)
        add_bar.addWidget(self._add_edit)

        # QCompleter with case-insensitive substring matching
        self._completer_model = QStringListModel()
        self._completer_proxy = QSortFilterProxyModel()
        self._completer_proxy.setSourceModel(self._completer_model)
        self._completer_proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)

        self._completer = QCompleter()
        self._completer.setModel(self._completer_proxy)
        self._completer.setCompletionMode(QCompleter.PopupCompletion)
        self._completer.setCaseSensitivity(Qt.CaseInsensitive)
        self._completer.setFilterMode(Qt.MatchContains)
        self._add_edit.setCompleter(self._completer)

        # Add button with checkmark icon
        self._add_btn = QPushButton()
        self._add_btn.setToolTip("Add analyzer to group")
        self._add_btn.setFixedSize(28, 28)
        style = self.style()
        if style is not None:
            icon = style.standardIcon(QStyle.SP_DialogApplyButton)
            if not icon.isNull():
                self._add_btn.setIcon(icon)
            else:
                self._add_btn.setText("Add")
        else:
            self._add_btn.setText("Add")
        self._add_btn.clicked.connect(self._on_add_analyzer)
        add_bar.addWidget(self._add_btn)

        outer.addLayout(add_bar)

        # -- Status label for warnings --------------------------------------
        self._status_label = QLabel()
        self._status_label.setStyleSheet("color: #c0392b; font-size: 11px;")
        self._status_label.setWordWrap(True)
        self._status_label.hide()
        outer.addWidget(self._status_label)

        # -- 3. Group member list -------------------------------------------
        members_label = QLabel("Group Members:")
        outer.addWidget(members_label)

        self._member_list = QListWidget()
        self._member_list.setSpacing(1)
        outer.addWidget(self._member_list, stretch=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_groups(self, groups_data: Dict[str, List[Dict[str, Any]]]) -> None:
        """Load groups from data as returned by :func:`load_groups_yaml`.

        Parameters
        ----------
        groups_data : Dict[str, List[dict]]
            Mapping of group name to list of ``{"id": str, "enabled": bool}``
            entries.
        """
        self._groups = {
            name: [dict(entry) for entry in members]
            for name, members in groups_data.items()
        }

        # Populate the group combo box
        self._group_combo.blockSignals(True)
        self._group_combo.clear()
        for name in self._groups:
            self._group_combo.addItem(name)
        self._group_combo.blockSignals(False)

        # Select the first group (or clear the member list)
        if self._groups:
            self._group_combo.setCurrentIndex(0)
            self._current_group = self._group_combo.currentText()
            self._refresh_member_list()
        else:
            self._current_group = None
            self._member_list.clear()

        self._update_button_states()

    def set_analyzer_ids(self, ids: List[str]) -> None:
        """Update the autocomplete suggestions for the add-analyzer bar.

        Parameters
        ----------
        ids : List[str]
            Sorted list of all known analyzer IDs from the library.
        """
        self._analyzer_ids = list(ids)
        self._completer_model.setStringList(self._analyzer_ids)

    def get_groups_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return the current groups state for saving.

        The returned structure matches the format expected by
        :func:`save_groups_yaml`.

        Returns
        -------
        Dict[str, List[dict]]
            Mapping of group name to list of ``{"id": str, "enabled": bool}``.
        """
        # Sync the currently displayed group back to the data model
        self._sync_current_group()
        return {
            name: [dict(entry) for entry in members]
            for name, members in self._groups.items()
        }

    def clear(self) -> None:
        """Reset the editor to an empty state."""
        self._groups.clear()
        self._current_group = None
        self._group_combo.blockSignals(True)
        self._group_combo.clear()
        self._group_combo.blockSignals(False)
        self._member_list.clear()
        self._add_edit.clear()
        self._status_label.hide()
        self._update_button_states()

    # ------------------------------------------------------------------
    # Group selector slots
    # ------------------------------------------------------------------

    def _on_group_selected(self, group_name: str) -> None:
        """Handle group combo-box selection change.

        Saves the state of the previously selected group before switching.
        """
        if not group_name:
            return

        # Sync previous group before switching
        self._sync_current_group()

        self._current_group = group_name
        self._refresh_member_list()
        self._status_label.hide()

    def _on_new_group(self) -> None:
        """Prompt the user for a new group name and create it."""
        name, ok = QInputDialog.getText(
            self,
            "New Group",
            "Enter group name:",
        )
        if not ok or not name.strip():
            return

        name = name.strip()
        if name in self._groups:
            QMessageBox.warning(
                self,
                "Duplicate Group",
                f'A group named "{name}" already exists.',
            )
            return

        self._sync_current_group()
        self._groups[name] = []

        self._group_combo.blockSignals(True)
        self._group_combo.addItem(name)
        self._group_combo.setCurrentText(name)
        self._group_combo.blockSignals(False)

        self._current_group = name
        self._refresh_member_list()
        self._update_button_states()
        self.groups_changed.emit()
        logger.debug("Created new group: %s", name)

    def _on_delete_group(self) -> None:
        """Delete the currently selected group after confirmation."""
        if self._current_group is None:
            return

        reply = QMessageBox.question(
            self,
            "Delete Group",
            f'Delete group "{self._current_group}"?\n\nThis cannot be undone.',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        name = self._current_group
        self._groups.pop(name, None)

        self._group_combo.blockSignals(True)
        idx = self._group_combo.findText(name)
        if idx >= 0:
            self._group_combo.removeItem(idx)
        self._group_combo.blockSignals(False)

        # Select the first remaining group (if any)
        if self._group_combo.count() > 0:
            self._group_combo.setCurrentIndex(0)
            self._current_group = self._group_combo.currentText()
            self._refresh_member_list()
        else:
            self._current_group = None
            self._member_list.clear()

        self._update_button_states()
        self.groups_changed.emit()
        logger.debug("Deleted group: %s", name)

    # ------------------------------------------------------------------
    # Autocomplete / add-analyzer slots
    # ------------------------------------------------------------------

    def _on_add_analyzer(self) -> None:
        """Add the current text as a new member to the active group."""
        text = self._add_edit.text().strip()
        if not text:
            return

        if self._current_group is None:
            self._show_status("No group selected. Create a group first.")
            return

        # Validate against known analyzer IDs
        if self._analyzer_ids and text not in self._analyzer_ids:
            self._show_status(f'"{text}" is not a recognized analyzer ID.')
            return

        # Check for duplicates in the current group
        members = self._groups.get(self._current_group, [])
        existing_ids = {entry["id"] for entry in members}
        if text in existing_ids:
            self._show_status(f'"{text}" is already in group "{self._current_group}".')
            return

        # Add the new entry
        members.append({"id": text, "enabled": True})
        self._groups[self._current_group] = members

        self._add_edit.clear()
        self._status_label.hide()
        self._refresh_member_list()
        self.groups_changed.emit()
        logger.debug("Added %s to group %s", text, self._current_group)

    # ------------------------------------------------------------------
    # Member list management
    # ------------------------------------------------------------------

    def _refresh_member_list(self) -> None:
        """Rebuild the member list widget from the current group's data."""
        self._member_list.clear()

        if self._current_group is None:
            return

        members = self._groups.get(self._current_group, [])
        for entry in members:
            analyzer_id = entry.get("id", "")
            enabled = entry.get("enabled", True)
            self._add_member_row(analyzer_id, enabled)

    def _add_member_row(self, analyzer_id: str, enabled: bool) -> None:
        """Add a single member row widget to the list.

        Parameters
        ----------
        analyzer_id : str
            The analyzer identifier.
        enabled : bool
            Whether the entry is enabled.
        """
        row_widget = _GroupMemberRow(analyzer_id, enabled)
        row_widget.enable_toggled.connect(self._on_member_toggled)
        row_widget.remove_clicked.connect(self._on_member_removed)

        item = QListWidgetItem(self._member_list)
        item.setSizeHint(row_widget.sizeHint())
        self._member_list.setItemWidget(item, row_widget)

    def _on_member_toggled(self, analyzer_id: str, enabled: bool) -> None:
        """Handle enable/disable checkbox toggle for a member."""
        if self._current_group is None:
            return

        members = self._groups.get(self._current_group, [])
        for entry in members:
            if entry["id"] == analyzer_id:
                entry["enabled"] = enabled
                break

        self.groups_changed.emit()

    def _on_member_removed(self, analyzer_id: str) -> None:
        """Remove a member from the current group (no confirmation)."""
        if self._current_group is None:
            return

        members = self._groups.get(self._current_group, [])
        self._groups[self._current_group] = [
            entry for entry in members if entry["id"] != analyzer_id
        ]

        self._refresh_member_list()
        self.groups_changed.emit()
        logger.debug(
            "Removed %s from group %s",
            analyzer_id,
            self._current_group,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_current_group(self) -> None:
        """Sync the displayed member list state back to ``_groups``.

        Iterates over the ``QListWidget`` rows and updates the enable
        state from each row's checkbox.  This ensures the data model
        reflects any checkbox changes before switching groups or saving.
        """
        if self._current_group is None:
            return

        members: List[Dict[str, Any]] = []
        for i in range(self._member_list.count()):
            item = self._member_list.item(i)
            if item is None:
                continue
            widget = self._member_list.itemWidget(item)
            if isinstance(widget, _GroupMemberRow):
                members.append({"id": widget.analyzer_id, "enabled": widget.is_enabled})

        self._groups[self._current_group] = members

    def _update_button_states(self) -> None:
        """Enable or disable buttons based on current state."""
        has_group = self._current_group is not None
        self._delete_group_btn.setEnabled(has_group)
        self._add_btn.setEnabled(has_group)
        self._add_edit.setEnabled(has_group)

    def _show_status(self, message: str) -> None:
        """Display a warning message in the status label.

        Parameters
        ----------
        message : str
            Warning text to display.
        """
        self._status_label.setText(message)
        self._status_label.show()
