"""Experiment config editor panel for the Scan Configuration GUI.

Provides a purpose-built editor for experiment YAML files — thin
configuration files that define an experiment name and a list of
``include`` directives referencing groups or individual analyzers.

Components
----------
* **Top-level fields** — ``experiment``, ``description``, ``version``,
  ``upload_to_scanlog``
* **Include directives list** — ``QListWidget`` showing group/ref entries
  with Add / Edit / Remove / Move Up / Move Down buttons
* **Include entry dialog** — Modal ``QDialog`` for adding or editing a
  single ``IncludeEntry``

The data model mirrors the Pydantic ``ExperimentAnalysisConfig`` and
``IncludeEntry`` models defined in
:mod:`scan_analysis.config.analyzer_config_models`.

Signals
-------
config_changed
    Emitted whenever any field or include directive is modified (for
    YAML preview updates).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: format an include entry for display
# ---------------------------------------------------------------------------


def _include_summary(entry: Dict[str, Any]) -> str:
    """Return a human-readable summary string for an include entry dict.

    Parameters
    ----------
    entry : dict
        An include entry with ``ref`` or ``group`` key.

    Returns
    -------
    str
        Display string such as ``"group: baseline (offset: +10)"``
        or ``"ref: Amp2Input (priority: 5)"``.
    """
    if entry.get("group"):
        label = f"group: {entry['group']}"
    elif entry.get("ref"):
        label = f"ref: {entry['ref']}"
    else:
        label = "<invalid entry>"

    extras: List[str] = []
    if entry.get("priority") is not None:
        extras.append(f"priority: {entry['priority']}")
    offset = entry.get("priority_offset", 0)
    if offset != 0:
        extras.append(f"offset: {offset:+d}")
    if entry.get("overrides"):
        extras.append("overrides")

    if extras:
        label += f" ({', '.join(extras)})"
    return label


# ---------------------------------------------------------------------------
# Include Entry Dialog
# ---------------------------------------------------------------------------


class IncludeEntryDialog(QDialog):
    """Modal dialog for adding or editing a single include directive.

    Parameters
    ----------
    group_names : List[str]
        Available group names to populate the combo box.
    analyzer_ids : List[str]
        Available analyzer IDs to populate the combo box.
    entry : dict, optional
        Existing include entry dict to edit.  ``None`` for a new entry.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        group_names: List[str],
        analyzer_ids: List[str],
        entry: Optional[Dict[str, Any]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Include Entry")
        self.setMinimumWidth(420)

        self._group_names = list(group_names)
        self._analyzer_ids = list(analyzer_ids)
        self._result_entry: Optional[Dict[str, Any]] = None

        self._setup_ui()

        if entry is not None:
            self._populate(entry)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build all child widgets and lay them out."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # -- Target type radio buttons --------------------------------------
        type_group_box = QGroupBox("Target Type")
        type_layout = QHBoxLayout(type_group_box)

        self._radio_group = QButtonGroup(self)
        self._radio_group_btn = QRadioButton("Group")
        self._radio_ref_btn = QRadioButton("Single Analyzer (ref)")
        self._radio_group.addButton(self._radio_group_btn, 0)
        self._radio_group.addButton(self._radio_ref_btn, 1)
        self._radio_group_btn.setChecked(True)

        type_layout.addWidget(self._radio_group_btn)
        type_layout.addWidget(self._radio_ref_btn)
        layout.addWidget(type_group_box)

        # -- Target value combo box -----------------------------------------
        target_layout = QFormLayout()

        self._target_combo = QComboBox()
        self._target_combo.setEditable(True)
        self._target_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        target_layout.addRow("Target:", self._target_combo)
        layout.addLayout(target_layout)

        # Populate initial combo contents
        self._update_target_combo()

        # Connect radio toggle to swap combo contents
        self._radio_group.buttonToggled.connect(self._on_type_changed)

        # -- Priority override ----------------------------------------------
        priority_layout = QHBoxLayout()

        self._priority_check = QCheckBox("Set priority:")
        self._priority_check.setChecked(False)
        priority_layout.addWidget(self._priority_check)

        self._priority_spin = QSpinBox()
        self._priority_spin.setRange(0, 10000)
        self._priority_spin.setValue(100)
        self._priority_spin.setEnabled(False)
        priority_layout.addWidget(self._priority_spin)
        priority_layout.addStretch()

        self._priority_check.toggled.connect(self._priority_spin.setEnabled)
        layout.addLayout(priority_layout)

        # -- Priority offset ------------------------------------------------
        offset_layout = QFormLayout()

        self._offset_spin = QSpinBox()
        self._offset_spin.setRange(-1000, 1000)
        self._offset_spin.setValue(0)
        offset_layout.addRow("Priority Offset:", self._offset_spin)
        layout.addLayout(offset_layout)

        # -- Overrides (collapsible) ----------------------------------------
        self._overrides_group = QGroupBox("Overrides (advanced)")
        self._overrides_group.setCheckable(True)
        self._overrides_group.setChecked(False)
        overrides_layout = QVBoxLayout(self._overrides_group)

        self._overrides_edit = QPlainTextEdit()
        self._overrides_edit.setPlaceholderText('JSON dict, e.g. {"key": "value"}')
        self._overrides_edit.setMaximumHeight(100)
        overrides_layout.addWidget(self._overrides_edit)

        layout.addWidget(self._overrides_group)

        # -- OK / Cancel buttons --------------------------------------------
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_type_changed(self) -> None:
        """Swap combo box contents when the radio button changes."""
        self._update_target_combo()

    def _update_target_combo(self) -> None:
        """Populate the target combo box based on the selected radio button."""
        self._target_combo.clear()
        if self._radio_group_btn.isChecked():
            self._target_combo.addItems(self._group_names)
        else:
            self._target_combo.addItems(self._analyzer_ids)

    def _on_accept(self) -> None:
        """Validate and accept the dialog."""
        target_value = self._target_combo.currentText().strip()
        if not target_value:
            QMessageBox.warning(
                self,
                "Validation Error",
                "Please select or enter a target value.",
            )
            return

        entry: Dict[str, Any] = {}
        if self._radio_group_btn.isChecked():
            entry["group"] = target_value
        else:
            entry["ref"] = target_value

        if self._priority_check.isChecked():
            entry["priority"] = self._priority_spin.value()

        entry["priority_offset"] = self._offset_spin.value()

        # Parse overrides JSON
        if self._overrides_group.isChecked():
            raw = self._overrides_edit.toPlainText().strip()
            if raw:
                try:
                    overrides = json.loads(raw)
                except json.JSONDecodeError as exc:
                    QMessageBox.warning(
                        self,
                        "Invalid JSON",
                        f"Overrides must be a valid JSON dict:\n{exc}",
                    )
                    return
                if not isinstance(overrides, dict):
                    QMessageBox.warning(
                        self,
                        "Invalid Overrides",
                        "Overrides must be a JSON object (dict), not a list or scalar.",
                    )
                    return
                if overrides:
                    entry["overrides"] = overrides

        self._result_entry = entry
        self.accept()

    # ------------------------------------------------------------------
    # Populate from existing entry
    # ------------------------------------------------------------------

    def _populate(self, entry: Dict[str, Any]) -> None:
        """Fill dialog fields from an existing include entry dict.

        Parameters
        ----------
        entry : dict
            Include entry with ``ref`` or ``group`` and optional fields.
        """
        if entry.get("group"):
            self._radio_group_btn.setChecked(True)
            self._update_target_combo()
            idx = self._target_combo.findText(entry["group"])
            if idx >= 0:
                self._target_combo.setCurrentIndex(idx)
            else:
                self._target_combo.setEditText(entry["group"])
        elif entry.get("ref"):
            self._radio_ref_btn.setChecked(True)
            self._update_target_combo()
            idx = self._target_combo.findText(entry["ref"])
            if idx >= 0:
                self._target_combo.setCurrentIndex(idx)
            else:
                self._target_combo.setEditText(entry["ref"])

        if entry.get("priority") is not None:
            self._priority_check.setChecked(True)
            self._priority_spin.setValue(entry["priority"])

        self._offset_spin.setValue(entry.get("priority_offset", 0))

        overrides = entry.get("overrides")
        if overrides:
            self._overrides_group.setChecked(True)
            self._overrides_edit.setPlainText(json.dumps(overrides, indent=2))

    # ------------------------------------------------------------------
    # Public result accessor
    # ------------------------------------------------------------------

    def get_entry(self) -> Optional[Dict[str, Any]]:
        """Return the validated include entry dict, or ``None`` if cancelled.

        Returns
        -------
        dict or None
            The include entry dict with ``ref`` or ``group`` key, or
            ``None`` if the dialog was cancelled.
        """
        return self._result_entry


# ---------------------------------------------------------------------------
# Main editor panel
# ---------------------------------------------------------------------------


class ExperimentEditorPanel(QWidget):
    """Editor panel for experiment configuration YAML files.

    Provides top-level metadata fields and a list of include directives
    with full CRUD controls (add, edit, remove, reorder).

    Parameters
    ----------
    parent : QWidget, optional
        Parent Qt widget.

    Signals
    -------
    config_changed
        Emitted whenever any field or include directive is modified.
    """

    config_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # -- data model -----------------------------------------------------
        self._config: Dict[str, Any] = {}
        self._includes: List[Dict[str, Any]] = []
        self._group_names: List[str] = []
        self._analyzer_ids: List[str] = []

        # -- build UI -------------------------------------------------------
        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build all child widgets and lay them out."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)

        # -- 1. Top-level fields --------------------------------------------
        fields_group = QGroupBox("Experiment Metadata")
        form = QFormLayout(fields_group)

        self._experiment_edit = QLineEdit()
        self._experiment_edit.setPlaceholderText("Experiment name (required)")
        form.addRow("Experiment:", self._experiment_edit)

        self._description_edit = QLineEdit()
        self._description_edit.setPlaceholderText("Optional description")
        form.addRow("Description:", self._description_edit)

        self._version_edit = QLineEdit()
        self._version_edit.setText("1.0")
        form.addRow("Version:", self._version_edit)

        self._upload_check = QCheckBox("Upload to scanlog")
        self._upload_check.setChecked(True)
        form.addRow("", self._upload_check)

        outer.addWidget(fields_group)

        # -- 2. Include directives list -------------------------------------
        includes_group = QGroupBox("Include Directives")
        includes_layout = QVBoxLayout(includes_group)

        self._include_list = QListWidget()
        self._include_list.setAlternatingRowColors(True)
        self._include_list.doubleClicked.connect(self._on_edit_include)
        includes_layout.addWidget(self._include_list, stretch=1)

        # Button bar
        btn_bar = QHBoxLayout()
        btn_bar.setSpacing(6)

        self._add_btn = QPushButton("Add Include…")
        self._add_btn.clicked.connect(self._on_add_include)
        btn_bar.addWidget(self._add_btn)

        self._edit_btn = QPushButton("Edit…")
        self._edit_btn.clicked.connect(self._on_edit_include)
        btn_bar.addWidget(self._edit_btn)

        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._on_remove_include)
        btn_bar.addWidget(self._remove_btn)

        btn_bar.addStretch()

        self._move_up_btn = QPushButton("Move Up")
        self._move_up_btn.clicked.connect(self._on_move_up)
        btn_bar.addWidget(self._move_up_btn)

        self._move_down_btn = QPushButton("Move Down")
        self._move_down_btn.clicked.connect(self._on_move_down)
        btn_bar.addWidget(self._move_down_btn)

        includes_layout.addLayout(btn_bar)
        outer.addWidget(includes_group, stretch=1)

        # -- Initial button state -------------------------------------------
        self._update_button_states()
        self._include_list.currentRowChanged.connect(self._update_button_states)

    def _connect_signals(self) -> None:
        """Wire up field-change signals to emit :pyqtSignal:`config_changed`."""
        self._experiment_edit.textChanged.connect(self._emit_changed)
        self._description_edit.textChanged.connect(self._emit_changed)
        self._version_edit.textChanged.connect(self._emit_changed)
        self._upload_check.toggled.connect(self._emit_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_config(self, data: Dict[str, Any]) -> None:
        """Populate the editor from a raw experiment YAML dict.

        Parameters
        ----------
        data : dict
            Experiment config dict as returned by
            :func:`scan_config_io.load_experiment_yaml`.
        """
        self._config = dict(data)
        self._experiment_edit.setText(data.get("experiment", ""))
        self._description_edit.setText(data.get("description", ""))
        self._version_edit.setText(str(data.get("version", "1.0")))
        self._upload_check.setChecked(data.get("upload_to_scanlog", True))

        # Load includes
        self._includes = []
        for inc in data.get("include", []):
            self._includes.append(dict(inc))

        self._refresh_include_list()
        self._emit_changed()

    def get_config_dict(self) -> Dict[str, Any]:
        """Return the current editor state as a dict suitable for saving.

        Returns
        -------
        dict
            Experiment config dict with keys ``experiment``,
            ``description``, ``version``, ``upload_to_scanlog``, and
            ``include``.
        """
        result: Dict[str, Any] = {}

        experiment = self._experiment_edit.text().strip()
        if experiment:
            result["experiment"] = experiment

        description = self._description_edit.text().strip()
        if description:
            result["description"] = description

        version = self._version_edit.text().strip()
        result["version"] = version if version else "1.0"

        result["upload_to_scanlog"] = self._upload_check.isChecked()

        # Build clean include list
        includes: List[Dict[str, Any]] = []
        for entry in self._includes:
            clean: Dict[str, Any] = {}
            if entry.get("group"):
                clean["group"] = entry["group"]
            elif entry.get("ref"):
                clean["ref"] = entry["ref"]
            else:
                continue  # skip invalid entries

            if entry.get("priority") is not None:
                clean["priority"] = entry["priority"]

            clean["priority_offset"] = entry.get("priority_offset", 0)

            overrides = entry.get("overrides")
            if overrides:
                clean["overrides"] = overrides

            includes.append(clean)

        if includes:
            result["include"] = includes

        return result

    def set_available_groups(self, group_names: List[str]) -> None:
        """Update the list of available group names for the include dialog.

        Parameters
        ----------
        group_names : List[str]
            Available group names.
        """
        self._group_names = list(group_names)

    def set_available_analyzers(self, analyzer_ids: List[str]) -> None:
        """Update the list of available analyzer IDs for the include dialog.

        Parameters
        ----------
        analyzer_ids : List[str]
            Available analyzer IDs.
        """
        self._analyzer_ids = list(analyzer_ids)

    def clear(self) -> None:
        """Reset the editor to its default empty state."""
        self._config = {}
        self._includes = []
        self._experiment_edit.clear()
        self._description_edit.clear()
        self._version_edit.setText("1.0")
        self._upload_check.setChecked(True)
        self._include_list.clear()
        self._update_button_states()
        self._emit_changed()

    # ------------------------------------------------------------------
    # Include list management
    # ------------------------------------------------------------------

    def _refresh_include_list(self) -> None:
        """Rebuild the QListWidget from ``self._includes``."""
        self._include_list.clear()
        for entry in self._includes:
            item = QListWidgetItem(_include_summary(entry))
            self._include_list.addItem(item)
        self._update_button_states()

    def _update_button_states(self) -> None:
        """Enable/disable buttons based on the current selection."""
        row = self._include_list.currentRow()
        count = self._include_list.count()
        has_selection = row >= 0

        self._edit_btn.setEnabled(has_selection)
        self._remove_btn.setEnabled(has_selection)
        self._move_up_btn.setEnabled(has_selection and row > 0)
        self._move_down_btn.setEnabled(has_selection and row < count - 1)

    # ------------------------------------------------------------------
    # Include list slots
    # ------------------------------------------------------------------

    def _on_add_include(self) -> None:
        """Open the include entry dialog to add a new entry."""
        dialog = IncludeEntryDialog(
            group_names=self._group_names,
            analyzer_ids=self._analyzer_ids,
            parent=self,
        )
        if dialog.exec_() == QDialog.Accepted:
            entry = dialog.get_entry()
            if entry is not None:
                self._includes.append(entry)
                self._refresh_include_list()
                self._include_list.setCurrentRow(len(self._includes) - 1)
                self._emit_changed()

    def _on_edit_include(self) -> None:
        """Open the include entry dialog to edit the selected entry."""
        row = self._include_list.currentRow()
        if row < 0 or row >= len(self._includes):
            return

        dialog = IncludeEntryDialog(
            group_names=self._group_names,
            analyzer_ids=self._analyzer_ids,
            entry=self._includes[row],
            parent=self,
        )
        if dialog.exec_() == QDialog.Accepted:
            entry = dialog.get_entry()
            if entry is not None:
                self._includes[row] = entry
                self._refresh_include_list()
                self._include_list.setCurrentRow(row)
                self._emit_changed()

    def _on_remove_include(self) -> None:
        """Remove the selected include entry."""
        row = self._include_list.currentRow()
        if row < 0 or row >= len(self._includes):
            return

        entry = self._includes[row]
        summary = _include_summary(entry)
        reply = QMessageBox.question(
            self,
            "Remove Include",
            f"Remove include entry?\n\n{summary}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            del self._includes[row]
            self._refresh_include_list()
            if self._includes:
                new_row = min(row, len(self._includes) - 1)
                self._include_list.setCurrentRow(new_row)
            self._emit_changed()

    def _on_move_up(self) -> None:
        """Move the selected include entry up by one position."""
        row = self._include_list.currentRow()
        if row <= 0:
            return
        self._includes[row - 1], self._includes[row] = (
            self._includes[row],
            self._includes[row - 1],
        )
        self._refresh_include_list()
        self._include_list.setCurrentRow(row - 1)
        self._emit_changed()

    def _on_move_down(self) -> None:
        """Move the selected include entry down by one position."""
        row = self._include_list.currentRow()
        if row < 0 or row >= len(self._includes) - 1:
            return
        self._includes[row], self._includes[row + 1] = (
            self._includes[row + 1],
            self._includes[row],
        )
        self._refresh_include_list()
        self._include_list.setCurrentRow(row + 1)
        self._emit_changed()

    # ------------------------------------------------------------------
    # Signal helpers
    # ------------------------------------------------------------------

    def _emit_changed(self, *_args: object) -> None:
        """Emit :pyqtSignal:`config_changed`."""
        self.config_changed.emit()
