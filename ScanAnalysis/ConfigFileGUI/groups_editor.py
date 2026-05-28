"""Single-group editor for the Scan Configuration GUI.

Edits one ``AnalysisGroupConfig`` YAML at a time. The post-PR-E
unified-diagnostic layout stores each group as its own file under
``groups/<facility>/<stem>.yaml``, replacing the pre-PR-E
single-file ``library/groups.yaml`` with comment-aware enable/disable
semantics. With the new layout the editor's job collapses:

* No more multi-group selector — the file tree picks the group.
* No more comment-aware parsing — ``AnalyzerRef.enabled`` is a proper
  Pydantic field, so enable/disable is just a checkbox value.

The form has:

* **General** — ``name``, ``description``, ``upload_to_scanlog``.
* **Add Analyzer** — autocomplete-backed line edit + an *Add* button.
* **Analyzers** — list of ``AnalyzerRef`` rows with an enable checkbox,
  the ref label, an optional priority override, and a remove button.

Public API:

* :meth:`load_config(data)` — populate the form from a raw YAML dict.
* :meth:`get_config_dict()` — return the form state as an
  ``AnalysisGroupConfig``-shaped dict.
* :meth:`set_analyzer_ids(ids)` — supply autocomplete suggestions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt, QSortFilterProxyModel, QStringListModel, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QCompleter,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Group member row
# ---------------------------------------------------------------------------


class _GroupMemberRow(QWidget):
    """Single ``AnalyzerRef`` row in the analyzers list.

    Layout (left → right):

    * Enable checkbox (``AnalyzerRef.enabled``)
    * Ref label (the diagnostic stem)
    * Priority override spinbox (``AnalyzerRef.priority``); ``-1``
      sentinel value represents the "use diagnostic default" (i.e.
      no override → ``None``).
    * Remove button.
    """

    PRIORITY_UNSET = -1  # spin-box value standing in for "use default"

    changed = pyqtSignal()
    remove_clicked = pyqtSignal(str)

    def __init__(
        self,
        ref: str,
        enabled: bool = True,
        priority: Optional[int] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._ref = ref

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        self._checkbox = QCheckBox()
        self._checkbox.setChecked(enabled)
        self._checkbox.setToolTip("Enable / disable this analyzer in the group")
        self._checkbox.toggled.connect(self.changed.emit)
        layout.addWidget(self._checkbox)

        self._label = QLabel(ref)
        self._label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(self._label)

        priority_label = QLabel("priority:")
        layout.addWidget(priority_label)
        self._priority_spin = QSpinBox()
        # -1 = "use diagnostic default" (None); 0..999 are explicit overrides.
        self._priority_spin.setRange(-1, 999)
        self._priority_spin.setSpecialValueText("—")  # shown when value == -1
        self._priority_spin.setValue(
            priority if priority is not None else self.PRIORITY_UNSET
        )
        self._priority_spin.setToolTip(
            "Per-group priority override. '—' means use the diagnostic's own "
            "scan.priority."
        )
        self._priority_spin.valueChanged.connect(self.changed.emit)
        layout.addWidget(self._priority_spin)

        self._remove_btn = QPushButton()
        self._remove_btn.setToolTip("Remove from group")
        self._remove_btn.setFixedSize(24, 24)
        self._remove_btn.setFlat(True)
        style = self.style()
        try:
            icon = style.standardIcon(QStyle.SP_TrashIcon) if style else None
            if icon is None or icon.isNull():
                raise AttributeError
            self._remove_btn.setIcon(icon)
        except AttributeError:
            self._remove_btn.setText("✕")
        self._remove_btn.clicked.connect(lambda: self.remove_clicked.emit(self._ref))
        layout.addWidget(self._remove_btn)

    @property
    def ref(self) -> str:
        return self._ref

    def to_dict(self) -> Dict[str, Any]:
        """Return the row's state as an ``AnalyzerRef`` dict."""
        prio = self._priority_spin.value()
        out: Dict[str, Any] = {"ref": self._ref}
        if not self._checkbox.isChecked():
            out["enabled"] = False
        if prio != self.PRIORITY_UNSET:
            out["priority"] = prio
        return out


# ---------------------------------------------------------------------------
# Main editor
# ---------------------------------------------------------------------------


class GroupEditorPanel(QWidget):
    """Editor for one ``AnalysisGroupConfig`` YAML.

    Signals
    -------
    config_changed
        Emitted whenever any field changes (name, description, member
        list, member enable/priority).
    """

    config_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._suppress_signals: bool = False
        self._analyzer_ids: List[str] = []
        self._member_rows: List[_GroupMemberRow] = []
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # ── General ─────────────────────────────────────────────────────
        general_box = QGroupBox("General")
        general_form = QFormLayout(general_box)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Group name (e.g. HTU_baseline)")
        self._name_edit.textChanged.connect(self._on_value_changed)
        general_form.addRow("name:", self._name_edit)

        self._description_edit = QPlainTextEdit()
        self._description_edit.setMaximumHeight(60)
        self._description_edit.setPlaceholderText("Optional free-text description")
        self._description_edit.textChanged.connect(self._on_value_changed)
        general_form.addRow("description:", self._description_edit)

        self._upload_check = QCheckBox("Upload to scan-log")
        self._upload_check.setChecked(True)
        self._upload_check.setToolTip(
            "Whether the runner uploads this group's outputs to the "
            "experiment scan-log on success."
        )
        self._upload_check.toggled.connect(self._on_value_changed)
        general_form.addRow("", self._upload_check)
        outer.addWidget(general_box)

        # ── Add Analyzer ────────────────────────────────────────────────
        add_box = QGroupBox("Add Analyzer")
        add_layout = QHBoxLayout(add_box)
        self._add_edit = QLineEdit()
        self._add_edit.setPlaceholderText("Type analyzer ID (filename stem)…")
        self._add_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._add_edit.returnPressed.connect(self._on_add_analyzer)
        add_layout.addWidget(self._add_edit)

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

        self._add_btn = QPushButton("Add")
        self._add_btn.setToolTip("Add analyzer to group")
        self._add_btn.clicked.connect(self._on_add_analyzer)
        add_layout.addWidget(self._add_btn)
        outer.addWidget(add_box)

        # ── Member list ─────────────────────────────────────────────────
        members_box = QGroupBox("Analyzers")
        members_layout = QVBoxLayout(members_box)
        self._member_list = QListWidget()
        self._member_list.setSpacing(1)
        members_layout.addWidget(self._member_list)
        outer.addWidget(members_box, stretch=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_config(self, data: Dict[str, Any]) -> None:
        """Populate the form from a raw group YAML dict.

        Parameters
        ----------
        data : dict
            ``AnalysisGroupConfig``-shaped dict. ``analyzers`` may
            contain plain strings (shorthand) or ``AnalyzerRef`` dicts;
            both are normalised to the dict form internally.
        """
        self._suppress_signals = True
        try:
            self._name_edit.setText(str(data.get("name", "")))
            self._description_edit.setPlainText(str(data.get("description", "") or ""))
            self._upload_check.setChecked(bool(data.get("upload_to_scanlog", True)))

            # Member list
            self._member_list.clear()
            self._member_rows = []
            for entry in data.get("analyzers", []) or []:
                ref, enabled, priority = _normalise_analyzer_entry(entry)
                if ref:
                    self._add_member_row(ref, enabled, priority)
        finally:
            self._suppress_signals = False

    def get_config_dict(self) -> Dict[str, Any]:
        """Return the form state as an ``AnalysisGroupConfig`` dict.

        Analyzer entries are emitted in their compact form when
        possible:

        * ``{ref: X}`` with no overrides → bare string ``"X"``.
        * Otherwise the full dict form.

        This matches the convention used in the on-disk groups (and
        means a clean round-trip for the common "all enabled, no
        priority overrides" case).
        """
        out: Dict[str, Any] = {}
        name = self._name_edit.text().strip()
        if name:
            out["name"] = name

        description = self._description_edit.toPlainText().strip()
        if description:
            out["description"] = description

        # Only emit upload_to_scanlog when it differs from the default.
        if not self._upload_check.isChecked():
            out["upload_to_scanlog"] = False

        analyzers: List[Any] = []
        for row in self._member_rows:
            entry = row.to_dict()
            if list(entry.keys()) == ["ref"]:
                # No overrides — emit the bare-string shorthand.
                analyzers.append(entry["ref"])
            else:
                analyzers.append(entry)
        out["analyzers"] = analyzers
        return out

    def set_analyzer_ids(self, ids: List[str]) -> None:
        """Update the autocomplete suggestions for the add-analyzer bar."""
        self._analyzer_ids = list(ids)
        self._completer_model.setStringList(self._analyzer_ids)

    # ------------------------------------------------------------------
    # Member-list management
    # ------------------------------------------------------------------

    def _add_member_row(
        self, ref: str, enabled: bool = True, priority: Optional[int] = None
    ) -> None:
        """Append a member row to the list widget."""
        row_widget = _GroupMemberRow(ref, enabled, priority)
        row_widget.changed.connect(self._on_value_changed)
        row_widget.remove_clicked.connect(self._on_member_removed)

        item = QListWidgetItem()
        item.setSizeHint(row_widget.sizeHint())
        self._member_list.addItem(item)
        self._member_list.setItemWidget(item, row_widget)
        self._member_rows.append(row_widget)

    def _on_add_analyzer(self) -> None:
        """Add the currently-typed analyzer ID as a new enabled member."""
        ref = self._add_edit.text().strip()
        if not ref:
            return
        # Refuse duplicates (silently keep existing).
        if any(r.ref == ref for r in self._member_rows):
            self._add_edit.clear()
            return
        self._add_member_row(ref, enabled=True, priority=None)
        self._add_edit.clear()
        self._on_value_changed()

    def _on_member_removed(self, ref: str) -> None:
        """Remove the row whose analyzer ref matches ``ref``."""
        for i, row in enumerate(self._member_rows):
            if row.ref == ref:
                # Find and remove the QListWidgetItem holding this row.
                for j in range(self._member_list.count()):
                    item = self._member_list.item(j)
                    if self._member_list.itemWidget(item) is row:
                        self._member_list.takeItem(j)
                        break
                self._member_rows.pop(i)
                break
        self._on_value_changed()

    # ------------------------------------------------------------------
    # Signal routing
    # ------------------------------------------------------------------

    def _on_value_changed(self, *_args) -> None:
        if not self._suppress_signals:
            self.config_changed.emit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_analyzer_entry(entry: Any) -> tuple[Optional[str], bool, Optional[int]]:
    """Normalise an ``analyzers:`` list entry to ``(ref, enabled, priority)``.

    Accepts both surface forms documented on ``AnalysisGroupConfig``:

    * Plain string → ``(string, True, None)``.
    * Dict → ``(dict['ref'], dict.get('enabled', True), dict.get('priority'))``.

    Returns ``(None, ..., ...)`` for malformed entries; callers skip those.
    """
    if isinstance(entry, str):
        return entry, True, None
    if isinstance(entry, dict):
        ref = entry.get("ref")
        if not isinstance(ref, str):
            return None, True, None
        enabled = bool(entry.get("enabled", True))
        priority = entry.get("priority")
        if priority is not None:
            try:
                priority = int(priority)
            except (TypeError, ValueError):
                priority = None
        return ref, enabled, priority
    return None, True, None
