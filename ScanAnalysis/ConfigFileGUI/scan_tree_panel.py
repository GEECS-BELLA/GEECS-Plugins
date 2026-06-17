"""Tree-view navigation panel for scan analysis configuration files.

Displays the scan_analysis_configs directory as a two-category,
facility-namespaced tree::

    Analyzers
      ├─ HTU
      │   ├─ UC_GaiaMode.yaml
      │   └─ ...
      ├─ HTT
      │   └─ ...
      └─ PW
    Groups
      ├─ HTU
      │   ├─ baseline.yaml
      │   └─ ...
      └─ ...

The pre-PR-E ``Experiments`` category is gone (the schema no longer
has a distinct experiment concept; groups cover that role). The
facility-namespace subnodes mirror the on-disk layout produced by
the unified-diagnostic migration.

This module is consumed by ``config_editor_window.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStyle,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .scan_config_io import (
    discover_scan_config_dirs,
    list_analyzer_configs,
    list_facilities,
    list_group_configs,
)

logger = logging.getLogger(__name__)


class ScanTreePanel(QWidget):
    """Tree-based file browser for scan analysis configurations.

    Two-category tree (Analyzers / Groups), each grouped by facility
    namespace (HTU / HTT / PW / UNCLASSIFIED) so on-disk organisation
    is visible at a glance. Only leaf items (individual files) are
    selectable.

    Parameters
    ----------
    parent : QWidget, optional
        Parent Qt widget.

    Signals
    -------
    file_selected(Path, str)
        Emitted when a leaf item is clicked. First argument is the
        full :class:`~pathlib.Path` to the file; second is the config
        type string (``"analyzer"`` or ``"group"``).
    root_changed(Path)
        Emitted when the root directory changes (via *Browse…* or
        :meth:`set_root`).
    """

    file_selected = pyqtSignal(Path, str)
    root_changed = pyqtSignal(Path)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._root: Optional[Path] = None

        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build the panel layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Top row: Browse button + path label
        top_row = QHBoxLayout()
        self._browse_btn = QPushButton("Browse...")
        top_row.addWidget(self._browse_btn)

        self._path_label = QLabel("No directory selected")
        self._path_label.setWordWrap(True)
        self._path_label.setStyleSheet("color: gray; font-size: 11px;")
        top_row.addWidget(self._path_label, stretch=1)
        layout.addLayout(top_row)

        # Tree widget
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setColumnCount(1)
        layout.addWidget(self._tree, stretch=1)

        # Bottom row: Refresh + Rename + New Config buttons
        bottom_row = QHBoxLayout()
        self._refresh_btn = QPushButton("Refresh")
        bottom_row.addWidget(self._refresh_btn)

        self._rename_btn = QPushButton("Rename…")
        self._rename_btn.setToolTip("Rename the selected config file")
        self._rename_btn.setEnabled(False)
        bottom_row.addWidget(self._rename_btn)

        self._new_config_btn = QPushButton("New Config…")
        self._new_config_btn.setToolTip("Create a new analyzer or group config file")
        bottom_row.addWidget(self._new_config_btn)

        layout.addLayout(bottom_row)

    def _connect_signals(self) -> None:
        """Wire up internal signals and slots."""
        self._browse_btn.clicked.connect(self._on_browse_clicked)
        self._refresh_btn.clicked.connect(self.refresh)
        self._rename_btn.clicked.connect(self._on_rename_clicked)
        self._new_config_btn.clicked.connect(self._on_new_config_clicked)
        self._tree.itemClicked.connect(self._on_item_clicked)
        self._tree.currentItemChanged.connect(self._on_current_item_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_root(self, root: Path) -> None:
        """Set the root directory and rebuild the tree."""
        self._root = root
        self._update_path_label()
        self._build_tree()
        self.root_changed.emit(root)

    def refresh(self) -> None:
        """Reload the tree from the current root directory."""
        if self._root is None:
            return
        self._build_tree()

    def get_root(self) -> Optional[Path]:
        """Return the current root directory, or ``None`` if not set."""
        return self._root

    # ------------------------------------------------------------------
    # Tree building
    # ------------------------------------------------------------------

    def _build_tree(self) -> None:
        """Clear and repopulate the tree from the current root."""
        self._tree.clear()

        if self._root is None:
            return

        # Validate directory structure
        try:
            dirs = discover_scan_config_dirs(self._root)
        except (FileNotFoundError, NotADirectoryError) as exc:
            logger.warning("Cannot discover scan config dirs: %s", exc)
            return

        style = self.style()
        folder_icon = style.standardIcon(QStyle.SP_DirIcon)
        file_icon = style.standardIcon(QStyle.SP_FileIcon)

        # --- Analyzers ---
        analyzers_item = QTreeWidgetItem(self._tree, ["Analyzers"])
        analyzers_item.setIcon(0, folder_icon)
        analyzers_item.setFlags(analyzers_item.flags() & ~Qt.ItemIsSelectable)

        if dirs.get("analyzers") is not None:
            try:
                self._populate_namespace_tree(
                    parent=analyzers_item,
                    files=list_analyzer_configs(self._root),
                    kind="analyzer",
                    file_icon=file_icon,
                    folder_icon=folder_icon,
                )
            except (FileNotFoundError, NotADirectoryError) as exc:
                logger.warning("Cannot list analyzer configs: %s", exc)

        # --- Groups ---
        groups_item = QTreeWidgetItem(self._tree, ["Groups"])
        groups_item.setIcon(0, folder_icon)
        groups_item.setFlags(groups_item.flags() & ~Qt.ItemIsSelectable)

        if dirs.get("groups") is not None:
            try:
                self._populate_namespace_tree(
                    parent=groups_item,
                    files=list_group_configs(self._root),
                    kind="group",
                    file_icon=file_icon,
                    folder_icon=folder_icon,
                )
            except (FileNotFoundError, NotADirectoryError) as exc:
                logger.warning("Cannot list group configs: %s", exc)

        # Expand all top-level items by default
        self._tree.expandAll()

    def _populate_namespace_tree(
        self,
        *,
        parent: QTreeWidgetItem,
        files: list[Path],
        kind: str,
        file_icon,
        folder_icon,
    ) -> None:
        """Group ``files`` by facility-namespace subdirectory and add to *parent*.

        Each file's immediate parent directory becomes a folder node
        under *parent*; the file itself becomes a selectable leaf.
        """
        namespaces: dict[str, QTreeWidgetItem] = {}
        for file_path in files:
            ns = file_path.parent.name
            if ns not in namespaces:
                ns_item = QTreeWidgetItem(parent, [ns])
                ns_item.setIcon(0, folder_icon)
                ns_item.setFlags(ns_item.flags() & ~Qt.ItemIsSelectable)
                namespaces[ns] = ns_item
            child = QTreeWidgetItem(namespaces[ns], [file_path.name])
            child.setIcon(0, file_icon)
            child.setData(0, Qt.UserRole, str(file_path))
            child.setData(0, Qt.UserRole + 1, kind)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_browse_clicked(self) -> None:
        """Open a directory picker and set the selected root."""
        start_dir = str(self._root) if self._root else ""
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select scan_analysis_configs directory",
            start_dir,
        )
        if chosen:
            self.set_root(Path(chosen))

    def _on_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        """Emit ``file_selected`` when a leaf item is clicked."""
        path_str = item.data(0, Qt.UserRole)
        config_type = item.data(0, Qt.UserRole + 1)

        if path_str is None or config_type is None:
            # Folder node — not selectable, ignore
            return

        file_path = Path(path_str)
        logger.debug("Tree item selected: %s (%s)", file_path, config_type)
        self.file_selected.emit(file_path, config_type)

    def _on_current_item_changed(
        self,
        current: Optional[QTreeWidgetItem],
        _previous: Optional[QTreeWidgetItem],
    ) -> None:
        """Enable Rename only when a renameable leaf item is selected."""
        if current is None:
            self._rename_btn.setEnabled(False)
            return
        config_type = current.data(0, Qt.UserRole + 1)
        self._rename_btn.setEnabled(config_type in ("analyzer", "group"))

    def _on_rename_clicked(self) -> None:
        """Rename the currently selected analyzer or group YAML on disk."""
        item = self._tree.currentItem()
        if item is None:
            return

        path_str = item.data(0, Qt.UserRole)
        config_type = item.data(0, Qt.UserRole + 1)

        if path_str is None or config_type not in ("analyzer", "group"):
            return

        old_path = Path(path_str)
        old_stem = old_path.stem

        new_stem, ok = QInputDialog.getText(
            self,
            "Rename Config",
            "New filename (without .yaml):",
            QLineEdit.Normal,
            old_stem,
        )
        if not ok or not new_stem.strip():
            return

        new_stem = new_stem.strip()
        if new_stem == old_stem:
            return

        new_path = old_path.parent / f"{new_stem}.yaml"

        if new_path.exists():
            QMessageBox.warning(
                self,
                "File Exists",
                f"A file named '{new_path.name}' already exists in\n{old_path.parent}",
            )
            return

        try:
            old_path.rename(new_path)
        except OSError as exc:
            QMessageBox.critical(
                self,
                "Rename Error",
                f"Failed to rename file:\n{exc}",
            )
            return

        # For analyzer configs, the filename stem is the analyzer ID
        # used by the analysis-group loader. The diagnostic's internal
        # ``name`` field is the device identity (different concept —
        # don't touch it on rename).
        logger.info("Renamed %s → %s", old_path.name, new_path.name)

        # Refresh tree and emit selection for the renamed file
        self._build_tree()
        self.file_selected.emit(new_path, config_type)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_path_label(self) -> None:
        """Update the path label with the current root directory."""
        if self._root is None:
            self._path_label.setText("No directory selected")
            self._path_label.setToolTip("")
            return

        dir_str = str(self._root)
        if len(dir_str) > 50:
            dir_str = "..." + dir_str[-47:]
        self._path_label.setText(dir_str)
        self._path_label.setToolTip(str(self._root))

    def _on_new_config_clicked(self) -> None:
        """Open a dialog to create a new analyzer or group config."""
        if self._root is None:
            QMessageBox.warning(
                self,
                "No Root Directory",
                "Please select a scan_analysis_configs directory first.",
            )
            return

        # Build facility lists for the dialog dropdowns.
        analyzer_facilities = list_facilities(self._root, "analyzers")
        group_facilities = list_facilities(self._root, "groups")
        # Make sure UNCLASSIFIED is offered as a fallback even if the
        # tree is otherwise empty for that kind.
        if "UNCLASSIFIED" not in analyzer_facilities:
            analyzer_facilities = analyzer_facilities + ["UNCLASSIFIED"]

        dialog = _NewConfigDialog(
            self,
            analyzer_facilities=analyzer_facilities,
            group_facilities=group_facilities,
        )
        if dialog.exec_() != QDialog.Accepted:
            return

        config_type = dialog.get_config_type()  # "Analyzer" | "Group"
        filename = dialog.get_filename()
        facility = dialog.get_facility()

        if not filename:
            return
        if not filename.endswith(".yaml"):
            filename += ".yaml"

        if config_type == "Analyzer":
            target_dir = self._root / "analyzers" / facility
            template = _new_analyzer_template(Path(filename).stem)
            kind = "analyzer"
        else:
            target_dir = self._root / "groups" / facility
            template = _new_group_template(Path(filename).stem)
            kind = "group"

        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / filename

        if file_path.exists():
            QMessageBox.warning(
                self,
                "File Exists",
                f"A file named '{filename}' already exists in\n{target_dir}",
            )
            return

        try:
            text = yaml.safe_dump(template, default_flow_style=False, sort_keys=False)
            file_path.write_text(text, encoding="utf-8")
            logger.info("Created new %s config: %s", config_type, file_path)
        except OSError as exc:
            QMessageBox.critical(
                self,
                "Creation Error",
                f"Failed to create config file:\n{exc}",
            )
            return

        # Refresh the tree and auto-select the new file
        self._build_tree()
        self.file_selected.emit(file_path, kind)


# ---------------------------------------------------------------------------
# Templates for newly-created configs
# ---------------------------------------------------------------------------


def _new_analyzer_template(stem: str) -> dict:
    """Return a minimal DiagnosticAnalysisConfig dict for a new analyzer."""
    return {
        "name": stem,
        "image_analyzer": "image_analysis.analyzers.beam_analyzer.BeamAnalyzer",
        "image": {
            "type": "camera",
        },
        "scan": {
            "priority": 100,
            "mode": "per_shot",
            "save": True,
        },
    }


def _new_group_template(stem: str) -> dict:
    """Return a minimal AnalysisGroupConfig dict for a new group."""
    return {
        "name": stem,
        "analyzers": [],
    }


# ---------------------------------------------------------------------------
# New Config Dialog
# ---------------------------------------------------------------------------


class _NewConfigDialog(QDialog):
    """Dialog to choose config type, facility namespace, and filename.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.
    analyzer_facilities : list of str
        Facility namespaces available under ``analyzers/``.
    group_facilities : list of str
        Facility namespaces available under ``groups/``.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        analyzer_facilities: list[str],
        group_facilities: list[str],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Config File")
        self.setMinimumWidth(380)

        self._analyzer_facilities = analyzer_facilities
        self._group_facilities = group_facilities

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._type_combo = QComboBox()
        self._type_combo.addItems(["Analyzer", "Group"])
        self._type_combo.currentTextChanged.connect(self._on_type_changed)
        form.addRow("Config type:", self._type_combo)

        self._facility_combo = QComboBox()
        self._facility_combo.setEditable(True)
        self._facility_combo.setToolTip(
            "Facility namespace (subdirectory). Type a new namespace name "
            "to create a new folder."
        )
        form.addRow("Facility:", self._facility_combo)
        self._refresh_facility_combo()

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Filename (without .yaml)")
        form.addRow("Filename:", self._name_edit)

        layout.addLayout(form)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _refresh_facility_combo(self) -> None:
        """Populate the facility combo based on the current config type."""
        current = self._facility_combo.currentText()
        self._facility_combo.clear()
        facilities = (
            self._analyzer_facilities
            if self._type_combo.currentText() == "Analyzer"
            else self._group_facilities
        )
        self._facility_combo.addItems(facilities)
        # Restore the previously-typed value if still applicable.
        if current:
            idx = self._facility_combo.findText(current)
            if idx >= 0:
                self._facility_combo.setCurrentIndex(idx)
            else:
                self._facility_combo.setEditText(current)

    def _on_type_changed(self, _text: str) -> None:
        """Refresh the facility combo when the type combo changes."""
        self._refresh_facility_combo()

    def _on_accept(self) -> None:
        """Validate and accept the dialog."""
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(
                self,
                "Validation Error",
                "Please enter a filename.",
            )
            return
        facility = self._facility_combo.currentText().strip()
        if not facility:
            QMessageBox.warning(
                self,
                "Validation Error",
                "Please choose or enter a facility namespace.",
            )
            return
        self.accept()

    def get_config_type(self) -> str:
        """Return the selected config type string (``"Analyzer"`` or ``"Group"``)."""
        return self._type_combo.currentText()

    def get_facility(self) -> str:
        """Return the chosen / entered facility namespace."""
        return self._facility_combo.currentText().strip()

    def get_filename(self) -> str:
        """Return the entered filename (stripped)."""
        return self._name_edit.text().strip()
