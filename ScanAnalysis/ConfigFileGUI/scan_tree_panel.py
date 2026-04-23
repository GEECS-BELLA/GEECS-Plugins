"""Tree-view navigation panel for scan analysis configuration files.

Displays the scan_analysis_configs directory as a three-category tree:

* **Experiments** — ``experiments/*.yaml``
* **Library Analyzers** — ``library/analyzers/*.yaml``
* **Groups** — ``library/groups.yaml``

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
    get_groups_file,
    list_analyzer_configs,
    list_experiment_configs,
)

logger = logging.getLogger(__name__)


class ScanTreePanel(QWidget):
    """Tree-based file browser for scan analysis configurations.

    Presents a hierarchical view of the ``scan_analysis_configs/`` directory
    with three top-level categories (Experiments, Library Analyzers, Groups).
    Only leaf items (individual files) are selectable.

    Parameters
    ----------
    parent : QWidget, optional
        Parent Qt widget.

    Signals
    -------
    file_selected(Path, str)
        Emitted when a leaf item is clicked.  First argument is the full
        :class:`~pathlib.Path` to the file; second is the config type
        string (``"experiment"``, ``"analyzer"``, or ``"groups"``).
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

        # Bottom row: Refresh + New Config buttons
        bottom_row = QHBoxLayout()
        self._refresh_btn = QPushButton("Refresh")
        bottom_row.addWidget(self._refresh_btn)

        self._new_config_btn = QPushButton("New Config…")
        self._new_config_btn.setToolTip(
            "Create a new library analyzer or experiment config file"
        )
        bottom_row.addWidget(self._new_config_btn)

        layout.addLayout(bottom_row)

    def _connect_signals(self) -> None:
        """Wire up internal signals and slots."""
        self._browse_btn.clicked.connect(self._on_browse_clicked)
        self._refresh_btn.clicked.connect(self.refresh)
        self._new_config_btn.clicked.connect(self._on_new_config_clicked)
        self._tree.itemClicked.connect(self._on_item_clicked)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_root(self, root: Path) -> None:
        """Set the root directory and rebuild the tree.

        Parameters
        ----------
        root : Path
            Path to the ``scan_analysis_configs/`` root directory.
        """
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
        """Return the current root directory.

        Returns
        -------
        Path or None
            The current root directory, or ``None`` if not yet set.
        """
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

        # --- Experiments ---
        experiments_item = QTreeWidgetItem(self._tree, ["Experiments"])
        experiments_item.setIcon(0, folder_icon)
        experiments_item.setFlags(experiments_item.flags() & ~Qt.ItemIsSelectable)

        if dirs.get("experiments") is not None:
            try:
                exp_files = list_experiment_configs(self._root)
                for file_path in exp_files:
                    child = QTreeWidgetItem(experiments_item, [file_path.name])
                    child.setIcon(0, file_icon)
                    child.setData(0, Qt.UserRole, str(file_path))
                    child.setData(0, Qt.UserRole + 1, "experiment")
            except (FileNotFoundError, NotADirectoryError) as exc:
                logger.warning("Cannot list experiment configs: %s", exc)

        # --- Library Analyzers ---
        analyzers_item = QTreeWidgetItem(self._tree, ["Library Analyzers"])
        analyzers_item.setIcon(0, folder_icon)
        analyzers_item.setFlags(analyzers_item.flags() & ~Qt.ItemIsSelectable)

        if dirs.get("analyzers") is not None:
            try:
                ana_files = list_analyzer_configs(self._root)
                for file_path in ana_files:
                    child = QTreeWidgetItem(analyzers_item, [file_path.name])
                    child.setIcon(0, file_icon)
                    child.setData(0, Qt.UserRole, str(file_path))
                    child.setData(0, Qt.UserRole + 1, "analyzer")
            except (FileNotFoundError, NotADirectoryError) as exc:
                logger.warning("Cannot list analyzer configs: %s", exc)

        # --- Groups ---
        groups_item = QTreeWidgetItem(self._tree, ["Groups"])
        groups_item.setIcon(0, folder_icon)
        groups_item.setFlags(groups_item.flags() & ~Qt.ItemIsSelectable)

        groups_path = get_groups_file(self._root)
        if groups_path is not None:
            child = QTreeWidgetItem(groups_item, [groups_path.name])
            child.setIcon(0, file_icon)
            child.setData(0, Qt.UserRole, str(groups_path))
            child.setData(0, Qt.UserRole + 1, "groups")

        # Expand all top-level items by default
        self._tree.expandAll()

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
        """Emit :pyqtSignal:`file_selected` when a leaf item is clicked.

        Parameters
        ----------
        item : QTreeWidgetItem
            The clicked tree item.
        _column : int
            The column index (unused; single-column tree).
        """
        path_str = item.data(0, Qt.UserRole)
        config_type = item.data(0, Qt.UserRole + 1)

        if path_str is None or config_type is None:
            # Top-level folder item — not selectable, ignore
            return

        file_path = Path(path_str)
        logger.debug("Tree item selected: %s (%s)", file_path, config_type)
        self.file_selected.emit(file_path, config_type)

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
        """Open a dialog to create a new analyzer or experiment config."""
        if self._root is None:
            QMessageBox.warning(
                self,
                "No Root Directory",
                "Please select a scan_analysis_configs directory first.",
            )
            return

        dialog = _NewConfigDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return

        config_type = dialog.get_config_type()
        filename = dialog.get_filename()

        if not filename:
            return

        # Ensure .yaml extension
        if not filename.endswith(".yaml"):
            filename += ".yaml"

        if config_type == "Library Analyzer":
            target_dir = self._root / "library" / "analyzers"
            target_dir.mkdir(parents=True, exist_ok=True)
            file_path = target_dir / filename
            template = _ANALYZER_TEMPLATE.copy()
            template["id"] = Path(filename).stem
            scan_type = "analyzer"
        else:
            target_dir = self._root / "experiments"
            target_dir.mkdir(parents=True, exist_ok=True)
            file_path = target_dir / filename
            template = _EXPERIMENT_TEMPLATE.copy()
            template["experiment"] = Path(filename).stem
            scan_type = "experiment"

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
        self.file_selected.emit(file_path, scan_type)


# ---------------------------------------------------------------------------
# New Config Dialog
# ---------------------------------------------------------------------------

_ANALYZER_TEMPLATE: dict = {
    "id": "NewAnalyzer",
    "type": "array2d",
    "device_name": "",
    "image_analyzer": {
        "analyzer_class": "",
        "config_name": "",
    },
}

_EXPERIMENT_TEMPLATE: dict = {
    "experiment": "new_experiment",
    "description": "",
    "version": "1.0",
    "include": [],
}


class _NewConfigDialog(QDialog):
    """Simple dialog to choose config type and filename for creation.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Config File")
        self.setMinimumWidth(350)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._type_combo = QComboBox()
        self._type_combo.addItems(["Library Analyzer", "Experiment"])
        form.addRow("Config type:", self._type_combo)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Filename (without .yaml)")
        form.addRow("Filename:", self._name_edit)

        layout.addLayout(form)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

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
        self.accept()

    def get_config_type(self) -> str:
        """Return the selected config type string.

        Returns
        -------
        str
            Either ``"Library Analyzer"`` or ``"Experiment"``.
        """
        return self._type_combo.currentText()

    def get_filename(self) -> str:
        """Return the entered filename (stripped).

        Returns
        -------
        str
            The filename without leading/trailing whitespace.
        """
        return self._name_edit.text().strip()
