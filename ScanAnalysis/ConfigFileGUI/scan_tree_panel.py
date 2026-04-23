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

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
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

        # Refresh button
        self._refresh_btn = QPushButton("Refresh")
        layout.addWidget(self._refresh_btn)

    def _connect_signals(self) -> None:
        """Wire up internal signals and slots."""
        self._browse_btn.clicked.connect(self._on_browse_clicked)
        self._refresh_btn.clicked.connect(self.refresh)
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
