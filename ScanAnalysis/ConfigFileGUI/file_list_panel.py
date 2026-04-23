"""Left-side file list panel for the Config File Editor GUI.

Lists YAML configuration files from a directory, supports filtering,
and provides buttons for creating new 2D/1D configs.

This module is consumed by ``config_editor_window.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QAction,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QComboBox,
    QVBoxLayout,
    QWidget,
)

from .config_io import (
    create_new_camera_config,
    create_new_line_config,
    detect_config_type,
    list_config_files,
    save_config,
)

logger = logging.getLogger(__name__)


class FileListPanel(QWidget):
    """Left-side panel listing YAML config files from a directory.

    Displays config files with type prefixes ([2D], [1D], [?]),
    supports real-time filtering, and provides buttons for creating
    new configurations.

    Parameters
    ----------
    parent : QWidget, optional
        Parent Qt widget.

    Signals
    -------
    configSelected(Path)
        Emitted with the full path when a config file is selected.
    configCreated(Path)
        Emitted when a new config file is created.
    """

    configSelected = pyqtSignal(Path)
    configCreated = pyqtSignal(Path)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config_dir: Optional[Path] = None
        self._mode: str = "DEVICE"  # "DEVICE" or "SCAN"
        self._file_paths: dict[str, Path] = {}  # display text -> full path

        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build the panel layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Directory label
        self._dir_label = QLabel("No directory selected")
        self._dir_label.setWordWrap(True)
        self._dir_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self._dir_label)

        # Category selector for Scan Analysis mode
        self._category_selector = QComboBox()
        self._category_selector.addItems(
            [
                "Experiments",
                "Library Analyzers",
                "Library Groups",
            ]
        )
        self._category_selector.hide()
        layout.addWidget(self._category_selector)

        # Search / filter
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Filter configs...")
        self._filter_edit.setClearButtonEnabled(True)
        layout.addWidget(self._filter_edit)

        # File list
        self._list_widget = QListWidget()
        self._list_widget.setContextMenuPolicy(3)  # Qt.CustomContextMenu
        layout.addWidget(self._list_widget, stretch=1)

        # Button row
        btn_layout = QHBoxLayout()
        self._new_2d_btn = QPushButton("New 2D")
        self._new_1d_btn = QPushButton("New 1D")
        self._refresh_btn = QPushButton("Refresh")
        btn_layout.addWidget(self._new_2d_btn)
        btn_layout.addWidget(self._new_1d_btn)
        btn_layout.addWidget(self._refresh_btn)
        layout.addLayout(btn_layout)

    def _connect_signals(self) -> None:
        """Wire up internal signals and slots."""
        self._filter_edit.textChanged.connect(self._on_filter_changed)
        self._list_widget.currentItemChanged.connect(self._on_item_selected)
        self._list_widget.customContextMenuRequested.connect(self._on_context_menu)
        self._new_2d_btn.clicked.connect(self._on_new_2d_clicked)
        self._new_1d_btn.clicked.connect(self._on_new_1d_clicked)
        self._refresh_btn.clicked.connect(self.refresh)
        self._category_selector.currentIndexChanged.connect(self.refresh)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Switch between 'DEVICE' and 'SCAN' modes.

        Parameters
        ----------
        mode : str
            The mode to switch to ("DEVICE" or "SCAN").
        """
        self._mode = mode
        if mode == "SCAN":
            self._category_selector.show()
        else:
            self._category_selector.hide()
        self.refresh()

    def set_config_dir(self, dir_path: Path) -> None:
        """Set the configuration directory and refresh the file list.

        Parameters
        ----------
        dir_path : Path
            Directory containing YAML configuration files.
        """
        self._config_dir = dir_path

        # Truncate long paths for display
        dir_str = str(dir_path)
        if len(dir_str) > 50:
            dir_str = "..." + dir_str[-47:]
        self._dir_label.setText(dir_str)
        self._dir_label.setToolTip(str(dir_path))

        self.refresh()

    def refresh(self) -> None:
        """Reload the file list from the current directory.

        In 'DEVICE' mode, lists files from the base config directory.
        In 'SCAN' mode, lists files from sub-directories based on the
        selected category.

        Calls :func:`config_io.list_config_files` and populates the
        list widget with type-prefixed filenames.
        """
        self._list_widget.clear()
        self._file_paths.clear()

        if self._config_dir is None:
            return

        # Determine target directory and filter based on mode
        target_dir = self._config_dir
        filter_filename: Optional[str] = None

        if self._mode == "SCAN":
            category = self._category_selector.currentText()
            if category == "Experiments":
                target_dir = self._config_dir / "experiments"
            elif category == "Library Analyzers":
                target_dir = self._config_dir / "library/analyzers"
            elif category == "Library Groups":
                target_dir = self._config_dir / "library"
                filter_filename = "groups.yaml"

        try:
            files = list_config_files(target_dir)
        except (FileNotFoundError, NotADirectoryError) as exc:
            logger.warning("Cannot list config files in %s: %s", target_dir, exc)
            return

        if filter_filename:
            files = [f for f in files if f.name == filter_filename]

        for file_path in files:
            try:
                config_type = detect_config_type(file_path)
            except Exception:
                config_type = "unknown"

            if config_type == "camera_2d":
                prefix = "[2D]"
            elif config_type == "line_1d":
                prefix = "[1D]"
            else:
                prefix = "[?]"

            display_text = f"{prefix} {file_path.stem}"
            item = QListWidgetItem(display_text)
            self._list_widget.addItem(item)
            self._file_paths[display_text] = file_path

        # Re-apply current filter
        self._on_filter_changed(self._filter_edit.text())

    def get_selected_path(self) -> Optional[Path]:
        """Return the full path of the currently selected config file.

        Returns
        -------
        Path or None
            The selected file path, or ``None`` if nothing is selected.
        """
        current = self._list_widget.currentItem()
        if current is None:
            return None
        return self._file_paths.get(current.text())

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_item_selected(
        self, current: QListWidgetItem, _previous: QListWidgetItem
    ) -> None:
        """Handle single-click selection of a list item.

        Parameters
        ----------
        current : QListWidgetItem
            The newly selected item.
        _previous : QListWidgetItem
            The previously selected item (unused).
        """
        if current is None:
            return
        path = self._file_paths.get(current.text())
        if path is not None:
            self.configSelected.emit(path)

    def _on_filter_changed(self, text: str) -> None:
        """Filter the list widget items by hiding non-matching rows.

        Parameters
        ----------
        text : str
            The current filter text (case-insensitive substring match).
        """
        filter_lower = text.lower()
        for row in range(self._list_widget.count()):
            item = self._list_widget.item(row)
            if item is not None:
                item.setHidden(filter_lower not in item.text().lower())

    def _on_new_2d_clicked(self) -> None:
        """Prompt for a name and create a new 2D camera config."""
        if self._config_dir is None:
            QMessageBox.warning(
                self, "No Directory", "Please select a config directory first."
            )
            return

        name, ok = QInputDialog.getText(
            self, "New 2D Camera Config", "Enter config name (without extension):"
        )
        if not ok or not name.strip():
            return

        name = name.strip()
        file_path = self._config_dir / f"{name}.yaml"

        if file_path.exists():
            QMessageBox.warning(
                self,
                "File Exists",
                f"A config file named '{name}.yaml' already exists.",
            )
            return

        try:
            config = create_new_camera_config(name)
            save_config(config, file_path)
            logger.info("Created new 2D config: %s", file_path)
            self.refresh()
            self.configCreated.emit(file_path)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to create config: {exc}")

    def _on_new_1d_clicked(self) -> None:
        """Prompt for a name and create a new 1D line config."""
        if self._config_dir is None:
            QMessageBox.warning(
                self, "No Directory", "Please select a config directory first."
            )
            return

        name, ok = QInputDialog.getText(
            self, "New 1D Line Config", "Enter config name (without extension):"
        )
        if not ok or not name.strip():
            return

        name = name.strip()
        file_path = self._config_dir / f"{name}.yaml"

        if file_path.exists():
            QMessageBox.warning(
                self,
                "File Exists",
                f"A config file named '{name}.yaml' already exists.",
            )
            return

        try:
            config = create_new_line_config(name)
            save_config(config, file_path)
            logger.info("Created new 1D config: %s", file_path)
            self.refresh()
            self.configCreated.emit(file_path)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to create config: {exc}")

    def _on_context_menu(self, position) -> None:
        """Show a right-click context menu with stub actions.

        Parameters
        ----------
        position : QPoint
            The position where the context menu was requested.
        """
        item = self._list_widget.itemAt(position)
        if item is None:
            return

        menu = QMenu(self)

        duplicate_action = QAction("Duplicate", self)
        duplicate_action.triggered.connect(
            lambda: QMessageBox.information(
                self,
                "Duplicate",
                "Duplicate functionality will be implemented in a future update.",
            )
        )
        menu.addAction(duplicate_action)

        rename_action = QAction("Rename", self)
        rename_action.triggered.connect(
            lambda: QMessageBox.information(
                self,
                "Rename",
                "Rename functionality will be implemented in a future update.",
            )
        )
        menu.addAction(rename_action)

        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(
            lambda: QMessageBox.information(
                self,
                "Delete",
                "Delete functionality will be implemented in a future update.",
            )
        )
        menu.addAction(delete_action)

        menu.exec_(self._list_widget.mapToGlobal(position))
