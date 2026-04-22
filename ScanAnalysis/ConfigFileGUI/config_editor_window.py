"""Main window for the Config File Editor GUI.

Assembles the :class:`FileListPanel`, :class:`ConfigEditorPanel`, and
:class:`YamlPreviewPanel` into a single ``QMainWindow`` with menus,
toolbar actions, and a status bar.

This is the top-level widget launched by ``main.py``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QAction,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
)

from .config_editor_panel import ConfigEditorPanel
from .config_io import detect_config_type, load_config, save_config
from .file_list_panel import FileListPanel
from .yaml_preview import YamlPreviewPanel

logger = logging.getLogger(__name__)


class ConfigEditorWindow(QMainWindow):
    """Main window for the Config File Editor.

    Provides a three-panel layout (file list, editor, YAML preview)
    with menus for file operations, validation, and view toggles.

    Parameters
    ----------
    config_dir : Path, optional
        Path to the directory containing device configuration YAML
        files.  If ``None``, the user is prompted to select a directory
        or the ``IMAGE_ANALYSIS_CONFIG_DIR`` environment variable is
        used.
    parent : QMainWindow, optional
        Parent widget, if any.
    """

    def __init__(
        self,
        config_dir: Optional[Path] = None,
        parent: Optional[QMainWindow] = None,
    ) -> None:
        super().__init__(parent)
        self._current_file: Optional[Path] = None
        self._current_config = None
        self._current_config_type: str = ""
        self._yaml_preview_visible: bool = False
        self._debounce_timer: Optional[QTimer] = None

        self._setup_window()
        self._setup_panels()
        self._setup_menus()
        self._setup_statusbar()
        self._connect_signals()
        self._setup_debounce_timer()

        # Determine initial config directory
        if config_dir is not None:
            self._file_list.set_config_dir(config_dir)
        else:
            env_dir = os.environ.get("IMAGE_ANALYSIS_CONFIG_DIR")
            if env_dir and Path(env_dir).is_dir():
                self._file_list.set_config_dir(Path(env_dir))
            else:
                # Defer directory selection — user can open via menu
                pass

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        """Configure window title, size, and properties."""
        self.setWindowTitle("Config File Editor")
        self.resize(1400, 900)

    def _setup_panels(self) -> None:
        """Create the three-panel splitter layout."""
        self._splitter = QSplitter(self)
        self.setCentralWidget(self._splitter)

        # Left: file list
        self._file_list = FileListPanel(self)
        self._file_list.setMinimumWidth(250)
        self._splitter.addWidget(self._file_list)

        # Center: editor
        self._editor_panel = ConfigEditorPanel(self)
        self._splitter.addWidget(self._editor_panel)

        # Right: YAML preview (initially hidden)
        self._yaml_preview = YamlPreviewPanel(self)
        self._splitter.addWidget(self._yaml_preview)
        self._yaml_preview.hide()

        # Set initial splitter sizes
        self._splitter.setSizes([280, 820, 300])

    def _setup_menus(self) -> None:
        """Build the menu bar with File, Edit, and View menus."""
        menubar = self.menuBar()

        # --- File menu ---
        file_menu = menubar.addMenu("&File")

        self._open_dir_action = QAction("Open &Directory...", self)
        self._open_dir_action.triggered.connect(self._on_open_directory)
        file_menu.addAction(self._open_dir_action)

        file_menu.addSeparator()

        self._save_action = QAction("&Save", self)
        self._save_action.setShortcut("Ctrl+S")
        self._save_action.triggered.connect(self._on_save)
        file_menu.addAction(self._save_action)

        self._save_as_action = QAction("Save &As...", self)
        self._save_as_action.triggered.connect(self._on_save_as)
        file_menu.addAction(self._save_as_action)

        self._reload_action = QAction("&Reload", self)
        self._reload_action.triggered.connect(self._on_reload)
        file_menu.addAction(self._reload_action)

        file_menu.addSeparator()

        self._exit_action = QAction("E&xit", self)
        self._exit_action.triggered.connect(self.close)
        file_menu.addAction(self._exit_action)

        # --- Edit menu ---
        edit_menu = menubar.addMenu("&Edit")

        self._validate_action = QAction("&Validate", self)
        self._validate_action.setShortcut("Ctrl+Shift+V")
        self._validate_action.triggered.connect(self._on_validate)
        edit_menu.addAction(self._validate_action)

        # --- View menu ---
        view_menu = menubar.addMenu("&View")

        self._toggle_yaml_action = QAction("Toggle YAML &Preview", self)
        self._toggle_yaml_action.setCheckable(True)
        self._toggle_yaml_action.setChecked(False)
        self._toggle_yaml_action.triggered.connect(self._on_toggle_yaml_preview)
        view_menu.addAction(self._toggle_yaml_action)

    def _setup_statusbar(self) -> None:
        """Create the status bar."""
        self._statusbar = QStatusBar(self)
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("Ready")

    def _connect_signals(self) -> None:
        """Wire up signals between panels."""
        self._file_list.configSelected.connect(self._on_config_selected)
        self._file_list.configCreated.connect(self._on_config_selected)
        self._editor_panel.dirtyStateChanged.connect(self._on_dirty_state_changed)

    def _setup_debounce_timer(self) -> None:
        """Set up a debounce timer for YAML preview updates."""
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(300)
        self._debounce_timer.timeout.connect(self._update_yaml_preview)

        # Connect editor valueChanged to debounced preview update
        self._editor_panel.valueChanged.connect(self._on_editor_value_changed)

    # ------------------------------------------------------------------
    # Slots: File operations
    # ------------------------------------------------------------------

    def _on_open_directory(self) -> None:
        """Open a directory picker and set the config directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Config Directory")
        if dir_path:
            self._file_list.set_config_dir(Path(dir_path))
            self._statusbar.showMessage(f"Opened directory: {dir_path}")

    def _on_config_selected(self, file_path: Path) -> None:
        """Handle selection of a config file from the file list.

        Parameters
        ----------
        file_path : Path
            The full path to the selected config file.
        """
        if not self._check_unsaved_changes():
            return

        try:
            config = load_config(file_path)
            config_type = detect_config_type(file_path)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Failed to load config:\n{exc}")
            self._statusbar.showMessage(f"Error loading: {file_path.name}")
            return

        self._current_file = file_path
        self._current_config = config
        self._current_config_type = config_type

        self._editor_panel.load_config(config, config_type, file_path)
        self._update_title()
        self._statusbar.showMessage(f"Loaded: {file_path.name} ({config_type})")

        # Update YAML preview if visible
        if self._yaml_preview_visible:
            self._update_yaml_preview()

    def _on_save(self) -> None:
        """Save the current config to its file path."""
        if self._current_file is None:
            self._statusbar.showMessage("No file loaded to save")
            return

        # Validate first
        errors = self._editor_panel.validate()
        if errors:
            error_text = "\n".join(f"• {e}" for e in errors[:10])
            if len(errors) > 10:
                error_text += f"\n... and {len(errors) - 10} more errors"
            QMessageBox.warning(
                self,
                "Validation Errors",
                f"Cannot save — fix the following errors:\n\n{error_text}",
            )
            self._statusbar.showMessage(
                f"Save failed: {len(errors)} validation error(s)"
            )
            return

        config_dict = self._editor_panel.get_config_dict()

        try:
            if self._current_config_type == "camera_2d":
                from image_analysis.processing.array2d.config_models import (
                    CameraConfig,
                )

                validated = CameraConfig.model_validate(config_dict)
            elif self._current_config_type == "line_1d":
                from image_analysis.processing.array1d.config_models import (
                    Line1DConfig,
                )

                validated = Line1DConfig.model_validate(config_dict)
            else:
                self._statusbar.showMessage("Unknown config type — cannot save")
                return

            save_config(validated, self._current_file)
            self._editor_panel.mark_clean()
            self._update_title()
            self._statusbar.showMessage(f"Saved: {self._current_file.name}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", f"Failed to save config:\n{exc}")
            self._statusbar.showMessage(f"Save error: {exc}")

    def _on_save_as(self) -> None:
        """Save the current config to a new file path."""
        if self._current_config_type == "":
            self._statusbar.showMessage("No config loaded")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Config As",
            str(self._current_file) if self._current_file else "",
            "YAML Files (*.yaml *.yml)",
        )
        if not file_path:
            return

        self._current_file = Path(file_path)
        self._on_save()
        self._file_list.refresh()

    def _on_reload(self) -> None:
        """Reload the current file from disk."""
        if self._current_file is None:
            self._statusbar.showMessage("No file loaded to reload")
            return

        if self._editor_panel.is_dirty():
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Reload from disk and lose changes?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self._on_config_selected(self._current_file)

    # ------------------------------------------------------------------
    # Slots: Edit operations
    # ------------------------------------------------------------------

    def _on_validate(self) -> None:
        """Run validation and show results."""
        if self._current_config_type == "":
            self._statusbar.showMessage("No config loaded to validate")
            return

        errors = self._editor_panel.validate()
        if not errors:
            self._statusbar.showMessage("✓ Validation passed")
            QMessageBox.information(self, "Validation", "Configuration is valid.")
        else:
            error_text = "\n".join(f"• {e}" for e in errors[:20])
            if len(errors) > 20:
                error_text += f"\n... and {len(errors) - 20} more errors"
            self._statusbar.showMessage(f"✗ {len(errors)} validation error(s)")
            QMessageBox.warning(
                self,
                "Validation Errors",
                f"Found {len(errors)} error(s):\n\n{error_text}",
            )

    # ------------------------------------------------------------------
    # Slots: View operations
    # ------------------------------------------------------------------

    def _on_toggle_yaml_preview(self, checked: bool) -> None:
        """Show or hide the YAML preview panel.

        Parameters
        ----------
        checked : bool
            Whether the preview should be visible.
        """
        self._yaml_preview_visible = checked
        if checked:
            self._yaml_preview.show()
            self._update_yaml_preview()
            # Restore splitter sizes to show preview
            self._splitter.setSizes([280, 820, 300])
        else:
            self._yaml_preview.hide()

    def _on_editor_value_changed(self) -> None:
        """Debounce YAML preview updates on editor changes."""
        if self._yaml_preview_visible and self._debounce_timer is not None:
            self._debounce_timer.start()

    def _update_yaml_preview(self) -> None:
        """Update the YAML preview panel from current editor values."""
        if not self._yaml_preview_visible:
            return
        try:
            config_dict = self._editor_panel.get_config_dict()
            self._yaml_preview.update_preview(config_dict)
        except Exception as exc:
            logger.warning("Failed to update YAML preview: %s", exc)

    # ------------------------------------------------------------------
    # Slots: Dirty state
    # ------------------------------------------------------------------

    def _on_dirty_state_changed(self, dirty: bool) -> None:
        """Handle dirty state changes from the editor panel.

        Parameters
        ----------
        dirty : bool
            Whether the editor has unsaved changes.
        """
        self._update_title()

    # ------------------------------------------------------------------
    # Window management
    # ------------------------------------------------------------------

    def _update_title(self) -> None:
        """Update the window title to reflect the current file and dirty state."""
        base = "Config File Editor"
        if self._current_file is not None:
            name = self._current_file.name
            dirty_marker = " *" if self._editor_panel.is_dirty() else ""
            self.setWindowTitle(f"{base} - {name}{dirty_marker}")
        else:
            self.setWindowTitle(base)

    def _check_unsaved_changes(self) -> bool:
        """Check for unsaved changes and prompt the user.

        Returns
        -------
        bool
            ``True`` if it is safe to proceed (no changes, or user
            chose to discard/save).  ``False`` if the user cancelled.
        """
        if not self._editor_panel.is_dirty():
            return True

        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved changes. Do you want to save before proceeding?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )

        if reply == QMessageBox.Save:
            self._on_save()
            # If still dirty after save attempt, the save failed
            return not self._editor_panel.is_dirty()
        elif reply == QMessageBox.Discard:
            return True
        else:
            # Cancel
            return False

    def closeEvent(self, event) -> None:
        """Handle window close — check for unsaved changes.

        Parameters
        ----------
        event : QCloseEvent
            The close event.
        """
        if self._check_unsaved_changes():
            event.accept()
        else:
            event.ignore()
