"""Main window for the Scan Config File Editor GUI.

Single-window editor for the post-PR-E unified-diagnostic config layout.
The window has three panels:

* **Tree** (left) — :class:`ScanTreePanel`, browses
  ``scan_analysis_configs/analyzers/<facility>/`` and
  ``scan_analysis_configs/groups/<facility>/``.
* **Editor stack** (center) — switches between
  :class:`ScanAnalyzerEditorPanel` (when an analyzer YAML is selected)
  and :class:`GroupEditorPanel` (when a group YAML is selected). A
  placeholder shows when nothing is selected.
* **YAML preview** (right, toggleable) — :class:`YamlPreviewPanel`
  rendering the current editor state.

Tools menu retains the **Analysis Preview** dialog — run a single-image
analysis test against the current diagnostic config without leaving
the GUI.

This is the top-level widget launched by ``main.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QAction,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .analysis_preview import AnalysisPreviewDialog
from .groups_editor import GroupEditorPanel
from .scan_analyzer_editor import ScanAnalyzerEditorPanel
from .scan_config_io import (
    list_all_analyzer_ids,
    load_analyzer_yaml,
    load_group_yaml,
    save_analyzer_yaml,
    save_group_yaml,
)
from .scan_tree_panel import ScanTreePanel
from .yaml_preview import YamlPreviewPanel

logger = logging.getLogger(__name__)


# Stacked-widget indices
_IDX_ANALYZER = 0
_IDX_GROUP = 1
_IDX_PLACEHOLDER = 2


class ConfigEditorWindow(QMainWindow):
    """Main window for the Scan Config File Editor.

    Parameters
    ----------
    scan_config_dir : Path, optional
        Path to the ``scan_analysis_configs/`` root directory. When
        provided, the tree panel is pre-populated.
    parent : QMainWindow, optional
        Parent widget, if any.
    """

    def __init__(
        self,
        scan_config_dir: Optional[Path] = None,
        parent: Optional[QMainWindow] = None,
    ) -> None:
        super().__init__(parent)

        # Current file state
        self._current_path: Optional[Path] = None
        self._current_type: str = ""  # "analyzer" | "group" | ""

        # YAML-preview state
        self._yaml_preview_visible: bool = False
        self._debounce_timer: Optional[QTimer] = None

        self._setup_window()
        self._setup_panels()
        self._setup_menus()
        self._setup_statusbar()
        self._setup_debounce_timer()

        if scan_config_dir is not None:
            self._tree.set_root(scan_config_dir)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        """Configure the window's basics."""
        self.setWindowTitle("Scan Config Editor")
        self.resize(1400, 900)

    def _setup_panels(self) -> None:
        """Build the three-panel layout (tree / editor stack / preview)."""
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._splitter = QSplitter(Qt.Horizontal)

        # Left: tree
        self._tree = ScanTreePanel(self)
        self._tree.setMinimumWidth(200)
        self._splitter.addWidget(self._tree)

        # Center: editor stack + save button
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        self._stack = QStackedWidget()
        self._analyzer_editor = ScanAnalyzerEditorPanel(self)
        self._stack.addWidget(self._analyzer_editor)  # _IDX_ANALYZER
        self._group_editor = GroupEditorPanel(self)
        self._stack.addWidget(self._group_editor)  # _IDX_GROUP
        placeholder = QLabel("Select a config file from the tree on the left")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: gray; font-size: 14px;")
        self._stack.addWidget(placeholder)  # _IDX_PLACEHOLDER
        self._stack.setCurrentIndex(_IDX_PLACEHOLDER)

        center_layout.addWidget(self._stack, stretch=1)

        self._save_btn = QPushButton("Save")
        self._save_btn.setToolTip("Save the current config file (Ctrl+S)")
        self._save_btn.clicked.connect(self._on_save)
        center_layout.addWidget(self._save_btn)

        self._splitter.addWidget(center)

        # Right: YAML preview (initially hidden)
        self._yaml_preview = YamlPreviewPanel(self)
        self._yaml_preview.hide()
        self._splitter.addWidget(self._yaml_preview)

        # Splitter sizes — tree narrower, editor takes most of the width
        self._splitter.setSizes([280, 820, 300])
        outer.addWidget(self._splitter)

        # Wire signals
        self._tree.file_selected.connect(self._on_file_selected)
        self._tree.root_changed.connect(self._on_root_changed)
        self._analyzer_editor.config_changed.connect(self._on_editor_changed)
        self._group_editor.config_changed.connect(self._on_editor_changed)

    def _setup_menus(self) -> None:
        """Build the File / Tools menu bar."""
        menubar = self.menuBar()

        # --- File ---
        file_menu = menubar.addMenu("&File")

        open_dir_action = QAction("Open &Directory…", self)
        open_dir_action.triggered.connect(self._on_open_directory)
        file_menu.addAction(open_dir_action)

        file_menu.addSeparator()

        save_action = QAction("&Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._on_save)
        file_menu.addAction(save_action)

        reload_action = QAction("&Reload", self)
        reload_action.triggered.connect(self._on_reload)
        file_menu.addAction(reload_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # --- Tools ---
        tools_menu = menubar.addMenu("&Tools")

        self._toggle_yaml_action = QAction("Toggle YAML &Preview", self)
        self._toggle_yaml_action.setCheckable(True)
        self._toggle_yaml_action.setChecked(False)
        self._toggle_yaml_action.triggered.connect(self._on_toggle_yaml_preview)
        tools_menu.addAction(self._toggle_yaml_action)

        tools_menu.addSeparator()

        analysis_preview_action = QAction("&Analysis Preview…", self)
        analysis_preview_action.setToolTip(
            "Run a single-image analysis test against the current diagnostic config."
        )
        analysis_preview_action.triggered.connect(self._on_analysis_preview)
        tools_menu.addAction(analysis_preview_action)

    def _setup_statusbar(self) -> None:
        """Create the status bar."""
        self._statusbar = QStatusBar(self)
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("Ready")

    def _setup_debounce_timer(self) -> None:
        """Set up a debounce timer for YAML preview updates."""
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(300)
        self._debounce_timer.timeout.connect(self._update_yaml_preview)

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def _on_open_directory(self) -> None:
        """Open a directory picker, set the tree root to the chosen folder."""
        from PyQt5.QtWidgets import QFileDialog

        dir_path = QFileDialog.getExistingDirectory(
            self, "Select scan_analysis_configs directory"
        )
        if dir_path:
            self._tree.set_root(Path(dir_path))
            self._statusbar.showMessage(f"Opened: {dir_path}")

    def _on_file_selected(self, file_path: Path, config_type: str) -> None:
        """Load the selected file into the appropriate editor."""
        try:
            if config_type == "analyzer":
                data = load_analyzer_yaml(file_path)
                self._analyzer_editor.load_config(data)
                self._stack.setCurrentIndex(_IDX_ANALYZER)
            elif config_type == "group":
                data = load_group_yaml(file_path)
                self._group_editor.load_config(data)
                root = self._tree.get_root()
                if root is not None:
                    self._group_editor.set_analyzer_ids(list_all_analyzer_ids(root))
                self._stack.setCurrentIndex(_IDX_GROUP)
            else:
                self._statusbar.showMessage(f"Unknown config type: {config_type}")
                return

            self._current_path = file_path
            self._current_type = config_type
            self._statusbar.showMessage(f"Loaded: {file_path.name} ({config_type})")
            self._update_title()
            self._update_yaml_preview()
        except Exception as exc:
            logger.exception("Failed to load %s", file_path)
            QMessageBox.critical(
                self, "Load Error", f"Failed to load {file_path.name}:\n{exc}"
            )
            self._statusbar.showMessage(f"Error loading: {file_path.name}")

    def _on_root_changed(self, root: Path) -> None:
        """Refresh autocomplete sources when the tree root changes."""
        try:
            self._group_editor.set_analyzer_ids(list_all_analyzer_ids(root))
        except Exception as exc:
            logger.warning("Error refreshing analyzer IDs: %s", exc)
        self._statusbar.showMessage(f"Scan config root: {root}")

    def _on_save(self) -> None:
        """Save the current editor's state to disk."""
        if self._current_path is None or not self._current_type:
            self._statusbar.showMessage("No file loaded to save")
            return

        try:
            if self._current_type == "analyzer":
                data = self._analyzer_editor.get_config_dict()
                save_analyzer_yaml(self._current_path, data)
            elif self._current_type == "group":
                data = self._group_editor.get_config_dict()
                save_group_yaml(self._current_path, data)
            else:
                self._statusbar.showMessage(
                    f"Unknown config type: {self._current_type}"
                )
                return
            self._statusbar.showMessage(f"Saved: {self._current_path.name}")
        except Exception as exc:
            logger.exception("Failed to save %s", self._current_path)
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save {self._current_path.name}:\n{exc}",
            )
            self._statusbar.showMessage(
                f"Save error: {self._current_path.name} — {exc}"
            )

    def _on_reload(self) -> None:
        """Reload the current file from disk, discarding in-memory edits."""
        if self._current_path is None:
            self._statusbar.showMessage("No file loaded to reload")
            return
        self._on_file_selected(self._current_path, self._current_type)

    # ------------------------------------------------------------------
    # Editor changes / YAML preview
    # ------------------------------------------------------------------

    def _on_editor_changed(self) -> None:
        """Debounce YAML preview updates when the editor reports a change."""
        if self._yaml_preview_visible and self._debounce_timer is not None:
            self._debounce_timer.start()

    def _on_toggle_yaml_preview(self, checked: bool) -> None:
        """Show or hide the YAML preview panel."""
        self._yaml_preview_visible = checked
        if checked:
            self._yaml_preview.show()
            self._update_yaml_preview()
        else:
            self._yaml_preview.hide()

    def _update_yaml_preview(self) -> None:
        """Update the YAML preview panel from the active editor."""
        if not self._yaml_preview_visible:
            return
        try:
            idx = self._stack.currentIndex()
            if idx == _IDX_ANALYZER:
                data = self._analyzer_editor.get_config_dict()
                self._yaml_preview.update_preview(data)
            elif idx == _IDX_GROUP:
                data = self._group_editor.get_config_dict()
                self._yaml_preview.update_preview(data)
            else:
                self._yaml_preview.clear()
        except Exception as exc:
            logger.warning("Failed to update YAML preview: %s", exc)

    # ------------------------------------------------------------------
    # Analysis preview (Tools menu)
    # ------------------------------------------------------------------

    def _on_analysis_preview(self) -> None:
        """Open the Analysis Preview dialog against the current diagnostic."""
        if self._current_type != "analyzer":
            QMessageBox.information(
                self,
                "Analysis Preview",
                "Select an analyzer config (not a group) to run an analysis preview.",
            )
            return

        root = self._tree.get_root()
        analyzers_dir = (root / "analyzers") if root is not None else None

        dlg = AnalysisPreviewDialog(
            config_dir=analyzers_dir,
            get_current_config=self._get_current_diagnostic_for_preview,
            parent=self,
        )
        dlg.show()

    def _get_current_diagnostic_for_preview(self) -> tuple:
        """Provide the live diagnostic dict to the Analysis Preview dialog.

        Returns ``(name, config_dict, config_type)`` where:

        * ``name`` is the diagnostic's filename stem (so the preview
          dialog can match the in-memory dict to the typed name).
        * ``config_dict`` is the editor's current dict — i.e. the
          unfiltered ``DiagnosticAnalysisConfig`` shape. The dialog's
          detector reads ``image.type`` to dispatch; ImageAnalysis's
          ``load_camera_config`` / ``load_line_config`` then unwrap
          the ``image:`` section.
        * ``config_type`` is ``"camera_2d"`` / ``"line_1d"`` /
          ``"unknown"`` derived from the same ``image.type`` field.
        """
        if self._current_path is None or self._current_type != "analyzer":
            return ("", {}, "unknown")

        name = self._current_path.stem
        config_dict = self._analyzer_editor.get_config_dict()
        image = config_dict.get("image") if isinstance(config_dict, dict) else None
        if isinstance(image, dict):
            kind = image.get("type")
            if kind == "camera":
                config_type = "camera_2d"
            elif kind == "line":
                config_type = "line_1d"
            else:
                config_type = "unknown"
        else:
            config_type = "unknown"
        return (name, config_dict, config_type)

    # ------------------------------------------------------------------
    # Title management
    # ------------------------------------------------------------------

    def _update_title(self) -> None:
        """Update the window title with the current file name."""
        base = "Scan Config Editor"
        if self._current_path is not None:
            self.setWindowTitle(f"{base} — {self._current_path.name}")
        else:
            self.setWindowTitle(base)
