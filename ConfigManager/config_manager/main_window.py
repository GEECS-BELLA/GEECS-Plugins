"""Main application window for the GEECS Config Manager."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Type

import yaml
from pydantic import BaseModel, ValidationError
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from image_analysis.processing.array2d.config_models import CameraConfig
from image_analysis.processing.array1d.config_models import Line1DConfig
from scan_analysis.config.analyzer_config_models import (
    Array1DAnalyzerConfig,
    Array2DAnalyzerConfig,
    ExperimentAnalysisConfig,
)

from .widgets import ModelFormWidget


# ---------------------------------------------------------------------------
# Special widget: Analyzers list (handles Union[Array2D, Array1D])
# ---------------------------------------------------------------------------

_ANALYZER_TYPES: Dict[str, Type[BaseModel]] = {
    "Array 2D (image)": Array2DAnalyzerConfig,
    "Array 1D (line/spectrum)": Array1DAnalyzerConfig,
}


class AnalyzerItemWidget(QWidget):
    """One analyzer entry: type selector + dynamic ModelFormWidget."""

    def __init__(self, data: Optional[dict] = None, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        self._type_combo = QComboBox()
        for label in _ANALYZER_TYPES:
            self._type_combo.addItem(label)
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        type_row.addWidget(self._type_combo)
        type_row.addStretch()
        layout.addLayout(type_row)

        self._form_container = QVBoxLayout()
        layout.addLayout(self._form_container)
        self._current_form: Optional[ModelFormWidget] = None

        # Determine initial type from data
        if data and data.get("type") == "array1d":
            self._type_combo.setCurrentIndex(1)
        else:
            self._type_combo.setCurrentIndex(0)

        self._rebuild_form()

        if data:
            self.set_value(data)

    def _on_type_changed(self) -> None:
        self._rebuild_form()

    def _rebuild_form(self) -> None:
        # Remove existing form
        while self._form_container.count():
            item = self._form_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._current_form = None

        label = self._type_combo.currentText()
        model_class = _ANALYZER_TYPES[label]
        self._current_form = ModelFormWidget(model_class)
        self._form_container.addWidget(self._current_form)

    def get_value(self) -> dict:
        """Return the current form values as a dict."""
        if self._current_form is None:
            return {}
        return self._current_form.get_value()

    def set_value(self, data: dict) -> None:
        """Populate the form from *data*."""
        if self._current_form:
            self._current_form.set_value(data)


class AnalyzersWidget(QWidget):
    """Add/remove list of AnalyzerItemWidgets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add Analyzer")
        add_btn.clicked.connect(lambda: self._add_item())
        btn_row.addWidget(add_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._items_layout = QVBoxLayout()
        layout.addLayout(self._items_layout)
        self._items: List[tuple] = []  # (QGroupBox, AnalyzerItemWidget)

    def _add_item(self, data: Optional[dict] = None) -> None:
        idx = len(self._items) + 1
        box = QGroupBox(f"Analyzer {idx}")
        box_layout = QVBoxLayout(box)

        item_widget = AnalyzerItemWidget(data)
        box_layout.addWidget(item_widget)

        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda: self._remove_item(box, item_widget))
        box_layout.addWidget(remove_btn)

        self._items_layout.addWidget(box)
        self._items.append((box, item_widget))

    def _remove_item(self, box: QGroupBox, item: AnalyzerItemWidget) -> None:
        self._items = [(b, w) for b, w in self._items if w is not item]
        box.setParent(None)

    def get_value(self) -> list:
        """Return all analyzer configs as a list of dicts."""
        return [w.get_value() for _, w in self._items]

    def set_value(self, data: list) -> None:
        """Populate the analyzer list from *data*."""
        for box, _ in list(self._items):
            box.setParent(None)
        self._items.clear()
        for entry in data:
            self._add_item(entry)


# ---------------------------------------------------------------------------
# Scrollable form tab
# ---------------------------------------------------------------------------


class _ScrollableForm(QScrollArea):
    """QScrollArea wrapping a ModelFormWidget."""

    def __init__(
        self,
        model_class: Type[BaseModel],
        extra_widgets: Optional[Dict[str, QWidget]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWidgetResizable(True)

        self._model_class = model_class
        self._extra = extra_widgets or {}

        inner = QWidget()
        self._layout = QVBoxLayout(inner)
        self._layout.setAlignment(Qt.AlignTop)

        # Build form, injecting extra_widgets as overrides
        self._form = ModelFormWidget(model_class)

        # Swap out extra widgets after construction
        for field_name, replacement in self._extra.items():
            if field_name in self._form._field_widgets:
                self._form._field_widgets[field_name] = replacement

        self._layout.addWidget(self._form)
        self.setWidget(inner)

    @property
    def form(self) -> ModelFormWidget:
        return self._form

    def get_value(self) -> dict:
        result = self._form.get_value()
        # Override with extra widgets
        for name, widget in self._extra.items():
            result[name] = widget.get_value()
        return result

    def set_value(self, data: dict) -> None:
        self._form.set_value(data)
        for name, widget in self._extra.items():
            if name in data:
                widget.set_value(data[name])

    def validate(self) -> Optional[str]:
        try:
            self._model_class.model_validate(self.get_value())
            return None
        except ValidationError as e:
            return str(e)


# ---------------------------------------------------------------------------
# Scan tab (ExperimentAnalysisConfig with custom analyzers widget)
# ---------------------------------------------------------------------------


class ScanConfigTab(QWidget):
    """Custom tab for ExperimentAnalysisConfig with AnalyzersWidget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self._analyzers_widget = AnalyzersWidget()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setAlignment(Qt.AlignTop)

        # Render all ExperimentAnalysisConfig fields except 'analyzers'
        self._form = ModelFormWidget(ExperimentAnalysisConfig)
        # Remove the analyzers widget that was auto-built (it uses DICT fallback)
        if "analyzers" in self._form._field_widgets:
            old_widget = self._form._field_widgets.pop("analyzers")
            old_widget.setParent(None)

        inner_layout.addWidget(self._form)

        # Add our custom analyzers section
        analyzers_box = QGroupBox("Analyzers")
        ab_layout = QVBoxLayout(analyzers_box)
        ab_layout.addWidget(self._analyzers_widget)
        inner_layout.addWidget(analyzers_box)

        scroll.setWidget(inner)
        layout.addWidget(scroll)

    def get_value(self) -> dict:
        """Return combined experiment + analyzers config as a dict."""
        result = self._form.get_value()
        result["analyzers"] = self._analyzers_widget.get_value()
        return result

    def set_value(self, data: dict) -> None:
        """Populate the scan config form from *data*."""
        self._form.set_value(data)
        if "analyzers" in data:
            self._analyzers_widget.set_value(data["analyzers"])

    def validate(self) -> Optional[str]:
        """Run Pydantic validation; return error string or None."""
        try:
            ExperimentAnalysisConfig.model_validate(self.get_value())
            return None
        except ValidationError as e:
            return str(e)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """Top-level Config Manager window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GEECS Config Manager")
        self.resize(900, 750)

        self._current_path: Optional[Path] = None

        self._build_toolbar()
        self._build_tabs()
        self._build_status_bar()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        for label, slot in [
            ("New", self._on_new),
            ("Open…", self._on_open),
            ("Save", self._on_save),
            ("Save As…", self._on_save_as),
            ("Validate", self._on_validate),
        ]:
            act = QAction(label, self)
            act.triggered.connect(slot)
            tb.addAction(act)

        tb.addSeparator()
        self._path_label = QLabel("  No file open")
        self._path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(self._path_label)

    def _build_tabs(self) -> None:
        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        # --- Image 2D tab ---
        self._image2d_scroll = _ScrollableForm(CameraConfig)
        self._tabs.addTab(self._image2d_scroll, "Image 2D (CameraConfig)")

        # --- Image 1D tab ---
        self._image1d_scroll = _ScrollableForm(Line1DConfig)
        self._tabs.addTab(self._image1d_scroll, "Image 1D (Line1DConfig)")

        # --- Scan tab ---
        self._scan_tab = ScanConfigTab()
        self._tabs.addTab(self._scan_tab, "Scan (ExperimentAnalysisConfig)")

    def _build_status_bar(self) -> None:
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

    # ------------------------------------------------------------------
    # Active tab helpers
    # ------------------------------------------------------------------

    def _active_tab(self):
        idx = self._tabs.currentIndex()
        if idx == 0:
            return self._image2d_scroll, CameraConfig
        elif idx == 1:
            return self._image1d_scroll, Line1DConfig
        else:
            return self._scan_tab, ExperimentAnalysisConfig

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_new(self) -> None:
        tab, _ = self._active_tab()
        # Re-build tab by resetting its form
        # Simplest: just clear text fields and uncheck optionals
        reply = QMessageBox.question(
            self,
            "New Config",
            "Clear the current form? Unsaved changes will be lost.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._current_path = None
            self._path_label.setText("  No file open")
            # Easiest reset: reload fresh widget into tab
            idx = self._tabs.currentIndex()
            if idx == 0:
                self._image2d_scroll = _ScrollableForm(CameraConfig)
                self._tabs.removeTab(0)
                self._tabs.insertTab(0, self._image2d_scroll, "Image 2D (CameraConfig)")
                self._tabs.setCurrentIndex(0)
            elif idx == 1:
                self._image1d_scroll = _ScrollableForm(Line1DConfig)
                self._tabs.removeTab(1)
                self._tabs.insertTab(1, self._image1d_scroll, "Image 1D (Line1DConfig)")
                self._tabs.setCurrentIndex(1)
            else:
                self._scan_tab = ScanConfigTab()
                self._tabs.removeTab(2)
                self._tabs.insertTab(
                    2, self._scan_tab, "Scan (ExperimentAnalysisConfig)"
                )
                self._tabs.setCurrentIndex(2)
            self._status.showMessage("New config created.")

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Config YAML", "", "YAML files (*.yaml *.yml);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise ValueError("YAML root must be a mapping.")

            # Auto-detect config type
            tab_idx = self._detect_config_type(data)
            self._tabs.setCurrentIndex(tab_idx)
            tab, _ = self._active_tab()
            tab.set_value(data)

            self._current_path = Path(path)
            self._path_label.setText(f"  {path}")
            self._status.showMessage(f"Opened: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Open failed", str(e))

    def _detect_config_type(self, data: dict) -> int:
        """Return tab index 0/1/2 based on YAML content."""
        if "data_loading" in data:
            return 1  # Line1DConfig
        if "analyzers" in data or "experiment" in data:
            return 2  # ExperimentAnalysisConfig
        return 0  # CameraConfig

    def _on_save(self) -> None:
        if self._current_path is None:
            self._on_save_as()
        else:
            self._save_to(self._current_path)

    def _on_save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Config YAML", "", "YAML files (*.yaml *.yml);;All files (*)"
        )
        if path:
            p = Path(path)
            if p.suffix not in (".yaml", ".yml"):
                p = p.with_suffix(".yaml")
            self._save_to(p)

    def _save_to(self, path: Path) -> None:
        tab, model_class = self._active_tab()
        data = tab.get_value()
        err = tab.validate()
        if err:
            reply = QMessageBox.question(
                self,
                "Validation errors",
                f"Config has validation errors. Save anyway?\n\n{err[:500]}",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        try:
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            self._current_path = path
            self._path_label.setText(f"  {path}")
            self._status.showMessage(f"Saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def _on_validate(self) -> None:
        tab, _ = self._active_tab()
        err = tab.validate()
        if err:
            self._status.showMessage(f"Validation errors: {err[:200]}")
            QMessageBox.warning(self, "Validation errors", err)
        else:
            self._status.showMessage("Valid!")
