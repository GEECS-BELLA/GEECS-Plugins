"""Library analyzer editor panel for the Scan Configuration GUI.

Provides a form-based editor for library analyzer YAML files
(``library/analyzers/*.yaml``).  The form is dynamically built from the
Pydantic models :class:`Array2DAnalyzerConfig` and
:class:`Array1DAnalyzerConfig` defined in
:mod:`scan_analysis.config.analyzer_config_models`.

Fields are grouped into collapsible sections:

* **General** — id, device_name, type, priority, analysis_mode, is_active
* **Image Analyzer** — analyzer_class, camera/line config name, kwargs
* **Output** — file_tail, flag_save_images/flag_save_data, gdoc_slot
* **Rendering** — renderer_kwargs

This module reuses :mod:`field_widgets` for widget creation and
:mod:`section_widget` for collapsible section rendering.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: JSON dict widget (inline, lightweight alternative to DictFieldWidget)
# ---------------------------------------------------------------------------


class _JsonDictWidget(QWidget):
    """Inline widget for editing ``Dict[str, Any]`` values as JSON text.

    Displays a ``QPlainTextEdit`` for entering JSON.  On every change
    the text is validated and an error label is shown when the JSON is
    malformed.

    Parameters
    ----------
    label_text : str
        Label displayed to the left of the text area.
    parent : QWidget, optional
        Parent Qt widget.

    Signals
    -------
    valueChanged
        Emitted whenever the text content changes.
    """

    valueChanged = pyqtSignal()

    def __init__(
        self,
        label_text: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        if label_text:
            lbl = QLabel(label_text)
            layout.addWidget(lbl)

        self._text_edit = QPlainTextEdit()
        self._text_edit.setMaximumHeight(90)
        layout.addWidget(self._text_edit)

        self._error_label = QLabel()
        self._error_label.setStyleSheet("color: red; font-size: 11px;")
        self._error_label.setWordWrap(True)
        self._error_label.hide()
        layout.addWidget(self._error_label)

        self._text_edit.textChanged.connect(self._on_text_changed)

    # -- internal -----------------------------------------------------------

    def _on_text_changed(self) -> None:
        """Validate JSON on every keystroke and emit valueChanged."""
        text = self._text_edit.toPlainText().strip()
        if text:
            try:
                json.loads(text)
                self._error_label.hide()
            except json.JSONDecodeError as exc:
                self._error_label.setText(f"Invalid JSON: {exc}")
                self._error_label.show()
        else:
            self._error_label.hide()
        self.valueChanged.emit()

    # -- public API ---------------------------------------------------------

    def get_value(self) -> Optional[Dict[str, Any]]:
        """Parse and return the JSON text as a dictionary.

        Returns
        -------
        dict or None
            Parsed dictionary, or ``None`` when the text is empty or
            cannot be parsed as a JSON object.
        """
        text = self._text_edit.toPlainText().strip()
        if not text:
            return None
        try:
            result = json.loads(text)
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            return None

    def set_value(self, value: Any) -> None:
        """Set the text from a dictionary or other value.

        Parameters
        ----------
        value : Any
            A dictionary to format as JSON, or ``None`` to clear.
        """
        if value is None or (isinstance(value, dict) and not value):
            self._text_edit.setPlainText("{}")
        elif isinstance(value, dict):
            self._text_edit.setPlainText(json.dumps(value, indent=2))
        else:
            self._text_edit.setPlainText(str(value))


# ---------------------------------------------------------------------------
# Helper: collapsible section frame
# ---------------------------------------------------------------------------


class _CollapsibleSection(QWidget):
    """Lightweight collapsible section with a toggle header.

    Parameters
    ----------
    title : str
        Section title displayed in the header bar.
    parent : QWidget, optional
        Parent Qt widget.
    """

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        frame = QFrame(self)
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Raised)
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)
        outer.addWidget(frame)

        # Header
        header = QFrame(frame)
        header.setObjectName("sectionHeader")
        header.setStyleSheet("QFrame#sectionHeader { background-color: #d8d8d8; }")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(4, 2, 4, 2)
        header_layout.setSpacing(4)

        self._toggle_btn = QToolButton(header)
        self._toggle_btn.setText("\u25b6")
        self._toggle_btn.setToolTip("Expand section")
        self._toggle_btn.setStyleSheet("QToolButton { border: none; font-size: 10px; }")
        self._toggle_btn.clicked.connect(self._on_toggle)
        header_layout.addWidget(self._toggle_btn)

        title_label = QLabel(title, header)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        frame_layout.addWidget(header)

        # Content area
        self._content = QWidget(frame)
        self._content_layout = QFormLayout(self._content)
        self._content_layout.setContentsMargins(8, 4, 8, 4)
        frame_layout.addWidget(self._content)

        # Start collapsed
        self._content.setVisible(False)

    @property
    def content_layout(self) -> QFormLayout:
        """Return the form layout inside the collapsible content area."""
        return self._content_layout

    def _on_toggle(self) -> None:
        """Toggle content visibility."""
        visible = not self._content.isVisible()
        self._content.setVisible(visible)
        self._toggle_btn.setText("\u25bc" if visible else "\u25b6")
        self._toggle_btn.setToolTip("Collapse section" if visible else "Expand section")


# ---------------------------------------------------------------------------
# Main editor panel
# ---------------------------------------------------------------------------


class ScanAnalyzerEditorPanel(QWidget):
    """Form-based editor for library analyzer YAML files.

    Dynamically builds form fields from the Pydantic
    ``Array2DAnalyzerConfig`` / ``Array1DAnalyzerConfig`` models and
    groups them into collapsible sections.

    Parameters
    ----------
    parent : QWidget, optional
        Parent Qt widget.

    Signals
    -------
    config_changed
        Emitted whenever any field value changes (for live YAML preview).
    """

    config_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._is_new: bool = True
        self._current_type: str = "array2d"

        # Widget references (populated by _build_form)
        self._type_combo: Optional[QComboBox] = None
        self._id_edit: Optional[QLineEdit] = None
        self._device_name_edit: Optional[QLineEdit] = None
        self._priority_spin: Optional[QSpinBox] = None
        self._analysis_mode_combo: Optional[QComboBox] = None
        self._is_active_cb: Optional[QCheckBox] = None

        # Image Analyzer section widgets
        self._analyzer_class_edit: Optional[QLineEdit] = None
        self._config_name_edit: Optional[QLineEdit] = None
        self._config_name_label: Optional[QLabel] = None
        self._ia_kwargs_widget: Optional[_JsonDictWidget] = None

        # Output section widgets
        self._file_tail_edit: Optional[QLineEdit] = None
        self._flag_save_cb: Optional[QCheckBox] = None
        self._flag_save_label: Optional[QLabel] = None
        self._gdoc_slot_spin: Optional[QSpinBox] = None
        self._gdoc_slot_cb: Optional[QCheckBox] = None

        # Rendering section widgets
        self._renderer_kwargs_widget: Optional[_JsonDictWidget] = None

        # Top-level kwargs section widgets
        self._kwargs_widget: Optional[_JsonDictWidget] = None

        # Section containers
        self._general_section: Optional[_CollapsibleSection] = None
        self._image_analyzer_section: Optional[_CollapsibleSection] = None
        self._output_section: Optional[_CollapsibleSection] = None
        self._rendering_section: Optional[_CollapsibleSection] = None
        self._kwargs_section: Optional[_CollapsibleSection] = None

        self._setup_ui()
        self._build_form()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build the outer scroll area."""
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        outer_layout.addWidget(self._scroll_area)

        self._scroll_content = QWidget()
        self._content_layout = QVBoxLayout(self._scroll_content)
        self._content_layout.setContentsMargins(8, 8, 8, 8)
        self._scroll_area.setWidget(self._scroll_content)

    def _build_form(self) -> None:
        """Build all form sections and populate with defaults."""
        self._build_type_selector()
        self._build_general_section()
        self._build_image_analyzer_section()
        self._build_output_section()
        self._build_rendering_section()
        self._build_kwargs_section()
        self._content_layout.addStretch()

    def _build_type_selector(self) -> None:
        """Build the type selector combo box at the top of the form."""
        type_group = QGroupBox("Analyzer Type")
        type_layout = QHBoxLayout(type_group)

        type_label = QLabel("Type:")
        self._type_combo = QComboBox()
        self._type_combo.addItems(["array2d", "array1d"])
        self._type_combo.currentTextChanged.connect(self._on_type_changed)

        type_layout.addWidget(type_label)
        type_layout.addWidget(self._type_combo, stretch=1)
        self._content_layout.addWidget(type_group)

    def _build_general_section(self) -> None:
        """Build the General section with id, device_name, priority, etc."""
        self._general_section = _CollapsibleSection("General")
        form = self._general_section.content_layout

        # id
        self._id_edit = QLineEdit()
        self._id_edit.setPlaceholderText("Unique analyzer identifier")
        self._id_edit.textChanged.connect(self._emit_changed)
        form.addRow("id:", self._id_edit)

        # device_name
        self._device_name_edit = QLineEdit()
        self._device_name_edit.setPlaceholderText("Device name (e.g. UC_GaiaMode)")
        self._device_name_edit.textChanged.connect(self._emit_changed)
        form.addRow("device_name:", self._device_name_edit)

        # priority
        self._priority_spin = QSpinBox()
        self._priority_spin.setRange(0, 999999)
        self._priority_spin.setValue(100)
        self._priority_spin.valueChanged.connect(self._emit_changed)
        form.addRow("priority:", self._priority_spin)

        # analysis_mode
        self._analysis_mode_combo = QComboBox()
        self._analysis_mode_combo.addItems(["per_shot", "per_bin"])
        self._analysis_mode_combo.currentTextChanged.connect(self._emit_changed)
        form.addRow("analysis_mode:", self._analysis_mode_combo)

        # is_active
        self._is_active_cb = QCheckBox("Enabled")
        self._is_active_cb.setChecked(True)
        self._is_active_cb.stateChanged.connect(self._emit_changed)
        form.addRow("is_active:", self._is_active_cb)

        self._content_layout.addWidget(self._general_section)

    def _build_image_analyzer_section(self) -> None:
        """Build the Image Analyzer collapsible section."""
        self._image_analyzer_section = _CollapsibleSection("Image Analyzer")
        form = self._image_analyzer_section.content_layout

        # analyzer_class
        self._analyzer_class_edit = QLineEdit()
        self._analyzer_class_edit.setPlaceholderText(
            "e.g. image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer"
        )
        self._analyzer_class_edit.textChanged.connect(self._emit_changed)
        form.addRow("analyzer_class:", self._analyzer_class_edit)

        # camera_config_name / line_config_name (label changes with type)
        self._config_name_label = QLabel("camera_config_name:")
        self._config_name_edit = QLineEdit()
        self._config_name_edit.setPlaceholderText("Config name (optional)")
        self._config_name_edit.textChanged.connect(self._emit_changed)
        form.addRow(self._config_name_label, self._config_name_edit)

        # image_analyzer kwargs
        self._ia_kwargs_widget = _JsonDictWidget("kwargs:")
        self._ia_kwargs_widget.valueChanged.connect(self._emit_changed)
        form.addRow(self._ia_kwargs_widget)

        self._content_layout.addWidget(self._image_analyzer_section)

    def _build_output_section(self) -> None:
        """Build the Output collapsible section."""
        self._output_section = _CollapsibleSection("Output")
        form = self._output_section.content_layout

        # file_tail
        self._file_tail_edit = QLineEdit()
        self._file_tail_edit.setPlaceholderText(
            "File suffix (e.g. .png). Leave empty for default."
        )
        self._file_tail_edit.textChanged.connect(self._emit_changed)
        form.addRow("file_tail:", self._file_tail_edit)

        # flag_save_images / flag_save_data (label changes with type)
        self._flag_save_label = QLabel("flag_save_images:")
        self._flag_save_cb = QCheckBox("Save")
        self._flag_save_cb.setChecked(True)
        self._flag_save_cb.stateChanged.connect(self._emit_changed)
        form.addRow(self._flag_save_label, self._flag_save_cb)

        # gdoc_slot (Optional[int]) — spinbox + "None" checkbox
        gdoc_row = QWidget()
        gdoc_layout = QHBoxLayout(gdoc_row)
        gdoc_layout.setContentsMargins(0, 0, 0, 0)

        self._gdoc_slot_cb = QCheckBox("Set")
        self._gdoc_slot_spin = QSpinBox()
        self._gdoc_slot_spin.setRange(0, 3)
        self._gdoc_slot_spin.setEnabled(False)

        self._gdoc_slot_cb.stateChanged.connect(self._on_gdoc_toggled)
        self._gdoc_slot_cb.stateChanged.connect(self._emit_changed)
        self._gdoc_slot_spin.valueChanged.connect(self._emit_changed)

        gdoc_layout.addWidget(self._gdoc_slot_cb)
        gdoc_layout.addWidget(self._gdoc_slot_spin, stretch=1)
        form.addRow("gdoc_slot:", gdoc_row)

        self._content_layout.addWidget(self._output_section)

    def _build_rendering_section(self) -> None:
        """Build the Rendering collapsible section."""
        self._rendering_section = _CollapsibleSection("Rendering")
        form = self._rendering_section.content_layout

        self._renderer_kwargs_widget = _JsonDictWidget("renderer_kwargs:")
        self._renderer_kwargs_widget.valueChanged.connect(self._emit_changed)
        form.addRow(self._renderer_kwargs_widget)

        self._content_layout.addWidget(self._rendering_section)

    def _build_kwargs_section(self) -> None:
        """Build the top-level kwargs collapsible section."""
        self._kwargs_section = _CollapsibleSection("Advanced (kwargs)")
        form = self._kwargs_section.content_layout

        self._kwargs_widget = _JsonDictWidget("kwargs:")
        self._kwargs_widget.valueChanged.connect(self._emit_changed)
        form.addRow(self._kwargs_widget)

        self._content_layout.addWidget(self._kwargs_section)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _emit_changed(self, *_args: object) -> None:
        """Re-emit as the public config_changed signal."""
        self.config_changed.emit()

    def _on_type_changed(self, new_type: str) -> None:
        """Handle type selector change between array2d and array1d.

        Updates labels that differ between the two types:

        * ``flag_save_images`` ↔ ``flag_save_data``
        * ``camera_config_name`` ↔ ``line_config_name``

        Parameters
        ----------
        new_type : str
            The newly selected type (``"array2d"`` or ``"array1d"``).
        """
        self._current_type = new_type
        if new_type == "array2d":
            if self._flag_save_label is not None:
                self._flag_save_label.setText("flag_save_images:")
            if self._config_name_label is not None:
                self._config_name_label.setText("camera_config_name:")
        else:
            if self._flag_save_label is not None:
                self._flag_save_label.setText("flag_save_data:")
            if self._config_name_label is not None:
                self._config_name_label.setText("line_config_name:")
        self._emit_changed()

    def _on_gdoc_toggled(self, state: int) -> None:
        """Enable/disable the gdoc_slot spinbox based on checkbox.

        Parameters
        ----------
        state : int
            Qt checkbox state.
        """
        enabled = bool(state)
        if self._gdoc_slot_spin is not None:
            self._gdoc_slot_spin.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_config(self, data: dict) -> None:
        """Populate the form from a raw YAML dict.

        Handles the ``LibraryAnalyzer`` wrapping convention where the
        YAML is flat (``id`` alongside analyzer fields) but the model
        wraps it under ``analyzer``.

        Parameters
        ----------
        data : dict
            Raw dictionary as returned by
            :func:`~ConfigFileGUI.scan_config_io.load_analyzer_yaml`.
        """
        self._is_new = False

        # --- Type ---
        analyzer_type = data.get("type", "array2d")
        if self._type_combo is not None:
            idx = self._type_combo.findText(analyzer_type)
            if idx >= 0:
                self._type_combo.setCurrentIndex(idx)
        self._current_type = analyzer_type

        # --- General ---
        if self._id_edit is not None:
            self._id_edit.setText(str(data.get("id", "")))
            self._id_edit.setReadOnly(True)
            self._id_edit.setStyleSheet("background-color: #f0f0f0;")

        if self._device_name_edit is not None:
            self._device_name_edit.setText(str(data.get("device_name", "")))

        if self._priority_spin is not None:
            self._priority_spin.setValue(int(data.get("priority", 100)))

        if self._analysis_mode_combo is not None:
            mode = data.get("analysis_mode", "per_shot")
            idx = self._analysis_mode_combo.findText(str(mode))
            if idx >= 0:
                self._analysis_mode_combo.setCurrentIndex(idx)

        if self._is_active_cb is not None:
            self._is_active_cb.setChecked(bool(data.get("is_active", True)))

        # --- Image Analyzer ---
        ia_data = data.get("image_analyzer", {}) or {}
        if self._analyzer_class_edit is not None:
            self._analyzer_class_edit.setText(str(ia_data.get("analyzer_class", "")))

        # camera_config_name or line_config_name
        if self._config_name_edit is not None:
            if analyzer_type == "array2d":
                config_name = ia_data.get("camera_config_name", "")
                # Also check inside ia kwargs for legacy format
                if not config_name:
                    ia_kwargs = ia_data.get("kwargs", {}) or {}
                    config_name = ia_kwargs.get("camera_config_name", "")
            else:
                config_name = ia_data.get("line_config_name", "")
                if not config_name:
                    ia_kwargs = ia_data.get("kwargs", {}) or {}
                    config_name = ia_kwargs.get("line_config_name", "")
            self._config_name_edit.setText(str(config_name or ""))

        if self._ia_kwargs_widget is not None:
            ia_kwargs = ia_data.get("kwargs", {}) or {}
            self._ia_kwargs_widget.set_value(ia_kwargs)

        # --- Output ---
        if self._file_tail_edit is not None:
            self._file_tail_edit.setText(str(data.get("file_tail", "") or ""))

        if self._flag_save_cb is not None:
            if analyzer_type == "array2d":
                self._flag_save_cb.setChecked(bool(data.get("flag_save_images", True)))
            else:
                self._flag_save_cb.setChecked(bool(data.get("flag_save_data", True)))

        gdoc_slot = data.get("gdoc_slot")
        if self._gdoc_slot_cb is not None and self._gdoc_slot_spin is not None:
            if gdoc_slot is not None:
                self._gdoc_slot_cb.setChecked(True)
                self._gdoc_slot_spin.setEnabled(True)
                self._gdoc_slot_spin.setValue(int(gdoc_slot))
            else:
                self._gdoc_slot_cb.setChecked(False)
                self._gdoc_slot_spin.setEnabled(False)

        # --- Rendering ---
        if self._renderer_kwargs_widget is not None:
            self._renderer_kwargs_widget.set_value(
                data.get("renderer_kwargs", {}) or {}
            )

        # --- Top-level kwargs ---
        if self._kwargs_widget is not None:
            self._kwargs_widget.set_value(data.get("kwargs", {}) or {})

    def get_config_dict(self) -> Dict[str, Any]:
        """Collect all form values into a flat dict for YAML saving.

        The output is the flat format expected by the YAML files (not
        wrapped under ``analyzer``), suitable for passing to
        :func:`~ConfigFileGUI.scan_config_io.save_analyzer_yaml`.

        Returns
        -------
        dict
            Analyzer configuration dictionary.
        """
        result: Dict[str, Any] = {}

        # id
        if self._id_edit is not None:
            result["id"] = self._id_edit.text().strip()

        # type
        analyzer_type = self._current_type
        result["type"] = analyzer_type

        # device_name
        if self._device_name_edit is not None:
            result["device_name"] = self._device_name_edit.text().strip()

        # priority
        if self._priority_spin is not None:
            result["priority"] = self._priority_spin.value()

        # analysis_mode
        if self._analysis_mode_combo is not None:
            result["analysis_mode"] = self._analysis_mode_combo.currentText()

        # file_tail
        if self._file_tail_edit is not None:
            ft = self._file_tail_edit.text().strip()
            if ft:
                result["file_tail"] = ft

        # flag_save_images / flag_save_data
        if self._flag_save_cb is not None:
            if analyzer_type == "array2d":
                result["flag_save_images"] = self._flag_save_cb.isChecked()
            else:
                result["flag_save_data"] = self._flag_save_cb.isChecked()

        # renderer_kwargs
        if self._renderer_kwargs_widget is not None:
            rk = self._renderer_kwargs_widget.get_value()
            if rk:
                result["renderer_kwargs"] = rk

        # image_analyzer
        ia: Dict[str, Any] = {}
        if self._analyzer_class_edit is not None:
            ia["analyzer_class"] = self._analyzer_class_edit.text().strip()

        if self._config_name_edit is not None:
            cn = self._config_name_edit.text().strip()
            if cn:
                if analyzer_type == "array2d":
                    ia["camera_config_name"] = cn
                else:
                    ia["line_config_name"] = cn

        if self._ia_kwargs_widget is not None:
            ia_kw = self._ia_kwargs_widget.get_value()
            if ia_kw:
                ia["kwargs"] = ia_kw

        result["image_analyzer"] = ia

        # kwargs (top-level)
        if self._kwargs_widget is not None:
            kw = self._kwargs_widget.get_value()
            if kw:
                result["kwargs"] = kw
            else:
                result["kwargs"] = {}

        # is_active
        if self._is_active_cb is not None:
            result["is_active"] = self._is_active_cb.isChecked()

        # gdoc_slot
        if self._gdoc_slot_cb is not None and self._gdoc_slot_spin is not None:
            if self._gdoc_slot_cb.isChecked():
                result["gdoc_slot"] = self._gdoc_slot_spin.value()

        return result

    def set_new_mode(self, is_new: bool) -> None:
        """Set whether the editor is in new-config or edit-existing mode.

        When *is_new* is ``False``, the ``id`` field becomes read-only.

        Parameters
        ----------
        is_new : bool
            ``True`` for a brand-new config, ``False`` for editing an
            existing file.
        """
        self._is_new = is_new
        if self._id_edit is not None:
            self._id_edit.setReadOnly(not is_new)
            if is_new:
                self._id_edit.setStyleSheet("")
            else:
                self._id_edit.setStyleSheet("background-color: #f0f0f0;")

    def clear(self) -> None:
        """Reset all fields to their default values."""
        self._is_new = True

        if self._type_combo is not None:
            self._type_combo.setCurrentIndex(0)

        if self._id_edit is not None:
            self._id_edit.clear()
            self._id_edit.setReadOnly(False)
            self._id_edit.setStyleSheet("")

        if self._device_name_edit is not None:
            self._device_name_edit.clear()

        if self._priority_spin is not None:
            self._priority_spin.setValue(100)

        if self._analysis_mode_combo is not None:
            self._analysis_mode_combo.setCurrentIndex(0)

        if self._is_active_cb is not None:
            self._is_active_cb.setChecked(True)

        if self._analyzer_class_edit is not None:
            self._analyzer_class_edit.clear()

        if self._config_name_edit is not None:
            self._config_name_edit.clear()

        if self._ia_kwargs_widget is not None:
            self._ia_kwargs_widget.set_value({})

        if self._file_tail_edit is not None:
            self._file_tail_edit.clear()

        if self._flag_save_cb is not None:
            self._flag_save_cb.setChecked(True)

        if self._gdoc_slot_cb is not None:
            self._gdoc_slot_cb.setChecked(False)

        if self._gdoc_slot_spin is not None:
            self._gdoc_slot_spin.setValue(0)
            self._gdoc_slot_spin.setEnabled(False)

        if self._renderer_kwargs_widget is not None:
            self._renderer_kwargs_widget.set_value({})

        if self._kwargs_widget is not None:
            self._kwargs_widget.set_value({})
