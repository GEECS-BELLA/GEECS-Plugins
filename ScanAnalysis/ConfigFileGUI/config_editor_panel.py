"""Main editor panel for the Config File Editor GUI.

Builds a scrollable form from a loaded Pydantic config model
(``CameraConfig`` or ``Line1DConfig``), with section widgets for each
sub-config and a pipeline widget at the bottom.

This module is consumed by ``config_editor_window.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .field_widgets import DictFieldWidget
from .pipeline_widget import PipelineWidget
from .section_widget import SectionWidget

logger = logging.getLogger(__name__)

# Lazy imports for config models — done inside methods to avoid
# import-time failures when image_analysis is not installed.

# 2D section order (matches CameraConfig field order)
_2D_SECTIONS = [
    "background",
    "roi",
    "crosshair_masking",
    "circular_mask",
    "vignette",
    "thresholding",
    "filtering",
    "normalization",
    "transforms",
]

# 1D section order (matches Line1DConfig field order)
_1D_SECTIONS = [
    "data_loading",
    "background",
    "roi",
    "filtering",
    "thresholding",
    "interpolation",
]


class ConfigEditorPanel(QWidget):
    """Scrollable editor panel that builds a form from a Pydantic config.

    Dynamically creates :class:`SectionWidget` instances for each
    sub-config and a :class:`PipelineWidget` for step ordering.

    Parameters
    ----------
    parent : QWidget, optional
        Parent Qt widget.

    Signals
    -------
    valueChanged()
        Emitted when any field value changes.
    dirtyStateChanged(bool)
        Emitted when the dirty state changes.
    """

    valueChanged = pyqtSignal()
    dirtyStateChanged = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config_type: str = ""
        self._file_path: Optional[Path] = None
        self._sections: Dict[str, SectionWidget] = {}
        self._pipeline: Optional[PipelineWidget] = None
        self._dirty: bool = False
        self._top_widgets: Dict[str, QWidget] = {}
        self._analysis_widget: Optional[DictFieldWidget] = None
        self._metadata_widget: Optional[DictFieldWidget] = None

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build the scroll area layout."""
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        outer_layout.addWidget(self._scroll_area)

        self._scroll_content = QWidget()
        self._content_layout = QVBoxLayout(self._scroll_content)
        self._content_layout.setContentsMargins(8, 8, 8, 8)
        self._scroll_area.setWidget(self._scroll_content)

        # Placeholder label shown when no config is loaded
        self._placeholder = QLabel(
            "Select a configuration file from the list to begin editing."
        )
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet("color: gray; font-size: 13px; padding: 20px;")
        self._content_layout.addWidget(self._placeholder)
        self._content_layout.addStretch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_config(self, config: BaseModel, config_type: str, file_path: Path) -> None:
        """Load a Pydantic config model and build the editor form.

        Parameters
        ----------
        config : BaseModel
            A validated Pydantic model (``CameraConfig`` or ``Line1DConfig``).
        config_type : str
            Either ``"camera_2d"`` or ``"line_1d"``.
        file_path : Path
            The file path of the loaded config.
        """
        self._clear()
        self._config_type = config_type
        self._file_path = file_path

        if config_type == "camera_2d":
            self._build_2d_editor(config)
        elif config_type == "line_1d":
            self._build_1d_editor(config)
        else:
            logger.warning("Unknown config type: %s", config_type)
            return

        # Add stretch at the bottom
        self._content_layout.addStretch()

        self._dirty = False
        self.dirtyStateChanged.emit(False)

    def get_config_dict(self) -> Dict[str, Any]:
        """Collect all values from top fields, sections, and pipeline.

        Returns
        -------
        dict
            A dictionary suitable for Pydantic model validation.
        """
        data: Dict[str, Any] = {}

        # Top-level fields
        for name, widget in self._top_widgets.items():
            if isinstance(widget, QLineEdit):
                data[name] = widget.text()
            elif isinstance(widget, QComboBox):
                text = widget.currentText()
                # Try to convert to int for bit_depth
                try:
                    data[name] = int(text)
                except (ValueError, TypeError):
                    data[name] = text

        # Section values
        for name, section in self._sections.items():
            data[name] = section.get_values()

        # Analysis field
        if self._analysis_widget is not None:
            data["analysis"] = self._analysis_widget.get_value()

        # Metadata / extra fields (2D only)
        if self._metadata_widget is not None:
            extra = self._metadata_widget.get_value()
            if extra and isinstance(extra, dict):
                data.update(extra)

        # Pipeline
        if self._pipeline is not None:
            step_order = self._pipeline.get_step_order()
            if self._config_type == "camera_2d":
                data["pipeline"] = {"steps": step_order}
            elif self._config_type == "line_1d":
                data["pipeline"] = {"steps": step_order}

        return data

    def is_dirty(self) -> bool:
        """Return whether any value has changed since last load/save.

        Returns
        -------
        bool
            ``True`` if the editor has unsaved changes.
        """
        return self._dirty

    def mark_clean(self) -> None:
        """Reset the dirty state to clean."""
        if self._dirty:
            self._dirty = False
            self.dirtyStateChanged.emit(False)

    def validate(self) -> List[str]:
        """Validate current values through Pydantic.

        Returns
        -------
        list of str
            Validation error messages. Empty list means valid.
        """
        data = self.get_config_dict()
        errors: List[str] = []

        try:
            if self._config_type == "camera_2d":
                from image_analysis.processing.array2d.config_models import (
                    CameraConfig,
                )

                CameraConfig.model_validate(data)
            elif self._config_type == "line_1d":
                from image_analysis.processing.array1d.config_models import (
                    Line1DConfig,
                )

                Line1DConfig.model_validate(data)
        except ValidationError as exc:
            for err in exc.errors():
                loc_parts = [str(p) for p in err.get("loc", [])]
                field_path = ".".join(loc_parts) if loc_parts else "unknown"
                message = err.get("msg", "Validation error")
                errors.append(f"{field_path}: {message}")
        except Exception as exc:
            errors.append(f"Unexpected validation error: {exc}")

        return errors

    # ------------------------------------------------------------------
    # Internal: build editors
    # ------------------------------------------------------------------

    def _build_2d_editor(self, config) -> None:
        """Build the editor form for a 2D camera config.

        Parameters
        ----------
        config : CameraConfig
            The loaded 2D camera configuration model.
        """
        from image_analysis.processing.array2d.config_models import (
            BackgroundConfig,
            CircularMaskConfig,
            CrosshairMaskingConfig,
            FilteringConfig,
            NormalizationConfig,
            ProcessingStepType,
            ROIConfig,
            ThresholdingConfig,
            TransformConfig,
            VignetteConfig,
        )

        # --- Top fields group ---
        top_group = QGroupBox("General Settings")
        top_layout = QFormLayout(top_group)

        # Name (read-only, derived from filename)
        name_edit = QLineEdit(self._file_path.stem if self._file_path else "")
        name_edit.setReadOnly(True)
        name_edit.setStyleSheet("background-color: #f0f0f0;")
        top_layout.addRow("Name:", name_edit)
        self._top_widgets["name"] = name_edit

        # Description
        desc_edit = QLineEdit(getattr(config, "description", "") or "")
        desc_edit.setPlaceholderText("Configuration description")
        desc_edit.textChanged.connect(self._on_value_changed)
        top_layout.addRow("Description:", desc_edit)
        self._top_widgets["description"] = desc_edit

        # Bit depth
        bit_depth_combo = QComboBox()
        bit_depth_values = ["8", "10", "12", "14", "16", "32"]
        bit_depth_combo.addItems(bit_depth_values)
        current_bd = str(getattr(config, "bit_depth", 16))
        idx = bit_depth_combo.findText(current_bd)
        if idx >= 0:
            bit_depth_combo.setCurrentIndex(idx)
        bit_depth_combo.currentTextChanged.connect(self._on_value_changed)
        top_layout.addRow("Bit Depth:", bit_depth_combo)
        self._top_widgets["bit_depth"] = bit_depth_combo

        self._content_layout.addWidget(top_group)

        # --- Section model mapping ---
        section_models: Dict[str, Type[BaseModel]] = {
            "background": BackgroundConfig,
            "roi": ROIConfig,
            "crosshair_masking": CrosshairMaskingConfig,
            "circular_mask": CircularMaskConfig,
            "vignette": VignetteConfig,
            "thresholding": ThresholdingConfig,
            "filtering": FilteringConfig,
            "normalization": NormalizationConfig,
            "transforms": TransformConfig,
        }

        # --- Build sections ---
        for section_name in _2D_SECTIONS:
            model_class = section_models.get(section_name)
            if model_class is None:
                continue

            section = SectionWidget(section_name, model_class, self)
            field_value = getattr(config, section_name, None)

            if field_value is not None:
                section.set_values(field_value.model_dump())
            else:
                section.setChecked(False)

            section.sectionEnabledChanged.connect(self._on_value_changed)
            section.valueChanged.connect(self._on_value_changed)
            self._sections[section_name] = section
            self._content_layout.addWidget(section)

        # --- Wire dynamic_computation auto-population for background ---
        self._connect_dynamic_computation_defaults()

        # --- Analysis (Dict field) ---
        self._build_analysis_widget(getattr(config, "analysis", None))

        # --- Metadata / Extra Fields ---
        self._build_metadata_widget(config)

        # --- Pipeline widget ---
        self._pipeline = PipelineWidget(ProcessingStepType, self)

        # Determine initially enabled steps from config
        pipeline_config = getattr(config, "pipeline", None)
        if pipeline_config is not None and hasattr(pipeline_config, "steps"):
            steps = pipeline_config.steps
            if steps:
                step_values = [
                    s.value if hasattr(s, "value") else str(s) for s in steps
                ]
                self._pipeline.set_step_order(step_values)
            else:
                self._set_pipeline_from_enabled_sections()
        else:
            self._set_pipeline_from_enabled_sections()

        # Wire section enable/disable to pipeline
        for section_name, section in self._sections.items():
            section.sectionEnabledChanged.connect(
                self._pipeline.on_section_enabled_changed
            )

        self._pipeline.orderChanged.connect(self._on_value_changed)
        self._content_layout.addWidget(self._pipeline)

    def _build_1d_editor(self, config) -> None:
        """Build the editor form for a 1D line config.

        Parameters
        ----------
        config : Line1DConfig
            The loaded 1D line configuration model.
        """
        from image_analysis.processing.array1d.config_models import (
            BackgroundConfig as BG1D,
            Data1DConfig,
            FilteringConfig as Filter1D,
            InterpolationConfig,
            PipelineStepType,
            ROI1DConfig,
            ThresholdingConfig as Thresh1D,
        )

        # --- Top fields group ---
        top_group = QGroupBox("General Settings")
        top_layout = QFormLayout(top_group)

        # Name (read-only)
        name_edit = QLineEdit(self._file_path.stem if self._file_path else "")
        name_edit.setReadOnly(True)
        name_edit.setStyleSheet("background-color: #f0f0f0;")
        top_layout.addRow("Name:", name_edit)
        self._top_widgets["name"] = name_edit

        # Description
        desc_edit = QLineEdit(getattr(config, "description", "") or "")
        desc_edit.setPlaceholderText("Configuration description")
        desc_edit.textChanged.connect(self._on_value_changed)
        top_layout.addRow("Description:", desc_edit)
        self._top_widgets["description"] = desc_edit

        # Data format
        data_format = QLineEdit(str(getattr(config, "data_format", "") or ""))
        data_format.setPlaceholderText("Data format (e.g. csv, tdms)")
        data_format.textChanged.connect(self._on_value_changed)
        top_layout.addRow("Data Format:", data_format)
        self._top_widgets["data_format"] = data_format

        # Processing dtype
        proc_dtype = QLineEdit(str(getattr(config, "processing_dtype", "") or ""))
        proc_dtype.setPlaceholderText("Processing dtype (e.g. float64)")
        proc_dtype.textChanged.connect(self._on_value_changed)
        top_layout.addRow("Processing Dtype:", proc_dtype)
        self._top_widgets["processing_dtype"] = proc_dtype

        # Storage dtype
        stor_dtype = QLineEdit(str(getattr(config, "storage_dtype", "") or ""))
        stor_dtype.setPlaceholderText("Storage dtype (e.g. float32)")
        stor_dtype.textChanged.connect(self._on_value_changed)
        top_layout.addRow("Storage Dtype:", stor_dtype)
        self._top_widgets["storage_dtype"] = stor_dtype

        self._content_layout.addWidget(top_group)

        # --- Section model mapping ---
        section_models: Dict[str, Type[BaseModel]] = {
            "data_loading": Data1DConfig,
            "background": BG1D,
            "roi": ROI1DConfig,
            "filtering": Filter1D,
            "thresholding": Thresh1D,
            "interpolation": InterpolationConfig,
        }

        # --- Build sections ---
        for section_name in _1D_SECTIONS:
            model_class = section_models.get(section_name)
            if model_class is None:
                continue

            section = SectionWidget(section_name, model_class, self)
            field_value = getattr(config, section_name, None)

            if field_value is not None:
                section.set_values(field_value.model_dump())
            else:
                section.setChecked(False)

            section.sectionEnabledChanged.connect(self._on_value_changed)
            section.valueChanged.connect(self._on_value_changed)
            self._sections[section_name] = section
            self._content_layout.addWidget(section)

        # --- Analysis (Dict field) ---
        self._build_analysis_widget(getattr(config, "analysis", None))

        # --- Pipeline widget ---
        self._pipeline = PipelineWidget(PipelineStepType, self)

        # Determine initially enabled steps from config
        pipeline_config = getattr(config, "pipeline", None)
        if pipeline_config is not None and hasattr(pipeline_config, "steps"):
            steps = pipeline_config.steps
            if steps:
                step_values = [
                    s.value if hasattr(s, "value") else str(s) for s in steps
                ]
                self._pipeline.set_step_order(step_values)
            else:
                self._set_pipeline_from_enabled_sections()
        else:
            self._set_pipeline_from_enabled_sections()

        # Wire section enable/disable to pipeline
        for section_name, section in self._sections.items():
            section.sectionEnabledChanged.connect(
                self._pipeline.on_section_enabled_changed
            )

        self._pipeline.orderChanged.connect(self._on_value_changed)
        self._content_layout.addWidget(self._pipeline)

    def _build_analysis_widget(self, analysis_data: Optional[Dict[str, Any]]) -> None:
        """Build the analysis dict widget.

        Parameters
        ----------
        analysis_data : dict or None
            The current analysis configuration data.
        """
        analysis_group = QGroupBox("Analysis")
        analysis_group.setCheckable(True)
        analysis_layout = QVBoxLayout(analysis_group)

        field_info = FieldInfo(description="Analysis configuration (free-form YAML)")
        self._analysis_widget = DictFieldWidget("analysis", field_info, self)

        if analysis_data is not None:
            analysis_group.setChecked(True)
            self._analysis_widget.set_value(analysis_data)
        else:
            analysis_group.setChecked(False)

        self._analysis_widget.valueChanged.connect(self._on_value_changed)
        analysis_layout.addWidget(self._analysis_widget)
        self._content_layout.addWidget(analysis_group)

    def _build_metadata_widget(self, config) -> None:
        """Build the metadata / extra fields widget for 2D configs.

        CameraConfig uses ``extra="allow"``, so any extra keys in the
        YAML are stored as model extras.  This widget lets users edit
        them as raw YAML.

        Parameters
        ----------
        config : CameraConfig
            The loaded camera configuration model.
        """
        # Collect extra fields (those not in the model's declared fields)
        extra_data = {}
        if hasattr(config, "model_extra") and config.model_extra:
            extra_data = dict(config.model_extra)
        elif hasattr(config, "__pydantic_extra__") and config.__pydantic_extra__:
            extra_data = dict(config.__pydantic_extra__)

        metadata_group = QGroupBox("Metadata / Extra Fields")
        metadata_group.setCheckable(True)
        metadata_layout = QVBoxLayout(metadata_group)

        field_info = FieldInfo(
            description="Extra metadata fields not part of the standard schema (YAML)"
        )
        self._metadata_widget = DictFieldWidget("metadata", field_info, self)

        if extra_data:
            metadata_group.setChecked(True)
            self._metadata_widget.set_value(extra_data)
        else:
            metadata_group.setChecked(False)

        self._metadata_widget.valueChanged.connect(self._on_value_changed)
        metadata_layout.addWidget(self._metadata_widget)
        self._content_layout.addWidget(metadata_group)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect_dynamic_computation_defaults(self) -> None:
        """Wire the background dynamic_computation toggle to auto-populate fields.

        When the ``dynamic_computation`` nested section inside the
        background section is toggled **on**, this automatically sets:

        * ``background.method`` → ``"from_file"``
        * ``background.file_path`` → ``"{scan_dir}/computed_background.npy"``
        * ``dynamic_computation.auto_save_path`` →
          ``"{scan_dir}/computed_background.npy"``

        This matches the well-established convention across all config
        files and saves the user from manually typing these values.
        """
        bg_section = self._sections.get("background")
        if bg_section is None:
            return

        dc_section = bg_section._nested_sections.get("dynamic_computation")
        if dc_section is None:
            return

        default_path = "{scan_dir}/computed_background.npy"

        def _on_dynamic_computation_toggled(checked: bool) -> None:
            if not checked:
                return

            # Set background.method to "from_file"
            method_widget = bg_section._field_widgets.get("method")
            if method_widget is not None:
                method_widget.set_value("from_file")

            # Set background.file_path to the conventional path
            file_path_widget = bg_section._field_widgets.get("file_path")
            if file_path_widget is not None:
                file_path_widget.set_value(default_path)

            # Set dynamic_computation.auto_save_path
            auto_save_widget = dc_section._field_widgets.get("auto_save_path")
            if auto_save_widget is not None:
                auto_save_widget.set_value(default_path)

        dc_section.toggled.connect(_on_dynamic_computation_toggled)

    def _set_pipeline_from_enabled_sections(self) -> None:
        """Populate the pipeline widget from currently enabled sections."""
        if self._pipeline is None:
            return
        enabled = [
            name for name, section in self._sections.items() if section.isChecked()
        ]
        self._pipeline.set_enabled_steps(enabled)

    def _clear(self) -> None:
        """Remove all widgets from the scroll area content."""
        self._sections.clear()
        self._pipeline = None
        self._top_widgets.clear()
        self._analysis_widget = None
        self._metadata_widget = None
        self._dirty = False

        # Remove all widgets from the content layout
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _on_value_changed(self, *_args) -> None:
        """Handle any value change in the editor.

        Sets the dirty flag and emits signals.
        """
        if not self._dirty:
            self._dirty = True
            self.dirtyStateChanged.emit(True)
        self.valueChanged.emit()
