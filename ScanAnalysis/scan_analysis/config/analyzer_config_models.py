"""
Pydantic models for scan analyzer configuration.

This module provides validated configuration models for defining scan analyzers
in YAML files, replacing the older NamedTuple-based ScanAnalyzerInfo system.
Follows the same patterns as ImageAnalysis config_loader.

Examples
--------
Example YAML configuration:

    experiment: Undulator
    description: Standard analysis for Undulator experiment
    version: "1.0"

    analyzers:
      - type: array2d
        device_name: UC_GaiaMode
        priority: 0
        image_analyzer:
          analyzer_class: image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer
          camera_config_name: UC_GaiaMode

      - type: array1d
        device_name: U_BCaveICT
        priority: 100
        image_analyzer:
          analyzer_class: image_analysis.offline_analyzers.standard_1d_analyzer.Standard1DAnalyzer
          kwargs:
            data_type: tdms
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

__all__ = [
    "ImageAnalyzerConfig",
    "Array2DAnalyzerConfig",
    "Array1DAnalyzerConfig",
    "PlotParameterConfig",
    "ScatterAnalyzerConfig",
    "ScanAnalyzerConfig",
    "LibraryAnalyzer",
    "IncludeEntry",
    "GroupsConfig",
    "ExperimentAnalysisConfig",
]


class ImageAnalyzerConfig(BaseModel):
    """
    Configuration for an image analyzer instance.

    Used by Array2D and Array1D scan analyzers to specify which
    image analyzer class to instantiate and how to configure it.

    Attributes
    ----------
    analyzer_class : str
        Fully qualified class name for the image analyzer.
        Example: 'image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer'
    camera_config_name : Optional[str]
        Name of camera/line config to load. Used by StandardAnalyzer,
        BeamAnalyzer, etc. If None, analyzer must handle config itself.
    kwargs : Dict[str, Any]
        Additional keyword arguments passed to the analyzer constructor.
        Can include custom parameters like data_type, file_tail, etc.

    Examples
    --------
    >>> config = ImageAnalyzerConfig(
    ...     analyzer_class="image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer",
    ...     camera_config_name="UC_GaiaMode"
    ... )
    """

    analyzer_class: str = Field(
        ...,
        description="Fully qualified class name (e.g., 'image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer')",
    )
    camera_config_name: Optional[str] = Field(
        None,
        description="Name of camera/line config to load (for StandardAnalyzer, BeamAnalyzer, etc.)",
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs passed to image analyzer constructor",
    )

    @field_validator("analyzer_class")
    @classmethod
    def validate_analyzer_class(cls, v: str) -> str:
        """Ensure analyzer_class is a valid import path."""
        if not v or "." not in v:
            raise ValueError(
                f"analyzer_class must be a fully qualified path (e.g., 'module.Class'), got: {v}"
            )
        return v


class Array2DAnalyzerConfig(BaseModel):
    """
    Configuration for Array2D scan analyzer.

    Defines a scan analyzer that processes 2D image data from a single device.
    Uses Array2DScanAnalyzer under the hood.

    Attributes
    ----------
    id : Optional[str]
        Unique analyzer identifier used for task-queue status tracking.
        Defaults to ``device_name`` when not provided.  Must be unique
        across all analyzers in an experiment config when multiple
        analyzers share the same ``device_name``.
    type : Literal["array2d"]
        Analyzer type identifier (must be "array2d")
    device_name : str
        Device name to analyze (e.g., "UC_GaiaMode")
    priority : int
        Execution priority (lower = higher priority, 0 = highest).
        Default is 100 (background priority).
    image_analyzer : ImageAnalyzerConfig
        Configuration for the image analyzer to use
    file_tail : Optional[str]
        Suffix/extension used to match data files (e.g., ".png").
        Uses analyzer default if omitted.
    skip_plt_show : bool
        Whether to suppress interactive plotting.
    flag_save_images : bool
        Whether to save images produced by the analyzer.
    renderer_kwargs : Dict[str, Any]
        Additional kwargs passed to the renderer.
    analysis_mode : Literal["per_shot", "per_bin"]
        Analysis mode for the analyzer.
    kwargs : Dict[str, Any]
        Additional kwargs for Array2DScanAnalyzer constructor (advanced)
    is_active : bool
        Whether this analyzer is enabled. Set to False to temporarily
        disable without removing from config.

    Examples
    --------
    >>> config = Array2DAnalyzerConfig(
    ...     device_name="UC_GaiaMode",
    ...     priority=0,
    ...     image_analyzer=ImageAnalyzerConfig(
    ...         analyzer_class="image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer",
    ...         camera_config_name="UC_GaiaMode"
    ...     )
    ... )
    """

    id: Optional[str] = Field(
        default=None,
        description="Unique analyzer identifier for task-queue tracking. Defaults to device_name.",
    )
    type: Literal["array2d"] = Field(
        default="array2d", description="Analyzer type identifier"
    )
    device_name: str = Field(..., description="Device name to analyze")
    priority: int = Field(
        default=100,
        ge=0,
        description="Analysis priority (lower number = higher priority, 0 = highest, 100 = default background)",
    )
    image_analyzer: ImageAnalyzerConfig = Field(
        ..., description="Image analyzer configuration"
    )
    file_tail: Optional[str] = Field(
        default=None,
        description="File suffix/extension to match (e.g., '.png'). Uses default if None.",
    )
    skip_plt_show: bool = Field(
        default=True, description="Whether to suppress interactive plotting"
    )
    flag_save_images: bool = Field(
        default=True, description="Whether to save images produced by the analyzer"
    )
    renderer_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Renderer kwargs"
    )
    analysis_mode: Literal["per_shot", "per_bin"] = Field(
        default="per_shot", description="Analysis mode"
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs for Array2DScanAnalyzer constructor",
    )
    is_active: bool = Field(
        default=True, description="Whether this analyzer is enabled"
    )
    gdoc_slot: Optional[int] = Field(
        default=None,
        ge=0,
        le=3,
        description=(
            "Table cell index (0–3) in the 2×2 display table embedded in each scan "
            "entry. When set, the last summary figure is inserted into that cell. "
            "When omitted and gdoc_enabled=True on the runner, all display files are "
            "uploaded to Drive and appended as hyperlinks instead."
        ),
    )

    @model_validator(mode="after")
    def default_id_to_device_name(self) -> "Array2DAnalyzerConfig":
        """Set ``id`` to ``device_name`` when the user omits it."""
        if self.id is None:
            self.id = self.device_name
        return self


class Array1DAnalyzerConfig(BaseModel):
    """
    Configuration for Array1D scan analyzer.

    Defines a scan analyzer that processes 1D line data from a single device.
    Uses Array1DScanAnalyzer under the hood.

    Attributes
    ----------
    id : Optional[str]
        Unique analyzer identifier used for task-queue status tracking.
        Defaults to ``device_name`` when not provided.  Must be unique
        across all analyzers in an experiment config when multiple
        analyzers share the same ``device_name``.
    type : Literal["array1d"]
        Analyzer type identifier (must be "array1d")
    device_name : str
        Device name to analyze
    priority : int
        Execution priority (lower = higher priority, 0 = highest).
        Default is 100 (background priority).
    image_analyzer : ImageAnalyzerConfig
        Configuration for the 1D image analyzer to use
    file_tail : Optional[str]
        Suffix/extension used to match data files (e.g., ".csv").
        Uses analyzer default if omitted.
    skip_plt_show : bool
        Whether to suppress interactive plotting.
    flag_save_data : bool
        Whether to save data/plots produced by the analyzer.
    renderer_kwargs : Dict[str, Any]
        Additional kwargs for the renderer.
    analysis_mode : Literal["per_shot", "per_bin"]
        Analysis mode for the analyzer.
    kwargs : Dict[str, Any]
        Additional kwargs for Array1DScanAnalyzer constructor (advanced)
    is_active : bool
        Whether this analyzer is enabled

    Examples
    --------
    >>> config = Array1DAnalyzerConfig(
    ...     device_name="U_BCaveICT",
    ...     priority=100,
    ...     image_analyzer=ImageAnalyzerConfig(
    ...         analyzer_class="image_analysis.offline_analyzers.standard_1d_analyzer.Standard1DAnalyzer",
    ...         kwargs={"data_type": "tdms"}
    ...     )
    ... )
    """

    id: Optional[str] = Field(
        default=None,
        description="Unique analyzer identifier for task-queue tracking. Defaults to device_name.",
    )
    type: Literal["array1d"] = Field(
        default="array1d", description="Analyzer type identifier"
    )
    device_name: str = Field(..., description="Device name to analyze")
    priority: int = Field(
        default=100,
        ge=0,
        description="Analysis priority (lower number = higher priority, 0 = highest, 100 = default background)",
    )
    image_analyzer: ImageAnalyzerConfig = Field(
        ..., description="Image analyzer (1D) configuration"
    )
    file_tail: Optional[str] = Field(
        default=None,
        description="File suffix/extension to match (e.g., '.csv'). Uses default if None.",
    )
    skip_plt_show: bool = Field(
        default=True, description="Whether to suppress interactive plotting"
    )
    flag_save_data: bool = Field(
        default=True, description="Whether to save data/plots produced by the analyzer"
    )
    renderer_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Renderer kwargs"
    )
    analysis_mode: Literal["per_shot", "per_bin"] = Field(
        default="per_shot", description="Analysis mode"
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs for Array1DScanAnalyzer constructor",
    )
    is_active: bool = Field(
        default=True, description="Whether this analyzer is enabled"
    )
    gdoc_slot: Optional[int] = Field(
        default=None,
        ge=0,
        le=3,
        description=(
            "Table cell index (0–3) in the 2×2 display table embedded in each scan "
            "entry. When set, the last summary figure is inserted into that cell. "
            "When omitted and gdoc_enabled=True on the runner, all display files are "
            "uploaded to Drive and appended as hyperlinks instead."
        ),
    )

    @model_validator(mode="after")
    def default_id_to_device_name(self) -> "Array1DAnalyzerConfig":
        """Set ``id`` to ``device_name`` when the user omits it."""
        if self.id is None:
            self.id = self.device_name
        return self


class PlotParameterConfig(BaseModel):
    """
    Configuration for a single y-series in a scatter plot.

    Attributes
    ----------
    key_name : str
        S-file column name to use as the y-axis data.
    label : Optional[str]
        Display name used for both the legend and the y-axis label.
        Defaults to ``key_name`` when not provided.
    color : str
        Matplotlib-compatible color string.
    y_range : Optional[list[float]]
        Fixed ``[min, max]`` limits for the y-axis. When omitted, matplotlib
        chooses the range automatically. Must have exactly two elements.

    Examples
    --------
    >>> p = PlotParameterConfig(key_name="UC_ModeImager_x_rms", label="x RMS (px)", color="blue")
    """

    key_name: str = Field(..., description="S-file column name for y-axis data")
    label: Optional[str] = Field(
        default=None,
        description="Display name for legend and y-axis label (defaults to key_name)",
    )
    color: str = Field(default="blue", description="Matplotlib color string")
    y_range: Optional[List[float]] = Field(
        default=None,
        description="Fixed [min, max] y-axis limits. Must have exactly 2 elements.",
    )

    @field_validator("y_range")
    @classmethod
    def validate_y_range(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Ensure y_range has exactly two elements when provided."""
        if v is not None and len(v) != 2:
            raise ValueError("y_range must have exactly 2 elements: [min, max]")
        return v


class ScatterAnalyzerConfig(BaseModel):
    """
    Configuration for a scatter-plot scan analyzer.

    Defines a standalone scatter plot that reads scalar columns from the s-file
    and produces a PNG summary figure. Supports multiple y-series on twinned axes,
    optional fixed y-axis ranges, and an optional custom x-axis column.

    Attributes
    ----------
    id : Optional[str]
        Unique identifier for task-queue tracking. Defaults to ``filename``.
    type : Literal["scatter"]
        Discriminator field (must be ``"scatter"``).
    title : str
        Figure title (date and scan number are prepended automatically).
    filename : str
        Output PNG filename stem (without ``.png`` extension).
    parameters : list of PlotParameterConfig
        One or more y-series to plot.
    x_column : Optional[str]
        S-file column to use as the x-axis. For 1D scans the per-bin statistic
        is used; for noscans per-shot values are used. Defaults to the scan
        parameter (1D scans) or shot number (noscans) when omitted.
    use_median : bool
        Use the median as the representative statistic. Default ``True``.
    priority : int
        Execution priority (lower = higher priority). Default ``200``.
    is_active : bool
        Set to ``False`` to disable without removing from config. Default ``True``.
    gdoc_slot : Optional[int]
        Table cell index (0–3) for GDoc upload. ``None`` uses hyperlink mode.

    Examples
    --------
    YAML entry::

        - type: scatter
          title: "ModeImager Beam Properties"
          filename: "mode_imager_rms"
          x_column: "U_ModeImagerESP Position.Axis 1 Alias:ModeImager"
          parameters:
            - key_name: UC_ModeImager_x_rms
              label: "x RMS (px)"
              color: "blue"
              y_range: [0, 50]
            - key_name: UC_ModeImage_y_rms
              label: "y RMS (px)"
              color: "red"
    """

    id: Optional[str] = Field(
        default=None,
        description="Unique identifier for task-queue tracking. Defaults to filename.",
    )
    type: Literal["scatter"] = Field(default="scatter", description="Analyzer type")
    title: str = Field(..., description="Figure title")
    filename: str = Field(..., description="Output PNG filename stem (no extension)")
    parameters: List[PlotParameterConfig] = Field(
        ..., description="Y-series specifications"
    )
    x_column: Optional[str] = Field(
        default=None,
        description="S-file column for x-axis. Falls back to scan parameter or shot number.",
    )
    use_median: bool = Field(
        default=True, description="Use median as representative statistic"
    )
    priority: int = Field(default=200, ge=0, description="Execution priority")
    is_active: bool = Field(
        default=True, description="Whether this analyzer is enabled"
    )
    gdoc_slot: Optional[int] = Field(
        default=None,
        ge=0,
        le=3,
        description="Table cell index (0–3) for GDoc upload. None = hyperlink mode.",
    )

    @model_validator(mode="after")
    def default_id_to_filename(self) -> "ScatterAnalyzerConfig":
        """Set ``id`` to ``filename`` when the user omits it."""
        if self.id is None:
            self.id = self.filename
        return self


# Union type for all analyzer configs
ScanAnalyzerConfig = Union[
    Array2DAnalyzerConfig, Array1DAnalyzerConfig, ScatterAnalyzerConfig
]


class LibraryAnalyzer(BaseModel):
    """Analyzer definition stored in the library with an identifier."""

    model_config = ConfigDict(extra="forbid")

    id: str
    analyzer: ScanAnalyzerConfig

    @model_validator(mode="before")
    @classmethod
    def wrap_analyzer(cls, values: Any) -> Any:
        """
        Allow YAML to specify `id` alongside analyzer fields (without nested key).

        If incoming mapping has `id` and analyzer fields, wrap them under `analyzer`.
        """
        if isinstance(values, dict) and "id" in values and "analyzer" not in values:
            data = values.copy()
            aid = data.pop("id")
            return {"id": aid, "analyzer": data}
        return values


class IncludeEntry(BaseModel):
    """
    Include directive for experiment configs.

    Exactly one of `ref` (single analyzer id) or `group` (named group of ids) must be set.
    Optionally apply explicit `priority` or a `priority_offset` to defaults, and shallow
    `overrides` for analyzer fields (e.g., kwargs).
    """

    model_config = ConfigDict(extra="forbid")

    ref: Optional[str] = Field(default=None, description="Analyzer id to include")
    group: Optional[str] = Field(default=None, description="Group name to include")
    priority: Optional[int] = Field(
        default=None, description="Explicit priority (overrides defaults/offsets)"
    )
    priority_offset: int = Field(
        default=0, description="Offset applied to library default priority"
    )
    overrides: Dict[str, Any] = Field(
        default_factory=dict, description="Shallow overrides for analyzer definition"
    )

    @model_validator(mode="after")
    def ensure_one_target(self) -> "IncludeEntry":
        """Ensure exactly one of ref/group is provided."""
        if bool(self.ref) == bool(self.group):
            raise ValueError("Exactly one of 'ref' or 'group' must be provided.")
        return self


class GroupsConfig(BaseModel):
    """Mapping of group name to list of analyzer ids."""

    model_config = ConfigDict(extra="forbid")
    groups: Dict[str, List[str]] = Field(default_factory=dict)


class ExperimentAnalysisConfig(BaseModel):
    """
    Top-level configuration for an experiment's analysis setup.

    This replaces the hardcoded map_*.py files with a flexible
    YAML-based configuration system that can be loaded from
    a separate configs repository.

    Attributes
    ----------
    experiment : str
        Experiment name (e.g., 'Undulator', 'HTU')
    description : Optional[str]
        Human-readable description of this analysis configuration
    version : str
        Config file version for tracking changes
    analyzers : List[ScanAnalyzerConfig]
        List of analyzer configurations (Array2D and/or Array1D)
    upload_to_scanlog : bool
        Whether to upload analysis results to scan log database
    include : List[IncludeEntry]
        Optional list of include directives (ref/group) for assembling analyzers

    Examples
    --------
    >>> config = ExperimentAnalysisConfig(
    ...     experiment="Undulator",
    ...     description="Standard Undulator analysis",
    ...     analyzers=[
    ...         Array2DAnalyzerConfig(
    ...             device_name="UC_GaiaMode",
    ...             priority=0,
    ...             image_analyzer=ImageAnalyzerConfig(
    ...                 analyzer_class="image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer",
    ...                 camera_config_name="UC_GaiaMode"
    ...             )
    ...         )
    ...     ]
    ... )
    """

    experiment: str = Field(..., description="Experiment name (e.g., 'Undulator')")
    description: Optional[str] = Field(None, description="Human-readable description")
    version: str = Field(default="1.0", description="Config version")
    analyzers: List[ScanAnalyzerConfig] = Field(
        default_factory=list, description="List of analyzer configurations"
    )
    upload_to_scanlog: bool = Field(
        default=True, description="Whether to upload results to scan log"
    )
    include: List[IncludeEntry] = Field(
        default_factory=list,
        description="Include directives (refs/groups) to assemble analyzers",
    )

    @property
    def active_analyzers(self) -> List[ScanAnalyzerConfig]:
        """Get all active analyzers."""
        return [a for a in self.analyzers if a.is_active]

    def get_analyzers_by_priority(
        self, max_priority: Optional[int] = None
    ) -> List[ScanAnalyzerConfig]:
        """
        Get active analyzers sorted by priority.

        Parameters
        ----------
        max_priority : Optional[int]
            If provided, only return analyzers with priority <= max_priority

        Returns
        -------
        List[ScanAnalyzerConfig]
            Analyzers sorted by priority (lowest first)
        """
        analyzers = self.active_analyzers
        if max_priority is not None:
            analyzers = [a for a in analyzers if a.priority <= max_priority]
        return sorted(analyzers, key=lambda a: a.priority)
