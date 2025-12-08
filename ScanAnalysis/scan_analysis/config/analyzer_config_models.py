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

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

__all__ = [
    "ImageAnalyzerConfig",
    "Array2DAnalyzerConfig",
    "Array1DAnalyzerConfig",
    "ScanAnalyzerConfig",
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
    type : Literal["array2d"]
        Analyzer type identifier (must be "array2d")
    device_name : str
        Device name to analyze (e.g., "UC_GaiaMode")
    priority : int
        Execution priority (lower = higher priority, 0 = highest).
        Default is 100 (background priority).
    image_analyzer : ImageAnalyzerConfig
        Configuration for the image analyzer to use
    requirements : Union[Dict[str, List[str]], List[str], str]
        Data requirements specifying which devices must be present.
        Can be a simple list of device names, or a dict mapping
        requirement types to device lists.
    kwargs : Dict[str, Any]
        Additional kwargs for Array2DScanAnalyzer constructor
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
    requirements: Union[Dict[str, List[str]], List[str], str] = Field(
        default_factory=list,
        description="Data requirements (devices that must be present)",
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs for Array2DScanAnalyzer constructor",
    )
    is_active: bool = Field(
        default=True, description="Whether this analyzer is enabled"
    )

    @field_validator("requirements", mode="before")
    @classmethod
    def ensure_requirements_include_device(cls, v: Any, info) -> Any:
        """Ensure requirements include the device_name."""
        device = info.data.get("device_name")
        if device:
            if isinstance(v, list):
                if device not in v:
                    v.append(device)
            elif isinstance(v, str):
                v = [device]
            elif isinstance(v, dict):
                # Complex requirements, assume user knows what they're doing
                pass
            else:
                v = [device]
        return v


class Array1DAnalyzerConfig(BaseModel):
    """
    Configuration for Array1D scan analyzer.

    Defines a scan analyzer that processes 1D line data from a single device.
    Uses Array1DScanAnalyzer under the hood.

    Attributes
    ----------
    type : Literal["array1d"]
        Analyzer type identifier (must be "array1d")
    device_name : str
        Device name to analyze
    priority : int
        Execution priority (lower = higher priority, 0 = highest).
        Default is 100 (background priority).
    image_analyzer : ImageAnalyzerConfig
        Configuration for the 1D image analyzer to use
    requirements : Union[Dict[str, List[str]], List[str], str]
        Data requirements specifying which devices must be present
    kwargs : Dict[str, Any]
        Additional kwargs for Array1DScanAnalyzer constructor
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
    requirements: Union[Dict[str, List[str]], List[str], str] = Field(
        default_factory=list, description="Data requirements"
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs for Array1DScanAnalyzer constructor",
    )
    is_active: bool = Field(
        default=True, description="Whether this analyzer is enabled"
    )

    @field_validator("requirements", mode="before")
    @classmethod
    def ensure_requirements_include_device(cls, v: Any, info) -> Any:
        """Ensure requirements include the device_name."""
        device = info.data.get("device_name")
        if device:
            if isinstance(v, list):
                if device not in v:
                    v.append(device)
            elif isinstance(v, str):
                v = [device]
            elif isinstance(v, dict):
                # Complex requirements, assume user knows what they're doing
                pass
            else:
                v = [device]
        return v


# Union type for all analyzer configs
ScanAnalyzerConfig = Union[Array2DAnalyzerConfig, Array1DAnalyzerConfig]


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
