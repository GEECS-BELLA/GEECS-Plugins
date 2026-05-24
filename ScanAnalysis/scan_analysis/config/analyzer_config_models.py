"""Pydantic models for scatter-plot scan analyzers.

After the unified-configs migration, image-analyzer-driven scan
analyzers (Array2D / Array1D) are configured via the unified
diagnostic schema in :mod:`scan_analysis.config.diagnostic_models`.
Scatter analyzers are different: they do not consume images, they
read scalar columns from the s-file and produce a single summary
plot. They stay on their own config shape because the unified
``image:`` / ``scan:`` layout does not naturally fit them.

This module defines just the scatter-specific models.
"""

from __future__ import annotations

import logging
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

__all__ = ["PlotParameterConfig", "ScatterAnalyzerConfig"]


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
