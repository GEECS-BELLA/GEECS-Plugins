"""Factory for scatter-plot scan analyzers.

The unified diagnostic factory in
:mod:`scan_analysis.config.diagnostic_factory` handles image-analyzer-
driven scan analyzers (Array2D / Array1D). Scatter analyzers stay on
their own model and factory because they do not consume images and
do not fit the unified ``image:`` / ``scan:`` schema.

Use :func:`create_analyzer` to instantiate a scatter analyzer from a
:class:`ScatterAnalyzerConfig`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .analyzer_config_models import ScatterAnalyzerConfig

if TYPE_CHECKING:
    from scan_analysis.base import ScanAnalyzer

logger = logging.getLogger(__name__)

__all__ = ["create_analyzer"]


def create_analyzer(config: ScatterAnalyzerConfig) -> "ScanAnalyzer":
    """Instantiate a scatter scan analyzer from configuration.

    Parameters
    ----------
    config : ScatterAnalyzerConfig
        Validated scatter analyzer configuration.

    Returns
    -------
    ScanAnalyzer
        Configured ScatterPlotterAnalysis instance with ``id``,
        ``priority``, and ``gdoc_slot`` attached.

    Raises
    ------
    ValueError
        If ``config`` is not a ``ScatterAnalyzerConfig``.
    TypeError
        If ScatterPlotterAnalysis cannot be instantiated with the
        configured parameters.
    """
    if not isinstance(config, ScatterAnalyzerConfig):
        raise ValueError(
            f"create_analyzer accepts only ScatterAnalyzerConfig; got "
            f"{type(config).__name__}. Use create_diagnostic_analyzer "
            f"from scan_analysis.config.diagnostic_factory for unified "
            f"image-analyzer-driven scan analyzers."
        )
    return _create_scatter_analyzer(config)


def _create_scatter_analyzer(config: ScatterAnalyzerConfig) -> "ScanAnalyzer":
    """Build a ScatterPlotterAnalysis from a ScatterAnalyzerConfig."""
    from scan_analysis.analyzers.common.scatter_plotter_analysis import (
        PlotParameter,
        ScatterPlotterAnalysis,
    )

    parameters = [
        PlotParameter(
            key_name=p.key_name,
            legend_label=p.label if p.label is not None else p.key_name,
            axis_label=p.label if p.label is not None else p.key_name,
            color=p.color,
            y_range=tuple(p.y_range) if p.y_range is not None else None,
        )
        for p in config.parameters
    ]

    try:
        logger.info(
            "Creating ScatterPlotterAnalysis '%s' (priority=%s)",
            config.filename,
            config.priority,
        )
        analyzer = ScatterPlotterAnalysis(
            use_median=config.use_median,
            title=config.title,
            parameters=parameters,
            filename=config.filename,
            x_column=config.x_column,
        )
        analyzer.id = config.id
        analyzer.priority = config.priority
        analyzer.gdoc_slot = config.gdoc_slot
        return analyzer
    except TypeError as exc:
        raise TypeError(
            f"Failed to instantiate ScatterPlotterAnalysis '{config.filename}': {exc}"
        ) from exc
