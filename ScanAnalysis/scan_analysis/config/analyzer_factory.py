"""
Factory for instantiating scan analyzers from configuration.

This module provides utilities to dynamically create scan analyzer instances
from Pydantic configuration models, using dynamic imports and validation.

Examples
--------
Basic usage:

    >>> from scan_analysis.config import load_experiment_config, create_analyzer
    >>>
    >>> # Load config
    >>> config = load_experiment_config("undulator")
    >>>
    >>> # Create analyzers from config
    >>> analyzers = [create_analyzer(cfg) for cfg in config.active_analyzers]
    >>>
    >>> # Use analyzers
    >>> for analyzer in analyzers:
    ...     analyzer.run_analysis(scan_tag)
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Type

from .analyzer_config_models import (
    Array1DAnalyzerConfig,
    Array2DAnalyzerConfig,
    ImageAnalyzerConfig,
    ScanAnalyzerConfig,
)

if TYPE_CHECKING:
    from scan_analysis.base import ScanAnalyzer

logger = logging.getLogger(__name__)

__all__ = [
    "create_analyzer",
    "create_image_analyzer",
]


# ----------------------------------------------------------------------
# Dynamic imports
# ----------------------------------------------------------------------


def _import_class(class_path: str) -> Type:
    """
    Dynamically import a class from a fully qualified path.

    Parameters
    ----------
    class_path : str
        Fully qualified class path (e.g., 'module.submodule.ClassName')

    Returns
    -------
    Type
        The imported class

    Raises
    ------
    ImportError
        If the module cannot be imported
    AttributeError
        If the class does not exist in the module

    Examples
    --------
    >>> BeamAnalyzer = _import_class(
    ...     'image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer'
    ... )
    """
    try:
        module_path, class_name = class_path.rsplit(".", 1)
    except ValueError as e:
        raise ValueError(
            f"Invalid class path '{class_path}'. "
            f"Must be fully qualified (e.g., 'module.Class')"
        ) from e

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Cannot import module '{module_path}' from class path '{class_path}': {e}"
        ) from e

    try:
        return getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_path}' has no class '{class_name}' "
            f"(from class path '{class_path}'): {e}"
        ) from e


# ----------------------------------------------------------------------
# Image analyzer creation
# ----------------------------------------------------------------------


def create_image_analyzer(config: ImageAnalyzerConfig):
    """
    Instantiate an image analyzer from configuration.

    Parameters
    ----------
    config : ImageAnalyzerConfig
        Configuration specifying the analyzer class and parameters

    Returns
    -------
    ImageAnalyzer
        Instantiated image analyzer ready to use

    Raises
    ------
    ImportError
        If the analyzer class cannot be imported
    TypeError
        If the analyzer class cannot be instantiated with the provided parameters

    Examples
    --------
    >>> from scan_analysis.config.analyzer_config_models import ImageAnalyzerConfig
    >>>
    >>> config = ImageAnalyzerConfig(
    ...     analyzer_class="image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer",
    ...     camera_config_name="UC_GaiaMode"
    ... )
    >>> analyzer = create_image_analyzer(config)
    """
    analyzer_class = _import_class(config.analyzer_class)

    # Build kwargs for instantiation
    kwargs = config.kwargs.copy()

    # Add camera_config_name if provided
    # This is used by StandardAnalyzer, BeamAnalyzer, etc.
    if config.camera_config_name:
        kwargs["camera_config_name"] = config.camera_config_name

    try:
        logger.info(
            f"Creating image analyzer: {config.analyzer_class} "
            f"with kwargs: {list(kwargs.keys())}"
        )
        return analyzer_class(**kwargs)
    except TypeError as e:
        raise TypeError(
            f"Failed to instantiate {config.analyzer_class} with kwargs {kwargs}: {e}"
        ) from e


# ----------------------------------------------------------------------
# Scan analyzer creation
# ----------------------------------------------------------------------


def create_analyzer(config: ScanAnalyzerConfig) -> "ScanAnalyzer":
    """
    Create scan analyzer instance from configuration.

    This is the main factory method that handles all analyzer types
    (Array2D, Array1D, etc.) and returns a ready-to-use ScanAnalyzer instance.

    Parameters
    ----------
    config : ScanAnalyzerConfig
        Analyzer configuration (Array2DAnalyzerConfig or Array1DAnalyzerConfig)

    Returns
    -------
    ScanAnalyzer
        Instantiated scan analyzer ready to use

    Raises
    ------
    ImportError
        If required analyzer classes cannot be imported
    TypeError
        If analyzers cannot be instantiated with the provided parameters
    ValueError
        If the config type is unknown

    Examples
    --------
    >>> from scan_analysis.config import load_experiment_config, create_analyzer
    >>>
    >>> config = load_experiment_config("undulator")
    >>>
    >>> # Create all analyzers
    >>> analyzers = [create_analyzer(cfg) for cfg in config.active_analyzers]
    >>>
    >>> # Create only high-priority analyzers
    >>> priority_analyzers = [
    ...     create_analyzer(cfg)
    ...     for cfg in config.get_analyzers_by_priority(max_priority=10)
    ... ]
    """
    if isinstance(config, Array2DAnalyzerConfig):
        return _create_array2d_analyzer(config)
    elif isinstance(config, Array1DAnalyzerConfig):
        return _create_array1d_analyzer(config)
    else:
        raise ValueError(f"Unknown analyzer config type: {type(config)}")


def _create_array2d_analyzer(config: Array2DAnalyzerConfig) -> "ScanAnalyzer":
    """Create Array2DScanAnalyzer from configuration."""
    # Import here to avoid circular imports
    from scan_analysis.analyzers.common.array2D_scan_analysis import (
        Array2DScanAnalyzer,
    )

    # Create the image analyzer
    image_analyzer = create_image_analyzer(config.image_analyzer)

    # Build kwargs for Array2DScanAnalyzer
    kwargs = config.kwargs.copy()
    kwargs["image_analyzer"] = image_analyzer
    kwargs["device_name"] = config.device_name

    # Add requirements if specified
    if config.requirements:
        kwargs["requirements"] = config.requirements

    try:
        logger.info(
            f"Creating Array2DScanAnalyzer for device '{config.device_name}' "
            f"(priority={config.priority})"
        )
        analyzer = Array2DScanAnalyzer(**kwargs)

        # Attach priority for later sorting
        # This is safe since we're just adding an attribute, not modifying the class
        analyzer.priority = config.priority

        return analyzer
    except TypeError as e:
        raise TypeError(
            f"Failed to instantiate Array2DScanAnalyzer for {config.device_name}: {e}"
        ) from e


def _create_array1d_analyzer(config: Array1DAnalyzerConfig) -> "ScanAnalyzer":
    """Create Array1DScanAnalyzer from configuration."""
    # Import here to avoid circular imports
    from scan_analysis.analyzers.common.array1d_scan_analysis import (
        Array1DScanAnalyzer,
    )

    # Create the image analyzer (1D)
    image_analyzer = create_image_analyzer(config.image_analyzer)

    # Build kwargs for Array1DScanAnalyzer
    kwargs = config.kwargs.copy()
    kwargs["image_analyzer"] = image_analyzer
    kwargs["device_name"] = config.device_name

    # Add requirements if specified
    if config.requirements:
        kwargs["requirements"] = config.requirements

    try:
        logger.info(
            f"Creating Array1DScanAnalyzer for device '{config.device_name}' "
            f"(priority={config.priority})"
        )
        analyzer = Array1DScanAnalyzer(**kwargs)

        # Attach priority for later sorting
        analyzer.priority = config.priority

        return analyzer
    except TypeError as e:
        raise TypeError(
            f"Failed to instantiate Array1DScanAnalyzer for {config.device_name}: {e}"
        ) from e
