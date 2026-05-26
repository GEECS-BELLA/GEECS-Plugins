"""
Configuration loading utilities for image analysis.

Single public entry point:
    load_camera_config(...) -> cfg_2d.CameraConfig

This version uses a single, explicit base directory for YAML configs, set via
environment variable ``IMAGE_ANALYSIS_CONFIG_DIR`` or passed per call.

Supports recursive search in subdirectories for organized config structures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import ValidationError

from . import array2d_processing as cfg_2d
from . import array1d_processing as cfg_1d
from geecs_data_utils.config_roots import image_analysis_config

logger = logging.getLogger(__name__)

__all__ = [
    "find_config_file",
    "load_camera_config",
    "load_line_config",
]

# ----------------------------------------------------------------------
# Base directory management
# ----------------------------------------------------------------------

_CONFIG_MANAGER = image_analysis_config
_CONFIG_CACHE: Dict[str, Path] = (
    _CONFIG_MANAGER.cache
)  # Cache for resolved config paths

# ----------------------------------------------------------------------
# Core helpers
# ----------------------------------------------------------------------


def find_config_file(
    camera_name: str, *, config_dir: Optional[Path] = None, use_cache: bool = True
) -> Path:
    r"""
    Resolve config file by name, searching recursively if needed.

    Search order:
    1. Check cache (if enabled)
    2. Direct children of base directory (fast path)
    3. Recursive search in subdirectories

    Parameters
    ----------
    camera_name : str
        Logical name of the camera configuration (file stem).
    config_dir : Optional[Path]
        Directory containing YAML config files. If None, uses the global base dir.
    use_cache : bool, default=True
        Whether to use cached paths for performance.

    Returns
    -------
    Path
        Path to the YAML configuration file.

    Raises
    ------
    ValueError
        If no base directory is available.
    FileNotFoundError
        If the file is not found under the base directory.

    Notes
    -----
    If multiple configs with the same name exist in different subdirectories,
    the first one found (alphabetically) is used and a warning is logged.
    """
    return _CONFIG_MANAGER.find_config(
        camera_name,
        patterns=[
            "{name}.yaml",
            "{name}.yml",
            "default_{name}_settings.yaml",
            "default_{name}_settings.yml",
        ],
        config_dir=config_dir,
        use_cache=use_cache,
        missing_base_message=(
            "config_dir is required (no global base dir set). "
            "Set IMAGE_ANALYSIS_CONFIG_DIR or pass config_dir explicitly."
        ),
        not_found_label="Config",
    )


def _load_camera_config_dict(
    config_source: Union[str, Path, Dict[str, Any]],
    *,
    config_dir: Optional[Path],
) -> Dict[str, Any]:
    """Internal: load raw config dict from name/path/dict.

    Unwraps the ``image:`` section if the YAML is a unified diagnostic,
    so callers always receive the flat camera/line config shape.
    """
    if isinstance(config_source, str):
        path = find_config_file(config_source, config_dir=config_dir)
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        logger.info("Loaded camera configuration from %s", path)
    elif isinstance(config_source, Path):
        if not config_source.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_source}")
        with open(config_source, "r") as f:
            data = yaml.safe_load(f)
        logger.info("Loaded configuration from %s", config_source)
    elif isinstance(config_source, dict):
        data = config_source.copy()
        logger.info("Using provided configuration dictionary")
    else:
        raise ValueError(f"Invalid config_source type: {type(config_source)}")

    if data is None:
        data = {}

    return _unwrap_diagnostic_image_section(data)


def _unwrap_diagnostic_image_section(data: Dict[str, Any]) -> Dict[str, Any]:
    r"""Return the ``image:`` subdict of a unified diagnostic YAML.

    A unified diagnostic config (per the schema in issue #400) lives in
    one file with both ImageAnalysis-side and ScanAnalysis-side
    sections. ImageAnalysis only consumes the ``image:`` body; the
    top-level ``name`` is injected as ``image.name`` so the embedded
    config validates the same way a standalone camera/line config
    would.

    Flat configs (no ``image:`` key) pass through unchanged so legacy
    standalone files in ``image_analysis_configs/`` keep working
    without modification.

    Parameters
    ----------
    data : dict
        Raw YAML-loaded dict.

    Returns
    -------
    dict
        Either the unwrapped image section (when ``image:`` is present)
        or the original dict (when it is not).

    Raises
    ------
    ValueError
        If ``image`` is present but is not a mapping.
    """
    if not isinstance(data, dict) or "image" not in data:
        return data

    image = data.get("image") or {}
    if not isinstance(image, dict):
        raise ValueError(
            f"Diagnostic config 'image' section must be a mapping, "
            f"got {type(image).__name__}"
        )

    image = dict(image)
    name = data.get("name")
    if name and "name" not in image:
        image["name"] = name
    return image


# ----------------------------------------------------------------------
# Public entry point (model-first)
# ----------------------------------------------------------------------


def load_camera_config(
    config_source: Union[str, Path, Dict[str, Any]],
    *,
    config_dir: Optional[Path] = None,
) -> cfg_2d.CameraConfig:
    """Load and validate a :class:`CameraConfig` from name / path / dict.

    Parameters
    ----------
    config_source : str, Path, or dict
        - ``str``: camera name (file stem), resolved against ``config_dir``
          (or the globally configured base dir). Unified diagnostic YAMLs
          and legacy bare-camera YAMLs are both handled — the ``image:``
          subsection is unwrapped transparently when present.
        - ``Path``: explicit ``.yaml`` / ``.yml`` file path.
        - ``dict``: already-loaded raw config (passed straight to
          ``CameraConfig.model_validate``).
    config_dir : Path, optional
        Directory containing YAML config files. Defaults to the global
        base dir.

    Returns
    -------
    CameraConfig
        Validated camera configuration model.
    """
    data = _load_camera_config_dict(config_source, config_dir=config_dir)
    try:
        return cfg_2d.CameraConfig.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Invalid camera configuration: {e}") from e


def load_line_config(
    config_source: Union[str, Path, Dict[str, Any]],
    *,
    config_dir: Optional[Path] = None,
) -> cfg_1d.Line1DConfig:
    """Load and validate a :class:`Line1DConfig` from name / path / dict.

    See :func:`load_camera_config` for parameter semantics — this is the
    1D counterpart.
    """
    data = _load_camera_config_dict(config_source, config_dir=config_dir)
    try:
        return cfg_1d.Line1DConfig.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Invalid line configuration: {e}") from e
