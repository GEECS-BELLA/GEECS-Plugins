"""Config I/O layer for the Config File Editor GUI.

Provides functions for listing, loading, saving, validating, and creating
device configuration YAML files.  All heavy lifting is delegated to the
existing ``image_analysis.config_loader`` module and the Pydantic config
models; this module adds convenience wrappers tailored to the GUI's needs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel

from image_analysis.config_loader import (
    load_camera_config,
    load_line_config,
    save_config_to_yaml,
)
from image_analysis.processing.array1d.config_models import (
    Data1DConfig,
    Data1DType,
    Line1DConfig,
)
from image_analysis.processing.array2d.config_models import CameraConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Directory listing
# ---------------------------------------------------------------------------


def list_config_files(config_dir: Path) -> List[Path]:
    """List all YAML configuration files in a directory.

    Scans the given directory (non-recursively) for files with ``.yaml`` or
    ``.yml`` extensions and returns them sorted alphabetically by filename.

    Parameters
    ----------
    config_dir : Path
        Directory to scan for configuration files.

    Returns
    -------
    List[Path]
        Sorted list of paths to YAML files found in *config_dir*.

    Raises
    ------
    FileNotFoundError
        If *config_dir* does not exist.
    NotADirectoryError
        If *config_dir* is not a directory.
    """
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    if not config_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {config_dir}")

    yaml_files: List[Path] = []
    for child in config_dir.iterdir():
        if child.is_file() and child.suffix.lower() in (".yaml", ".yml"):
            yaml_files.append(child)

    yaml_files.sort(key=lambda p: p.name.lower())
    logger.debug("Found %d YAML files in %s", len(yaml_files), config_dir)
    return yaml_files


# ---------------------------------------------------------------------------
# Type detection
# ---------------------------------------------------------------------------


def detect_config_type(file_path: Path) -> Literal["camera_2d", "line_1d", "unknown"]:
    """Determine whether a YAML file is a 2D camera or 1D line config.

    The detection heuristic inspects top-level keys in the YAML document:

    * If the file contains a ``bit_depth`` key it is classified as
      ``"camera_2d"`` (a :class:`CameraConfig`).
    * If the file contains a ``data_loading`` key it is classified as
      ``"line_1d"`` (a :class:`Line1DConfig`).
    * Otherwise the type is ``"unknown"``.

    Parameters
    ----------
    file_path : Path
        Path to the YAML configuration file.

    Returns
    -------
    Literal["camera_2d", "line_1d", "unknown"]
        Detected configuration type.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(file_path, "r") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        logger.warning("YAML file does not contain a mapping: %s", file_path)
        return "unknown"

    if "bit_depth" in data:
        logger.debug("Detected camera_2d config: %s", file_path)
        return "camera_2d"

    if "data_loading" in data:
        logger.debug("Detected line_1d config: %s", file_path)
        return "line_1d"

    logger.debug("Unknown config type: %s", file_path)
    return "unknown"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_config(file_path: Path) -> Union[CameraConfig, Line1DConfig]:
    """Load and validate a configuration file, auto-detecting its type.

    Uses :func:`detect_config_type` to determine whether the file is a 2D
    camera config or a 1D line config, then delegates to the appropriate
    loader from ``image_analysis.config_loader``.

    Parameters
    ----------
    file_path : Path
        Path to the YAML configuration file.

    Returns
    -------
    Union[CameraConfig, Line1DConfig]
        Validated Pydantic configuration model.

    Raises
    ------
    ValueError
        If the configuration type cannot be determined.
    FileNotFoundError
        If *file_path* does not exist.
    """
    config_type = detect_config_type(file_path)

    if config_type == "camera_2d":
        logger.info("Loading camera config from %s", file_path)
        return load_camera_config(file_path)

    if config_type == "line_1d":
        logger.info("Loading line config from %s", file_path)
        return load_line_config(file_path)

    raise ValueError(
        f"Cannot determine config type for '{file_path}'. "
        "Expected a YAML file with 'bit_depth' (camera) or "
        "'data_loading' (line) top-level key."
    )


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def save_config(config: BaseModel, output_path: Path) -> None:
    """Save a Pydantic configuration model to a YAML file.

    Delegates to :func:`image_analysis.config_loader.save_config_to_yaml`,
    which serialises the model via ``model_dump()`` and writes it with
    ``yaml.dump``.  Parent directories are created automatically.

    Parameters
    ----------
    config : BaseModel
        A validated Pydantic config model (e.g. :class:`CameraConfig` or
        :class:`Line1DConfig`).
    output_path : Path
        Destination file path.  Will be created or overwritten.
    """
    logger.info("Saving config to %s", output_path)
    save_config_to_yaml(config, output_path)


# ---------------------------------------------------------------------------
# Factory helpers for new configs
# ---------------------------------------------------------------------------


def create_new_camera_config(name: str) -> CameraConfig:
    """Create a new :class:`CameraConfig` with sensible defaults.

    The returned config has ``bit_depth=16`` and no processing sections
    enabled, providing a clean starting point for the user to customise.

    Parameters
    ----------
    name : str
        Camera identifier / name for the new configuration.

    Returns
    -------
    CameraConfig
        A minimal, valid camera configuration.
    """
    logger.info("Creating new camera config: %s", name)
    return CameraConfig(
        name=name,
        description="",
        bit_depth=16,
    )


def create_new_line_config(name: str) -> Line1DConfig:
    """Create a new :class:`Line1DConfig` with sensible defaults.

    The returned config uses ``data_type="csv"`` for the data-loading
    section and leaves all processing sections disabled.

    Parameters
    ----------
    name : str
        Configuration identifier / name.

    Returns
    -------
    Line1DConfig
        A minimal, valid 1D line configuration.
    """
    logger.info("Creating new line config: %s", name)
    return Line1DConfig(
        name=name,
        description="",
        data_loading=Data1DConfig(data_type=Data1DType.CSV),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_config(file_path: Path) -> Tuple[bool, Optional[str]]:
    """Validate a configuration file against its Pydantic schema.

    Attempts to load and validate the file using :func:`load_config`.
    Returns a tuple indicating success or failure with an error message.

    Parameters
    ----------
    file_path : Path
        Path to the YAML configuration file to validate.

    Returns
    -------
    Tuple[bool, Optional[str]]
        ``(True, None)`` if the file is valid, or
        ``(False, error_message)`` if validation fails.
    """
    try:
        load_config(file_path)
        logger.info("Validation passed: %s", file_path)
        return True, None
    except FileNotFoundError as exc:
        msg = f"File not found: {exc}"
        logger.error("Validation failed for %s: %s", file_path, msg)
        return False, msg
    except ValueError as exc:
        msg = str(exc)
        logger.error("Validation failed for %s: %s", file_path, msg)
        return False, msg
    except Exception as exc:
        msg = f"Unexpected error: {exc}"
        logger.error("Validation failed for %s: %s", file_path, msg)
        return False, msg
