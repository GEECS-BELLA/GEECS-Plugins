"""
Configuration loading utilities for image analysis.

Single public entry point:
    load_camera_config(...) -> cfg_2d.CameraConfig

This version uses a single, explicit base directory for YAML configs:
- Set once via `set_config_base_dir("/path/to/image_analysis_configs")`
  or environment variable `IMAGE_ANALYSIS_CONFIG_DIR`.
- Or pass `config_dir=Path(...)` per call.

No multi-location "search"; resolution is deterministic.
Supports nested overrides via double-underscore keys, e.g., background__method="mean".
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, TypeVar, Union

import yaml
import numpy as np
from pydantic import BaseModel, ValidationError

# All config models in one namespace
import image_analysis.processing.array2d.config_models as cfg_2d

if TYPE_CHECKING:
    from .types import Array2D

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

__all__ = [
    "set_config_base_dir",
    "get_config_base_dir",
    "load_camera_config",
    "find_config_file",
    "create_processing_configs",
    "save_config_to_yaml",
    "validate_config_file",
    "get_config_schema",
    "convert_from_processing_dtype",
    "list_available_configs",
]

# ----------------------------------------------------------------------
# Base directory management (Option B)
# ----------------------------------------------------------------------

_DEFAULT_CONFIG_DIR: Optional[Path] = None


def set_config_base_dir(path: Union[str, Path]) -> None:
    """
    Set a global base directory for camera config files.

    Parameters
    ----------
    path : Union[str, Path]
        Directory containing YAML config files (e.g., "UC_VisaEBeam1.yaml").
    """
    global _DEFAULT_CONFIG_DIR
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config base dir does not exist: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"Config base dir is not a directory: {p}")
    _DEFAULT_CONFIG_DIR = p
    logger.info("Config base dir set to %s", p)


def get_config_base_dir() -> Optional[Path]:
    """Return the currently configured base directory (or None if unset)."""
    return _DEFAULT_CONFIG_DIR


# Initialize from environment, if present
_env_dir = os.getenv("IMAGE_ANALYSIS_CONFIG_DIR")
if _env_dir:
    try:
        set_config_base_dir(_env_dir)
    except Exception as e:
        logger.warning("IMAGE_ANALYSIS_CONFIG_DIR invalid: %s", e)

# ----------------------------------------------------------------------
# Core helpers
# ----------------------------------------------------------------------


def _apply_nested_overrides(
    data: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply nested overrides using double-underscore syntax."""
    result = data.copy()
    for key, value in overrides.items():
        if "__" in key:
            parts = key.split("__")
            cur = result
            for part in parts[:-1]:
                if part not in cur or not isinstance(cur[part], dict):
                    cur[part] = {}
                cur = cur[part]
            cur[parts[-1]] = value
        else:
            result[key] = value
    return result


def find_config_file(camera_name: str, *, config_dir: Optional[Path] = None) -> Path:
    r"""
    Resolve `<base>/<camera_name>.{yaml|yml}` (and default_* variants) deterministically.

    Parameters
    ----------
    camera_name : str
        Logical name of the camera configuration (file stem).
    config_dir : Optional[Path]
        Directory containing YAML config files. If None, uses the global base dir.

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
    """
    base = config_dir or _DEFAULT_CONFIG_DIR
    if base is None:
        raise ValueError(
            "config_dir is required (no global base dir set). "
            "Call set_config_base_dir(...) or set IMAGE_ANALYSIS_CONFIG_DIR."
        )

    patterns = [
        f"{camera_name}.yaml",
        f"{camera_name}.yml",
        f"default_{camera_name}_settings.yaml",
        f"default_{camera_name}_settings.yml",
    ]
    tried: List[str] = []
    for pat in patterns:
        p = base / pat
        tried.append(str(p))
        if p.exists():
            logger.info("Found configuration file: %s", p)
            return p

    raise FileNotFoundError(
        "Config not found. Tried:\n" + "\n".join(f"  - {t}" for t in tried)
    )


def _load_camera_config_dict(
    config_source: Union[str, Path, Dict[str, Any]],
    *,
    config_dir: Optional[Path],
    **overrides: Any,
) -> Dict[str, Any]:
    r"""Internal: load raw config dict from name/path/dict and apply __ overrides."""
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

    return _apply_nested_overrides(data, overrides)


# ----------------------------------------------------------------------
# Public entry point (model-first)
# ----------------------------------------------------------------------


def load_camera_config(
    config_source: Union[str, Path, Dict[str, Any]],
    *,
    config_dir: Optional[Path] = None,
    **overrides: Any,
) -> cfg_2d.CameraConfig:
    r"""
    Load and validate a CameraConfig model from name/path/dict, with __ overrides.

    Parameters
    ----------
    config_source : Union[str, Path, Dict[str, Any]]
        - str: camera name (file stem) under the base directory
        - Path: explicit .yaml/.yml file path
        - dict: already-loaded configuration dictionary
    config_dir : Optional[Path]
        Directory containing YAML config files. If omitted, uses the global base dir.
    **overrides : Any
        Nested overrides using double-underscore syntax (e.g., background__method="mean").

    Returns
    -------
    cfg_2d.CameraConfig
        Validated camera configuration model.
    """
    data = _load_camera_config_dict(config_source, config_dir=config_dir, **overrides)
    try:
        return cfg_2d.CameraConfig.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Invalid camera configuration: {e}") from e


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------


def create_processing_configs(config_data: Dict[str, Any]) -> Dict[str, BaseModel]:
    """Create validated processing sub-configs from a dict (useful outside CameraConfig)."""
    mapping: Dict[str, Type[BaseModel]] = {
        "background": cfg_2d.BackgroundConfig,
        "crosshair_masking": cfg_2d.CrosshairMaskingConfig,
        "roi": cfg_2d.ROIConfig,
        "filtering": cfg_2d.FilteringConfig,
        "transforms": cfg_2d.TransformConfig,
        "circular_mask": cfg_2d.CircularMaskConfig,
        "thresholding": cfg_2d.ThresholdingConfig,
    }
    out: Dict[str, BaseModel] = {}
    for key, model_cls in mapping.items():
        if key in config_data:
            try:
                out[key] = model_cls.model_validate(config_data[key])
                logger.debug("Created %s configuration", key)
            except ValidationError as e:
                logger.error("Invalid %s configuration: %s", key, e)
                raise ValueError(f"Invalid {key} configuration: {e}") from e
    return out


def save_config_to_yaml(config: BaseModel, output_path: Path) -> None:
    """Save a Pydantic configuration object to YAML."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, indent=2)
    logger.info("Saved configuration to %s", output_path)


def validate_config_file(config_path: Path, config_class: Type[T]) -> bool:
    """Validate a configuration file without loading it into your app."""
    try:
        _ = load_config_from_yaml(config_path, config_class)
        return True
    except (FileNotFoundError, ValueError) as e:
        logger.error("Configuration validation failed: %s", e)
        return False


def load_config_from_yaml(config_path: Path, config_class: Type[T]) -> T:
    """Load and validate configuration from a YAML file into a Pydantic model."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        if data is None:
            raise ValueError(f"Empty or invalid YAML file: {config_path}")
        return config_class.model_validate(data)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {config_path}: {e}") from e
    except ValidationError as e:
        raise ValueError(f"Invalid configuration in {config_path}: {e}") from e
    except Exception as e:
        raise ValueError(f"Error loading {config_path}: {e}") from e


def get_config_schema(config_class: Type[BaseModel]) -> Dict[str, Any]:
    """Return the JSON schema for a configuration class."""
    return config_class.model_json_schema()


def convert_from_processing_dtype(image: "Array2D", target_dtype: str) -> "Array2D":
    """Convert image from float64 processing dtype to target dtype."""
    if target_dtype == "float64":
        return image
    if target_dtype == "uint8":
        return np.clip(image, 0, 255).astype(np.uint8)
    if target_dtype == "uint16":
        return np.clip(image, 0, 65535).astype(np.uint16)
    if target_dtype == "uint32":
        return np.clip(image, 0, 4294967295).astype(np.uint32)
    raise ValueError(f"Unsupported target dtype: {target_dtype}")


def list_available_configs(config_dir: Optional[Path] = None) -> List[str]:
    """List YAML config filenames (stems) in the base directory (no recursion)."""
    base = config_dir or _DEFAULT_CONFIG_DIR
    if base is None:
        raise ValueError(
            "config_dir is required (no global base dir set). "
            "Call set_config_base_dir(...) or set IMAGE_ANALYSIS_CONFIG_DIR."
        )

    names: set[str] = set()
    for p in base.glob("*.yaml"):
        names.add(p.stem)
    for p in base.glob("*.yml"):
        names.add(p.stem)
    return sorted(names)
