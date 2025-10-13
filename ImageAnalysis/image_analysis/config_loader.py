"""
Configuration loading utilities for image analysis.

Single public entry point:
    load_camera_config(...) -> cfg_2d.CameraConfig

This version uses a single, explicit base directory for YAML configs:
- Set once via `set_config_base_dir("/path/to/image_analysis_configs")`
  or environment variable `IMAGE_ANALYSIS_CONFIG_DIR`.
- Or pass `config_dir=Path(...)` per call.

Supports recursive search in subdirectories for organized config structures.
Supports nested overrides via double-underscore keys, e.g., background__method="mean".
"""

from __future__ import annotations

import os
import logging
import configparser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, TypeVar, Union

import yaml
import numpy as np
from pydantic import BaseModel, ValidationError

# All config models in one namespace
import image_analysis.processing.array2d.config_models as cfg_2d
import image_analysis.processing.array1d.config_models as cfg_1d

if TYPE_CHECKING:
    from .types import Array2D

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

__all__ = [
    "set_config_base_dir",
    "get_config_base_dir",
    "load_camera_config",
    "load_line_config",
    "find_config_file",
    "create_processing_configs",
    "save_config_to_yaml",
    "validate_config_file",
    "get_config_schema",
    "convert_from_processing_dtype",
    "list_available_configs",
    "clear_config_cache",
]

# ----------------------------------------------------------------------
# Base directory management
# ----------------------------------------------------------------------

_DEFAULT_CONFIG_DIR: Optional[Path] = None
_CONFIG_CACHE: Dict[str, Path] = {}  # Cache for resolved config paths


def _resolve_config_dir_from_ini() -> Optional[Path]:
    """
    Resolve config directory from ~/.config/geecs_python_api/config.ini.

    Returns
    -------
    Optional[Path]
        Path to image_analysis configs if found in config file, None otherwise
    """
    config_path = Path("~/.config/geecs_python_api/config.ini").expanduser()
    if not config_path.exists():
        return None

    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        if config_root := config.get("Paths", "config_root", fallback=None):
            if config_root:  # Not empty string
                image_dir = (
                    Path(config_root).expanduser().resolve()
                    / "image_analysis"
                    / "cameras"
                )
                if image_dir.exists():
                    return image_dir
    except Exception as e:
        logger.warning(f"Error reading config from {config_path}: {e}")

    return None


def set_config_base_dir(path: Union[str, Path]) -> None:
    """
    Set a global base directory for camera config files.

    Parameters
    ----------
    path : Union[str, Path]
        Directory containing YAML config files (e.g., "UC_VisaEBeam1.yaml").
        Can contain subdirectories - recursive search is supported.
    """
    global _DEFAULT_CONFIG_DIR
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config base dir does not exist: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"Config base dir is not a directory: {p}")
    _DEFAULT_CONFIG_DIR = p
    clear_config_cache()  # Clear cache when base dir changes
    logger.info("Config base dir set to %s", p)


def get_config_base_dir() -> Optional[Path]:
    """Return the currently configured base directory (or None if unset)."""
    return _DEFAULT_CONFIG_DIR


def clear_config_cache() -> None:
    """Clear the config file path cache. Useful after adding new configs."""
    global _CONFIG_CACHE
    _CONFIG_CACHE.clear()
    logger.info("Config cache cleared")


# Initialize from environment or config file
_env_dir = os.getenv("IMAGE_ANALYSIS_CONFIG_DIR")
if _env_dir:
    try:
        set_config_base_dir(_env_dir)
        logger.info(f"Loaded config dir from IMAGE_ANALYSIS_CONFIG_DIR: {_env_dir}")
    except Exception as e:
        logger.warning("IMAGE_ANALYSIS_CONFIG_DIR invalid: %s", e)
else:
    # Try config file
    config_dir = _resolve_config_dir_from_ini()
    if config_dir:
        try:
            set_config_base_dir(config_dir)
            logger.info(f"Loaded config dir from config.ini: {config_dir}")
        except Exception as e:
            logger.warning(f"Config dir from ini invalid: {e}")

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
    base = config_dir or _DEFAULT_CONFIG_DIR
    if base is None:
        raise ValueError(
            "config_dir is required (no global base dir set). "
            "Call set_config_base_dir(...) or set IMAGE_ANALYSIS_CONFIG_DIR."
        )

    # Check cache first
    cache_key = f"{base}::{camera_name}"
    if use_cache and cache_key in _CONFIG_CACHE:
        cached_path = _CONFIG_CACHE[cache_key]
        if cached_path.exists():
            return cached_path
        else:
            # File was deleted, remove from cache
            del _CONFIG_CACHE[cache_key]

    patterns = [
        f"{camera_name}.yaml",
        f"{camera_name}.yml",
        f"default_{camera_name}_settings.yaml",
        f"default_{camera_name}_settings.yml",
    ]

    # Fast path: check direct children first
    for pat in patterns:
        p = base / pat
        if p.exists():
            logger.info(f"Found config (direct): {p}")
            _CONFIG_CACHE[cache_key] = p
            return p

    # Recursive search
    all_matches = []
    for pat in patterns:
        matches = list(base.rglob(pat))
        all_matches.extend(matches)

    if not all_matches:
        raise FileNotFoundError(
            f"Config '{camera_name}' not found under {base}\n"
            f"Searched recursively for patterns:\n"
            + "\n".join(f"  - {pat}" for pat in patterns)
        )

    # Sort for deterministic behavior
    all_matches.sort()

    if len(all_matches) > 1:
        logger.warning(
            f"Multiple configs found for '{camera_name}':\n"
            + "\n".join(f"  - {m}" for m in all_matches)
            + f"\nUsing: {all_matches[0]}"
        )

    result = all_matches[0]
    logger.info(f"Found config (recursive): {result}")
    _CONFIG_CACHE[cache_key] = result
    return result


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


def load_line_config(
    config_source: Union[str, Path, Dict[str, Any]],
    *,
    config_dir: Optional[Path] = None,
    **overrides: Any,
) -> cfg_1d.Line1DConfig:
    r"""
    Load and validate a Line1DConfig model from name/path/dict, with __ overrides.

    Parameters
    ----------
    config_source : Union[str, Path, Dict[str, Any]]
        - str: line config name (file stem) under the base directory
        - Path: explicit .yaml/.yml file path
        - dict: already-loaded configuration dictionary
    config_dir : Optional[Path]
        Directory containing YAML config files. If omitted, uses the global base dir.
    **overrides : Any
        Nested overrides using double-underscore syntax (e.g., data_loading__data_type="csv").

    Returns
    -------
    cfg_1d.Line1DConfig
        Validated line configuration model.
    """
    data = _load_camera_config_dict(config_source, config_dir=config_dir, **overrides)
    try:
        return cfg_1d.Line1DConfig.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Invalid line configuration: {e}") from e


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


def list_available_configs(
    config_dir: Optional[Path] = None, recursive: bool = True
) -> Dict[str, List[Path]]:
    """
    List available configs, optionally searching recursively.

    Parameters
    ----------
    config_dir : Optional[Path]
        Directory to search. If None, uses global base dir.
    recursive : bool, default=True
        Whether to search subdirectories recursively.

    Returns
    -------
    Dict[str, List[Path]]
        Mapping from config name (stem) to list of paths.
        If multiple configs have same name, all paths are listed.
    """
    base = config_dir or _DEFAULT_CONFIG_DIR
    if base is None:
        raise ValueError(
            "config_dir is required (no global base dir set). "
            "Call set_config_base_dir(...) or set IMAGE_ANALYSIS_CONFIG_DIR."
        )

    configs: Dict[str, List[Path]] = {}

    if recursive:
        for p in base.rglob("*.yaml"):
            configs.setdefault(p.stem, []).append(p)
        for p in base.rglob("*.yml"):
            configs.setdefault(p.stem, []).append(p)
    else:
        for p in base.glob("*.yaml"):
            configs.setdefault(p.stem, []).append(p)
        for p in base.glob("*.yml"):
            configs.setdefault(p.stem, []).append(p)

    return {k: sorted(v) for k, v in sorted(configs.items())}
