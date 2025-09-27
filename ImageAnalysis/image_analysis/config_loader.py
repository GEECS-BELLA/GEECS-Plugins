"""
Configuration loading utilities for image analysis.

This module provides functions for loading and validating Pydantic configuration
models from YAML files, with support for camera-specific configurations and
configuration inheritance.
"""

from __future__ import annotations

import yaml
import logging
from pathlib import Path

from typing import TYPE_CHECKING, Type, TypeVar, Dict, Any, Optional, List, Union

if TYPE_CHECKING:
    from .types import Array2D

from pydantic import BaseModel, ValidationError
import numpy as np


# Import all config models for loading
from .processing.config_models import (
    BackgroundConfig,
    CrosshairMaskingConfig,
    ROIConfig,
    FilteringConfig,
    TransformConfig,
    CircularMaskConfig,
    CameraConfig,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def load_config_from_yaml(config_path: Path, config_class: Type[T]) -> T:
    """
    Load and validate configuration from YAML file.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.
    config_class : Type[T]
        Pydantic model class to validate the configuration against.

    Returns
    -------
    T
        Validated configuration instance.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the configuration is invalid or cannot be parsed.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"Empty or invalid YAML file: {config_path}")

        return config_class(**data)

    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {config_path}: {e}")
    except ValidationError as e:
        raise ValueError(f"Invalid configuration in {config_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error loading {config_path}: {e}")


def load_config_from_dict(config_data: Dict[str, Any], config_class: Type[T]) -> T:
    """
    Load and validate configuration from dictionary.

    Parameters
    ----------
    config_data : Dict[str, Any]
        Configuration data as dictionary.
    config_class : Type[T]
        Pydantic model class to validate the configuration against.

    Returns
    -------
    T
        Validated configuration instance.

    Raises
    ------
    ValueError
        If the configuration is invalid.
    """
    try:
        return config_class(**config_data)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration data: {e}")


def get_config_search_paths(config_dir: Optional[Path] = None) -> List[Path]:
    """
    Get list of directories to search for configuration files.

    This function provides robust path resolution that works regardless of
    the current working directory by searching in multiple locations.

    Parameters
    ----------
    config_dir : Optional[Path]
        Specific directory to search. If provided, only this directory is used.

    Returns
    -------
    List[Path]
        List of directories to search, in order of preference.
    """
    if config_dir is not None:
        return [config_dir]

    search_paths = []

    # 1. Try relative to current working directory
    search_paths.append(Path.cwd() / "image_analysis_configs")

    # 2. Try to find project root by looking for pyproject.toml or other markers
    current_dir = Path.cwd()
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            # Found project root
            search_paths.append(parent / "image_analysis_configs")
            break

    # 3. Try relative to this module's location
    module_dir = Path(__file__).parent
    for parent in [module_dir] + list(module_dir.parents):
        potential_config_dir = parent / "image_analysis_configs"
        if potential_config_dir.exists():
            search_paths.append(potential_config_dir)
            break

    # 4. Try inside ImageAnalysis package
    search_paths.append(module_dir / "image_analysis_configs")

    # Remove duplicates while preserving order
    unique_paths = []
    for path in search_paths:
        if path not in unique_paths:
            unique_paths.append(path)

    return unique_paths


def find_config_file(camera_name: str, config_dir: Optional[Path] = None) -> Path:
    """
    Find configuration file for a camera, searching multiple locations.

    Parameters
    ----------
    camera_name : str
        Name of the camera configuration to find.
    config_dir : Optional[Path]
        Specific directory to search. If None, searches multiple locations.

    Returns
    -------
    Path
        Path to the found configuration file.

    Raises
    ------
    FileNotFoundError
        If configuration file is not found in any search location.
    """
    search_paths = get_config_search_paths(config_dir)

    # Try different filename patterns
    filename_patterns = [
        f"{camera_name}.yaml",
        f"{camera_name}.yml",
        f"default_{camera_name}_settings.yaml",
        f"default_{camera_name}_settings.yml",
    ]

    attempted_paths = []

    for search_dir in search_paths:
        if not search_dir.exists():
            continue

        for pattern in filename_patterns:
            config_path = search_dir / pattern
            attempted_paths.append(str(config_path))

            if config_path.exists():
                logger.info(f"Found configuration file: {config_path}")
                return config_path

    # If we get here, file was not found
    error_msg = f"Config file not found for camera '{camera_name}'. Searched in:\n"
    for path in attempted_paths:
        error_msg += f"  - {path}\n"

    # Also check for .ini files and provide helpful message
    for search_dir in search_paths:
        if not search_dir.exists():
            continue
        ini_path = search_dir / f"default_{camera_name}_settings.ini"
        if ini_path.exists():
            error_msg += (
                f"\nFound .ini file at {ini_path}, but YAML format is required. "
            )
            error_msg += "Consider converting to YAML format."
            break

    raise FileNotFoundError(error_msg)


def load_camera_config(
    config_source: Union[str, Path, Dict[str, Any]],
    config_dir: Optional[Path] = None,
    **overrides,
) -> Dict[str, Any]:
    """
    Load camera configuration with support for multiple sources and overrides.

    This function supports loading configurations from:
    1. Camera name (looks for {camera_name}.yaml in multiple directories)
    2. File path (loads specific YAML file)
    3. Dictionary (uses data directly)

    The function uses smart path resolution that works regardless of the current
    working directory by searching in multiple locations including:
    - Current working directory
    - Project root (detected by pyproject.toml or .git)
    - Relative to the ImageAnalysis package
    - Inside the ImageAnalysis package

    Parameters
    ----------
    config_source : Union[str, Path, Dict[str, Any]]
        Configuration source. Can be:
        - String: camera name to look up in config directories
        - Path: path to specific configuration file
        - Dict: configuration data directly
    config_dir : Optional[Path]
        Specific directory to search for camera configuration files.
        If None, searches multiple standard locations.
    **overrides
        Additional configuration parameters to override.

    Returns
    -------
    Dict[str, Any]
        Loaded and merged configuration data.

    Raises
    ------
    FileNotFoundError
        If configuration file is not found.
    ValueError
        If configuration is invalid.
    """
    # Load base configuration data
    if isinstance(config_source, str):
        # Treat as camera name - use smart path resolution
        config_path = find_config_file(config_source, config_dir)

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        logger.info(f"Loaded camera configuration from {config_path}")

    elif isinstance(config_source, Path):
        # Load from specific file path
        if not config_source.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_source}")

        with open(config_source, "r") as f:
            config_data = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {config_source}")

    elif isinstance(config_source, dict):
        # Use dictionary directly
        config_data = config_source.copy()
        logger.info("Using provided configuration dictionary")

    else:
        raise ValueError(f"Invalid config_source type: {type(config_source)}")

    if config_data is None:
        config_data = {}

    # Apply overrides using nested key syntax
    config_data = _apply_nested_overrides(config_data, overrides)

    return config_data


def create_processing_configs(config_data: Dict[str, Any]) -> Dict[str, BaseModel]:
    """
    Create validated processing configuration objects from configuration data.

    Parameters
    ----------
    config_data : Dict[str, Any]
        Configuration data containing processing parameters.

    Returns
    -------
    Dict[str, BaseModel]
        Dictionary of validated configuration objects keyed by processing type.
    """
    configs = {}

    # Map configuration keys to their corresponding Pydantic models
    config_mapping = {
        "background": BackgroundConfig,
        "crosshair_masking": CrosshairMaskingConfig,
        "roi": ROIConfig,
        "filtering": FilteringConfig,
        "transforms": TransformConfig,
        "circular_mask": CircularMaskConfig,
    }

    for key, config_class in config_mapping.items():
        if key in config_data:
            try:
                configs[key] = config_class(**config_data[key])
                logger.debug(f"Created {key} configuration")
            except ValidationError as e:
                logger.error(f"Invalid {key} configuration: {e}")
                raise ValueError(f"Invalid {key} configuration: {e}")

    return configs


def save_config_to_yaml(config: BaseModel, output_path: Path) -> None:
    """
    Save configuration object to YAML file.

    Parameters
    ----------
    config : BaseModel
        Pydantic configuration object to save.
    output_path : Path
        Path where to save the YAML file.
    """
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dictionary and save as YAML
    config_dict = config.model_dump()

    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    logger.info(f"Saved configuration to {output_path}")


def validate_config_file(config_path: Path, config_class: Type[T]) -> bool:
    """
    Validate a configuration file without loading it.

    Parameters
    ----------
    config_path : Path
        Path to the configuration file to validate.
    config_class : Type[T]
        Pydantic model class to validate against.

    Returns
    -------
    bool
        True if configuration is valid, False otherwise.
    """
    try:
        load_config_from_yaml(config_path, config_class)
        return True
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


def get_config_schema(config_class: Type[BaseModel]) -> Dict[str, Any]:
    """
    Get JSON schema for a configuration class.

    Parameters
    ----------
    config_class : Type[BaseModel]
        Pydantic model class to get schema for.

    Returns
    -------
    Dict[str, Any]
        JSON schema dictionary.
    """
    return config_class.model_json_schema()


def _apply_nested_overrides(
    config_data: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply nested configuration overrides using double underscore syntax.

    Parameters
    ----------
    config_data : Dict[str, Any]
        Base configuration data.
    overrides : Dict[str, Any]
        Override parameters using nested key syntax (e.g., "background__method").

    Returns
    -------
    Dict[str, Any]
        Configuration data with overrides applied.
    """
    result = config_data.copy()

    for key, value in overrides.items():
        if "__" in key:
            # Handle nested keys
            parts = key.split("__")
            current = result

            # Navigate to the nested dictionary
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set the final value
            current[parts[-1]] = value
        else:
            # Handle top-level keys
            result[key] = value

    return result


def load_camera_config_model(
    config_source: Union[str, Path, Dict[str, Any]],
    config_dir: Optional[Path] = None,
    **overrides,
) -> CameraConfig:
    """
    Load and validate complete camera configuration using CameraConfig model.

    Parameters
    ----------
    config_source : Union[str, Path, Dict[str, Any]]
        Configuration source (camera name, file path, or dictionary).
    config_dir : Optional[Path]
        Directory to search for camera configuration files.
    **overrides
        Additional configuration parameters to override.

    Returns
    -------
    CameraConfig
        Validated camera configuration object.
    """
    config_data = load_camera_config(config_source, config_dir, **overrides)

    try:
        return CameraConfig(**config_data)
    except ValidationError as e:
        raise ValueError(f"Invalid camera configuration: {e}")


def convert_from_processing_dtype(image: "Array2D", target_dtype: str) -> "Array2D":
    """
    Convert image from processing dtype (float64) back to target dtype.

    Parameters
    ----------
    image : Array2D
        Processed image in float64 format.
    target_dtype : str
        Target dtype for output ('uint8', 'uint16', 'uint32', or 'float64').

    Returns
    -------
    Array2D
        Image converted to target dtype with appropriate clipping.
    """
    if target_dtype == "float64":
        return image
    elif target_dtype == "uint8":
        return np.clip(image, 0, 255).astype(np.uint8)
    elif target_dtype == "uint16":
        return np.clip(image, 0, 65535).astype(np.uint16)
    elif target_dtype == "uint32":
        return np.clip(image, 0, 4294967295).astype(np.uint32)
    else:
        raise ValueError(f"Unsupported target dtype: {target_dtype}")


def list_available_configs(config_dir: Optional[Path] = None) -> List[str]:
    """
    List available camera configuration files using smart path resolution.

    Parameters
    ----------
    config_dir : Optional[Path]
        Specific directory to search for configuration files.
        If None, searches multiple standard locations.

    Returns
    -------
    List[str]
        List of available camera configuration names (without file extensions).
    """
    search_paths = get_config_search_paths(config_dir)

    config_files = set()  # Use set to avoid duplicates

    for search_dir in search_paths:
        if not search_dir.exists():
            continue

        # Look for YAML files
        for yaml_file in search_dir.glob("*.yaml"):
            config_files.add(yaml_file.stem)

        # Also check for .yml extension
        for yml_file in search_dir.glob("*.yml"):
            config_files.add(yml_file.stem)

    return sorted(list(config_files))
