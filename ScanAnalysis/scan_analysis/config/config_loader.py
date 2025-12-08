"""
Configuration loader for scan analysis.

Provides utilities for loading and managing experiment analysis configurations
from YAML files. Follows the same patterns as ImageAnalysis config_loader.

Single public entry point:
    load_experiment_config(...) -> ExperimentAnalysisConfig

Configuration Directory Management:
- Set once via `set_config_base_dir("/path/to/scan_analysis_configs")`
  or environment variable `SCAN_ANALYSIS_CONFIG_DIR`.
- Or pass `config_dir=Path(...)` per call.

Supports recursive search in subdirectories for organized config structures.

Examples
--------
Basic usage:

    >>> from scan_analysis.config import load_experiment_config, set_config_base_dir
    >>>
    >>> # Set config directory once
    >>> set_config_base_dir("/path/to/configs/scan_analysis")
    >>>
    >>> # Load experiment config
    >>> config = load_experiment_config("undulator")
    >>>
    >>> # Get analyzers sorted by priority
    >>> analyzers = config.get_analyzers_by_priority()
    >>> print(f"Found {len(analyzers)} active analyzers")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import ValidationError

from .analyzer_config_models import ExperimentAnalysisConfig
from geecs_data_utils.config_base import ConfigDirManager

logger = logging.getLogger(__name__)

__all__ = [
    "set_config_base_dir",
    "get_config_base_dir",
    "find_config_file",
    "load_experiment_config",
    "list_available_configs",
    "clear_config_cache",
]

# ----------------------------------------------------------------------
# Base directory management
# ----------------------------------------------------------------------

_CONFIG_MANAGER = ConfigDirManager(
    env_var="SCAN_ANALYSIS_CONFIG_DIR",
    logger=logger,
    name="Scan analysis config",
)
_CONFIG_CACHE: dict[str, Path] = (
    _CONFIG_MANAGER.cache
)  # Cache for resolved config paths


def set_config_base_dir(path: Union[str, Path]) -> None:
    """
    Set a global base directory for scan analysis config files.

    Parameters
    ----------
    path : Union[str, Path]
        Directory containing YAML config files (e.g., "undulator_analysis.yaml").
        Can contain subdirectories - recursive search is supported.

    Raises
    ------
    FileNotFoundError
        If the specified directory does not exist
    NotADirectoryError
        If the specified path is not a directory
    """
    _CONFIG_MANAGER.set_base_dir(path)


def get_config_base_dir() -> Optional[Path]:
    """
    Return the currently configured base directory.

    Returns
    -------
    Optional[Path]
        Current config base directory, or None if not set
    """
    return _CONFIG_MANAGER.base_dir


def clear_config_cache() -> None:
    """Clear the config file path cache. Useful after adding new configs."""
    _CONFIG_MANAGER.clear_cache()


_CONFIG_MANAGER.bootstrap_from_env_or_fallback()


# ----------------------------------------------------------------------
# Config file discovery
# ----------------------------------------------------------------------


def find_config_file(
    experiment_name: str, *, config_dir: Optional[Path] = None, use_cache: bool = True
) -> Path:
    """
    Resolve config file by experiment name, searching recursively if needed.

    Search order:
    1. Check cache (if enabled)
    2. Direct children of base directory (fast path)
    3. Recursive search in subdirectories

    Searches for patterns:
    - {experiment}_analysis.yaml
    - {experiment}.yaml
    - {experiment}_analysis.yml
    - {experiment}.yml

    Parameters
    ----------
    experiment_name : str
        Name of the experiment (e.g., "undulator", "HTU")
    config_dir : Optional[Path]
        Directory containing YAML config files. If None, uses the global base dir.
    use_cache : bool, default=True
        Whether to use cached paths for performance

    Returns
    -------
    Path
        Path to the YAML configuration file

    Raises
    ------
    ValueError
        If no base directory is available
    FileNotFoundError
        If the file is not found under the base directory

    Notes
    -----
    If multiple configs with the same name exist in different subdirectories,
    the first one found (alphabetically) is used and a warning is logged.
    """
    return _CONFIG_MANAGER.find_config(
        experiment_name,
        patterns=[
            "{name}_analysis.yaml",
            "{name}.yaml",
            "{name}_analysis.yml",
            "{name}.yml",
        ],
        config_dir=config_dir,
        use_cache=use_cache,
        missing_base_message=(
            "config_dir is required (no global base dir set). "
            "Call set_config_base_dir(...) or set SCAN_ANALYSIS_CONFIG_DIR."
        ),
        not_found_label="Config for experiment",
    )


# ----------------------------------------------------------------------
# Config loading
# ----------------------------------------------------------------------


def load_experiment_config(
    config_source: Union[str, Path], *, config_dir: Optional[Path] = None
) -> ExperimentAnalysisConfig:
    """
    Load and validate an experiment analysis configuration.

    Parameters
    ----------
    config_source : Union[str, Path]
        - str: experiment name (searches for config file)
        - Path: explicit path to YAML file
    config_dir : Optional[Path]
        Directory containing config files. If omitted, uses the global base dir.

    Returns
    -------
    ExperimentAnalysisConfig
        Validated experiment analysis configuration

    Raises
    ------
    FileNotFoundError
        If the config file cannot be found
    ValueError
        If the config file is invalid (YAML syntax or validation errors)

    Examples
    --------
    Load by experiment name:

        >>> config = load_experiment_config("undulator")

    Load by explicit path:

        >>> from pathlib import Path
        >>> config = load_experiment_config(Path("/path/to/config.yaml"))

    Load with custom config directory:

        >>> config = load_experiment_config(
        ...     "undulator",
        ...     config_dir=Path("/custom/configs")
        ... )
    """
    if isinstance(config_source, str):
        path = find_config_file(config_source, config_dir=config_dir)
    else:
        path = Path(config_source)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"Empty or invalid YAML file: {path}")

        logger.info(f"Loaded experiment configuration from {path}")
        return ExperimentAnalysisConfig.model_validate(data)

    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {path}: {e}") from e
    except ValidationError as e:
        raise ValueError(f"Invalid configuration in {path}: {e}") from e
    except Exception as e:
        raise ValueError(f"Error loading {path}: {e}") from e


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------


def list_available_configs(
    config_dir: Optional[Path] = None, recursive: bool = True
) -> dict[str, list[Path]]:
    """
    List available experiment configs, optionally searching recursively.

    Parameters
    ----------
    config_dir : Optional[Path]
        Directory to search. If None, uses global base dir.
    recursive : bool, default=True
        Whether to search subdirectories recursively

    Returns
    -------
    dict[str, list[Path]]
        Mapping from config name (stem) to list of paths.
        If multiple configs have same name, all paths are listed.

    Raises
    ------
    ValueError
        If no config directory is available

    Examples
    --------
    >>> configs = list_available_configs()
    >>> for name, paths in configs.items():
    ...     print(f"{name}: {paths[0]}")
    """
    base = config_dir or _CONFIG_MANAGER.base_dir
    if base is None:
        raise ValueError(
            "config_dir is required (no global base dir set). "
            "Call set_config_base_dir(...) or set SCAN_ANALYSIS_CONFIG_DIR."
        )

    configs: dict[str, list[Path]] = {}

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
