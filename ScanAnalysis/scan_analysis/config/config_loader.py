"""
Configuration loader for scan analysis.

Provides utilities for loading and managing experiment analysis configurations
from YAML files. Supports recursive search in subdirectories for organized
config structures. Use environment variable ``SCAN_ANALYSIS_CONFIG_DIR`` or
pass ``config_dir=Path(...)`` per call.

Examples
--------
Basic usage:

    >>> from scan_analysis.config import load_experiment_config
    >>> from geecs_data_utils.config_roots import scan_analysis_config
    >>>
    >>> # Set config directory once (or set SCAN_ANALYSIS_CONFIG_DIR)
    >>> scan_analysis_config.set_base_dir("/path/to/configs/scan_analysis")
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
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import ValidationError

from .analyzer_config_models import (
    Array1DAnalyzerConfig,
    Array2DAnalyzerConfig,
    ExperimentAnalysisConfig,
    IncludeEntry,
    LibraryAnalyzer,
)
from geecs_data_utils.config_roots import scan_analysis_config

logger = logging.getLogger(__name__)

__all__ = [
    "find_config_file",
    "load_experiment_config",
    "list_available_configs",
]

# ----------------------------------------------------------------------
# Base directory management
# ----------------------------------------------------------------------

_CONFIG_MANAGER = scan_analysis_config
_CONFIG_CACHE: dict[str, Path] = (
    _CONFIG_MANAGER.cache
)  # Cache for resolved config paths

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries (updates win)."""
    result = base.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_library_analyzers(base_dir: Path) -> Dict[str, LibraryAnalyzer]:
    """Load analyzer definitions from library/analyzers."""
    lib_dir = base_dir / "library" / "analyzers"
    analyzers: Dict[str, LibraryAnalyzer] = {}
    if not lib_dir.exists():
        raise FileNotFoundError(f"Analyzer library not found: {lib_dir}")

    for path in list(lib_dir.glob("*.yaml")) + list(lib_dir.glob("*.yml")):
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        lib_an = LibraryAnalyzer.model_validate(data)
        analyzers[lib_an.id] = lib_an
    if not analyzers:
        raise ValueError(f"No analyzers found in library: {lib_dir}")
    return analyzers


def _load_groups(base_dir: Path) -> Dict[str, List[str]]:
    """Load group definitions from library/groups.yaml if present."""
    groups_path_yaml = base_dir / "library" / "groups.yaml"
    groups_path_yml = base_dir / "library" / "groups.yml"
    path = None
    if groups_path_yaml.exists():
        path = groups_path_yaml
    elif groups_path_yml.exists():
        path = groups_path_yml

    if not path:
        return {}

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict) or "groups" not in data:
        raise ValueError(f"Invalid groups file (missing 'groups'): {path}")
    groups = data.get("groups", {})
    if not isinstance(groups, dict):
        raise ValueError(f"Invalid groups structure in {path}")
    return groups


def _validate_analyzer(data: Dict[str, Any]):
    """Validate analyzer dict into the correct Pydantic model."""
    analyzer_type = data.get("type")
    if analyzer_type == "array1d":
        return Array1DAnalyzerConfig.model_validate(data)
    if analyzer_type == "array2d":
        return Array2DAnalyzerConfig.model_validate(data)
    raise ValueError(f"Unknown analyzer type '{analyzer_type}' in {data}")


def _resolve_includes(
    include_entries: List[IncludeEntry],
    library: Dict[str, LibraryAnalyzer],
    groups: Dict[str, List[str]],
) -> List[Union[Array1DAnalyzerConfig, Array2DAnalyzerConfig]]:
    """Resolve include directives into concrete analyzer configs."""
    resolved = []
    for entry in include_entries:
        targets: List[str] = []
        if entry.ref:
            targets = [entry.ref]
        elif entry.group:
            if entry.group not in groups:
                raise ValueError(f"Group '{entry.group}' not defined in groups file.")
            targets = groups[entry.group]

        for analyzer_id in targets:
            if analyzer_id not in library:
                raise ValueError(f"Analyzer id '{analyzer_id}' not found in library.")
            base_model = library[analyzer_id].analyzer
            data = base_model.model_dump()
            if entry.overrides:
                data = _deep_merge(data, entry.overrides)

            if entry.priority is not None:
                data["priority"] = entry.priority
            else:
                default_priority = data.get("priority", 100)
                data["priority"] = default_priority + entry.priority_offset

            resolved.append(_validate_analyzer(data))

    # Sort for deterministic ordering (priority ascending, then device/analyzer id)
    resolved.sort(key=lambda a: (a.priority, getattr(a, "device_name", "")))
    return resolved


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
            "Set SCAN_ANALYSIS_CONFIG_DIR or pass config_dir explicitly."
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
    base_dir = config_dir or _CONFIG_MANAGER.base_dir
    if base_dir is None:
        raise ValueError(
            "config_dir is required (no global base dir set). "
            "Set SCAN_ANALYSIS_CONFIG_DIR or pass config_dir explicitly."
        )

    if isinstance(config_source, str):
        path = find_config_file(config_source, config_dir=base_dir)
    else:
        path = Path(config_source)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"Empty or invalid YAML file: {path}")

        # Resolve include directives if present
        include_entries_raw = data.get("include", [])
        if include_entries_raw:
            include_entries = [
                IncludeEntry.model_validate(entry) for entry in include_entries_raw
            ]
            library = _load_library_analyzers(base_dir)
            groups = _load_groups(base_dir)
            resolved_analyzers = _resolve_includes(include_entries, library, groups)
            data["analyzers"] = resolved_analyzers

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
            "Set SCAN_ANALYSIS_CONFIG_DIR or pass config_dir explicitly."
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
