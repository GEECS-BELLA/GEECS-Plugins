"""Scan Configuration I/O layer for the Config File Editor GUI.

Provides functions for discovering, listing, loading, saving, and classifying
the three tiers of scan analysis configuration files:

1. **Library Analyzers** — individual ``library/analyzers/*.yaml`` files
2. **Groups** — ``library/groups.yaml`` mapping group names to analyzer ID lists
3. **Experiment Configs** — ``experiments/*.yaml`` composing analyzers via includes

The groups layer requires comment-aware parsing because users enable/disable
analyzers by commenting or uncommenting list entries (``# - AnalyzerName``).
Standard ``yaml.safe_load`` strips comments, so :func:`load_groups_yaml` and
:func:`save_groups_yaml` operate on raw text.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml

from ConfigFileGUI.config_io import sanitize_for_yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directory structure constants
# ---------------------------------------------------------------------------

_EXPERIMENTS_DIR = "experiments"
_LIBRARY_DIR = "library"
_ANALYZERS_DIR = "analyzers"
_GROUPS_FILE = "groups.yaml"


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------


def discover_scan_config_dirs(root: Path) -> Dict[str, Optional[Path]]:
    """Discover the scan-analysis config directory structure under *root*.

    Expects the layout::

        <root>/
            experiments/          ← experiment YAML files
            library/
                analyzers/        ← individual analyzer YAML files
                groups.yaml       ← group definitions

    Parameters
    ----------
    root : Path
        Path to the ``scan_analysis_configs/`` root directory.

    Returns
    -------
    Dict[str, Optional[Path]]
        Dictionary with keys ``"experiments"``, ``"analyzers"``, and
        ``"groups"`` pointing to the respective directories or file.
        A value is ``None`` when the expected path does not exist.

    Raises
    ------
    FileNotFoundError
        If *root* itself does not exist.
    NotADirectoryError
        If *root* is not a directory.
    """
    if not root.exists():
        raise FileNotFoundError(f"Scan config root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    experiments_dir = root / _EXPERIMENTS_DIR
    analyzers_dir = root / _LIBRARY_DIR / _ANALYZERS_DIR
    groups_file = root / _LIBRARY_DIR / _GROUPS_FILE

    result: Dict[str, Optional[Path]] = {
        "experiments": experiments_dir if experiments_dir.is_dir() else None,
        "analyzers": analyzers_dir if analyzers_dir.is_dir() else None,
        "groups": groups_file if groups_file.is_file() else None,
    }

    missing = [k for k, v in result.items() if v is None]
    if missing:
        logger.warning(
            "Missing expected scan config paths in %s: %s",
            root,
            ", ".join(missing),
        )

    return result


# ---------------------------------------------------------------------------
# Listing helpers
# ---------------------------------------------------------------------------


def _list_yaml_files(directory: Path) -> List[Path]:
    """Return a sorted list of ``.yaml`` / ``.yml`` files in *directory*.

    Parameters
    ----------
    directory : Path
        Directory to scan (non-recursively).

    Returns
    -------
    List[Path]
        Sorted list of YAML file paths.

    Raises
    ------
    FileNotFoundError
        If *directory* does not exist.
    NotADirectoryError
        If *directory* exists but is not a directory.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    files = [
        child
        for child in directory.iterdir()
        if child.is_file() and child.suffix.lower() in (".yaml", ".yml")
    ]
    files.sort(key=lambda p: p.name.lower())
    return files


def list_experiment_configs(root: Path) -> List[Path]:
    """List all experiment YAML files under ``<root>/experiments/``.

    Parameters
    ----------
    root : Path
        Path to the ``scan_analysis_configs/`` root directory.

    Returns
    -------
    List[Path]
        Sorted list of experiment config file paths.

    Raises
    ------
    FileNotFoundError
        If the ``experiments/`` subdirectory does not exist.
    """
    experiments_dir = root / _EXPERIMENTS_DIR
    result = _list_yaml_files(experiments_dir)
    logger.debug("Found %d experiment configs in %s", len(result), experiments_dir)
    return result


def list_analyzer_configs(root: Path) -> List[Path]:
    """List all analyzer YAML files under ``<root>/library/analyzers/``.

    Parameters
    ----------
    root : Path
        Path to the ``scan_analysis_configs/`` root directory.

    Returns
    -------
    List[Path]
        Sorted list of analyzer config file paths.

    Raises
    ------
    FileNotFoundError
        If the ``library/analyzers/`` subdirectory does not exist.
    """
    analyzers_dir = root / _LIBRARY_DIR / _ANALYZERS_DIR
    result = _list_yaml_files(analyzers_dir)
    logger.debug("Found %d analyzer configs in %s", len(result), analyzers_dir)
    return result


def get_groups_file(root: Path) -> Optional[Path]:
    """Return the path to ``<root>/library/groups.yaml`` if it exists.

    Parameters
    ----------
    root : Path
        Path to the ``scan_analysis_configs/`` root directory.

    Returns
    -------
    Optional[Path]
        Path to ``groups.yaml``, or ``None`` if the file does not exist.
    """
    groups_path = root / _LIBRARY_DIR / _GROUPS_FILE
    if groups_path.is_file():
        return groups_path
    logger.debug("groups.yaml not found at %s", groups_path)
    return None


# ---------------------------------------------------------------------------
# Groups YAML — comment-aware load / save
# ---------------------------------------------------------------------------

# Matches an enabled entry like "    - Amp2Input"
_ENABLED_RE = re.compile(r"^\s+-\s+(\S+)\s*$")
# Matches a disabled (commented-out) entry like "#    - HasoLift" or "# - HasoLift"
_DISABLED_RE = re.compile(r"^#\s*-\s+(\S+)\s*$")
# Matches a group header like "  baseline:" (indented under `groups:`)
_GROUP_HEADER_RE = re.compile(r"^\s{2}(\S+):\s*$")


def load_groups_yaml(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load ``groups.yaml`` preserving commented-out (disabled) entries.

    Standard ``yaml.safe_load`` discards comments, so this function parses
    the raw text to detect lines like ``# - AnalyzerName`` and treats them as
    disabled members of the enclosing group.

    Parameters
    ----------
    path : Path
        Path to the ``groups.yaml`` file.

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        Mapping of group name to a list of member dicts.  Each member dict
        has the keys:

        * ``"id"`` (``str``) — the analyzer identifier
        * ``"enabled"`` (``bool``) — ``True`` if the entry is uncommented

        Example::

            {
                "baseline": [
                    {"id": "Amp2Input", "enabled": True},
                    {"id": "Amp4Output", "enabled": False},
                ],
                ...
            }

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Groups file not found: {path}")

    text = path.read_text(encoding="utf-8")

    groups: Dict[str, List[Dict[str, Any]]] = {}
    current_group: Optional[str] = None
    in_groups_block = False

    for line in text.splitlines():
        stripped = line.strip()

        # Detect the top-level `groups:` key
        if stripped == "groups:":
            in_groups_block = True
            continue

        if not in_groups_block:
            continue

        # Detect a group header (indented key under `groups:`)
        group_match = _GROUP_HEADER_RE.match(line)
        if group_match:
            current_group = group_match.group(1)
            groups[current_group] = []
            continue

        if current_group is None:
            continue

        # Check for an enabled entry
        enabled_match = _ENABLED_RE.match(line)
        if enabled_match:
            groups[current_group].append(
                {"id": enabled_match.group(1), "enabled": True}
            )
            continue

        # Check for a disabled (commented-out) entry
        disabled_match = _DISABLED_RE.match(line)
        if disabled_match:
            groups[current_group].append(
                {"id": disabled_match.group(1), "enabled": False}
            )
            continue

    logger.debug("Loaded groups.yaml with %d groups from %s", len(groups), path)
    return groups


def save_groups_yaml(path: Path, groups: Dict[str, List[Dict[str, Any]]]) -> None:
    """Save groups back to YAML, converting disabled entries to comments.

    Enabled entries are written as ``    - AnalyzerId`` and disabled entries
    as ``#    - AnalyzerId``, preserving the convention used by the scan
    analysis tooling.

    Parameters
    ----------
    path : Path
        Destination file path (will be created or overwritten).
    groups : Dict[str, List[Dict[str, Any]]]
        Group data in the same format returned by :func:`load_groups_yaml`.
    """
    lines: List[str] = ["groups:"]

    for group_name, members in groups.items():
        lines.append(f"  {group_name}:")
        for member in members:
            analyzer_id = member["id"]
            enabled = member.get("enabled", True)
            if enabled:
                lines.append(f"    - {analyzer_id}")
            else:
                lines.append(f"#    - {analyzer_id}")

    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    logger.info("Saved groups.yaml to %s", path)


# ---------------------------------------------------------------------------
# Analyzer YAML — load / save
# ---------------------------------------------------------------------------


def load_analyzer_yaml(path: Path) -> dict:
    """Load a single library analyzer YAML file.

    Parameters
    ----------
    path : Path
        Path to the analyzer ``.yaml`` file.

    Returns
    -------
    dict
        Raw dictionary parsed from the YAML file.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    yaml.YAMLError
        If the file contains invalid YAML.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Analyzer config not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        logger.warning("Analyzer YAML is not a mapping: %s", path)
        return {}

    logger.debug("Loaded analyzer config: %s", path)
    return data


def save_analyzer_yaml(path: Path, data: dict) -> None:
    """Save an analyzer configuration dict to a YAML file.

    The data is sanitized via :func:`~ConfigFileGUI.config_io.sanitize_for_yaml`
    before writing to ensure all values are YAML-safe primitives.

    Parameters
    ----------
    path : Path
        Destination file path (will be created or overwritten).
    data : dict
        Analyzer configuration dictionary.
    """
    sanitized = sanitize_for_yaml(data)
    text = yaml.safe_dump(sanitized, default_flow_style=False, sort_keys=False)
    path.write_text(text, encoding="utf-8")
    logger.info("Saved analyzer config to %s", path)


# ---------------------------------------------------------------------------
# Experiment YAML — load / save
# ---------------------------------------------------------------------------


def load_experiment_yaml(path: Path) -> dict:
    """Load an experiment configuration YAML file.

    Parameters
    ----------
    path : Path
        Path to the experiment ``.yaml`` file.

    Returns
    -------
    dict
        Raw dictionary parsed from the YAML file.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    yaml.YAMLError
        If the file contains invalid YAML.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        logger.warning("Experiment YAML is not a mapping: %s", path)
        return {}

    logger.debug("Loaded experiment config: %s", path)
    return data


def save_experiment_yaml(path: Path, data: dict) -> None:
    """Save an experiment configuration dict to a YAML file.

    The data is sanitized via :func:`~ConfigFileGUI.config_io.sanitize_for_yaml`
    before writing to ensure all values are YAML-safe primitives.

    Parameters
    ----------
    path : Path
        Destination file path (will be created or overwritten).
    data : dict
        Experiment configuration dictionary.
    """
    sanitized = sanitize_for_yaml(data)
    text = yaml.safe_dump(sanitized, default_flow_style=False, sort_keys=False)
    path.write_text(text, encoding="utf-8")
    logger.info("Saved experiment config to %s", path)


# ---------------------------------------------------------------------------
# Config type detection
# ---------------------------------------------------------------------------


def detect_scan_config_type(
    path: Path,
) -> Literal["experiment", "analyzer", "groups", "unknown"]:
    """Determine the scan config type of a YAML file based on its location.

    The detection uses the file's position in the directory hierarchy:

    * Files under ``experiments/`` → ``"experiment"``
    * Files under ``library/analyzers/`` → ``"analyzer"``
    * A file named ``groups.yaml`` under ``library/`` → ``"groups"``

    If location-based detection fails, a content-based fallback inspects
    top-level YAML keys:

    * ``experiment`` + ``include`` → ``"experiment"``
    * ``id`` + ``type`` + ``device_name`` → ``"analyzer"``
    * ``groups`` → ``"groups"``

    Parameters
    ----------
    path : Path
        Path to the YAML configuration file.

    Returns
    -------
    Literal["experiment", "analyzer", "groups", "unknown"]
        Detected configuration type.
    """
    # --- Location-based detection ---
    parts = path.resolve().parts

    # Check for experiments/ in path
    if _EXPERIMENTS_DIR in parts:
        return "experiment"

    # Check for library/analyzers/ in path
    if _LIBRARY_DIR in parts and _ANALYZERS_DIR in parts:
        return "analyzer"

    # Check for library/groups.yaml
    if _LIBRARY_DIR in parts and path.name == _GROUPS_FILE:
        return "groups"

    # --- Content-based fallback ---
    logger.debug("Location-based detection inconclusive for %s; trying content", path)
    try:
        if not path.is_file():
            return "unknown"
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            return "unknown"

        if "experiment" in data and "include" in data:
            return "experiment"
        if "id" in data and "type" in data and "device_name" in data:
            return "analyzer"
        if "groups" in data:
            return "groups"
    except (yaml.YAMLError, OSError) as exc:
        logger.warning("Failed to read %s for type detection: %s", path, exc)

    return "unknown"


# ---------------------------------------------------------------------------
# Analyzer ID enumeration
# ---------------------------------------------------------------------------


def list_all_analyzer_ids(root: Path) -> List[str]:
    """Return a sorted list of analyzer IDs derived from filenames.

    Uses the filename stem (without ``.yaml``) of each file in
    ``<root>/library/analyzers/`` as the canonical analyzer ID.  This
    matches what the user sees in the tree panel and what
    ``groups.yaml`` references, so the autocomplete is always
    up-to-date even when internal ``id`` fields are stale.

    Parameters
    ----------
    root : Path
        Path to the ``scan_analysis_configs/`` root directory.

    Returns
    -------
    List[str]
        Alphabetically sorted list of unique analyzer IDs.
    """
    try:
        analyzer_files = list_analyzer_configs(root)
    except (FileNotFoundError, NotADirectoryError) as exc:
        logger.warning("Cannot list analyzer configs: %s", exc)
        return []

    ids = sorted({f.stem for f in analyzer_files}, key=str.lower)
    logger.debug("Found %d unique analyzer IDs", len(ids))
    return ids
