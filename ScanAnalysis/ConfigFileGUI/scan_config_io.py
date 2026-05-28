"""Scan Configuration I/O layer for the Config File Editor GUI.

Provides functions for discovering, listing, loading, saving, and
classifying the two tiers of scan-analysis configuration files under
the post-PR-E unified-diagnostic layout::

    <scan_analysis_configs>/
        analyzers/
            HTU/
                UC_GaiaMode.yaml
                ...
            HTT/
                ...
            PW/
                ...
        groups/
            HTU/
                baseline.yaml
                full.yaml
            HTT/
                ...

* **Analyzer YAMLs** validate as
  :class:`image_analysis.config.DiagnosticAnalysisConfig` — one unified
  diagnostic per device (image + scan sections fused).
* **Group YAMLs** validate as
  :class:`scan_analysis.config.diagnostic_models.AnalysisGroupConfig`
  — one file per group, each listing analyzer references with
  per-analyzer ``enabled`` flags.

The pre-PR-E ``experiments/`` + ``library/`` layout is gone; this
module reflects that. The old comment-aware ``groups.yaml`` parsing
is also gone — ``AnalyzerRef.enabled`` is a proper schema field now.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml

from ConfigFileGUI.config_io import sanitize_for_yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Directory structure constants
# ---------------------------------------------------------------------------


_ANALYZERS_DIR = "analyzers"
_GROUPS_DIR = "groups"


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------


def discover_scan_config_dirs(root: Path) -> Dict[str, Optional[Path]]:
    """Discover the scan-analysis config directory structure under *root*.

    Expects the post-PR-E unified-diagnostic layout::

        <root>/
            analyzers/      ← per-device DiagnosticAnalysisConfig YAMLs (facility-namespaced)
            groups/         ← per-group AnalysisGroupConfig YAMLs (facility-namespaced)

    Parameters
    ----------
    root : Path
        Path to the ``scan_analysis_configs/`` root directory.

    Returns
    -------
    Dict[str, Optional[Path]]
        Dictionary with keys ``"analyzers"`` and ``"groups"`` pointing
        to the respective directories, or ``None`` when the expected
        directory does not exist.

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

    analyzers_dir = root / _ANALYZERS_DIR
    groups_dir = root / _GROUPS_DIR

    result: Dict[str, Optional[Path]] = {
        "analyzers": analyzers_dir if analyzers_dir.is_dir() else None,
        "groups": groups_dir if groups_dir.is_dir() else None,
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


def _list_yaml_files_recursive(directory: Path) -> List[Path]:
    """Return all ``.yaml`` / ``.yml`` files under *directory*, sorted.

    Recursive — needed because the new layout puts files under
    facility subdirectories (``analyzers/HTU/...``,
    ``groups/HTU/...``). Hidden directories (``.git`` etc.) are
    skipped via ``rglob`` default behaviour.

    Parameters
    ----------
    directory : Path
        Root directory to scan.

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

    files = sorted(
        list(directory.rglob("*.yaml")) + list(directory.rglob("*.yml")),
        key=lambda p: str(p).lower(),
    )
    return files


def list_analyzer_configs(root: Path) -> List[Path]:
    """List every analyzer YAML under ``<root>/analyzers/`` recursively.

    Walks the facility namespace subdirectories
    (``analyzers/HTU/``, ``analyzers/HTT/``, …) and returns absolute
    paths. Callers can derive the facility namespace from
    ``path.parent.name`` and the analyzer ID from ``path.stem``.

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
        If the ``analyzers/`` subdirectory does not exist.
    """
    analyzers_dir = root / _ANALYZERS_DIR
    result = _list_yaml_files_recursive(analyzers_dir)
    logger.debug("Found %d analyzer configs under %s", len(result), analyzers_dir)
    return result


def list_group_configs(root: Path) -> List[Path]:
    """List every group YAML under ``<root>/groups/`` recursively.

    Same facility-namespaced layout as the analyzers tree.

    Parameters
    ----------
    root : Path
        Path to the ``scan_analysis_configs/`` root directory.

    Returns
    -------
    List[Path]
        Sorted list of group config file paths.

    Raises
    ------
    FileNotFoundError
        If the ``groups/`` subdirectory does not exist.
    """
    groups_dir = root / _GROUPS_DIR
    result = _list_yaml_files_recursive(groups_dir)
    logger.debug("Found %d group configs under %s", len(result), groups_dir)
    return result


# ---------------------------------------------------------------------------
# Analyzer YAML — load / save
# ---------------------------------------------------------------------------


def load_analyzer_yaml(path: Path) -> dict:
    """Load a single analyzer (unified diagnostic) YAML file.

    Parameters
    ----------
    path : Path
        Path to the analyzer ``.yaml`` file.

    Returns
    -------
    dict
        Raw dictionary parsed from the YAML file. Caller is responsible
        for validating against
        :class:`image_analysis.config.DiagnosticAnalysisConfig` if a
        typed view is needed.

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
        Analyzer configuration dictionary matching
        :class:`image_analysis.config.DiagnosticAnalysisConfig`.
    """
    sanitized = sanitize_for_yaml(data)
    text = yaml.safe_dump(sanitized, default_flow_style=False, sort_keys=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Saved analyzer config to %s", path)


# ---------------------------------------------------------------------------
# Group YAML — load / save
# ---------------------------------------------------------------------------


def load_group_yaml(path: Path) -> dict:
    """Load a single group YAML file.

    No comment-aware parsing is needed in the new layout: each
    :class:`scan_analysis.config.diagnostic_models.AnalyzerRef` has a
    proper ``enabled: bool`` field that the schema honours, so the
    YAML uses normal lists and Pydantic handles enable/disable
    semantics. The old ``# - AnalyzerName`` convention is dead.

    Parameters
    ----------
    path : Path
        Path to the group ``.yaml`` file.

    Returns
    -------
    dict
        Raw dictionary parsed from the YAML file. Caller is responsible
        for validating against
        :class:`scan_analysis.config.diagnostic_models.AnalysisGroupConfig`
        if a typed view is needed.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    yaml.YAMLError
        If the file contains invalid YAML.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Group config not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        logger.warning("Group YAML is not a mapping: %s", path)
        return {}

    logger.debug("Loaded group config: %s", path)
    return data


def save_group_yaml(path: Path, data: dict) -> None:
    """Save a group configuration dict to a YAML file.

    Parameters
    ----------
    path : Path
        Destination file path (will be created or overwritten).
    data : dict
        Group configuration dictionary matching
        :class:`scan_analysis.config.diagnostic_models.AnalysisGroupConfig`.
    """
    sanitized = sanitize_for_yaml(data)
    text = yaml.safe_dump(sanitized, default_flow_style=False, sort_keys=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Saved group config to %s", path)


# ---------------------------------------------------------------------------
# Config type detection
# ---------------------------------------------------------------------------


def detect_scan_config_type(
    path: Path,
) -> Literal["analyzer", "group", "unknown"]:
    """Determine the scan-config type of a YAML file from its location.

    Pure location-based check — works because the new layout is
    unambiguous (every config lives under ``analyzers/`` or
    ``groups/``; there's no overlap).

    Parameters
    ----------
    path : Path
        Path to the YAML configuration file.

    Returns
    -------
    Literal["analyzer", "group", "unknown"]
        Detected configuration type.
    """
    parts = path.resolve().parts
    if _ANALYZERS_DIR in parts:
        return "analyzer"
    if _GROUPS_DIR in parts:
        return "group"
    return "unknown"


# ---------------------------------------------------------------------------
# Analyzer ID enumeration
# ---------------------------------------------------------------------------


def list_all_analyzer_ids(root: Path) -> List[str]:
    """Return a sorted list of analyzer IDs derived from filenames.

    Walks ``<root>/analyzers/`` recursively and returns the stem of
    each YAML file (without the ``.yaml`` extension). Used by the
    groups editor to autocomplete analyzer references — the stem is
    what ``AnalyzerRef.ref`` should be set to.

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


# ---------------------------------------------------------------------------
# Facility namespace helpers
# ---------------------------------------------------------------------------


def list_facilities(root: Path, kind: Literal["analyzers", "groups"]) -> List[str]:
    """Return the facility-namespace subdirectories under ``<root>/<kind>/``.

    The new layout organises configs by facility (``HTU``, ``HTT``,
    ``PW``, ``UNCLASSIFIED``, …) as immediate subdirectories of
    ``analyzers/`` and ``groups/``. This helper enumerates those
    subdirectory names so the tree panel can show facility headers
    and the groups editor can populate a facility picker.

    Parameters
    ----------
    root : Path
        Path to the ``scan_analysis_configs/`` root directory.
    kind : {"analyzers", "groups"}
        Which top-level subtree to enumerate.

    Returns
    -------
    List[str]
        Sorted list of facility names (subdirectory names). Empty
        list if the kind directory is missing.
    """
    subtree = root / kind
    if not subtree.is_dir():
        return []
    return sorted(
        (child.name for child in subtree.iterdir() if child.is_dir()),
        key=str.lower,
    )
