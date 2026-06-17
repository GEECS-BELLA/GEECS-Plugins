"""Loader for the unified analysis-group config layout (issue #400).

A scan-analysis config tree under this scheme looks like:

.. code-block:: text

    scan_analysis_configs/
      analyzers/
        HTU/
          GaiaMode.yaml
          U_FROG_Beam.yaml
        PW/
          MagSpectStitcher.yaml
      groups/
        HTU/
          laser_diagnostics.yaml
          ebeam_diagnostics.yaml
        PW/
          frog_only.yaml

``load_analysis_group(name_or_path)`` is the entry point. It locates a
group file under ``groups/**/*.yaml``, resolves each analyzer
reference against the diagnostics discovered under ``analyzers/**/*.yaml``,
applies per-group overrides, sorts by priority, and returns a
:class:`LoadedAnalysisGroup`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from image_analysis.config import load_diagnostic

from .diagnostic_models import (
    AnalysisGroupConfig,
    ResolvedDiagnosticConfig,
    ScanRuntimeConfig,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LoadedAnalysisGroup",
    "discover_analyzers",
    "discover_groups",
    "load_analysis_group",
    "resolve_group",
]


class LoadedAnalysisGroup(BaseModel):
    """Loader output: a group plus its resolved, prioritised diagnostics.

    Attributes
    ----------
    name : str
        Display name from the group config's ``name`` field.
    description : str, optional
        Free-text description from the group config.
    upload_to_scanlog : bool
        Whether the runner should upload outputs to the scan log.
    analyzers : list of ResolvedDiagnosticConfig
        Diagnostics referenced by the group, in execution order. Sorted
        by effective priority ascending. Entries whose group reference
        had ``enabled: false`` are excluded.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: Optional[str] = None
    upload_to_scanlog: bool = True
    analyzers: List[ResolvedDiagnosticConfig] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_analyzers(base_dir: Path) -> Dict[str, Path]:
    """Build a mapping from diagnostic ID to its YAML path.

    Recurses ``base_dir / "analyzers"`` and indexes every
    ``*.yaml`` / ``*.yml`` by its file stem (the canonical diagnostic
    ID per issue #400's naming rules).

    Parameters
    ----------
    base_dir : Path
        Root of the scan-analysis configs tree (typically the value of
        ``SCAN_ANALYSIS_CONFIG_DIR``).

    Returns
    -------
    dict
        ``{diagnostic_id: path}``.

    Raises
    ------
    FileNotFoundError
        If the ``analyzers/`` subdirectory does not exist.
    ValueError
        If two analyzer YAMLs share the same file stem.
    """
    analyzers_dir = base_dir / "analyzers"
    if not analyzers_dir.is_dir():
        raise FileNotFoundError(
            f"Analyzer directory not found: {analyzers_dir}. "
            f"Expected the unified-configs layout under {base_dir}."
        )

    index: Dict[str, Path] = {}
    for path in sorted(_iter_yaml_files(analyzers_dir)):
        stem = path.stem
        if stem in index:
            raise ValueError(
                f"Duplicate diagnostic ID '{stem}' at {path} and "
                f"{index[stem]}. Diagnostic file stems must be unique "
                f"across the entire 'analyzers/' tree."
            )
        index[stem] = path
    return index


def discover_groups(base_dir: Path) -> Dict[str, Path]:
    """Build a mapping from group name to its YAML path.

    Two index keys are written per group:

    * The filename stem (``"frog_only"``).
    * The path-like name relative to ``groups/`` (``"PW/frog_only"``).

    Either form is acceptable input to :func:`load_analysis_group`. If
    two groups share a filename stem under different namespaces, only
    the path-like name disambiguates; the stem-only key raises on
    lookup.

    Parameters
    ----------
    base_dir : Path
        Root of the scan-analysis configs tree.

    Returns
    -------
    dict
        ``{key: path}``. Keys include both stem-only and path-like
        forms; ambiguous stems map to a sentinel that raises on access
        (handled by :func:`_resolve_group_path`).

    Raises
    ------
    FileNotFoundError
        If the ``groups/`` subdirectory does not exist.
    """
    groups_dir = base_dir / "groups"
    if not groups_dir.is_dir():
        raise FileNotFoundError(
            f"Group directory not found: {groups_dir}. Expected the "
            f"unified-configs layout under {base_dir}."
        )

    index: Dict[str, Path] = {}
    stem_counts: Dict[str, int] = {}
    for path in sorted(_iter_yaml_files(groups_dir)):
        stem = path.stem
        rel = path.relative_to(groups_dir).with_suffix("")
        path_like = str(rel).replace("\\", "/")

        # Path-like form is always unique (it includes the namespace).
        index[path_like] = path

        # Stem-only form is convenient when the group name is unique.
        stem_counts[stem] = stem_counts.get(stem, 0) + 1
        if stem_counts[stem] == 1:
            index[stem] = path
        else:
            # Two or more groups share this stem; remove the stem-only
            # entry to force callers to use the path-like form.
            index.pop(stem, None)

    return index


def _iter_yaml_files(directory: Path) -> Iterable[Path]:
    """Yield ``*.yaml`` and ``*.yml`` files anywhere under ``directory``."""
    yield from directory.rglob("*.yaml")
    yield from directory.rglob("*.yml")


# ---------------------------------------------------------------------------
# Group resolution
# ---------------------------------------------------------------------------


def resolve_group(
    group_cfg: AnalysisGroupConfig,
    analyzer_index: Dict[str, Path],
) -> LoadedAnalysisGroup:
    """Resolve a parsed group config against a diagnostic index.

    Each ``AnalyzerRef`` in the group is looked up in ``analyzer_index``
    (by stem), the referenced diagnostic YAML is loaded and validated,
    per-group overrides (``enabled``, ``priority``) are applied, and the
    result is collected into a :class:`ResolvedDiagnosticConfig`. The
    final list is filtered by ``enabled`` and sorted by effective
    priority ascending.

    Parameters
    ----------
    group_cfg : AnalysisGroupConfig
        Parsed group YAML (already validated by Pydantic).
    analyzer_index : dict
        ``{diagnostic_id: path}`` from :func:`discover_analyzers`.

    Returns
    -------
    LoadedAnalysisGroup
        Group with resolved diagnostics ordered by priority.

    Raises
    ------
    ValueError
        On unresolved refs or duplicate IDs within the resolved group.
    """
    seen: Dict[str, int] = {}
    resolved: List[ResolvedDiagnosticConfig] = []

    for ref_entry in group_cfg.analyzers:
        ref = ref_entry.ref
        if ref not in analyzer_index:
            raise ValueError(
                f"Group '{group_cfg.name}' references unknown analyzer "
                f"'{ref}'. Known IDs: {sorted(analyzer_index)}"
            )
        if ref in seen:
            raise ValueError(
                f"Group '{group_cfg.name}' contains duplicate analyzer "
                f"reference '{ref}'. Each ID may appear at most once "
                f"per resolved group."
            )
        seen[ref] = 1

        diagnostic = load_diagnostic(analyzer_index[ref])
        # diagnostic.scan is weakly typed at the ImageAnalysis layer;
        # validate to read the priority field. ``or {}`` covers the
        # legitimate "no scan block in the YAML" case.
        scan_cfg = ScanRuntimeConfig.model_validate(diagnostic.scan or {})
        effective_priority = (
            ref_entry.priority if ref_entry.priority is not None else scan_cfg.priority
        )

        if not ref_entry.enabled:
            continue

        resolved.append(
            ResolvedDiagnosticConfig(
                id=ref,
                enabled=True,
                priority=effective_priority,
                diagnostic=diagnostic,
            )
        )

    resolved.sort(key=lambda r: (r.priority, r.id))

    return LoadedAnalysisGroup(
        name=group_cfg.name,
        description=group_cfg.description,
        upload_to_scanlog=group_cfg.upload_to_scanlog,
        analyzers=resolved,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def load_analysis_group(
    name_or_path: Union[str, Path],
    *,
    config_dir: Optional[Path] = None,
) -> LoadedAnalysisGroup:
    """Load an analysis group by name or absolute path.

    Parameters
    ----------
    name_or_path : str or Path
        Either a group name resolved against ``config_dir / "groups/"``,
        or an explicit ``Path`` to a group YAML. Names can be the file
        stem (``"frog_only"``) or path-like (``"PW/frog_only"``); the
        path-like form is required to disambiguate when two groups in
        different namespaces share a stem.
    config_dir : Path, optional
        Root of the scan-analysis configs tree. Required when
        ``name_or_path`` is a string (no global default is provided
        here — the call site decides).

    Returns
    -------
    LoadedAnalysisGroup
        Group with resolved diagnostics ordered by priority.

    Raises
    ------
    FileNotFoundError
        If the group file or analyzer directory can't be located.
    ValueError
        On invalid YAML, validation errors, missing refs, or duplicates.
    """
    if isinstance(name_or_path, Path):
        group_path = name_or_path
        if not group_path.exists():
            raise FileNotFoundError(f"Group config not found: {group_path}")
        # Find the configs root by walking up from the group file to a
        # ``groups/`` ancestor; the parent of ``groups/`` is the root.
        base_dir = _infer_config_root(group_path)
    else:
        if config_dir is None:
            raise ValueError(
                "config_dir is required when name_or_path is a string. "
                "Pass an explicit Path to load_analysis_group(), or "
                "supply config_dir."
            )
        base_dir = Path(config_dir)
        group_path = _resolve_group_path(name_or_path, base_dir)

    analyzer_index = discover_analyzers(base_dir)

    with open(group_path, "r") as f:
        data = yaml.safe_load(f) or {}

    try:
        group_cfg = AnalysisGroupConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid group config at {group_path}: {exc}") from exc

    return resolve_group(group_cfg, analyzer_index)


def _resolve_group_path(name: str, base_dir: Path) -> Path:
    """Look up a group by stem or path-like name; raise on miss/ambiguity."""
    index = discover_groups(base_dir)
    if name in index:
        return index[name]

    # If the stem was dropped due to ambiguity, give a useful error
    # listing the qualified names the caller could use instead.
    candidates = [k for k in index if k.endswith(f"/{name}") or k == name]
    if candidates:
        raise ValueError(
            f"Group name '{name}' is ambiguous; matches: {sorted(candidates)}. "
            f"Use the path-like form (e.g. 'HTU/{name}') to disambiguate."
        )
    raise FileNotFoundError(
        f"Group '{name}' not found under {base_dir / 'groups'}. "
        f"Known groups: {sorted(index)}"
    )


def _infer_config_root(group_path: Path) -> Path:
    """Walk up from a group YAML to the directory containing ``groups/``."""
    for ancestor in group_path.parents:
        if ancestor.name == "groups":
            return ancestor.parent
    raise ValueError(
        f"Could not infer config root for {group_path}: no 'groups/' "
        f"ancestor directory found."
    )
