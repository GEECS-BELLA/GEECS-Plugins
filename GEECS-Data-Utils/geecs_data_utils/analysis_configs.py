"""Read-only diagnostic discovery and YAML loading, independent of analysis.

These functions resolve and read raw documents. Schema validation belongs to
consumers (GEECS-Schemas for v2); this foundational package depends on neither
schemas nor numerical analyzers. The config root is explicit for stem lookup.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

# Any is intentional here: these are raw YAML documents awaiting the consumer's
# schema validation, not an alternative configuration model.
RawDocument = dict[str, Any]


def deep_merge(base: RawDocument, overlay: RawDocument) -> RawDocument:
    """Merge nested mappings key-by-key; replace lists/scalars without aliasing."""
    out = deepcopy(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def read_yaml_mapping(path: Path) -> RawDocument:
    """Read one fresh YAML mapping; empty YAML becomes an empty document."""
    with path.open("r") as stream:
        data = yaml.safe_load(stream)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at {path}")
    return data


def discover_diagnostics(config_dir: Path) -> dict[str, Path]:
    """Map unique file stems under analyzers/ to paths, refusing duplicates."""
    analyzers_dir = config_dir / "analyzers"
    if not analyzers_dir.is_dir():
        raise FileNotFoundError(
            f"Analyzer directory not found: {analyzers_dir}. "
            f"Expected the unified-configs layout under {config_dir}."
        )
    index = {}
    for path in sorted(
        list(analyzers_dir.rglob("*.yaml")) + list(analyzers_dir.rglob("*.yml"))
    ):
        if path.stem in index:
            raise ValueError(
                f"Duplicate diagnostic ID '{path.stem}' at {path} and {index[path.stem]}. "
                "Diagnostic file stems must be unique across the entire 'analyzers/' tree."
            )
        index[path.stem] = path
    return index


def read_diagnostic(
    name_or_path: str | Path,
    *,
    config_dir: Path | None = None,
    overrides: RawDocument | None = None,
) -> tuple[Path, RawDocument]:
    """Resolve a stem or explicit Path and read its raw document with overrides.

    Returns the source path alongside the data so callers retain filename
    identity and can include it in validation errors. String inputs are stems,
    not paths. Every read is fresh; no directories are created or files changed.
    """
    if isinstance(name_or_path, Path):
        path = name_or_path
        if not path.exists():
            raise FileNotFoundError(f"Diagnostic config not found: {path}")
    else:
        if config_dir is None:
            raise ValueError("config_dir is required when loading a diagnostic by stem")
        index = discover_diagnostics(config_dir)
        if name_or_path not in index:
            raise KeyError(
                f"Diagnostic '{name_or_path}' not found under {config_dir / 'analyzers'}. "
                f"Known diagnostics: {sorted(index)}"
            )
        path = index[name_or_path]
    data = read_yaml_mapping(path)
    return path, deep_merge(data, overrides) if overrides else data
