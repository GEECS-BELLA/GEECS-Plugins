"""Loader functions: YAML file → typed config model.

Three public entry points:

* :func:`load_diagnostic` — a diagnostic YAML (by stem or path) →
  :class:`~geecs_schemas.analysis.AnalysisDiagnostic` (format v2; a pre-v2
  file is refused with a pointer to the one-shot converter).
* :func:`load_camera_config` / :func:`load_line_config` — the ``image:``
  section of a diagnostic (by stem or path), or a bare camera / line
  YAML or dict, → :class:`CameraConfig` / :class:`Line1DConfig`.  The
  notebook convenience: get the processing section without building an
  analyzer.
* :func:`list_diagnostics` — the diagnostic IDs under a configs tree.

Plus :func:`find_config_file` for resolving a stem to its path.  Lookup
uses the scan-analysis configs root (``SCAN_ANALYSIS_CONFIG_DIR`` or
``scan_analysis_configs_path`` in the shared GEECS user config), searched
recursively.

For typed config → live analyzer see
:func:`image_analysis.config.factory.create_image_analyzer`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from geecs_schemas.analysis import AnalysisDiagnostic, CameraConfig, Line1DConfig
from pydantic import ValidationError

from geecs_data_utils.config_roots import scan_analysis_config

logger = logging.getLogger(__name__)

__all__ = [
    "find_config_file",
    "list_diagnostics",
    "load_camera_config",
    "load_diagnostic",
    "load_line_config",
]

_CONFIG_MANAGER = scan_analysis_config

#: Keys whose presence marks a YAML as a diagnostic document rather than a
#: bare processing section: the analyzer block, the ``image:`` wrapper, or
#: the pre-v2 class path (so a stale file gets the model's converter hint).
_DIAGNOSTIC_MARKERS = ("analyzer", "image_analyzer", "image")


def find_config_file(
    name: str, *, config_dir: Optional[Path] = None, use_cache: bool = True
) -> Path:
    """Resolve a config stem to its YAML path, searching the configs root recursively.

    Parameters
    ----------
    name : str
        File stem (``UC_TopView``).
    config_dir : Path, optional
        Root to search; defaults to the configured scan-analysis root.
    use_cache : bool, default True
        Reuse previously resolved paths.

    Raises
    ------
    ValueError
        If no root is configured and ``config_dir`` is not given.
    FileNotFoundError
        If nothing matches.
    """
    return _CONFIG_MANAGER.find_config(
        name,
        patterns=["{name}.yaml", "{name}.yml"],
        config_dir=config_dir,
        use_cache=use_cache,
        missing_base_message=(
            "config_dir is required (no unified analysis config root set). "
            "Set SCAN_ANALYSIS_CONFIG_DIR or pass config_dir explicitly."
        ),
        not_found_label="Config",
    )


def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return {} if data is None else data


def _load_image_section(
    config_source: Union[str, Path, Dict[str, Any]],
    *,
    config_dir: Optional[Path],
    expected: type,
    label: str,
) -> Any:
    """Shared body of :func:`load_camera_config` / :func:`load_line_config`."""
    if isinstance(config_source, dict):
        data: Dict[str, Any] = dict(config_source)
        source = "<dict>"
    else:
        path = (
            Path(config_source)
            if isinstance(config_source, Path)
            else find_config_file(config_source, config_dir=config_dir)
        )
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        data = _read_yaml(path)
        source = str(path)
        logger.info("Loaded %s configuration from %s", label, path)

    if any(marker in data for marker in _DIAGNOSTIC_MARKERS):
        # A diagnostic document: validate the whole thing and hand back its
        # image section.
        try:
            diag = AnalysisDiagnostic.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"Invalid diagnostic config at {source}: {exc}") from exc
        if not isinstance(diag.image, expected):
            raise ValueError(
                f"{source}: expected a {label} image section, got "
                f"{type(diag.image).__name__ if diag.image is not None else 'none'}"
            )
        return diag.image

    # A bare processing section (camera / line) — validate directly.
    try:
        return expected.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid {label} configuration at {source}: {exc}") from exc


def load_camera_config(
    config_source: Union[str, Path, Dict[str, Any]],
    *,
    config_dir: Optional[Path] = None,
) -> CameraConfig:
    """Load a :class:`CameraConfig` from a diagnostic stem / path, a bare camera YAML, or a dict.

    Parameters
    ----------
    config_source : str, Path, or dict
        A stem resolved under ``config_dir`` (or the configured root), an
        explicit YAML path, or an already-loaded mapping.  Diagnostic
        documents (v1 or v2) yield their ``image:`` section; a bare camera
        mapping validates directly.
    config_dir : Path, optional
        Root to search for a stem.

    Raises
    ------
    ValueError
        If the document is invalid or its image section is not a camera.
    """
    return _load_image_section(
        config_source, config_dir=config_dir, expected=CameraConfig, label="camera"
    )


def load_line_config(
    config_source: Union[str, Path, Dict[str, Any]],
    *,
    config_dir: Optional[Path] = None,
) -> Line1DConfig:
    """Load a :class:`Line1DConfig`; the 1D counterpart of :func:`load_camera_config`."""
    return _load_image_section(
        config_source, config_dir=config_dir, expected=Line1DConfig, label="line"
    )


def load_diagnostic(
    name_or_path: Union[str, Path],
    *,
    config_dir: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> AnalysisDiagnostic:
    """Load a diagnostic YAML by stem or path.

    Parameters
    ----------
    name_or_path : str or Path
        Diagnostic ID (filename stem, unique across the ``analyzers/``
        tree) or an explicit path.
    config_dir : Path, optional
        Root of the scan-analysis configs tree (the parent of
        ``analyzers/``); defaults to
        ``ScanPaths.paths_config.scan_analysis_configs_path``.
    overrides : dict, optional
        Deep-merged into the raw YAML before validation (nested dicts
        key-by-key, everything else replaced wholesale), so a consumer can
        run a per-call variant — the optimizer's ``scan: {mode: per_bin}``
        — without forking the file.  Override typos surface exactly like a
        bad YAML.

    Returns
    -------
    AnalysisDiagnostic
        The validated, fully typed document.

    Raises
    ------
    FileNotFoundError
        If a path does not exist.
    KeyError
        If a stem is not present under ``analyzers/``.
    ValueError
        On invalid YAML or validation errors, or when no root is available.
    """
    if isinstance(name_or_path, Path):
        diag_path = name_or_path
        if not diag_path.exists():
            raise FileNotFoundError(f"Diagnostic config not found: {diag_path}")
    else:
        base_dir = _resolve_default_config_dir(config_dir)
        index = _discover_analyzers(base_dir)
        if name_or_path not in index:
            raise KeyError(
                f"Diagnostic '{name_or_path}' not found under "
                f"{base_dir / 'analyzers'}. Known diagnostics: "
                f"{sorted(set(index))}"
            )
        diag_path = index[name_or_path]

    data = _read_yaml(diag_path)
    if overrides:
        data = _deep_merge(data, overrides)

    try:
        diagnostic = AnalysisDiagnostic.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid diagnostic config at {diag_path}: {exc}") from exc
    diagnostic._source_id = diag_path.stem
    return diagnostic


def list_diagnostics(*, config_dir: Optional[Path] = None) -> List[str]:
    """List the diagnostic IDs (YAML stems) under a configs tree, sorted.

    Listing proves discovery, not validity: each stem resolves via
    :func:`load_diagnostic`, which may still raise on a malformed file.
    """
    base_dir = _resolve_default_config_dir(config_dir)
    return sorted(_discover_analyzers(base_dir))


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``base`` with ``overlay`` merged in: nested dicts key-by-key, else replaced.

    Always a new dict; neither input is mutated.
    """
    out: Dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_default_config_dir(config_dir: Optional[Path]) -> Path:
    """Return ``config_dir`` if set, else the globally configured root."""
    if config_dir is not None:
        return Path(config_dir)
    try:
        from geecs_data_utils import ScanPaths

        root = ScanPaths.paths_config.scan_analysis_configs_path
    except Exception as exc:
        raise ValueError(
            "config_dir was not provided and no default could be "
            "resolved from ScanPaths.paths_config."
        ) from exc
    if root is None:
        raise ValueError(
            "config_dir was not provided and "
            "ScanPaths.paths_config.scan_analysis_configs_path is unset."
        )
    return Path(root)


def _discover_analyzers(base_dir: Path) -> Dict[str, Path]:
    """Map diagnostic stem → YAML path under ``<base_dir>/analyzers``; stems must be unique."""
    analyzers_dir = base_dir / "analyzers"
    if not analyzers_dir.is_dir():
        raise FileNotFoundError(
            f"Analyzer directory not found: {analyzers_dir}. "
            f"Expected the unified-configs layout under {base_dir}."
        )
    index: Dict[str, Path] = {}
    for path in sorted(
        list(analyzers_dir.rglob("*.yaml")) + list(analyzers_dir.rglob("*.yml"))
    ):
        if path.stem in index:
            raise ValueError(
                f"Duplicate diagnostic ID '{path.stem}' at {path} and "
                f"{index[path.stem]}. Diagnostic file stems must be unique "
                f"across the entire 'analyzers/' tree."
            )
        index[path.stem] = path
    return index
