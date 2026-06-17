"""Loader functions: YAML file → typed config model.

Three public entry points:

* :func:`load_camera_config` — bare camera YAML or unified diagnostic
  YAML → :class:`CameraConfig`.
* :func:`load_line_config` — bare line YAML or unified diagnostic YAML
  → :class:`Line1DConfig`.
* :func:`load_diagnostic` — unified diagnostic YAML (by stem or path) →
  :class:`DiagnosticAnalysisConfig`.

Plus the low-level :func:`find_config_file` for resolving a camera /
line config name to its path on disk (used by the bare-name forms of
``load_camera_config`` / ``load_line_config``).

Lookup of standalone camera / line configs uses a single, explicit
base directory set via the environment variable
``IMAGE_ANALYSIS_CONFIG_DIR`` or passed per call. The base directory
is searched recursively.

For the typed-config → live-analyzer step, see
:func:`image_analysis.config.factory.create_image_analyzer`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import ValidationError

from . import array2d_processing as cfg_2d
from . import array1d_processing as cfg_1d
from .diagnostic import DiagnosticAnalysisConfig
from geecs_data_utils.config_roots import image_analysis_config

logger = logging.getLogger(__name__)

__all__ = [
    "find_config_file",
    "load_camera_config",
    "load_diagnostic",
    "load_line_config",
]

# ----------------------------------------------------------------------
# Base directory management
# ----------------------------------------------------------------------

_CONFIG_MANAGER = image_analysis_config
_CONFIG_CACHE: Dict[str, Path] = (
    _CONFIG_MANAGER.cache
)  # Cache for resolved config paths

# ----------------------------------------------------------------------
# Core helpers
# ----------------------------------------------------------------------


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
    return _CONFIG_MANAGER.find_config(
        camera_name,
        patterns=[
            "{name}.yaml",
            "{name}.yml",
            "default_{name}_settings.yaml",
            "default_{name}_settings.yml",
        ],
        config_dir=config_dir,
        use_cache=use_cache,
        missing_base_message=(
            "config_dir is required (no global base dir set). "
            "Set IMAGE_ANALYSIS_CONFIG_DIR or pass config_dir explicitly."
        ),
        not_found_label="Config",
    )


def _load_camera_config_dict(
    config_source: Union[str, Path, Dict[str, Any]],
    *,
    config_dir: Optional[Path],
) -> Dict[str, Any]:
    """Internal: load raw config dict from name/path/dict.

    Unwraps the ``image:`` section if the YAML is a unified diagnostic,
    so callers always receive the flat camera/line config shape. Per
    #412, ``CameraConfig`` / ``Line1DConfig`` no longer carry a
    ``name`` field — analyzer identity is set at construction time by
    the diagnostic factory via the ``output_name`` kwarg.
    """
    source_path: Optional[Path] = None
    if isinstance(config_source, str):
        source_path = find_config_file(config_source, config_dir=config_dir)
        with open(source_path, "r") as f:
            data = yaml.safe_load(f)
        logger.info("Loaded camera configuration from %s", source_path)
    elif isinstance(config_source, Path):
        if not config_source.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_source}")
        source_path = config_source
        with open(source_path, "r") as f:
            data = yaml.safe_load(f)
        logger.info("Loaded configuration from %s", source_path)
    elif isinstance(config_source, dict):
        data = config_source.copy()
        logger.info("Using provided configuration dictionary")
    else:
        raise ValueError(f"Invalid config_source type: {type(config_source)}")

    if data is None:
        data = {}

    data = _unwrap_diagnostic_image_section(data)
    # Filename-stem name injection used to live here as a cosmetic
    # fallback; removed in #412 along with the ``name`` field itself.
    # If a legacy standalone YAML still carries a top-level ``name``,
    # CameraConfig's ``extra="allow"`` policy keeps it as model_extra
    # without affecting behavior.
    return data


def _unwrap_diagnostic_image_section(data: Dict[str, Any]) -> Dict[str, Any]:
    r"""Return the ``image:`` subdict of a unified diagnostic YAML.

    A unified diagnostic config (per the schema in issue #400) lives in
    one file with both ImageAnalysis-side and ScanAnalysis-side
    sections. ImageAnalysis only consumes the ``image:`` body.

    Per #412, the previous "inject top-level ``name`` as ``image.name``"
    behavior is gone — ``CameraConfig`` / ``Line1DConfig`` no longer
    have a ``name`` field. Analyzer identity flows through the
    diagnostic factory's ``output_name`` constructor kwarg instead.

    Flat configs (no ``image:`` key) pass through unchanged so legacy
    standalone files in ``image_analysis_configs/`` keep working
    without modification.

    Parameters
    ----------
    data : dict
        Raw YAML-loaded dict.

    Returns
    -------
    dict
        Either the unwrapped image section (when ``image:`` is present)
        or the original dict (when it is not).

    Raises
    ------
    ValueError
        If ``image`` is present but is not a mapping.
    """
    if not isinstance(data, dict) or "image" not in data:
        return data

    image = data.get("image") or {}
    if not isinstance(image, dict):
        raise ValueError(
            f"Diagnostic config 'image' section must be a mapping, "
            f"got {type(image).__name__}"
        )

    return dict(image)


# ----------------------------------------------------------------------
# Public entry point (model-first)
# ----------------------------------------------------------------------


def load_camera_config(
    config_source: Union[str, Path, Dict[str, Any]],
    *,
    config_dir: Optional[Path] = None,
) -> cfg_2d.CameraConfig:
    """Load and validate a :class:`CameraConfig` from name / path / dict.

    Parameters
    ----------
    config_source : str, Path, or dict
        - ``str``: camera name (file stem), resolved against ``config_dir``
          (or the globally configured base dir). Unified diagnostic YAMLs
          and legacy bare-camera YAMLs are both handled — the ``image:``
          subsection is unwrapped transparently when present.
        - ``Path``: explicit ``.yaml`` / ``.yml`` file path.
        - ``dict``: already-loaded raw config (passed straight to
          ``CameraConfig.model_validate``).
    config_dir : Path, optional
        Directory containing YAML config files. Defaults to the global
        base dir.

    Returns
    -------
    CameraConfig
        Validated camera configuration model.
    """
    data = _load_camera_config_dict(config_source, config_dir=config_dir)
    try:
        return cfg_2d.CameraConfig.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Invalid camera configuration: {e}") from e


def load_line_config(
    config_source: Union[str, Path, Dict[str, Any]],
    *,
    config_dir: Optional[Path] = None,
) -> cfg_1d.Line1DConfig:
    """Load and validate a :class:`Line1DConfig` from name / path / dict.

    See :func:`load_camera_config` for parameter semantics — this is the
    1D counterpart.
    """
    data = _load_camera_config_dict(config_source, config_dir=config_dir)
    try:
        return cfg_1d.Line1DConfig.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Invalid line configuration: {e}") from e


# ----------------------------------------------------------------------
# Unified diagnostic loader
# ----------------------------------------------------------------------


def load_diagnostic(
    name_or_path: Union[str, Path],
    *,
    config_dir: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> DiagnosticAnalysisConfig:
    """Load a unified diagnostic YAML by name or path.

    Parameters
    ----------
    name_or_path : str or Path
        Diagnostic ID (filename stem, e.g. ``"UC_VisaEBeam1"``) or an
        absolute path to a unified diagnostic YAML. Filename stems must
        be globally unique across the ``analyzers/`` tree, so the bare
        stem suffices — no namespace prefix needed.
    config_dir : Path, optional
        Root of the scan-analysis configs tree (the parent of
        ``analyzers/``). When ``None`` and ``name_or_path`` is a
        string, falls back to
        ``ScanPaths.paths_config.scan_analysis_configs_path`` — the
        same default the task queue uses.
    overrides : dict, optional
        Deep-merged into the raw YAML before validation. Nested dicts
        are merged key-by-key (so ``{"scan": {"mode": "per_bin"}}``
        replaces only ``scan.mode`` and leaves the rest of ``scan``
        untouched); anything else (scalars, lists) replaces wholesale.
        Pydantic re-validates the merged result, so override typos and
        type mismatches surface with the same error path as a bad YAML
        on disk. Use this when a single consumer (e.g. the optimizer's
        ``MultiDeviceScanEvaluator``) needs a per-call variant of a
        diagnostic without forking the YAML.

    Returns
    -------
    DiagnosticAnalysisConfig
        Validated top-level config. The discriminated
        ``image:`` field has been routed to a typed :class:`CameraConfig`
        or :class:`Line1DConfig`; the ``scan:`` field is left as a raw
        dict at this layer (ScanAnalysis validates it against its own
        ``ScanRuntimeConfig`` at build time).

    Raises
    ------
    FileNotFoundError
        If the diagnostic can't be located.
    KeyError
        If the named stem isn't present under ``analyzers/``.
    ValueError
        On invalid YAML, validation errors, or when ``config_dir`` is
        needed but no default is available.
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

    with open(diag_path, "r") as f:
        data = yaml.safe_load(f) or {}

    if overrides:
        data = _deep_merge(data, overrides)

    try:
        return DiagnosticAnalysisConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid diagnostic config at {diag_path}: {exc}") from exc


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``base`` with ``overlay`` recursively merged in.

    Nested dicts are merged key-by-key. Anything else — scalars, lists,
    None — replaces wholesale (no list concatenation; that would create
    surprising semantics when an overlay tries to *replace* a list).
    Always returns a new dict; ``base`` and ``overlay`` are not mutated.

    Used by :func:`load_diagnostic` to apply per-call overrides on top
    of the on-disk YAML before Pydantic validation, but generic enough
    to apply to any dict pair.
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
    """Build a mapping from diagnostic stem to its YAML path.

    Mirrors :func:`scan_analysis.config.analysis_group_loader.discover_analyzers`
    but lives here so ImageAnalysis can load diagnostics without
    importing from ScanAnalysis.
    """
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
        stem = path.stem
        if stem in index:
            raise ValueError(
                f"Duplicate diagnostic ID '{stem}' at {path} and "
                f"{index[stem]}. Diagnostic file stems must be unique "
                f"across the entire 'analyzers/' tree."
            )
        index[stem] = path
    return index
