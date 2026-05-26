"""Loader and factory for unified diagnostic configs.

This module is the public Mode-2 entry point on the ImageAnalysis side
of the unified-configs API. Two functions:

* :func:`load_diagnostic` — find a unified YAML by name (or path),
  parse, validate. Returns a :class:`DiagnosticAnalysisConfig`.
* :func:`create_image_analyzer` — take a validated diagnostic config,
  import the named class, instantiate it with the embedded image
  config. Returns a live ``ImageAnalyzer``.

The two compose naturally:

>>> diag = load_diagnostic("UC_Amp4_IR_Input")
>>> diag.image.roi.x_max = 200    # optional notebook tweak
>>> analyzer = create_image_analyzer(diag)
>>> result = analyzer.analyze_image_file(some_path)

For Mode 1 (no YAML), construct a ``DiagnosticAnalysisConfig`` in code
and hand it to ``create_image_analyzer`` directly — same factory, same
result.

The ScanAnalysis-side ``create_scan_analyzer`` builds on this factory:
it calls ``create_image_analyzer`` for the inner analyzer, then wraps
it in the dimension-specific ``Array{1,2}DScanAnalyzer``.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import yaml
from pydantic import ValidationError

from .aliases import ImageAnalyzerSpec, ImageKind
from .diagnostic_models import DiagnosticAnalysisConfig

logger = logging.getLogger(__name__)

__all__ = ["create_image_analyzer", "load_diagnostic"]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_diagnostic(
    name_or_path: Union[str, Path],
    *,
    config_dir: Optional[Path] = None,
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

    Returns
    -------
    DiagnosticAnalysisConfig
        Validated top-level config. Pydantic recursively validates the
        ``image_analyzer`` field against :class:`ImageAnalyzerSpec`,
        but leaves the ``image:`` and ``scan:`` sub-dicts weakly typed
        at this layer — they're validated by their respective consumers
        (:func:`create_image_analyzer` for image,
        ``ScanRuntimeConfig.model_validate`` for scan).

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

    try:
        return DiagnosticAnalysisConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid diagnostic config at {diag_path}: {exc}") from exc


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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_image_analyzer(diag: DiagnosticAnalysisConfig) -> Any:
    """Build a live ImageAnalyzer instance from a validated diagnostic config.

    Imports the class named by ``diag.image_analyzer.class_path``,
    validates the embedded ``image:`` section through ImageAnalysis's
    typed loader (``load_camera_config`` for 2D, ``load_line_config``
    for 1D), and instantiates the class with the right constructor
    kwargs (``camera_config_name``, ``line_config_name``, or none —
    determined by ``diag.image_analyzer.image_kind``).

    Parameters
    ----------
    diag : DiagnosticAnalysisConfig
        Validated diagnostic. The ``image_analyzer`` spec tells us
        which class to import and how it consumes ``image``.

    Returns
    -------
    ImageAnalyzer
        Fully-configured analyzer instance. Caller can immediately
        call ``analyze_image`` / ``analyze_image_file`` on it.

    Raises
    ------
    ValueError
        If the embedded image section's shape doesn't match what the
        alias expects (missing when required, present when forbidden,
        or invalid against the ImageAnalysis config models).
    ImportError, AttributeError
        If ``class_path`` does not resolve to an importable class.
    TypeError
        If the resolved class can't be instantiated with the inferred
        kwargs.
    """
    spec = diag.image_analyzer
    image_config = _validate_embedded_image_section(diag, spec.image_kind)
    return _instantiate_image_analyzer(spec, image_config)


def _validate_embedded_image_section(
    diag: DiagnosticAnalysisConfig, image_kind: ImageKind
) -> Any:
    """Validate ``diag.image`` through ImageAnalysis, or ensure absence.

    Returns the validated ``CameraConfig`` / ``Line1DConfig`` object,
    or ``None`` when the analyzer takes no embedded image config.
    """
    # Lazy import: ``image_analysis.config_loader`` pulls in heavy
    # processing modules; defer until we actually need it.
    from image_analysis.config_loader import load_camera_config, load_line_config

    if image_kind == ImageKind.NONE:
        if diag.image is not None:
            raise ValueError(
                f"Diagnostic '{diag.name}': image_analyzer has "
                f"image_kind='none' but the YAML provided an 'image:' "
                f"section. Either remove the section or use an analyzer "
                f"that consumes it."
            )
        return None

    image_dict: Dict[str, Any] = dict(diag.image or {})
    # Top-level diagnostic name is the default metric prefix / device
    # identity. Explicit image.name (rare) takes precedence.
    image_dict.setdefault("name", diag.name)

    if image_kind == ImageKind.CAMERA:
        return load_camera_config(image_dict)
    if image_kind == ImageKind.LINE:
        return load_line_config(image_dict)

    raise ValueError(f"Unknown image_kind: {image_kind}")


def _instantiate_image_analyzer(spec: ImageAnalyzerSpec, image_config: Any) -> Any:
    """Import the class and instantiate with kwargs + image config."""
    analyzer_class = _import_class(spec.class_path)
    kwargs = dict(spec.kwargs)

    if spec.image_kind == ImageKind.CAMERA:
        kwargs["camera_config"] = image_config
    elif spec.image_kind == ImageKind.LINE:
        kwargs["line_config"] = image_config
    # ImageKind.NONE: kwargs is exactly what the YAML supplied.

    logger.info(
        "Instantiating %s with kwargs %s",
        spec.class_path,
        sorted(kwargs),
    )
    try:
        return analyzer_class(**kwargs)
    except TypeError as exc:
        raise TypeError(
            f"Failed to instantiate {spec.class_path} with kwargs "
            f"{sorted(kwargs)}: {exc}"
        ) from exc


def _import_class(class_path: str) -> Type:
    """Resolve a dotted import path to a class object."""
    try:
        module_path, class_name = class_path.rsplit(".", 1)
    except ValueError as exc:
        raise ValueError(
            f"Invalid class path '{class_path}'. Must be fully qualified "
            f"(e.g. 'module.Class')."
        ) from exc

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Cannot import module '{module_path}' for class path '{class_path}': {exc}"
        ) from exc

    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Module '{module_path}' has no class '{class_name}' "
            f"(from class path '{class_path}'): {exc}"
        ) from exc
