"""Factory for instantiating scan analyzers from unified diagnostic configs.

The factory takes a :class:`ResolvedDiagnosticConfig` produced by the
analysis-group loader and returns a runnable
:class:`~scan_analysis.base.ScanAnalyzer`. It performs three steps:

1. Validate the embedded ``image:`` section by calling the appropriate
   ImageAnalysis loader (``load_camera_config`` or ``load_line_config``)
   so the analyzer receives a pre-built ``CameraConfig`` /
   ``Line1DConfig`` object rather than a name string.
2. Import the ImageAnalyzer class named by the alias's class path and
   instantiate it with the validated image config plus any
   alias-specific kwargs.
3. Wrap the ImageAnalyzer in the matching scan-analyzer
   (``Array2DScanAnalyzer`` or ``Array1DScanAnalyzer``) and attach
   the per-group ``id``, effective ``priority``, and ``gdoc_slot``.

Aliases with ``image_kind: none`` (currently just ``haso``) skip step
1 and pass kwargs through directly.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any, Dict, Type

from image_analysis.config_loader import load_camera_config, load_line_config

from .aliases import ImageAnalyzerSpec, ImageKind, ScanType
from .diagnostic_models import DiagnosticAnalysisConfig, ResolvedDiagnosticConfig

if TYPE_CHECKING:
    from scan_analysis.base import ScanAnalyzer

logger = logging.getLogger(__name__)

__all__ = ["create_diagnostic_analyzer"]


def create_diagnostic_analyzer(resolved: ResolvedDiagnosticConfig) -> "ScanAnalyzer":
    """Build a runnable scan analyzer from a resolved diagnostic config.

    Parameters
    ----------
    resolved : ResolvedDiagnosticConfig
        The loader-output form: a validated :class:`DiagnosticAnalysisConfig`
        plus its filename-derived ``id`` and group-effective ``priority``.

    Returns
    -------
    ScanAnalyzer
        A fully configured ``Array2DScanAnalyzer`` or
        ``Array1DScanAnalyzer`` ready to run.

    Raises
    ------
    ValueError
        If the diagnostic's ``image:`` section is inconsistent with the
        analyzer's declared ``image_kind`` (e.g. an ``image:`` section
        is present for an ``image_kind: none`` analyzer).
    ImportError
        If the alias's class path cannot be imported.
    TypeError
        If the ImageAnalyzer constructor rejects the kwargs the alias
        and YAML produced.
    """
    diag = resolved.diagnostic
    spec = diag.image_analyzer

    image_config = _validate_embedded_image_section(diag, spec.image_kind)
    image_analyzer = _instantiate_image_analyzer(spec, image_config)
    return _wrap_in_scan_analyzer(resolved, image_analyzer)


# ---------------------------------------------------------------------------
# Step 1: embedded image section
# ---------------------------------------------------------------------------


def _validate_embedded_image_section(
    diag: DiagnosticAnalysisConfig, image_kind: ImageKind
) -> Any:
    """Validate ``diag.image`` through ImageAnalysis, or ensure absence.

    Returns the validated ``CameraConfig`` / ``Line1DConfig`` object,
    or ``None`` when the analyzer takes no embedded image config.
    """
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


# ---------------------------------------------------------------------------
# Step 2: ImageAnalyzer
# ---------------------------------------------------------------------------


def _instantiate_image_analyzer(spec: ImageAnalyzerSpec, image_config: Any) -> Any:
    """Import the alias's class and instantiate with kwargs + image config."""
    analyzer_class = _import_class(spec.class_path)
    kwargs = dict(spec.kwargs)

    if spec.image_kind == ImageKind.CAMERA:
        kwargs["camera_config_name"] = image_config
    elif spec.image_kind == ImageKind.LINE:
        kwargs["line_config_name"] = image_config
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


# ---------------------------------------------------------------------------
# Step 3: scan analyzer wrapper
# ---------------------------------------------------------------------------


def _wrap_in_scan_analyzer(
    resolved: ResolvedDiagnosticConfig, image_analyzer: Any
) -> "ScanAnalyzer":
    """Construct the dimension-specific scan-analyzer wrapper.

    The 1D and 2D wrappers differ only in their save-flag kwarg name
    (``flag_save_data`` vs ``flag_save_images``). Everything else maps
    directly from the ``scan:`` section.
    """
    diag = resolved.diagnostic
    spec = diag.image_analyzer
    scan_cfg = diag.scan

    if spec.scan_type == ScanType.ARRAY2D:
        from scan_analysis.analyzers.common.array2D_scan_analysis import (
            Array2DScanAnalyzer,
        )

        wrapper_class: Type["ScanAnalyzer"] = Array2DScanAnalyzer
        save_kwarg = "flag_save_images"
    elif spec.scan_type == ScanType.ARRAY1D:
        from scan_analysis.analyzers.common.array1d_scan_analysis import (
            Array1DScanAnalyzer,
        )

        wrapper_class = Array1DScanAnalyzer
        save_kwarg = "flag_save_data"
    else:
        raise ValueError(f"Unknown scan_type: {spec.scan_type}")

    wrapper_kwargs: Dict[str, Any] = {
        "device_name": scan_cfg.device or diag.name,
        "image_analyzer": image_analyzer,
        "renderer_kwargs": scan_cfg.renderer_kwargs,
        "analysis_mode": scan_cfg.mode,
        save_kwarg: scan_cfg.save,
    }
    if scan_cfg.file_tail is not None:
        wrapper_kwargs["file_tail"] = scan_cfg.file_tail

    try:
        analyzer = wrapper_class(**wrapper_kwargs)
    except TypeError as exc:
        raise TypeError(
            f"Failed to wrap {spec.class_path} in {wrapper_class.__name__} "
            f"for diagnostic '{diag.name}': {exc}"
        ) from exc

    # Task-queue / scan-log metadata attached as instance attributes.
    # These attributes already exist on analyzers built via the legacy
    # factory; mirroring them here keeps the rest of the runtime
    # unchanged.
    analyzer.id = resolved.id
    analyzer.priority = resolved.priority
    analyzer.gdoc_slot = scan_cfg.gdoc_slot

    return analyzer
