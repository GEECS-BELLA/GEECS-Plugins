"""Build a runnable scan-analyzer from a unified diagnostic config.

This module is thin by design: the image-analyzer instantiation lives
in ImageAnalysis (:func:`image_analysis.config.create_image_analyzer`),
and this factory only adds the scan-side wrapping —
:class:`Array2DScanAnalyzer` or :class:`Array1DScanAnalyzer` —
populated from the validated ``scan:`` block.

Pattern:

>>> from image_analysis.config import load_diagnostic
>>> from scan_analysis.config import create_scan_analyzer
>>> diag = load_diagnostic("UC_VisaEBeam1")
>>> analyzer = create_scan_analyzer(diag)
>>> analyzer.run_analysis(scan_tag)

For the group-loader path (production), the wrapping ``id`` and
``priority`` come from the group's :class:`ResolvedDiagnosticConfig`
rather than from the diagnostic's own ``scan.priority`` — pass them
explicitly via the keyword arguments.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

from image_analysis.config import (
    DiagnosticAnalysisConfig,
    Line1DConfig,
    create_image_analyzer,
)

from .diagnostic_models import ScanRuntimeConfig

if TYPE_CHECKING:
    from scan_analysis.base import ScanAnalyzer

logger = logging.getLogger(__name__)

__all__ = ["create_scan_analyzer"]


def create_scan_analyzer(
    diag: DiagnosticAnalysisConfig,
    *,
    id: Optional[str] = None,
    priority: Optional[int] = None,
) -> "ScanAnalyzer":
    """Build a ScanAnalyzer from a validated diagnostic config.

    Composition: ``create_image_analyzer(diag)`` builds the inner
    ``ImageAnalyzer``, then this function wraps it in the
    dimension-specific scan-analyzer class (chosen by the type of
    ``diag.image`` — :class:`Line1DConfig` → 1D wrapper, anything else
    → 2D wrapper) and attaches the runtime metadata the task queue
    needs.

    Parameters
    ----------
    diag : DiagnosticAnalysisConfig
        Validated diagnostic. Both halves consumed here:
        ``image_analyzer`` + ``image`` go to
        :func:`image_analysis.config.create_image_analyzer`, ``scan``
        is parsed into a :class:`ScanRuntimeConfig`.
    id : str, optional
        Task-queue ID for the analyzer instance. Defaults to
        ``diag.name``. The group loader passes the filename stem here
        (which may differ from ``diag.name``).
    priority : int, optional
        Effective execution priority. Defaults to the diagnostic's own
        ``scan.priority``. The group loader passes the
        per-group-overridden value when present.

    Returns
    -------
    ScanAnalyzer
        Configured ``Array1DScanAnalyzer`` or ``Array2DScanAnalyzer``
        with ``id`` / ``priority`` / ``gdoc_slot`` / ``background_source``
        attached as instance attributes.

    Raises
    ------
    ValueError
        Propagated from :func:`create_image_analyzer` or from
        validating ``diag.scan`` against :class:`ScanRuntimeConfig`.
    TypeError
        If the resolved wrapper class can't be instantiated with the
        inferred kwargs.
    """
    image_analyzer = create_image_analyzer(diag)
    # diag.scan is weakly typed at the ImageAnalysis layer; validate
    # against the scan-side runtime model here.
    scan_cfg = ScanRuntimeConfig.model_validate(diag.scan or {})

    effective_id = id if id is not None else diag.name
    effective_priority = priority if priority is not None else scan_cfg.priority

    return _wrap_in_scan_analyzer(
        diag=diag,
        scan_cfg=scan_cfg,
        image_analyzer=image_analyzer,
        analyzer_id=effective_id,
        priority=effective_priority,
    )


def _wrap_in_scan_analyzer(
    *,
    diag: DiagnosticAnalysisConfig,
    scan_cfg: ScanRuntimeConfig,
    image_analyzer: Any,
    analyzer_id: str,
    priority: int,
) -> "ScanAnalyzer":
    """Construct the dimension-specific scan-analyzer wrapper.

    The 1D and 2D wrappers differ only in their save-flag kwarg name
    (``flag_save_data`` vs ``flag_save_images``). Everything else maps
    directly from the ``scan:`` section.

    Dispatch is on the type of ``diag.image``:
    :class:`Line1DConfig` → :class:`Array1DScanAnalyzer`; anything else
    (including the ``image`` is None HASO case) → :class:`Array2DScanAnalyzer`.
    """
    if isinstance(diag.image, Line1DConfig):
        from scan_analysis.analyzers.common.array1d_scan_analysis import (
            Array1DScanAnalyzer,
        )

        wrapper_class: Type["ScanAnalyzer"] = Array1DScanAnalyzer
        save_kwarg = "flag_save_data"
    else:
        from scan_analysis.analyzers.common.array2D_scan_analysis import (
            Array2DScanAnalyzer,
        )

        wrapper_class = Array2DScanAnalyzer
        save_kwarg = "flag_save_images"

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
            f"Failed to wrap {diag.image_analyzer.class_path} in "
            f"{wrapper_class.__name__} for diagnostic '{diag.name}': {exc}"
        ) from exc

    # Task-queue / scan-log metadata attached as instance attributes.
    analyzer.id = analyzer_id
    analyzer.priority = priority
    analyzer.gdoc_slot = scan_cfg.gdoc_slot
    # The directive is consumed at run time inside
    # SingleDeviceScanAnalyzer._resolve_background_paths. ``None`` is the
    # common case (no scan-context bg needed); the runtime check is
    # ``getattr(analyzer, "background_source", None)``.
    analyzer.background_source = scan_cfg.background_source

    return analyzer
