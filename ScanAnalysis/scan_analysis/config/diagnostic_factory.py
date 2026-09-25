"""Build a runnable scan-analyzer from either analysis document.

Two routes sit behind one call. An analysis recipe (the v3
:class:`~geecs_schemas.analysis.AnalysisRecipe`) always runs on the core; a
v2 diagnostic the analysis core can run
(:func:`scan_analysis.core_analyzer.core_supports`: beam/line/standard/trace
kinds, ported processing steps, no scan-context background) becomes a
:class:`~scan_analysis.core_analyzer.CoreScanAnalyzer` too. Every other v2
recipe, and any caller passing ``use_injected_data=True``, keeps the legacy wrapping:
the image-analyzer instantiation in ImageAnalysis
(:func:`image_analysis.config.create_image_analyzer`) inside
:class:`Array2DScanAnalyzer` or :class:`Array1DScanAnalyzer`, populated from
the typed ``scan:`` section. Both routes honour the same ``ScanAnalyzer``
contract, so the task queue, the portal and MCP never see the difference.

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
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Type

from geecs_schemas.analysis import (
    AnalysisDiagnostic,
    AnalysisRecipe,
    Line1DConfig,
    ScanRuntime,
)
from image_analysis.config import create_image_analyzer

from geecs_analysis.compat.v2 import compile_v2
from geecs_analysis.recipe import AnalysisDocument

from scan_analysis.core_analyzer import CoreScanAnalyzer, core_supports

if TYPE_CHECKING:
    from scan_analysis.base import ScanAnalyzer

logger = logging.getLogger(__name__)

__all__ = ["create_scan_analyzer"]


def create_scan_analyzer(
    diag: AnalysisDocument,
    *,
    id: Optional[str] = None,
    priority: Optional[int] = None,
    use_injected_data: bool = False,
    route: Literal["auto", "core", "legacy"] = "auto",
) -> "ScanAnalyzer":
    """Build a ScanAnalyzer from a validated analysis document.

    An :class:`AnalysisRecipe` (v3) always becomes a :class:`CoreScanAnalyzer`;
    it has no legacy route, so ``route="legacy"`` and ``use_injected_data``
    are refused for it and a recipe that does not bind to the core's
    registry raises ``RecipeError`` here. For a v2 diagnostic, routing
    (``route="auto"``): when :func:`core_supports` accepts the recipe
    (a compile-only check, no reads) and ``use_injected_data`` is False, the
    result is a :class:`CoreScanAnalyzer` running on ``geecs_analysis``.
    ``route="legacy"`` forces the wrapper for a supported recipe (the
    observation-period escape hatch and the comparison harness's oracle);
    ``route="core"`` forces the core and raises ``UnsupportedRecipe`` for a
    recipe it cannot run. Otherwise
    ``create_image_analyzer(diag)`` builds the inner ``ImageAnalyzer`` and
    this function wraps it in the dimension-specific legacy scan-analyzer
    class (chosen by the type of ``diag.image`` — :class:`Line1DConfig` →
    1D wrapper, anything else → 2D wrapper). Both carry the runtime
    metadata the task queue needs.

    Parameters
    ----------
    diag : AnalysisRecipe or AnalysisDiagnostic
        Validated document. For a v2 diagnostic, ``analyzer`` + ``image`` go to
        :func:`image_analysis.config.create_image_analyzer`; the typed
        ``scan`` section (:class:`ScanRuntime`) drives the wrapper.
    id : str, optional
        Task-queue ID for the analyzer instance. Defaults to
        ``diag.source_id`` when the diagnostic was loaded from a YAML
        file, otherwise ``diag.name``. The group loader passes the
        filename stem here explicitly (which may differ from
        ``diag.name``).
    priority : int, optional
        Effective execution priority. Defaults to the diagnostic's own
        ``scan.priority``. The group loader passes the
        per-group-overridden value when present.
    use_injected_data : bool, default=False
        When ``False`` (default), the wrapper analyzer loads its s-file
        from disk after the scan completes. When ``True``, the caller is
        responsible for setting ``analyzer.auxiliary_data`` before
        each ``run_analysis`` call. See
        :class:`scan_analysis.base.ScanAnalyzer` for the full contract.
        Legacy wrappers only.
    route : {"auto", "core", "legacy"}, default="auto"
        Which implementation runs the recipe; see above.

    Returns
    -------
    ScanAnalyzer
        A ``CoreScanAnalyzer`` carrying ``id`` / ``priority``, or a
        configured ``Array1DScanAnalyzer`` / ``Array2DScanAnalyzer`` with
        ``id`` / ``priority`` / ``background_source`` attached as instance
        attributes.

    Raises
    ------
    ValueError
        Propagated from :func:`create_image_analyzer` or from
        building the inner analyzer.
    TypeError
        If the resolved wrapper class can't be instantiated with the
        inferred kwargs.
    """
    source_id = getattr(diag, "source_id", None)
    effective_priority = priority if priority is not None else diag.scan.priority

    if route not in ("auto", "core", "legacy"):
        raise ValueError(f"route must be auto, core or legacy, not {route!r}")
    if isinstance(diag, AnalysisRecipe):
        if route == "legacy":
            raise ValueError("An analysis recipe (v3) has no legacy route")
        if use_injected_data:
            raise ValueError("The core route has no injected-data mode")
        effective_id = id if id is not None else source_id or diag.device
        logger.debug("Routing recipe %r to the analysis core", effective_id)
        return CoreScanAnalyzer(diag, id=effective_id, priority=effective_priority)

    scan_cfg: ScanRuntime = diag.scan  # typed in-document since v2
    effective_id = id if id is not None else source_id or diag.name
    if route == "core" or (
        route == "auto" and not use_injected_data and core_supports(diag)
    ):
        if use_injected_data:
            raise ValueError("The core route has no injected-data mode")
        if route == "core":
            compile_v2(diag, allow_file_backgrounds=True)  # surfaces UnsupportedRecipe
        logger.debug("Routing diagnostic %r to the analysis core", effective_id)
        return CoreScanAnalyzer(diag, id=effective_id, priority=effective_priority)

    image_analyzer = create_image_analyzer(diag)
    return _wrap_in_scan_analyzer(
        diag=diag,
        scan_cfg=scan_cfg,
        image_analyzer=image_analyzer,
        analyzer_id=effective_id,
        priority=effective_priority,
        use_injected_data=use_injected_data,
    )


def _wrap_in_scan_analyzer(
    *,
    diag: AnalysisDiagnostic,
    scan_cfg: ScanRuntime,
    image_analyzer: Any,
    analyzer_id: str,
    priority: int,
    use_injected_data: bool,
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

    # ``device_name`` is the GEECS device identifier (used as the auxiliary
    # data lookup prefix and the background-image folder name). ``scan.device``
    # is an *optional* data subfolder override for the rare case where the
    # data folder name differs from the device's metric prefix
    # (e.g. post-processed/stitched outputs that live in a sibling folder).
    # Keep them separate — the previous form `scan_cfg.device or diag.name`
    # collapsed them, which silently misrouted background-image paths
    # whenever ``scan.device`` was set.
    wrapper_kwargs: Dict[str, Any] = {
        "device_name": diag.name,
        "data_device_name": scan_cfg.device,
        "image_analyzer": image_analyzer,
        # Only the options the YAML set — the renderer's own defaults apply
        # to the rest (RendererOptions replaced the renderer_kwargs dict).
        "renderer_kwargs": scan_cfg.renderer.as_kwargs(),
        "analysis_mode": scan_cfg.mode,
        "use_injected_data": use_injected_data,
        # Output naming (#412). ImageAnalysis emits bare scalar keys;
        # ScanAnalysis is the sole authority for naming them on the way
        # to disk / memory consumers, and for labelling per-analyzer
        # output directories. ``effective_output_name`` returns
        # ``output_name`` when set on the diagnostic, otherwise falls
        # back to ``diag.name`` — so YAMLs that don't set
        # ``output_name`` produce columns/dirs identical to the
        # device name.
        "output_name": diag.effective_output_name,
        "metric_suffix": diag.metric_suffix or "",
        save_kwarg: scan_cfg.save,
    }
    if scan_cfg.file_tail is not None:
        wrapper_kwargs["file_tail"] = scan_cfg.file_tail
    if scan_cfg.data_format is not None:
        wrapper_kwargs["data_format"] = scan_cfg.data_format

    try:
        analyzer = wrapper_class(**wrapper_kwargs)
    except TypeError as exc:
        raise TypeError(
            f"Failed to wrap analyzer kind {diag.analyzer.kind!r} in "
            f"{wrapper_class.__name__} for diagnostic '{diag.name}': {exc}"
        ) from exc

    # Task-queue / scan-log metadata attached as instance attributes.
    analyzer.id = analyzer_id
    analyzer.priority = priority
    # The directive is consumed at run time inside
    # SingleDeviceScanAnalyzer._resolve_background_paths. ``None`` is the
    # common case (no scan-context bg needed); the runtime check is
    # ``getattr(analyzer, "background_source", None)``.
    analyzer.background_source = scan_cfg.background_source

    return analyzer
