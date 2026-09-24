"""Portal processing/preview routing during the analysis-core migration.

Analysis recipes (v3) and supported v2 recipes use geecs-analysis. Only a v2
compilation-time capability refusal selects the legacy write-free route;
numerical/render failures never retry against another backend. Configuration
reads use an explicit tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
from geecs_data_utils.analysis_configs import discover_diagnostics, read_diagnostic
from geecs_data_utils.frames import Frame
from geecs_schemas.analysis import (
    AnalysisDiagnostic,
    AnalysisDocument,
    load_analysis_document,
)
from pydantic import ValidationError

from geecs_analysis.compat.v2 import UnsupportedRecipe
from geecs_analysis.measurement import Measurement
from geecs_analysis.recipe import figure_of
from geecs_analysis.render import RenderError, single
from geecs_analysis.render.specs import FigureSpec

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def list_diagnostics(*, config_dir: Path) -> list[str]:
    """List diagnostic stems under the portal's explicitly configured tree."""
    return sorted(discover_diagnostics(Path(config_dir)))


def load_diagnostic(name: str, *, config_dir: Path) -> AnalysisDocument:
    """Read and validate a fresh document (either format) without importing legacy analyzers."""
    path, data = read_diagnostic(name, config_dir=Path(config_dir))
    try:
        document = load_analysis_document(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid diagnostic config at {path}: {exc}") from exc
    document._source_id = path.stem
    return document


def process_images(
    name: str, arrays: Sequence[np.ndarray], *, config_dir: Path
) -> list[np.ndarray | None]:
    """Process each member independently; None preserves the non-image refusal."""
    from scan_analysis.core_preview import measure_frame, prepare_document

    document = load_diagnostic(name, config_dir=config_dir)
    try:
        # the batch route has no scan folder in hand: a recipe's frame inputs
        # keep their placeholder here (fallback level, or an error), as before
        prepared = prepare_document(document)
    except UnsupportedRecipe:
        from image_analysis.ephemeral import run_document_ephemeral

        return [
            getattr(result, "processed_image", None)
            for result in run_document_ephemeral(document, arrays)
        ]
    results = [measure_frame(prepared, array) for array in arrays]
    return [r.frame.data if r.frame.data.ndim == 2 else None for r in results]


def _style(
    array: np.ndarray,
    *,
    window: tuple[float, float] | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> FigureSpec:
    """Preserve the portal's finite-pixel window and explicit-limit precedence."""
    limits = {}
    if array.ndim == 2:
        if window is not None:
            finite = np.asarray(array, dtype=float)[np.isfinite(array)]
            if finite.size >= 2:
                lo, hi = np.percentile(finite, window)
                if lo < hi:
                    limits.update(vmin=float(lo), vmax=float(hi))
        if vmin is not None:
            limits["vmin"] = vmin
        if vmax is not None:
            limits["vmax"] = vmax
    palette = {"cmap": cmap or "plasma", **limits}
    return FigureSpec(
        fig={"figsize": (5.0, 4.2), "dpi": 110},
        imshow=palette,
        pcolormesh=palette,
        colorbar={"shrink": 0.65},
    )


def _render(
    document: AnalysisDocument,
    arrays: Sequence[np.ndarray],
    style_for: Callable[[Measurement], FigureSpec],
    *,
    scan_folder: Path | None = None,
    auxiliary_data: dict | None = None,
    legacy: dict | None = None,
) -> list[Figure]:
    """Draw each frame through the run's own analysis, or the legacy route.

    The analysis is ScanAnalysis' (``core_preview.prepare_document`` +
    ``measure_frame``: the run's compile, frame inputs resolved under
    *scan_folder*, the run's per-frame measure); only the style differs
    per caller. Only a compilation-time capability refusal
    (``UnsupportedRecipe``: a v2 kind the core has not ported) selects the
    legacy write-free route, with ``legacy`` as its keyword arguments;
    numerical and render failures never retry against another backend.
    """
    from scan_analysis.core_preview import measure_frame, prepare_document

    try:
        prepared = prepare_document(document, scan_folder=scan_folder)
    except UnsupportedRecipe:
        from image_analysis import ephemeral

        try:
            return ephemeral.render_document_ephemeral(
                document, arrays, auxiliary_data=auxiliary_data, **(legacy or {})
            )
        except ephemeral.RenderError as exc:
            raise RenderError(str(exc)) from exc
    figures = []
    for array in arrays:
        result = measure_frame(prepared, array)
        figures.append(single(result, style_for(result)))
    return figures


def _legacy_palette(document: AnalysisDocument) -> dict | None:
    """The v2 renderer's palette for the legacy route; a recipe never takes it."""
    if not isinstance(document, AnalysisDiagnostic):
        return None
    opts = document.scan.renderer
    return {"cmap": opts.cmap, "vmin": opts.vmin, "vmax": opts.vmax}


def render_document_ephemeral(
    document: AnalysisDocument,
    arrays: Sequence[np.ndarray],
    *,
    window: tuple[float, float] | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    auxiliary_data: dict | None = None,
) -> list[Figure]:
    """Draw the supplied (possibly unsaved) document with the portal's own styling.

    The Images tab's processing view: the portal's finite-pixel window and
    palette over the processed frame. ``auxiliary_data`` reaches only the
    legacy fallback (a line trace's auxiliary columns, which e.g. the FROG
    phase analyzer reads); the core route reads the primary trace alone,
    exactly as its scan run does.
    """
    return _render(
        document,
        arrays,
        lambda r: _style(r.frame.data, window=window, cmap=cmap, vmin=vmin, vmax=vmax),
        auxiliary_data=auxiliary_data,
        legacy={"window": window, "cmap": cmap, "vmin": vmin, "vmax": vmax},
    )


def render_document_as_run(
    document: AnalysisDocument,
    arrays: Sequence[np.ndarray],
    *,
    scan_folder: Path | None = None,
    auxiliary_data: dict | None = None,
) -> list[Figure]:
    """Draw each frame the way a scan run of *document* draws its products.

    ScanAnalysis' ``core_preview.preview_frame`` — the sink's per-frame
    call with the document's own figure block, the recipe's frame inputs
    loaded from its device folder under *scan_folder* as the run loads
    them — so the editor's preview IS the product image; no portal palette
    or window reaches it. Kinds the core does not serve fall back to the
    legacy write-free route, which draws its own figure from the
    diagnostic's renderer fields.
    """
    from scan_analysis.core_preview import preview_frame

    try:
        return [
            preview_frame(document, array, scan_folder=scan_folder) for array in arrays
        ]
    except UnsupportedRecipe:
        return _render(
            document,
            arrays,
            lambda _r: figure_of(document),
            scan_folder=scan_folder,
            auxiliary_data=auxiliary_data,
            legacy=_legacy_palette(document),
        )


def render_summary_as_run(
    document: AnalysisDocument,
    arrays: Sequence[np.ndarray],
    positions: Sequence[float | None],
    label: str,
    index: int,
    *,
    scan_folder: Path | None = None,
) -> Figure:
    """The document's ``index``-th summary over *arrays*, as the run's sink draws it.

    ScanAnalysis' ``core_preview.preview_summary``: one panel per array at
    its position for a kind that consumes panels, the arrays' average for
    the ``average`` kind. A v2 kind the core does not serve has no summary
    preview (``ValueError``): the legacy route draws no scan figure from a
    handful of frames.
    """
    from scan_analysis.core_preview import preview_summary

    try:
        return preview_summary(
            document, arrays, positions, label, index, scan_folder=scan_folder
        )
    except UnsupportedRecipe as exc:
        raise ValueError(
            "no summary preview for an analyzer kind the analysis core does not "
            f"run yet: {exc}"
        ) from exc


def render_diagnostic_ephemeral(
    name: str,
    arrays: Sequence[np.ndarray],
    *,
    config_dir: Path,
    window: tuple[float, float] | None = None,
    cmap: str | None = None,
) -> list[Figure]:
    """Draw a named recipe using the same path as unsaved editor previews."""
    return render_document_ephemeral(
        load_diagnostic(name, config_dir=config_dir), arrays, window=window, cmap=cmap
    )


def render_frame_figure(
    array: np.ndarray,
    *,
    window: tuple[float, float] | None = None,
    cmap: str | None = None,
) -> Figure:
    """Draw a processed bin average without inventing per-shot overlays."""
    try:
        if array.ndim != 2:
            raise ValueError("A bin image must be 2D")
        return single(
            Measurement(scalars={}, frame=Frame.from_array(array)),
            _style(array, window=window, cmap=cmap),
        )
    except (TypeError, ValueError) as exc:
        raise RenderError(str(exc)) from exc
