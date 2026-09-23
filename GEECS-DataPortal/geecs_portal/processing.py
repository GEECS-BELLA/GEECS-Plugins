"""Portal processing/preview routing during the analysis-core migration.

Supported v2 recipes use geecs-analysis. Only a compilation-time capability
refusal selects the legacy write-free route; numerical/render failures never
retry against another backend. Configuration reads use an explicit tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from geecs_data_utils.analysis_configs import discover_diagnostics, read_diagnostic
from geecs_data_utils.frames import Frame
from geecs_schemas.analysis import AnalysisDiagnostic
from pydantic import ValidationError

from geecs_analysis.compat.v2 import UnsupportedRecipe, analyze_v2, compile_v2
from geecs_analysis.measurement import Measurement
from geecs_analysis.render import RenderError, single
from geecs_analysis.render.specs import FigureSpec

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def list_diagnostics(*, config_dir: Path) -> list[str]:
    """List diagnostic stems under the portal's explicitly configured tree."""
    return sorted(discover_diagnostics(Path(config_dir)))


def load_diagnostic(name: str, *, config_dir: Path) -> AnalysisDiagnostic:
    """Read and validate a fresh diagnostic without importing legacy analyzers."""
    path, data = read_diagnostic(name, config_dir=Path(config_dir))
    try:
        document = AnalysisDiagnostic.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid diagnostic config at {path}: {exc}") from exc
    document._source_id = path.stem
    return document


def process_images(
    name: str, arrays: Sequence[np.ndarray], *, config_dir: Path
) -> list[np.ndarray | None]:
    """Process each member independently; None preserves the non-image refusal."""
    document = load_diagnostic(name, config_dir=config_dir)
    try:
        recipe = compile_v2(document)
    except UnsupportedRecipe:
        from image_analysis.ephemeral import run_document_ephemeral

        return [
            getattr(result, "processed_image", None)
            for result in run_document_ephemeral(document, arrays)
        ]
    results = [analyze_v2(array, recipe) for array in arrays]
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


def render_document_ephemeral(
    document: AnalysisDiagnostic,
    arrays: Sequence[np.ndarray],
    *,
    window: tuple[float, float] | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> list[Figure]:
    """Draw the supplied (possibly unsaved) document without filesystem writes."""
    try:
        recipe = compile_v2(document)
    except UnsupportedRecipe:
        from image_analysis import ephemeral

        try:
            return ephemeral.render_document_ephemeral(
                document, arrays, window=window, cmap=cmap, vmin=vmin, vmax=vmax
            )
        except ephemeral.RenderError as exc:
            raise RenderError(str(exc)) from exc
    figures = []
    for array in arrays:
        result = analyze_v2(array, recipe)
        figures.append(
            single(
                result,
                _style(
                    result.frame.data, window=window, cmap=cmap, vmin=vmin, vmax=vmax
                ),
            )
        )
    return figures


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
