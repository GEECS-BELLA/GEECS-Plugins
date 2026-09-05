"""Figure options for the per-scan summary renderers — ``scan.renderer`` on a diagnostic.

One typed option set covers both renderers.  Fields that apply only to
trace (1D) figures or only to camera (2D) figures are named as such and
:class:`~geecs_schemas.analysis.diagnostic.AnalysisDiagnostic` refuses one
set on the wrong image kind.  Every field is optional: an unset field lets
the renderer use its own default, so a diagnostic states only what it
wants changed — the same behaviour as the untyped ``renderer_kwargs``
dictionary this replaces.
"""

from __future__ import annotations

from typing import ClassVar, Literal, Optional, Tuple

from pydantic import Field

from geecs_schemas._base import SchemaModel


class RendererOptions(SchemaModel):
    """Cosmetic overrides for the scan summary figures (colormap, labels, layout).

    Leave a field unset to keep the renderer's default.  The waterfall_*
    fields and ``mode`` apply to trace diagnostics only; ``figsize`` and
    ``figsize_inches`` to camera diagnostics only.
    """

    colormap_mode: Optional[Literal["auto", "sequential", "diverging", "custom"]] = (
        Field(
            None,
            description=(
                "'sequential' runs 0 to max; 'diverging' is symmetric about zero; "
                "'auto' picks diverging when the data crosses zero; 'custom' uses "
                "vmin/vmax as given."
            ),
        )
    )
    cmap: Optional[str] = Field(
        None, description="Matplotlib colormap name (e.g. 'plasma', 'RdBu_r')."
    )
    vmin: Optional[float] = Field(None, description="Colour scale minimum.")
    vmax: Optional[float] = Field(
        None, description="Colour scale maximum (2D: the old plot_scale)."
    )
    duration: Optional[float] = Field(
        None, gt=0, description="Animation frame duration, ms."
    )
    dpi: Optional[int] = Field(
        None, gt=0, description="Figure resolution, dots per inch."
    )
    xlabel: Optional[str] = Field(None, description="X-axis label.")
    ylabel: Optional[str] = Field(None, description="Y-axis label.")
    colorbar_label: Optional[str] = Field(None, description="Colour bar label.")

    # ---- trace (1D) figures only ----
    mode: Optional[Literal["waterfall", "overlay", "grid"]] = Field(
        None,
        description=(
            "Trace summary layout: 'waterfall' heat map (x vs scan parameter), "
            "'overlay' of all bins, or a 'grid' of one plot per bin. 1D only."
        ),
    )
    waterfall_sort_key: Optional[str] = Field(
        None,
        description=(
            "For a noscan waterfall, order rows by this s-file column instead "
            "of shot number ('Device:Var' or a substring). 1D only."
        ),
    )
    waterfall_sort_sigma: Optional[float] = Field(
        None,
        description=(
            "Drop shots whose sort-key value lies outside mean ± this many "
            "standard deviations. 1D only."
        ),
    )
    waterfall_sort_bounds: Optional[Tuple[float, float]] = Field(
        None,
        description="Explicit (low, high) bounds on the sort key; overrides the sigma cut. 1D only.",
    )
    waterfall_even_y_spacing: Optional[bool] = Field(
        None,
        description="Draw waterfall rows at equal height regardless of sort-key spacing. 1D only.",
    )

    # ---- camera (2D) figures only ----
    figsize: Optional[Tuple[float, float]] = Field(
        None, description="Panel (width, height) in inches for grid montages. 2D only."
    )
    figsize_inches: Optional[float] = Field(
        None, gt=0, description="Side of the square animation frames, inches. 2D only."
    )

    LINE_ONLY: ClassVar[frozenset[str]] = frozenset(
        {
            "mode",
            "waterfall_sort_key",
            "waterfall_sort_sigma",
            "waterfall_sort_bounds",
            "waterfall_even_y_spacing",
        }
    )
    CAMERA_ONLY: ClassVar[frozenset[str]] = frozenset({"figsize", "figsize_inches"})

    def as_kwargs(self) -> dict:
        """Return only the options that were set, as renderer constructor kwargs."""
        return self.model_dump(exclude_none=True)

    def fields_set_for(self, only: frozenset) -> list[str]:
        """Return the names in *only* that carry a value."""
        return sorted(name for name in only if getattr(self, name) is not None)


__all__ = ["RendererOptions"]
