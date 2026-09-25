"""Legacy post-analysis averages, distinct from raw-bin evaluation.

This compatibility operation deliberately averages stored trace coordinates as
well as samples. It is not a general physical-grid alignment or resampler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

if TYPE_CHECKING:
    from geecs_analysis.compat.v2 import V2Recipe
    from geecs_analysis.measurement import Measurement


def average_results(
    results: Sequence[Measurement],
    recipe: V2Recipe,
    *,
    mode: Literal["noscan", "bin"],
) -> Measurement | None:
    """Average processed results using the selected legacy summary convention.

    Noscan uses mean for samples/scalars and omits shot overlays. Bin summaries
    use nanmean, retain the first nonempty scalar map's keys and average
    projection overlays. Neither mode re-runs a measure on the averaged frame.
    Trace coordinates and samples reduce at the v2 storage dtype, including
    float32 accumulation. Empty or mixed-shape inputs return None so the host
    can skip the averaged figure without losing per-shot scalar products.
    Units/rank must agree; camera axes must match. Trace axes are intentionally
    averaged index-wise for v2 compatibility, not silently interpolated.
    """
    import numpy as np
    from geecs_data_utils.frames import Frame
    from geecs_analysis.measurement import Marker, Measurement, Projection

    if mode not in ("noscan", "bin"):
        raise ValueError("Average mode must be noscan or bin")
    if not results:
        return None
    first = results[0].frame
    rank = 1 if recipe.input_kind == "line" else 2
    if any(r.frame.data.ndim != rank for r in results):
        raise ValueError("Result rank must match the recipe input kind")
    if len({r.frame.data.shape for r in results}) != 1:
        return None
    if any(
        r.frame.unit != first.unit
        or tuple(a.unit for a in r.frame.axes) != tuple(a.unit for a in first.axes)
        for r in results
    ):
        raise ValueError("Averaged results must have matching units")
    if rank == 2 and any(
        not all(
            np.array_equal(a.values, b.values)
            for a, b in zip(r.frame.axes, first.axes, strict=True)
        )
        for r in results
    ):
        raise ValueError("Averaged camera results must have matching axes")
    reduce = np.mean if mode == "noscan" else np.nanmean
    if rank == 1:
        trace = reduce(
            [r.frame.as_trace().astype(recipe.storage_dtype) for r in results], axis=0
        )
        frame = Frame.from_trace(
            trace,
            x_unit=first.axes[0].unit,
            x_label=first.axes[0].label,
            y_unit=first.unit,
            y_label=first.label,
        )
    else:
        frame = Frame.from_array(
            reduce([r.frame.data for r in results], axis=0),
            axes=first.axes,
            unit=first.unit,
            label=first.label,
        )
    maps = [r.scalars for r in results if r.scalars]
    if mode == "noscan" and not results[0].scalars:
        maps = []
    keys = (
        dict.fromkeys(key for values in maps for key in values)
        if mode == "noscan"
        else (maps[0] if maps else {})
    )
    scalars = {}
    for key in keys:
        values = [values[key] for values in maps if key in values]
        scalars[key] = (
            float("nan") if all(np.isnan(v) for v in values) else float(reduce(values))
        )
    overlays = []
    if mode == "bin":
        by_id = {}
        for result in results:
            for overlay in result.overlays:
                by_id.setdefault(overlay.id, []).append(overlay)
        for id, group in by_id.items():
            template = group[0]
            if isinstance(template, Projection):
                if any(
                    not isinstance(o, Projection) or o.axis != template.axis
                    for o in group
                ):
                    raise ValueError(
                        "Averaged overlay ids must retain their type and axis"
                    )
                if len({o.frame.data.shape for o in group}) != 1:
                    continue
                if any(
                    o.frame.unit != template.frame.unit
                    or o.frame.axes[0].unit != template.frame.axes[0].unit
                    or not np.array_equal(
                        o.frame.axes[0].values, template.frame.axes[0].values
                    )
                    for o in group
                ):
                    raise ValueError(
                        "Averaged projections must share coordinates and units"
                    )
                projection = Frame.from_array(
                    np.nanmean([o.frame.data for o in group], axis=0),
                    axes=template.frame.axes,
                    unit=template.frame.unit,
                    label=template.frame.label,
                )
                overlays.append(Projection(id, template.axis, projection))
            else:
                if any(not isinstance(o, Marker) for o in group):
                    raise ValueError("Averaged overlay ids must retain their type")
                x, y = np.nanmean([(o.x, o.y) for o in group], axis=0)
                if np.isfinite(x) and np.isfinite(y):
                    overlays.append(Marker(id, float(x), float(y)))
    return Measurement(scalars, frame, tuple(overlays))
