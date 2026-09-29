"""Legacy post-analysis averages, distinct from raw-bin evaluation.

This compatibility operation deliberately averages stored trace coordinates as
well as samples. It is not a general physical-grid alignment or resampler.

:class:`RunningAverage` folds one measurement at a time so a scan's average
needs one frame of memory however many shots it has; :func:`average_results`
is the same reduction over a sequence already in hand, built on it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

if TYPE_CHECKING:
    import numpy as np
    from geecs_data_utils.frames import Frame
    from geecs_analysis.compat.v2 import V2Recipe
    from geecs_analysis.measurement import Marker, Measurement, Projection

AverageMode = Literal["noscan", "bin"]


class _Sum:
    """A float64 running sum of equally shaped arrays, with a per-element count.

    In ``bin`` mode NaN samples are left out of both the sum and the count
    (``nanmean``); in ``noscan`` mode every sample counts and NaN propagates
    (``mean``). Numpy reduces a stack along its first axis in exactly this
    sequential order, so the quotient equals the stacked reduction bit for
    bit — it is a memory shape, not a numerical approximation. ``dtype`` is
    the accumulator's (float64 unless a caller matches another reduction's
    intermediate, as the raw-bin mean does for float32 samples).
    """

    def __init__(
        self,
        first: np.ndarray,
        *,
        skip_nan: bool,
        dtype: np.dtype | type | None = None,
    ) -> None:
        import numpy as np

        self.skip_nan = skip_nan
        self.total = np.array(first, dtype=dtype or np.float64, copy=True)
        self.count = 1
        if skip_nan:
            missing = np.isnan(self.total)
            self.total[missing] = 0.0
            self.counts = np.asarray(~missing, dtype=np.float64)

    def add(self, data: np.ndarray) -> None:
        """Fold one more array of the shape the first one had."""
        import numpy as np

        self.count += 1
        if not self.skip_nan:
            self.total += data
            return
        missing = np.isnan(data)
        self.total += np.where(missing, 0.0, data)
        self.counts += ~missing

    def mean(self) -> np.ndarray:
        """The mean so far; an all-NaN element is NaN, as ``nanmean`` gives."""
        import numpy as np

        if not self.skip_nan:
            return self.total / self.count
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.total / self.counts


class _ProjectionAverage:
    """The bin-mode average of one projection id across measurements."""

    def __init__(self, template: Projection) -> None:
        self.template = template
        self.sum = _Sum(template.frame.data, skip_nan=True)
        self.mixed = False

    def add(self, overlay: Projection) -> None:
        import numpy as np

        template = self.template
        if not isinstance(overlay, type(template)) or overlay.axis != template.axis:
            raise ValueError("Averaged overlay ids must retain their type and axis")
        if self.mixed:
            return
        if overlay.frame.data.shape != template.frame.data.shape:
            self.mixed = True
            return
        if (
            overlay.frame.unit != template.frame.unit
            or overlay.frame.axes[0].unit != template.frame.axes[0].unit
            or not np.array_equal(
                overlay.frame.axes[0].values, template.frame.axes[0].values
            )
        ):
            raise ValueError("Averaged projections must share coordinates and units")
        self.sum.add(overlay.frame.data)

    def result(self) -> Projection | None:
        from geecs_data_utils.frames import Frame
        from geecs_analysis.measurement import Projection

        if self.mixed:
            return None
        template = self.template
        frame = Frame.from_array(
            self.sum.mean(),
            axes=template.frame.axes,
            unit=template.frame.unit,
            label=template.frame.label,
        )
        return Projection(template.id, template.axis, frame)


class _MarkerAverage:
    """The bin-mode average of one marker id: each coordinate over its finite values."""

    def __init__(self, template: Marker) -> None:
        import numpy as np

        self.id = template.id
        self.sum = _Sum(np.array([template.x, template.y]), skip_nan=True)

    def add(self, overlay: Marker) -> None:
        import numpy as np
        from geecs_analysis.measurement import Marker

        if not isinstance(overlay, Marker):
            raise ValueError("Averaged overlay ids must retain their type")
        self.sum.add(np.array([overlay.x, overlay.y]))

    def result(self) -> Marker | None:
        import numpy as np
        from geecs_analysis.measurement import Marker

        x, y = self.sum.mean()
        if np.isfinite(x) and np.isfinite(y):
            return Marker(self.id, float(x), float(y))
        return None


class RunningAverage:
    """Fold processed results one at a time into a legacy summary average.

    The conventions are :func:`average_results`'s (it is built on this
    class): noscan uses mean for samples/scalars and omits shot overlays;
    bin summaries use nanmean, retain the first nonempty scalar map's keys
    and average projection overlays. Neither mode re-runs a measure on the
    averaged frame. Camera frames, projections and markers accumulate as
    float64 sums, so memory is one frame however many results are folded,
    and the sequential fold is numpy's own order for reducing a stack along
    its first axis: the quotient equals ``np.mean`` / ``np.nanmean`` over the
    stack bit for bit for any frame of more than one element (numpy reduces a
    stack of 1×1 frames along a contiguous axis, pairwise). Scalars are a few
    floats per result and are kept and
    reduced at the end, because numpy's pairwise sum over a 1-D vector is
    not a running sum. Trace results (rank 1) are kept and reduced at the
    storage dtype at the end, exactly as before; a scan keeps every trace
    for its waterfall anyway.

    :meth:`add` refuses a result whose rank, units or camera axes disagree
    with the first one. A result of another shape marks the average mixed:
    :meth:`result` is then ``None`` so the host can skip the averaged
    figure without losing per-shot scalar products (further results are
    still folded for their scalars only). :attr:`count` is the number of
    results folded.
    """

    def __init__(self, recipe: V2Recipe, *, mode: AverageMode) -> None:
        if mode not in ("noscan", "bin"):
            raise ValueError("Average mode must be noscan or bin")
        self.recipe = recipe
        self.mode = mode
        self.rank = 1 if recipe.input_kind == "line" else 2
        self.count = 0
        self.mixed = False
        self._first: Frame | None = None
        self._traces: list[np.ndarray] = []
        self._sum: _Sum | None = None
        # Scalar values by key in first-seen order; the first result's
        # emptiness and the first nonempty map's keys decide which are kept.
        self._values: dict[str, list[float]] = {}
        self._first_has_scalars = False
        self._bin_keys: tuple[str, ...] | None = None
        self._overlays: dict[str, _ProjectionAverage | _MarkerAverage] = {}

    def add(self, result: Measurement) -> None:
        """Fold one more processed result."""
        import numpy as np
        from geecs_analysis.measurement import Projection

        frame = result.frame
        if frame.data.ndim != self.rank:
            raise ValueError("Result rank must match the recipe input kind")
        first = self._first
        if first is None:
            self._first = first = frame
            self._first_has_scalars = bool(result.scalars)
        self.count += 1
        if frame.data.shape != first.data.shape:
            self.mixed = True
        elif self.count > 1:
            if frame.unit != first.unit or tuple(a.unit for a in frame.axes) != tuple(
                a.unit for a in first.axes
            ):
                raise ValueError("Averaged results must have matching units")
            if self.rank == 2 and not all(
                np.array_equal(a.values, b.values)
                for a, b in zip(frame.axes, first.axes, strict=True)
            ):
                raise ValueError("Averaged camera results must have matching axes")
        if not self.mixed:
            if self.rank == 1:
                self._traces.append(frame.as_trace().astype(self.recipe.storage_dtype))
            elif self._sum is None:
                self._sum = _Sum(frame.data, skip_nan=self.mode == "bin")
            else:
                self._sum.add(frame.data)
        if result.scalars:
            if self._bin_keys is None:
                self._bin_keys = tuple(result.scalars)
            for key, value in result.scalars.items():
                self._values.setdefault(key, []).append(value)
        if self.mode == "bin":
            for overlay in result.overlays:
                running = self._overlays.get(overlay.id)
                if running is None:
                    self._overlays[overlay.id] = (
                        _ProjectionAverage(overlay)
                        if isinstance(overlay, Projection)
                        else _MarkerAverage(overlay)
                    )
                else:
                    running.add(overlay)

    def result(self) -> Measurement | None:
        """The average of everything folded; ``None`` when empty or mixed."""
        import numpy as np
        from geecs_data_utils.frames import Frame
        from geecs_analysis.measurement import Measurement

        first = self._first
        if first is None or self.mixed:
            return None
        reduce = np.mean if self.mode == "noscan" else np.nanmean
        if self.rank == 1:
            frame = Frame.from_trace(
                reduce(self._traces, axis=0),
                x_unit=first.axes[0].unit,
                x_label=first.axes[0].label,
                y_unit=first.unit,
                y_label=first.label,
            )
        else:
            frame = Frame.from_array(
                self._sum.mean(), axes=first.axes, unit=first.unit, label=first.label
            )
        if self.mode == "noscan":
            keys = () if not self._first_has_scalars else tuple(self._values)
        else:
            keys = self._bin_keys or ()
        scalars = {}
        for key in keys:
            values = self._values[key]
            scalars[key] = (
                float("nan")
                if all(np.isnan(v) for v in values)
                else float(reduce(values))
            )
        overlays = tuple(
            averaged
            for averaged in (running.result() for running in self._overlays.values())
            if averaged is not None
        )
        return Measurement(scalars, frame, overlays)


def average_results(
    results: Sequence[Measurement],
    recipe: V2Recipe,
    *,
    mode: AverageMode,
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
    The fold is :class:`RunningAverage`'s, one definition for both.
    """
    running = RunningAverage(recipe, mode=mode)
    for result in results:
        running.add(result)
    return running.result()
