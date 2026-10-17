"""The core's runtime: the in-memory compiled recipe and the evaluator that runs it.

``geecs_analysis.recipe.compile_recipe`` produces a :class:`V2Recipe`;
:func:`analyze_v2` runs it on already-loaded samples, without loading
configs or input files. The legacy conventions the ported analyzers pinned
stay at this boundary: trace storage rounding before measurement, float64
trace-result negative clipping, and the beam ROI-origin convention. (The
module keeps its historical name until the rename that closes #1039.)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Mapping, Sequence


from geecs_analysis.pipeline import (
    apply_measure,
    apply_step,
    bind_inputs,
    process_measure_input,
)
from geecs_analysis.registry import MeasureSpec, StepSpec, measure_definition
from geecs_analysis.specs import Analysis
from geecs_analysis.steps.interpolate import InterpolateSpec
from geecs_analysis.steps.roi import RoiSpec

if TYPE_CHECKING:
    import numpy as np
    from geecs_data_utils.frames import Frame, ShotMeta
    from geecs_analysis.measurement import Measurement


class UnsupportedRecipe(ValueError):
    """A document the core cannot run (a ``.himg`` job, an unloadable trace channel)."""


@dataclass(frozen=True)
class FileBackground:
    """A source-layer request; the core never opens or resolves this path."""

    key: str
    path: str
    #: Constant to subtract when the file cannot be read; ``None`` makes a
    #: failed read an error (a v3 recipe's choice; v2 always has a level).
    fallback_level: float | None


@dataclass(frozen=True)
class ScanBackground:
    """A source-layer request for a frame computed from a scan's frames.

    ``scan_number`` ``None`` is the scan being analyzed. The core never reads
    a scan: the host computes the per-pixel ``statistic`` (``mean``,
    ``median`` or ``percentile``) and binds the frame under ``key``.
    """

    key: str
    scan_number: int | None
    statistic: Literal["mean", "median", "percentile"]
    percentile: float | None = None


@dataclass(frozen=True)
class TableInput:
    """A source-layer request for a tab-delimited table; the core never opens the path.

    The host reads it into a ``geecs_data_utils.tables.Table`` (a relative
    path against the analysis-configs root) and binds it under ``key``.
    """

    key: str
    path: str


@dataclass(frozen=True)
class V2Recipe:
    """An immutable in-memory recipe and explicit legacy conversion conventions."""

    analysis: Analysis
    input_kind: Literal["camera", "line"]
    device: str
    output_name: str
    metric_suffix: str | None
    storage_dtype: Literal["float32", "float64"] = "float64"
    x_scale: float = 1.0
    y_scale: float = 1.0
    x_unit: str = ""
    y_unit: str = ""
    label: str = ""
    camera_origin: tuple[int, int] = (0, 0)
    file_backgrounds: tuple[FileBackground, ...] = ()
    scan_backgrounds: tuple[ScanBackground, ...] = ()
    #: A line input's named auxiliary channels the measure reads, in the
    #: order the source appends them to the trace as columns 2, 3, …
    channels: tuple[str, ...] = ()
    #: The measure's table inputs (the magspec calibration surveys), read
    #: and bound by the host under their keys before the run.
    tables: tuple[TableInput, ...] = ()


def trace_channels(
    measure: MeasureSpec, auxiliary_columns: Mapping[str, int], steps: Sequence
) -> tuple[str, ...]:
    """The input's named auxiliary columns the measure reads, in declared order.

    A column the measure does not read is not loaded (the legacy line
    analyzers ignored theirs too). A channel follows the trace's sample
    selection (``roi``) but not a resample, so ``interpolate`` with a
    channel is refused, as the legacy preprocessing refused it.
    """
    declared = measure_definition(measure).channels
    channels = tuple(name for name in auxiliary_columns if name in declared)
    if channels and any(isinstance(spec, InterpolateSpec) for spec in steps):
        raise UnsupportedRecipe(
            f"Auxiliary channels {list(channels)} cannot follow interpolation"
        )
    return channels


def analyze_v2(
    data: np.ndarray,
    recipe: V2Recipe,
    *,
    shot: ShotMeta | None = None,
    inputs: Mapping[str, object] | None = None,
) -> Measurement:
    """Run already-loaded raw samples with legacy scaling/rounding conventions.

    Images use HxW arrays; traces use Nx2 arrays, plus one column per
    declared auxiliary channel (``recipe.channels``, in order). Sources keep
    native dtype until this boundary so trace axis scaling occurs in the
    same precision as before. Input arrays are always copied and never
    mutated. No paths are accepted.
    """
    import numpy as np
    from geecs_data_utils.frames import Axis, Frame

    bound = bind_inputs(recipe.analysis.steps, inputs, measure=recipe.analysis.measure)
    channels: dict[str, Frame] = {}
    if recipe.input_kind == "camera":
        if data.ndim != 2:
            raise ValueError("Camera input must be HxW")
        frame = Frame.from_array(data, shot=shot)
    else:
        frame, channels = _trace_input(data, recipe, shot)

    def process(frame: Frame) -> Frame:
        for spec in recipe.analysis.steps:
            if recipe.input_kind == "camera" and isinstance(spec, RoiSpec):  # noqa: SIM102 — reads clearer nested
                # Legacy returns the full image for a crop wholly outside the input.
                if any(
                    lo >= min(hi, size)
                    for (lo, hi), size in zip(
                        spec.bounds, frame.data.shape, strict=True
                    )
                ):
                    continue
            frame = apply_step(frame, spec, inputs=bound)
        return frame

    frame = process(frame)
    channels = _follow_selection(channels, recipe.analysis.steps, frame.data.size)
    # A measure's frame input (the haso reference) sees the shot's steps.
    measured = process_measure_input(recipe.analysis.measure, bound, process)
    if recipe.input_kind == "line":
        # Legacy rounds coordinates AND samples before calculating statistics.
        # The steps own the signal unit (a derivative makes it V/s), so the
        # rebuilt frame keeps the processed frame's unit, not the input's.
        stored = frame.as_trace().astype(recipe.storage_dtype)
        frame = Frame.from_trace(
            stored,
            x_unit=recipe.x_unit,
            y_unit=frame.unit,
            y_label=recipe.label,
            shot=shot,
        )
    elif recipe.analysis.measure.kind in {"beam", "hi_res_mag_cam"}:
        # This is deliberately v2-only: BeamAnalyzer (and the HiResMagCam
        # analyzer built on it) uses the configured origin once even when ROI
        # is skipped or repeated, unlike the pure Frame API.
        frame = frame.replace(
            data=frame.data,
            axes=tuple(
                Axis(np.arange(size) + origin, label=label)
                for size, origin, label in zip(
                    frame.data.shape, recipe.camera_origin, ("y", "x"), strict=True
                )
            ),
        )
    result = apply_measure(
        frame, recipe.analysis.measure, inputs=measured, channels=channels
    )
    if (
        recipe.input_kind == "line"
        and recipe.storage_dtype == "float64"
        and recipe.analysis.measure.kind == "line"
    ):
        # Old float64 results alias the RMS scratch; float32 results do not.
        # Preserve output values without reproducing the alias or mutating input.
        data = frame.data.copy()
        data[data < 0] = 0
        result = replace(result, frame=frame.replace(data=data), notes=())
    return result


def _trace_input(
    data: np.ndarray, recipe: V2Recipe, shot: ShotMeta | None
) -> tuple[Frame, dict[str, Frame]]:
    """The scaled trace and its auxiliary channels on the same x axis.

    Channel samples are taken as loaded (the legacy analyzer never scaled
    or rounded its auxiliary columns).
    """
    from geecs_data_utils.frames import Frame

    width = 2 + len(recipe.channels)
    if data.ndim != 2 or data.shape[1] != width:
        names = ", ".join(("x", "y", *recipe.channels))
        raise ValueError(f"Trace input must be Nx{width} ({names})")
    scaled = data[:, :2].copy()
    # Legacy performs these before its float64 processing conversion.
    scaled[:, 0] *= recipe.x_scale
    scaled[:, 1] *= recipe.y_scale
    frame = Frame.from_trace(
        scaled,
        x_unit=recipe.x_unit,
        y_unit=recipe.y_unit,
        y_label=recipe.label,
        shot=shot,
    )
    channels = {
        name: Frame.from_array(data[:, column], axes=frame.axes, shot=shot, label=name)
        for column, name in enumerate(recipe.channels, start=2)
    }
    return frame, channels


def _follow_selection(
    channels: dict[str, Frame], steps: Sequence[StepSpec], size: int
) -> dict[str, Frame]:
    """Cut each channel by the recipe's sample selections (``roi``), nothing else.

    The legacy analyzer cut its auxiliary columns by the line ROI only; the
    other steps change the trace's values, not which samples it has.
    """
    cut = {}
    for name, channel in channels.items():
        for spec in steps:
            if isinstance(spec, RoiSpec):
                channel = apply_step(channel, spec)
        if channel.data.size != size:
            raise ValueError(f"Channel {name!r} no longer matches the trace's samples")
        cut[name] = channel
    return cut
