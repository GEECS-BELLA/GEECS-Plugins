"""Compile supported v2 documents without loading configs or input files.

Unsupported recipes fail explicitly so callers can keep their existing route.
No fallback executes here. Compatibility quirks stay at this boundary: trace
storage rounding before measurement, float64 trace-result negative clipping,
and the legacy beam ROI-origin convention (even for inactive/repeated ROIs).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Mapping

from geecs_schemas.analysis import AnalysisDiagnostic
from geecs_schemas.analysis.processing_1d import Line1DConfig
from geecs_schemas.analysis.processing_2d import CameraConfig

from geecs_analysis.measures.beam import BeamSpec
from geecs_analysis.measures.line import LineSpec
from geecs_analysis.measures.none import NoneSpec
from geecs_analysis.pipeline import apply_step, bind_inputs
from geecs_analysis.registry import StepSpec, measure_definition
from geecs_analysis.specs import Analysis
from geecs_analysis.steps.background_constant import BackgroundConstantSpec
from geecs_analysis.steps.background_frame import BackgroundFrameSpec
from geecs_analysis.steps.circular_mask import CircularMaskSpec
from geecs_analysis.steps.clip_above import ClipAboveSpec
from geecs_analysis.steps.clip_below import ClipBelowSpec
from geecs_analysis.steps.crosshair_mask import CrosshairMaskSpec
from geecs_analysis.steps.gaussian import GaussianSpec
from geecs_analysis.steps.interpolate import InterpolateSpec
from geecs_analysis.steps.median import MedianSpec
from geecs_analysis.steps.roi import RoiSpec
from geecs_analysis.steps.rotate import RotateSpec
from geecs_analysis.steps.zero_below import ZeroBelowSpec

if TYPE_CHECKING:
    import numpy as np
    from geecs_data_utils.frames import Frame, ShotMeta
    from geecs_analysis.measurement import Measurement


class UnsupportedRecipe(ValueError):
    """The v2 recipe needs a capability not yet ported to the new core."""


@dataclass(frozen=True)
class FileBackground:
    """A source-layer request; the core never opens or resolves this path."""

    key: str
    path: str
    fallback_level: float


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


def compile_v2(
    document: AnalysisDiagnostic, *, allow_file_backgrounds: bool = False
) -> V2Recipe:
    """Translate supported beam/line/standard/trace recipes, without file access.

    Currently covers constant backgrounds, ROI, circular/crosshair masks, trace
    interpolation, Gaussian/median filtering, fixed-canvas rotation,
    absolute trace clipping and non-inverted constant image thresholds
    (to_zero/truncate/truncate_inv). Trace processing must be float64 and
    storage float32/float64. Other active features are refused before execution.
    Preprocessing-only trace ROIs remain unported because empty legacy outputs
    cannot be represented by Frame. Inactive sections are ignored as before.
    File backgrounds require explicit source-layer opt-in; the compiled recipe
    then declares requests and expects loaded Frame inputs at execution time.
    """
    kind = document.analyzer.kind
    if kind not in {"beam", "line", "standard", "trace"}:
        raise UnsupportedRecipe(f"Analyzer not ported: {kind}")
    if document.scan.background_source is not None:
        raise UnsupportedRecipe("Scan backgrounds must be resolved by a source")
    config = document.image
    if not isinstance(config, (CameraConfig, Line1DConfig)):
        raise UnsupportedRecipe("A camera or line processing section is required")
    if kind in {"beam", "standard"} and not isinstance(config, CameraConfig):
        raise UnsupportedRecipe(f"{kind} requires a camera input")
    if kind in {"line", "trace"} and not isinstance(config, Line1DConfig):
        raise UnsupportedRecipe(f"{kind} requires a line input")
    # Legacy preprocessing-only traces can successfully return empty Nx2 data.
    # Frame cannot represent that result. Until an empty-result contract exists,
    # leave ALL trace ROI recipes on the old route rather than conditionally fail
    # when one shot has no samples in the requested physical range.
    if kind == "trace" and config.roi is not None and "roi" in config.pipeline:
        raise UnsupportedRecipe(
            "Preprocessing-only trace ROI may produce an empty result"
        )
    steps = []
    for name in config.pipeline:
        section = getattr(config, name.value)
        if section is None:
            continue
        steps.extend(
            _camera_steps(
                name.value, config, allow_file_backgrounds=allow_file_backgrounds
            )
            if isinstance(config, CameraConfig)
            else _line_steps(name.value, config)
        )
    if kind == "beam":
        measure = BeamSpec(
            enabled_stats=document.analyzer.enabled_stats,
            compute_slopes=document.analyzer.compute_slopes,
        )
    elif kind == "line":
        measure = LineSpec()
    else:
        measure = NoneSpec()
    common = dict(
        analysis=Analysis(steps=tuple(steps), measure=measure),
        input_kind=config.type,
        device=document.name,
        output_name=document.effective_output_name,
        metric_suffix=document.metric_suffix,
    )
    if isinstance(config, CameraConfig):
        origin = (
            (config.roi.y_min, config.roi.x_min) if config.roi is not None else (0, 0)
        )
        requests = ()
        if any(isinstance(spec, BackgroundFrameSpec) for spec in steps):
            background = config.background
            requests = (
                FileBackground(
                    key="camera_background",
                    path=str(background.file_path),
                    fallback_level=background.constant_level,
                ),
            )
        return V2Recipe(**common, camera_origin=origin, file_backgrounds=requests)
    if config.processing_dtype != "float64" or config.storage_dtype not in {
        "float32",
        "float64",
    }:
        raise UnsupportedRecipe(
            "Trace adapter requires float64 processing and float32/float64 storage"
        )
    return V2Recipe(
        **common,
        storage_dtype=config.storage_dtype,
        x_scale=config.x_scale_factor,
        y_scale=config.y_scale_factor,
        x_unit=config.x_units or "",
        y_unit=config.y_units or "",
        label=config.label,
    )


def _camera_steps(
    name: str, config: CameraConfig, *, allow_file_backgrounds: bool
) -> list[StepSpec]:
    section = getattr(config, name)
    if name == "transforms":
        if (
            section.flip_horizontal
            or section.flip_vertical
            or section.distortion_correction
        ):
            raise UnsupportedRecipe(
                "Camera processing step not ported: transforms (flip/distortion)"
            )
        return (
            [RotateSpec(angle=section.rotation_angle)]
            if section.rotation_angle != 0
            else []
        )
    if name == "crosshair_masking":
        return [
            CrosshairMaskSpec(
                center=(cross.center[1], cross.center[0]),
                width=cross.width,
                height=cross.height,
                thickness=cross.thickness,
                angle=cross.angle,
                value=section.mask_value,
            )
            for cross in section.crosshairs
        ]
    if name == "background":
        if section.method not in {None, "constant"} and not (
            section.method == "from_file" and allow_file_backgrounds
        ):
            raise UnsupportedRecipe(f"Camera background not ported: {section.method}")
        steps = []
        if section.method == "from_file":
            steps.append(
                BackgroundFrameSpec(source="camera_background", alignment="samples")
            )
        if section.method == "constant" and section.constant_level > 0:
            steps.append(BackgroundConstantSpec(level=section.constant_level))
        if section.additional_constant != 0:
            steps.append(BackgroundConstantSpec(level=section.additional_constant))
        return steps
    if name == "roi":
        return [
            RoiSpec(
                bounds=((section.y_min, section.y_max), (section.x_min, section.x_max))
            )
        ]
    if name == "circular_mask":
        return [
            CircularMaskSpec(
                center=(section.center[1], section.center[0]),
                radius=section.radius,
                mask_outside=section.mask_outside,
                value=section.mask_value,
            )
        ]
    if name == "filtering":
        steps = []
        if section.gaussian_sigma is not None:
            steps.append(GaussianSpec(sigma=section.gaussian_sigma))
        if section.median_kernel_size is not None:
            steps.append(MedianSpec(kernel=section.median_kernel_size))
        return steps
    if name == "thresholding":
        if section.method != "constant" or section.invert:
            raise UnsupportedRecipe(
                "Only non-inverted constant camera thresholds are ported"
            )
        specs = {
            "to_zero": ZeroBelowSpec,
            "truncate": ClipAboveSpec,
            "truncate_inv": ClipBelowSpec,
        }
        if section.mode not in specs:
            raise UnsupportedRecipe(f"Camera threshold mode not ported: {section.mode}")
        return [specs[section.mode](level=section.value)]
    raise UnsupportedRecipe(f"Camera processing step not ported: {name}")


def _line_steps(name: str, config: Line1DConfig) -> list[StepSpec]:
    section = getattr(config, name)
    if name == "background":
        if section.method == "none":
            return []
        if section.method == "constant":
            return [BackgroundConstantSpec(level=section.constant_level)]
    elif name == "roi":
        return [RoiSpec(bounds=((section.x_min, section.x_max),), units="axis")]
    elif name == "interpolation":
        return [
            InterpolateSpec(
                count=section.num_points, lower=section.x_min, upper=section.x_max
            )
        ]
    elif name == "filtering":
        if section.method == "none":
            return []
        if section.method == "median":
            return [MedianSpec(kernel=section.kernel_size)]
        if section.method == "gaussian":
            return [GaussianSpec(sigma=section.sigma)]
    elif name == "thresholding":
        if section.method == "none":
            return []
        if section.method == "absolute":
            spec = ClipBelowSpec if section.clip_below else ClipAboveSpec
            return [spec(level=section.threshold_value)]
    raise UnsupportedRecipe(
        f"Line processing step not ported: {name}/{getattr(section, 'method', '')}"
    )


def analyze_v2(
    data: np.ndarray,
    recipe: V2Recipe,
    *,
    shot: ShotMeta | None = None,
    inputs: Mapping[str, Frame] | None = None,
) -> Measurement:
    """Run already-loaded raw samples with legacy scaling/rounding conventions.

    Images use HxW arrays; traces use Nx2 arrays. Sources keep native dtype until
    this boundary so trace axis scaling occurs in the same precision as before.
    Input arrays are always copied and never mutated. No paths are accepted.
    """
    import numpy as np
    from geecs_data_utils.frames import Axis, Frame

    bound = bind_inputs(recipe.analysis.steps, inputs)
    if recipe.input_kind == "camera":
        if data.ndim != 2:
            raise ValueError("Camera input must be HxW")
        frame = Frame.from_array(data, shot=shot)
    else:
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("Trace input must be Nx2")
        scaled = data.copy()
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
    for spec in recipe.analysis.steps:
        if recipe.input_kind == "camera" and isinstance(spec, RoiSpec):
            # Legacy returns the full image for a crop wholly outside the input.
            if any(
                lo >= min(hi, size)
                for (lo, hi), size in zip(spec.bounds, frame.data.shape, strict=True)
            ):
                continue
        frame = apply_step(frame, spec, inputs=bound)
    if recipe.input_kind == "line":
        # Legacy rounds coordinates AND samples before calculating statistics.
        stored = frame.as_trace().astype(recipe.storage_dtype)
        frame = Frame.from_trace(
            stored,
            x_unit=recipe.x_unit,
            y_unit=recipe.y_unit,
            y_label=recipe.label,
            shot=shot,
        )
    elif recipe.analysis.measure.kind == "beam":
        # This is deliberately v2-only: BeamAnalyzer uses the configured origin
        # once even when ROI is skipped or repeated, unlike the pure Frame API.
        frame = frame.replace(
            data=frame.data,
            axes=tuple(
                Axis(np.arange(size) + origin, label=label)
                for size, origin, label in zip(
                    frame.data.shape, recipe.camera_origin, ("y", "x"), strict=True
                )
            ),
        )
    result = measure_definition(recipe.analysis.measure).function(
        frame, recipe.analysis.measure
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
