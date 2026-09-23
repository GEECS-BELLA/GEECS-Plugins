"""Pure pipeline execution over caller-supplied frames."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Iterable, Mapping

from geecs_analysis.registry import StepSpec, definition
from geecs_analysis.specs import Pipeline

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame


def bind_inputs(
    steps: Iterable[StepSpec], inputs: Mapping[str, Frame] | None = None
) -> Mapping[str, Frame]:
    """Snapshot required in-memory inputs, refusing absent or untyped frames.

    Names in specs are binding keys, never file paths to open. Sources load
    backgrounds before calling this function. Only required keys are retained;
    caller mutations to the mapping cannot rebind an in-flight execution.
    """
    from geecs_data_utils.frames import Frame

    supplied = inputs if inputs is not None else {}
    resolved = {}
    for spec in steps:
        field = definition(spec).input_field
        if field is None:
            continue
        key = getattr(spec, field)
        if key not in supplied:
            raise ValueError(f"Missing frame input: {key}")
        value = supplied[key]
        if not isinstance(value, Frame):
            raise TypeError(f"Frame input {key!r} must be a Frame")
        resolved[key] = value
    return MappingProxyType(resolved)


def apply_step(
    frame: Frame, spec: StepSpec, *, inputs: Mapping[str, Frame] | None = None
) -> Frame:
    """Apply one declared step with an optional already-bound input frame."""
    declared = definition(spec)
    if frame.data.ndim not in declared.ndim:
        raise ValueError(f"{spec.step} does not support {frame.data.ndim}D frames")
    if declared.input_field is None:
        return declared.function(frame, spec)
    bound = bind_inputs((spec,), inputs)
    return declared.function(frame, spec, bound[getattr(spec, declared.input_field)])


def apply_pipeline(
    frame: Frame, pipeline: Pipeline, *, inputs: Mapping[str, Frame] | None = None
) -> Frame:
    """Validate dimensions and fold steps without modifying the input frame."""
    for spec in pipeline.steps:
        if frame.data.ndim not in definition(spec).ndim:
            raise ValueError(f"{spec.step} does not support {frame.data.ndim}D frames")
    bound = bind_inputs(pipeline.steps, inputs)
    for spec in pipeline.steps:
        frame = apply_step(frame, spec, inputs=bound)
    return frame
