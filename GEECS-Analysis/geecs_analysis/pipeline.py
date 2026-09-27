"""Pure pipeline execution over caller-supplied frames."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Iterable, Mapping

from geecs_analysis.registry import (
    MeasureSpec,
    StepSpec,
    definition,
    measure_definition,
)
from geecs_analysis.specs import Pipeline

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame
    from geecs_analysis.measurement import Measurement


def bind_inputs(
    steps: Iterable[StepSpec],
    inputs: Mapping[str, object] | None = None,
    *,
    measure: MeasureSpec | None = None,
) -> Mapping[str, object]:
    """Snapshot required in-memory inputs, refusing absent or untyped values.

    Names in specs are binding keys, never file paths to open. Sources load
    backgrounds before calling this function. Only required keys are retained;
    caller mutations to the mapping cannot rebind an in-flight execution.
    A step's input must be a Frame. A ``measure`` whose registration names a
    service also binds that key: the host's collaborator (any object that is
    not a Frame, e.g. the FROG retrieval), which the core calls but never
    builds, configures or inspects.
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
    service = measure_definition(measure).service if measure is not None else None
    if service is not None:
        if service not in supplied:
            raise ValueError(
                f"Missing service: {service} (the {measure.kind!r} measure needs "
                "the host to supply it)"
            )
        value = supplied[service]
        if value is None or isinstance(value, Frame):
            raise TypeError(f"Service {service!r} must be the host's collaborator")
        resolved[service] = value
    return MappingProxyType(resolved)


def apply_measure(
    frame: Frame, spec: MeasureSpec, *, inputs: Mapping[str, object] | None = None
) -> Measurement:
    """Measure one processed frame, handing a service measure its bound service."""
    declared = measure_definition(spec)
    if declared.service is None:
        return declared.function(frame, spec)
    bound = bind_inputs((), inputs, measure=spec)
    return declared.function(frame, spec, bound[declared.service])


def apply_step(
    frame: Frame, spec: StepSpec, *, inputs: Mapping[str, object] | None = None
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
    frame: Frame, pipeline: Pipeline, *, inputs: Mapping[str, object] | None = None
) -> Frame:
    """Validate dimensions and fold steps without modifying the input frame."""
    for spec in pipeline.steps:
        if frame.data.ndim not in definition(spec).ndim:
            raise ValueError(f"{spec.step} does not support {frame.data.ndim}D frames")
    bound = bind_inputs(pipeline.steps, inputs)
    for spec in pipeline.steps:
        frame = apply_step(frame, spec, inputs=bound)
    return frame
