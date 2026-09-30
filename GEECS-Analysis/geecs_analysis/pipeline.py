"""Pure pipeline execution over caller-supplied frames."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Iterable, Mapping

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
    builds, configures or inspects. A ``measure`` whose spec names a frame
    input (:func:`measure_input`) binds that Frame too.
    """
    from geecs_data_utils.frames import Frame

    supplied = inputs if inputs is not None else {}
    resolved = {}

    def bind_frame(key: str) -> None:
        if key not in supplied:
            raise ValueError(f"Missing frame input: {key}")
        value = supplied[key]
        if not isinstance(value, Frame):
            raise TypeError(f"Frame input {key!r} must be a Frame")
        resolved[key] = value

    for spec in steps:
        field = definition(spec).input_field
        if field is not None:
            bind_frame(getattr(spec, field))
    key = measure_input(measure) if measure is not None else None
    if key is not None:
        bind_frame(key)
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


def measure_input(spec: MeasureSpec) -> str | None:
    """The frame-binding key the measure's spec names, or ``None``.

    ``None`` both for a measure registered without an ``input_field`` and
    for one whose field is unset in this spec (a ``haso`` measure without
    a reference).
    """
    field = measure_definition(spec).input_field
    return None if field is None else getattr(spec, field)


def process_measure_input(
    spec: MeasureSpec,
    bound: Mapping[str, object],
    process: Callable[[Frame], Frame],
) -> Mapping[str, object]:
    """Bindings with the measure's frame input replaced by ``process(frame)``.

    The evaluator passes its own step fold as ``process``, so the
    comparison frame is processed exactly as the measured frame is (a
    reference must see the same background subtraction as every shot);
    the steps themselves keep the loaded frames (``compile_recipe`` refuses
    a key bound by both a step and the measure).
    Unchanged when the measure takes no frame input.
    """
    key = measure_input(spec)
    if key is None:
        return bound
    return MappingProxyType({**bound, key: process(bound[key])})


def apply_measure(
    frame: Frame, spec: MeasureSpec, *, inputs: Mapping[str, object] | None = None
) -> Measurement:
    """Measure one processed frame with its bound service and frame input.

    The frame input is handed as bound: the evaluator has already processed
    it (:func:`process_measure_input`).
    """
    declared = measure_definition(spec)
    if declared.service is None and declared.input_field is None:
        return declared.function(frame, spec)
    bound = bind_inputs((), inputs, measure=spec)
    arguments: list[object] = []
    if declared.service is not None:
        arguments.append(bound[declared.service])
    if declared.input_field is not None:
        key = measure_input(spec)
        arguments.append(None if key is None else bound[key])
    return declared.function(frame, spec, *arguments)


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
