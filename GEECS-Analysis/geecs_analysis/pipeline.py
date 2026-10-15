"""Pure pipeline execution over caller-supplied frames."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Iterable, Mapping

from geecs_analysis.registry import (
    MeasureDefinition,
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
    input (:func:`measure_input`) binds that Frame too, and one whose spec
    names tables (:func:`measure_tables`) binds a ``Table`` under each key.
    """
    from geecs_data_utils.frames import Frame
    from geecs_data_utils.tables import Table

    supplied = inputs if inputs is not None else {}
    wanted: list[tuple[str, type, str]] = []
    for spec in steps:
        field = definition(spec).input_field
        if field is not None:
            wanted.append((getattr(spec, field), Frame, "frame"))
    if measure is not None:
        key = measure_input(measure)
        if key is not None:
            wanted.append((key, Frame, "frame"))
        wanted.extend((key, Table, "table") for key in measure_tables(measure).values())
    resolved = {
        key: _supplied(supplied, key, kind, label) for key, kind, label in wanted
    }
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


def _supplied(
    supplied: Mapping[str, object], key: str, kind: type, label: str
) -> object:
    """The caller's value under ``key``, which must be a ``kind``."""
    if key not in supplied:
        raise ValueError(f"Missing {label} input: {key}")
    value = supplied[key]
    if not isinstance(value, kind):
        raise TypeError(f"{label.capitalize()} input {key!r} must be a {kind.__name__}")
    return value


def measure_input(spec: MeasureSpec) -> str | None:
    """The frame-binding key the measure's spec names, or ``None``.

    ``None`` both for a measure registered without an ``input_field`` and
    for one whose field is unset in this spec (a ``haso`` measure without
    a reference).
    """
    field = measure_definition(spec).input_field
    return None if field is None else getattr(spec, field)


def measure_tables(spec: MeasureSpec) -> Mapping[str, str]:
    """The measure's table bindings, ``{spec field: binding key}``, for the fields set."""
    return {
        field: getattr(spec, field)
        for field in measure_definition(spec).tables
        if getattr(spec, field) is not None
    }


def shot_scalar_columns(spec: MeasureSpec) -> tuple[str, ...]:
    """The scan's recorded columns the measure reads per shot, as the spec names them.

    What a scan host binds on each shot's ``ShotMeta.scalars`` for this
    recipe; empty for a measure that reads none, or whose spec leaves the
    fields unset. A cell the scan did not record for a shot (NaN in the
    s-file) is left out, so the measure takes its own fallback for that
    shot; a per-shot column cannot be read on averaged frames
    (``compile_recipe`` refuses it with ``scan.average_frames_first``).
    """
    columns = (getattr(spec, field) for field in measure_definition(spec).shot_scalars)
    return tuple(column for column in columns if column is not None)


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
    frame: Frame,
    spec: MeasureSpec,
    *,
    inputs: Mapping[str, object] | None = None,
    channels: Mapping[str, Frame] | None = None,
) -> Measurement:
    """Measure one processed frame with its bound service, frame input and channels.

    The frame input is handed as bound: the evaluator has already processed
    it (:func:`process_measure_input`). ``channels`` are the trace's
    auxiliary channels, already cut to the frame's samples; a measure that
    reads channels receives the ones it declared that are present, last.
    Tables arrive as keyword arguments named by the spec field that binds
    each (``None`` for a field left unset).
    """
    declared = measure_definition(spec)
    if (
        declared.service is None
        and declared.input_field is None
        and not declared.tables
    ):
        if not declared.channels:
            return declared.function(frame, spec)
        return declared.function(frame, spec, _declared_channels(declared, channels))
    bound = bind_inputs((), inputs, measure=spec)
    arguments: list[object] = []
    if declared.service is not None:
        arguments.append(bound[declared.service])
    if declared.input_field is not None:
        key = measure_input(spec)
        arguments.append(None if key is None else bound[key])
    if declared.channels:
        arguments.append(_declared_channels(declared, channels))
    # Tables go by the field's name; a field left unset hands over None.
    tables = measure_tables(spec)
    keywords = {
        field: bound[tables[field]] if field in tables else None
        for field in declared.tables
    }
    return declared.function(frame, spec, *arguments, **keywords)


def _declared_channels(
    declared: MeasureDefinition, channels: Mapping[str, Frame] | None
) -> Mapping[str, Frame]:
    """The supplied channels the measure declared, read-only."""
    supplied = channels or {}
    return MappingProxyType(
        {name: supplied[name] for name in declared.channels if name in supplied}
    )


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
