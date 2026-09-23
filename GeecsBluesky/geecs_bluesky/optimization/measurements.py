"""Compile live measurements once, then evaluate bins without filesystem reads."""

from __future__ import annotations

import importlib
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from geecs_schemas import OptimizerConfig, optimizer_required_devices
from geecs_schemas.analysis import AnalysisDiagnostic
from geecs_schemas.optimizer_config import (
    DiagnosticMeasurement,
    SignalMeasurement,
    PythonDerived,
    MEASUREMENT_MATH,
    expression_references,
)
from geecs_schemas.restricted_expr import compile_expression
from geecs_schemas.scan_variables import split_device_variable
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_core.pv_naming import pv_name
from geecs_core.db.variable_types import LABVIEW_EPOCH_OFFSET

from .live_frames import FrameSource, LiveFrameSource

if TYPE_CHECKING:
    from geecs_bluesky.config_resolver import ConfigsRepoResolver
    from geecs_bluesky.namespace import GeecsNamespace

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompiledMeasurement:
    """Resolved document, event key and output names for one measurement."""

    name: str
    spec: SignalMeasurement | DiagnosticMeasurement
    device: str
    event_key: str
    keys: frozenset[str]
    diagnostic: AnalysisDiagnostic | None = None
    readable: object | None = None


@dataclass(frozen=True)
class BinResult:
    """Reduced scalar values and their per-measurement valid-shot counts."""

    outputs: dict[str, float]
    valid_shots: dict[str, int]


@dataclass
class CompiledMeasurements:
    """A run's immutable measurement interpretation and held frame sources."""

    measurements: tuple[CompiledMeasurement, ...]
    derived: dict[str, Callable[[Mapping[str, float]], float]]
    sources: dict[str, FrameSource]
    required_devices: frozenset[str]
    output_names: tuple[str, ...]

    def frames_for(
        self, rows: Sequence[Mapping[str, float]], timeout: float = 2.0
    ) -> dict[str, dict[float, NDArray]]:
        """Join raw LabVIEW reading stamps to Unix PVA frame stamps."""
        frames = {}
        for measurement in self.measurements:
            if measurement.diagnostic is not None and measurement.device not in frames:
                stamps = [
                    float(row[measurement.event_key]) - LABVIEW_EPOCH_OFFSET
                    for row in rows
                    if math.isfinite(float(row.get(measurement.event_key, math.nan)))
                ]
                frames[measurement.device] = self.sources[
                    measurement.device
                ].await_frames(stamps, timeout)
        return frames

    def evaluate_bin(
        self,
        rows: Sequence[Mapping[str, float]],
        frames: Mapping[str, Mapping[float, NDArray]],
    ) -> BinResult:
        """Reduce valid measurements, then evaluate derived outputs in order."""
        from image_analysis.ephemeral import run_document_ephemeral

        outputs: dict[str, float] = {}
        counts: dict[str, int] = {}
        for m in self.measurements:
            values: list[dict[str, float]] = []
            if isinstance(m.spec, SignalMeasurement):
                values = [
                    {m.name: float(row[m.event_key])}
                    for row in rows
                    if math.isfinite(float(row.get(m.event_key, math.nan)))
                ]
                valid = len(values)
            else:
                selected = []
                seen = set()
                for row in rows:
                    stamp = float(row.get(m.event_key, math.nan)) - LABVIEW_EPOCH_OFFSET
                    if stamp in seen:
                        continue
                    frame = frames.get(m.device, {}).get(stamp)
                    if frame is not None:
                        selected.append(frame)
                        seen.add(stamp)
                valid = len(selected)
                if m.spec.frames == "per_bin" and selected:
                    selected = [np.mean(selected, axis=0)]
                for frame in selected:
                    try:
                        (result,) = run_document_ephemeral(m.diagnostic, [frame])
                        value = {
                            f"{m.name}.{key}": float(result.scalars[key])
                            for key in m.keys
                        }
                        if not all(math.isfinite(v) for v in value.values()):
                            raise ValueError("non-finite analyzer output")
                        values.append(value)
                    except Exception:
                        logger.warning(
                            "measurement %s: analysis failed", m.name, exc_info=True
                        )
                if m.spec.frames == "per_shot":
                    valid = len(values)
                elif not values:
                    valid = 0
            counts[m.name] = valid
            keys = (
                [m.name]
                if isinstance(m.spec, SignalMeasurement)
                else [f"{m.name}.{k}" for k in m.keys]
            )
            reducer = getattr(np, m.spec.reduce)
            for key in keys:
                outputs[key] = (
                    float(reducer([v[key] for v in values]))
                    if valid >= m.spec.min_shots and values
                    else math.nan
                )
        for name, evaluate in self.derived.items():
            try:
                outputs[name] = float(evaluate(dict(outputs)))
            except Exception:
                logger.warning("derived %s failed", name, exc_info=True)
                outputs[name] = math.nan
        return BinResult(outputs, counts)


def compile_measurements(
    cfg: OptimizerConfig,
    *,
    namespace: GeecsNamespace,
    resolver: ConfigsRepoResolver,
    shots_per_step: int,
    source_factory: Callable[..., FrameSource] = LiveFrameSource,
) -> CompiledMeasurements:
    """Resolve signals, diagnostics, output references and imports before a run opens."""
    from image_analysis.config import load_diagnostic
    from image_analysis.ephemeral import EPHEMERAL_DENYLIST
    from geecs_bluesky.namespace import capture_streams

    measurements = []
    sources = {}
    diagnostic_devices = {}
    names = set()
    try:
        for name, spec in cfg.measurements.items():
            if isinstance(spec, SignalMeasurement):
                device, variable = split_device_variable(spec.signal)
                signal = namespace.resolve(spec.signal)
                m = CompiledMeasurement(
                    name,
                    spec,
                    device,
                    getattr(signal, "reading_key", signal.name),
                    frozenset(),
                    readable=signal,
                )
                names.add(name)
            else:
                diag = load_diagnostic(
                    spec.diagnostic,
                    config_dir=resolver.analysis_config_dir,
                    overrides=spec.overrides,
                )
                if (
                    diag.analyzer.kind in EPHEMERAL_DENYLIST
                    or diag.analyzer.image_kind != "camera"
                ):
                    raise ValueError(
                        f"{name}: diagnostic {spec.diagnostic} is not supported on live camera frames"
                    )
                keys = diag.analyzer.emitted_scalars()
                if not keys:
                    raise ValueError(f"{name}: diagnostic declares no scalar outputs")
                detector = namespace.resolve(diag.name)
                images = capture_streams(
                    namespace.roster.variables[diag.name],
                    namespace.roster.types.get(diag.name, ""),
                    diag.name,
                )
                if not images:
                    raise ValueError(
                        f"{diag.name}: no image variable in the device roster"
                    )
                diagnostic_devices[spec.diagnostic] = diag.name
                if diag.name not in sources:
                    sources[diag.name] = source_factory(
                        pv_name(namespace.experiment, diag.name, images[0]),
                        keep=max(64, shots_per_step * 3 + 8),
                    )
                m = CompiledMeasurement(
                    name, spec, diag.name, detector.acq_timestamp.name, keys, diag
                )
                names.update(f"{name}.{key}" for key in keys)
            measurements.append(m)
        derived = {}
        for name, expression in cfg.derived.items():
            if isinstance(expression, PythonDerived):
                module, function = expression.python.split(":")
                evaluate = getattr(importlib.import_module(module), function)
                if not callable(evaluate):
                    raise ValueError(f"{expression.python} is not callable")
            else:
                compiled_expression = compile_expression(
                    expression, names, MEASUREMENT_MATH
                )
                refs = expression_references(expression)

                def evaluate(
                    values, compiled_expression=compiled_expression, refs=refs
                ):
                    if any(not math.isfinite(values[r]) for r in refs):
                        return math.nan
                    return compiled_expression.evaluate(values)

            derived[name] = evaluate
            names.add(name)
        for name in [
            *cfg.vocs.objectives,
            *cfg.vocs.observables,
            *cfg.vocs.constraints,
        ]:
            if name not in names:
                raise ValueError(
                    f"unknown reference {name!r}; emitted names: {sorted(names)}"
                )
        return CompiledMeasurements(
            tuple(measurements),
            derived,
            sources,
            optimizer_required_devices(cfg, diagnostic_devices),
            tuple(sorted(names)),
        )
    except Exception as exc:
        raise GeecsConfigurationError(f"optimizer measurements: {exc}") from exc
