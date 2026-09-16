"""Predetermined motion through scan_nd, with relative cleanup inside the run."""

from collections.abc import Callable, Mapping
from typing import Any

from bluesky import plans as bp, preprocessors as bpp
from bluesky.protocols import Movable, Readable
from geecs_schemas import Sweep
from geecs_schemas.sweep import AxisSweep

from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.trajectory import axis_positions, sweep_to_cycler

from .strict import name_failed_status


def sweep_plan(namespace: Mapping[str, Any]) -> Callable:
    """Bind expanded axis names to the worker's existing namespace objects.

    ``namespace`` may be a GeecsNamespace or a mapping in hermetic tests.
    Device protocols and heterogeneous RunEngine metadata require Any here.
    """

    def resolve(name: str) -> Movable:
        parts = name.split(".")
        if any(not p.isidentifier() or p.startswith("_") for p in parts):
            raise GeecsConfigurationError(f"Invalid expanded axis binding: {name!r}")
        try:
            obj = namespace[parts[0]]
            for part in parts[1:]:
                obj = getattr(obj, part)
        except (KeyError, AttributeError, TypeError) as exc:
            raise GeecsConfigurationError(f"Unknown sweep axis: {name!r}") from exc
        if not isinstance(obj, Movable) or not isinstance(obj, Readable):
            raise GeecsConfigurationError(
                f"Sweep axis {name!r} is not movable/readable"
            )
        return obj

    # Deliberately not a generator function: validation and expansion run when
    # the acquisition binder constructs the plan, BEFORE its first box move.
    def sweep(detectors, *, sweep: dict, per_step=None, md=None):
        """Run a validated JSON Sweep with the worker's acquisition hook."""
        payload = Sweep.model_validate(sweep)
        trajectory = sweep_to_cycler(payload, resolve)
        refs = payload.axis_references()
        motors = [resolve(ref.axis) for ref in refs]
        relative = [motor for motor, ref in zip(motors, refs) if ref.relative]
        points = trajectory.by_key()
        spec = payload.trajectory
        grid = isinstance(spec, AxisSweep) and spec.combine == "product"
        first = (
            axis_positions(spec.axes[0])
            if isinstance(spec, AxisSweep)
            else points[motors[0]]
        )
        shape = (
            [len(axis_positions(a)) for a in spec.axes] if grid else [len(trajectory)]
        )
        fields = [getattr(m, "hints", {}).get("fields", []) for m in motors]
        dimensions = (
            [(f, "primary") for f in fields] if grid else [(sum(fields, []), "primary")]
        )
        metadata = dict(md or {})
        metadata.update(
            plan_name="sweep",
            sweep=payload.model_dump(mode="json"),
            sweep_first_axis=[
                first[0],
                first[-1],
                first[1] - first[0] if len(first) > 1 else 0.0,
            ],
            detectors=[d.name for d in detectors],
            motors=[m.name for m in motors],
            num_points=len(trajectory),
            num_intervals=len(trajectory) - 1,
            shape=shape,
            extents=[[min(points[m]), max(points[m])] for m in motors],
            snaking=[False, *([spec.snake] * (len(motors) - 1))] if grid else [False],
            hints={"dimensions": dimensions} if all(fields) else {},
        )
        # Retain Bluesky's traversal and hook invocation. Own the lifecycle so
        # reset happens in the staged coordinate frame, before close_run. An
        # outer reset_positions_wrapper(scan_nd(...)) would reset AFTER unstage.
        inner = bpp.stub_wrapper(bp.scan_nd(detectors, trajectory, per_step=per_step))
        if relative:
            inner = bpp.relative_set_wrapper(inner, relative)
            inner = bpp.reset_positions_wrapper(inner, relative)
        inner = bpp.run_wrapper(name_failed_status(inner), md=metadata)
        return bpp.stage_wrapper(inner, [*detectors, *motors])

    return sweep
