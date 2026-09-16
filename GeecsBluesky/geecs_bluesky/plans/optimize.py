"""Native strict optimization: one run, one bin per adaptive iteration."""

from __future__ import annotations

import inspect
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from collections.abc import Callable, Generator
from pathlib import Path
from time import time
from typing import TYPE_CHECKING

from bluesky import plan_stubs as bps, preprocessors as bpp
from bluesky.protocols import Movable
from bluesky.utils import Msg
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_schemas import TriggerState

from .gated import non_essential_wrapper, run_bracket
from .strict import BinCounter, geecs_take_reading, name_failed_status

if TYPE_CHECKING:
    from geecs_bluesky.config_resolver import ConfigsRepoResolver
    from geecs_bluesky.namespace import GeecsNamespace
    from .registry import TriggerProfiles

logger = logging.getLogger(__name__)


def _column(prefix: str, name: str) -> str:
    # event-model forbids dots and slashes in every document key, even nested.
    encoded = name.replace("%", "%25").replace(".", "%2E").replace("/", "%2F")
    return f"{prefix}:{encoded}"


class OptimizationRecord:
    """Fixed columns for proposed/measured variables, outputs and best-so-far."""

    name = "optimization"
    parent = None

    def __init__(self, names: list[str]) -> None:
        self.values = dict.fromkeys(names, math.nan)

    async def read(self) -> dict:
        """Return the latest iteration record."""
        return {
            name: {"value": value, "timestamp": time()}
            for name, value in self.values.items()
        }

    async def describe(self) -> dict:
        """Declare the complete record shape before its first event."""
        return {
            name: {
                "source": f"soft://optimization/{name}",
                "dtype": "number",
                "shape": [],
            }
            for name in self.values
        }


class _MeasurementReadback:
    """Read a missing scalar without being pruned as a detector's child."""

    parent = None

    def __init__(self, readable, key: str, header: str) -> None:
        self.name = "measurement_" + key
        self._readable = readable
        self._key = key
        self._column_headers = {key: header}

    async def read(self) -> dict:
        """Read the declared measurement column."""
        return {self._key: (await self._readable.read())[self._key]}

    async def describe(self) -> dict:
        """Describe only the additional column."""
        return {self._key: (await self._readable.describe())[self._key]}


class _RunDocuments:
    """Capture claimed metadata and the actual rows emitted, including replayed shots."""

    def __init__(self, shots: int, timestamps: tuple[str, ...]) -> None:
        self.timestamps = timestamps
        self.folder: Path | None = None
        self.primary: set[str] = set()
        self.rows: deque[dict] = deque(
            maxlen=shots
        )  # Bluesky event data is a heterogeneous mapping.

    def __call__(self, name: str, doc: dict) -> None:
        if name == "start" and doc.get("scan_folder"):
            self.folder = Path(doc["scan_folder"])
        elif name == "descriptor" and doc["name"] == "primary":
            self.primary.add(doc["uid"])
        elif name == "event" and doc["descriptor"] in self.primary:
            row = doc["data"]
            if all(
                math.isfinite(float(row.get(key, math.nan))) for key in self.timestamps
            ):
                self.rows.append(row)


def _on_thread(pool, function, *args):
    future = pool.submit(function, *args)
    while not future.done():
        yield from bps.sleep(0.05)
    return future.result()


def optimize_plan(
    profiles: TriggerProfiles,
    resolver: ConfigsRepoResolver | None,
    namespace: GeecsNamespace | None,
) -> Callable[..., Generator]:
    """Bind the optimize verb without importing the optional analysis/Xopt stack."""

    def optimize(
        detectors,
        *,
        optimizer_config: str,
        max_iterations: int | None = None,
        shots_per_step: int | None = None,
        trigger_profile: str | None = None,
        shot_period: float | None = None,
        non_essential=None,
        md=None,
    ):
        """Optimize a configured objective through strict acquisition.

        Parameters
        ----------
        detectors : list
            Required devices plus any extra devices to record.
        optimizer_config : str
            OptimizerConfig document ID in the experiment configs.
        max_iterations : int, optional
            Override the document's finite iteration budget.
        shots_per_step : int, optional
            Successful acquisitions per iteration.
        trigger_profile : str, optional
            Trigger profile; defaults to the experiment's profile.
        shot_period : float, optional
            Minimum time between strict fires, seconds.
        non_essential : list, optional
            Extra cameras streamed independently of the objective.
        md : dict, optional
            Additional run metadata.
        """
        from .registry import liveness_gate
        from geecs_bluesky.optimization.driver import XoptDriver
        from geecs_bluesky.optimization.measurements import compile_measurements

        if resolver is None or namespace is None:
            raise GeecsConfigurationError(
                "optimize needs a configs resolver and device namespace"
            )
        cfg = resolver.resolve_optimizer_config(optimizer_config)
        iterations = (
            cfg.run.max_iterations if max_iterations is None else max_iterations
        )
        shots = cfg.run.shots_per_step if shots_per_step is None else shots_per_step
        for label, value in (("max_iterations", iterations), ("shots_per_step", shots)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise GeecsConfigurationError(f"{label} must be a positive integer")
        if shot_period is not None and (
            not math.isfinite(shot_period) or shot_period <= 0
        ):
            raise GeecsConfigurationError("shot_period must be finite and positive")
        detectors = list(detectors)
        non_essential = list(non_essential or ())
        catalog = resolver.scan_variable_catalog().variables
        movables = {}
        move_references = {}
        for name in cfg.vocs.variables:
            spec = catalog.get(name)
            target = (
                spec.target if spec is not None and hasattr(spec, "target") else name
            )
            obj = namespace.resolve(target)
            if not isinstance(obj, Movable):
                raise GeecsConfigurationError(
                    f"optimizer variable {name!r} is not movable"
                )
            movables[name] = obj
            move_references[name] = target
        if len({id(obj) for obj in movables.values()}) != len(movables):
            raise GeecsConfigurationError(
                "optimizer variables resolve to the same movable"
            )
        physical_components = [
            namespace.resolve(target)
            for name, obj in movables.items()
            for target in getattr(obj, "component_targets", (move_references[name],))
        ]
        if len({id(component) for component in physical_components}) != len(
            physical_components
        ):
            raise GeecsConfigurationError(
                "optimizer variables share physical components"
            )
        if cfg.run.seed_dumps and any(
            getattr(obj, "relative", False) for obj in movables.values()
        ):
            raise GeecsConfigurationError(
                "seed dumps cannot be reused across relative pseudo zeroing frames"
            )
        compiled = compile_measurements(
            cfg, namespace=namespace, resolver=resolver, shots_per_step=shots
        )
        for device in compiled.required_devices:
            owner = namespace.resolve(device)
            matches = [
                d
                for d in detectors
                if d is owner or getattr(d, "_owner", None) is owner
            ]
            if not matches:
                raise GeecsConfigurationError(
                    f"optimizer requires essential device {device}"
                )
            if device in compiled.sources and (
                owner not in detectors or not (owner.native_save or owner.plugin_backed)
            ):
                raise GeecsConfigurationError(
                    f"optimizer requires images saved for {device}"
                )
        owners = {id(getattr(d, "_owner", d)) for d in detectors}
        if any(id(getattr(d, "_owner", d)) in owners for d in non_essential):
            raise GeecsConfigurationError(
                "a device cannot be both essential and non-essential"
            )
        seed_paths = [Path(p).expanduser() for p in cfg.run.seed_dumps]
        config_dir = resolver.optimizer_config_path(optimizer_config).parent
        try:
            driver = XoptDriver(
                cfg.vocs,
                cfg.generator,
                [p if p.is_absolute() else config_dir / p for p in seed_paths],
            )
        except Exception as exc:
            raise GeecsConfigurationError(f"optimizer generator: {exc}") from exc
        sc = profiles.resolve(trigger_profile)
        metadata = dict(md or {})
        metadata.update(
            plan_name="optimize",
            optimizer_config=optimizer_config,
            max_iterations=iterations,
            shots_per_step=shots,
            acquisition="strict",
            trigger_profile=trigger_profile or profiles.default,
            non_essential=[d.name for d in non_essential],
            shot_period=shot_period,
        )
        metadata["geecs"] = {
            **metadata.get("geecs", {}),
            "optimizer_json": cfg.model_dump_json(),
        }
        metadata["optimization_variables"] = list(movables)
        metadata["optimization_objectives"] = list(cfg.vocs.objectives)
        move_targets = sorted(
            {
                target
                for name, obj in movables.items()
                for target in getattr(
                    obj, "component_targets", (move_references[name],)
                )
            }
        )
        if sum(
            len(getattr(obj, "component_targets", (move_references[name],)))
            for name, obj in movables.items()
        ) != len(move_targets):
            raise GeecsConfigurationError(
                "optimizer variables share physical components"
            )
        metadata["optimization_move_targets"] = move_targets
        timestamps = tuple(
            getattr(d, "_owner", d).acq_timestamp.name
            for d in detectors
            if hasattr(getattr(d, "_owner", d), "acq_timestamp")
        )
        records = _RunDocuments(shots, timestamps)
        measurement_signals = []
        record = OptimizationRecord(
            [
                "iteration",
                *[_column("best_move", n) for n in move_targets],
                *[_column("proposal", n) for n in movables],
                *[_column("measured", n) for n in movables],
                *[_column("output", n) for n in compiled.output_names],
                *[_column("n_valid_shots", n) for n in cfg.measurements],
                *[_column("best", n) for n in [*movables, *cfg.vocs.objectives]],
            ]
        )
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="geecs-optimize")
        take = geecs_take_reading(sc, shot_period=shot_period)
        bins = BinCounter()

        def inner():
            if records.folder is not None:
                driver.bind_folder(records.folder)
            initial = {}
            reading_keys = {}
            for name, obj in movables.items():
                reading = yield from bps.read(obj)
                if len(reading) != 1:
                    raise GeecsConfigurationError(
                        f"{name}: expected one numeric readback"
                    )
                key, value = next(iter(reading.items()))
                reading_keys[name] = key
                initial[name] = float(value["value"])
            yield from bps.declare_stream(record, name="optimization")
            for iteration in range(1, iterations + 1):
                yield from bps.checkpoint()
                proposal = yield from _on_thread(pool, driver.ask)
                bins.value = iteration
                yield from bps.mv(
                    *[
                        item
                        for name, obj in movables.items()
                        for item in (obj, proposal[name])
                    ]
                )
                completed = []
                for _ in range(shots):
                    reading = yield from take([*detectors, *measurement_signals, bins])
                    completed.append(
                        {key: value["value"] for key, value in reading.items()}
                    )
                # Commit the acquired bin: calculation and tell must never be replayed.
                # Changing rewindability clears the acquisition replay buffer.
                yield Msg("rewindable", None, False)
                # The callback sees fresh rows even when the RE replays acquisition messages.
                rows = [r for r in records.rows if r.get(bins.name) == iteration][
                    -shots:
                ] or completed
                frames = yield from _on_thread(pool, compiled.frames_for, rows)
                result = yield from _on_thread(
                    pool, compiled.evaluate_bin, rows, frames
                )
                measured = {
                    name: sum(float(r[key]) for r in rows) / len(rows)
                    for name, key in reading_keys.items()
                }
                driver.tell(measured, result.outputs)
                best = driver.best or {}
                physical_best = {}
                if best:
                    for name, obj in movables.items():
                        if hasattr(obj, "targets_for"):
                            tasks = yield from bps.wait_for(
                                [
                                    lambda obj=obj, value=best[name]: obj.targets_for(
                                        value
                                    )
                                ]
                            )
                            physical_best.update(tasks[0].result())
                        else:
                            physical_best[move_references[name]] = best[name]
                record.values.update(
                    {
                        _column("best_move", n): physical_best.get(n, math.nan)
                        for n in move_targets
                    }
                )
                record.values.update(iteration=float(iteration))
                record.values.update(
                    {_column("proposal", n): v for n, v in proposal.items()}
                )
                record.values.update(
                    {_column("measured", n): v for n, v in measured.items()}
                )
                record.values.update(
                    {_column("output", n): v for n, v in result.outputs.items()}
                )
                record.values.update(
                    {
                        _column("n_valid_shots", n): float(v)
                        for n, v in result.valid_shots.items()
                    }
                )
                record.values.update(
                    {
                        _column("best", n): best.get(n, math.nan)
                        for n in [*movables, *cfg.vocs.objectives]
                    }
                )
                yield from bps.trigger_and_read([record], name="optimization")
                yield Msg("rewindable", None, True)
            if cfg.run.on_finish == "best":
                target = driver.best or initial
                moves = [
                    item
                    for name, obj in movables.items()
                    if not getattr(obj, "relative", False)
                    for item in (obj, target[name])
                ]
                if moves:
                    yield from bps.mv(*moves)

        def cleanup():
            yield Msg("rewindable", None, True)
            yield from _on_thread(pool, lambda: None)
            for source in compiled.sources.values():
                source.close()
            # A running calculation is allowed to finish; do not block the RE while it does.
            pool.shutdown(wait=False, cancel_futures=True)
            yield from bps.null()

        def _dump_record():
            # Dump after all use of the driver, before the stop document closes its log.
            yield from _on_thread(pool, lambda: None)
            if records.folder is not None:
                driver.dump(records.folder / "xopt_dump.yaml")
            yield from bps.null()

        def prepared():
            # Read only scalar surfaces before claim. These messages also invoke
            # connect_on_demand; a camera's full describe requires prepare first.
            keys = set()
            for obj in detectors:
                keys.update((yield from bps.read(getattr(obj, "scalars", obj))))
            for name, obj in movables.items():
                reading = yield from bps.read(obj)
                if len(reading) != 1:
                    raise GeecsConfigurationError(
                        f"{name}: expected one numeric readback"
                    )
                key = next(iter(reading))
                if key not in keys:
                    measurement_signals.append(_MeasurementReadback(obj, key, name))
                    keys.add(key)
            for measurement in compiled.measurements:
                if (
                    measurement.readable is not None
                    and measurement.event_key not in keys
                ):
                    reading = yield from bps.read(measurement.readable)
                    if measurement.event_key not in reading:
                        raise GeecsConfigurationError(
                            f"missing measurement column {measurement.event_key}"
                        )
                    measurement_signals.append(
                        _MeasurementReadback(
                            measurement.readable,
                            measurement.event_key,
                            measurement.spec.signal.replace(":", " ", 1),
                        )
                    )
                    keys.add(measurement.event_key)
            for source in compiled.sources.values():
                source.open()
                yield from _on_thread(pool, source.wait_connected, 5.0)
            yield from liveness_gate(
                sc, [*detectors, *movables.values(), *non_essential]
            )
            run = bpp.run_wrapper(
                name_failed_status(bpp.finalize_wrapper(inner(), _dump_record())),
                md=metadata,
            )
            staged = bpp.stage_wrapper(
                non_essential_wrapper(run, non_essential),
                [*detectors, *movables.values()],
            )
            yield from name_failed_status(run_bracket(staged, sc, TriggerState.ARMED))

        # Subscribe outside open_run to see the claim preprocessor's augmented start document.
        yield from bpp.subs_wrapper(
            bpp.finalize_wrapper(prepared(), cleanup()), records
        )

    # Device annotations are interpreted specially by the manager. Leave the list
    # arguments unannotated, like the bound stock plans.
    optimize.__signature__ = inspect.signature(optimize, eval_str=True)
    return optimize
