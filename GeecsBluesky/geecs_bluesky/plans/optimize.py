"""Native strict optimization: one run, one bin per adaptive iteration.

The run is a strict shot-table run (``plans/on_demand.py``): ``primary``
is one row per fire, every per-shot device records into its own stream,
and the start document says so.  At the end of each bin the objective's
scalars come from an **in-plan join** — the run's documents so far,
through the s-file's own :func:`geecs_data_utils.shot_table.shot_table`
(:class:`_BinJoin`) — over the bin's complete rows.  A camera objective
reads the live PVA frames matched by stamp (``optimization/live_frames``);
a plugin camera's stamp reaches the plan through its companion stamp
stream (``plans/streams.py``).
"""

from __future__ import annotations

import inspect
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Generator, Mapping, Sequence
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any

from bluesky import plan_stubs as bps, preprocessors as bpp
from bluesky.protocols import Movable, Readable
from bluesky.utils import Msg
from geecs_bluesky.devices.ca._view import owner_of
from geecs_bluesky.devices.native import refuse_native_cameras
from geecs_bluesky.native_scope import NOT_OPTIMIZE
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.optimization.events import (
    OptimizationRole,
    optimization_column as _column,
)
from geecs_bluesky.callbacks._base import _StreamCallback
from geecs_data_utils.shot_table import fresh_rows, shot_table
from geecs_schemas import TriggerState

from .gated import run_bracket
from .on_demand import OnDemand, _Progress, start_keys
from .streams import DeviceStreams, is_per_shot
from .scan_mode import (
    EnterScanMode,
    run_device_names,
    scan_mode_bracket,
    scan_mode_preflight,
    scan_parameter_for,
)
from .strict import BinCounter, name_failed_status

if TYPE_CHECKING:
    from geecs_bluesky.config_resolver import ConfigsRepoResolver
    from geecs_bluesky.namespace import GeecsNamespace
    from .registry import TriggerProfiles

logger = logging.getLogger(__name__)


class OptimizationRecord:
    """Fixed columns for proposed/measured variables, outputs and best-so-far."""

    name = "optimization"
    parent = None

    def __init__(self, names: list[str]) -> None:
        # Tiled adds ts_ columns and SQL identifiers are case-insensitive.
        if any(len(name) > 60 for name in names):
            raise GeecsConfigurationError(
                "optimizer column names must fit 60 characters after escaping"
            )
        if len({name.casefold() for name in names}) != len(names):
            raise GeecsConfigurationError(
                "optimizer column names must be distinct ignoring case"
            )
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


class _BinJoin(_StreamCallback):
    """The in-plan join: the run's documents so far, through the s-file's shot table.

    The s-file's own document buffer (``callbacks/_base.py``), read before
    the stop document: :meth:`bin_rows` joins every per-shot event stream —
    and each plugin camera's companion stamp stream, under the camera's
    name — onto ``primary`` with :func:`~geecs_data_utils.shot_table.shot_table`
    and returns one bin's complete rows (every row of the bin when none is
    complete: the measurement's ``min_shots`` decides).
    """

    def __init__(self, companions: Mapping[str, str]) -> None:
        super().__init__()
        self.companions = dict(companions)
        self.folder: Path | None = None
        self._start: dict[str, Any] | None = None

    def on_start(self, start: dict[str, Any]) -> None:
        """Buffer the run; its companions are event streams of their cameras."""
        super().on_start(start)
        self._start = start
        self._runs[str(start["uid"])].event_streams.update(self.companions)
        if start.get("scan_folder"):
            self.folder = Path(start["scan_folder"])

    def bin_rows(self, column: str, value: int) -> list[dict[str, Any]]:
        """The rows of bin *value* (``column``), every device's scalars joined."""
        import pandas as pd

        start = self._start
        run = self._runs.get(str(start["uid"])) if start is not None else None
        rows = run.stream_rows("primary") if run is not None else []
        if start is None or run is None or not rows:
            return []
        table = shot_table(
            start,
            pd.DataFrame(rows),
            run.event_stream_records(start),
            drain_offsets=run.drain_offsets,
        )
        # A stale sample is not the shot's measurement: never tell it.
        joined = fresh_rows(table)
        mine = joined[joined[column] == value]
        complete = mine[mine["complete"].astype(bool)]
        # NaN is the join's gap; the measurements skip it.
        return (complete if len(complete) else mine).to_dict("records")


#: Seconds the bin's end waits for the companions' last stamps and the
#: device streams' last snapshots.
SETTLE_S = 1.0


class _Settle:
    """Plan: until every stamp the bin's devices published is recorded, bounded.

    Two waits in one bound.  A companion: until it recorded a stamp per
    frame its camera wrote — the camera's frame count and its gateway stamp
    travel separate paths, so a stamp still in flight at the bin's end
    would cost that shot's frame.  A stamp that never comes (the gateway
    dropped it) is a frame without a stamp: on expiry the shortfall is
    logged once per camera and carried, so a later bin waits only for its
    own frames.  A device stream (``streams.stamped``): until no stamp is
    pending — its stamp completes the slot the moment it lands, while the
    stream is still reading that shot's scalars, and an event not yet
    recorded would be missing from the bin; on expiry it is logged once per
    stream (the event joins a later collect, not this bin).
    """

    def __init__(self, streams: DeviceStreams) -> None:
        self.streams = streams
        self.shortfall = {id(c): 0 for c in streams.companions}
        self._warned: set[int] = set()
        self._warned_stamped: set[int] = set()

    def _missing(self, companion: Any, frames: tuple[float, ...]) -> int:
        if not frames or companion.pending:
            return int(bool(companion.pending))
        settled = companion.published + companion.failed + self.shortfall[id(companion)]
        return max(0, int(min(frames)) - settled)

    def _report_unrecorded(self, stamped: Any) -> None:
        """Log once per device stream a snapshot still pending at the bound."""
        for s in stamped:
            if s.pending and id(s) not in self._warned_stamped:
                self._warned_stamped.add(id(s))
                logger.warning(
                    "%s: a snapshot still unrecorded after %.1f s at the bin's "
                    "end — that shot is not in the bin (logged once)",
                    s.device_name,
                    SETTLE_S,
                )

    def __call__(self):
        import asyncio

        companions = self.streams.companions
        stamped = self.streams.stamped
        missing: dict[int, int] = {}

        async def settle() -> None:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + SETTLE_S
            while True:
                for c in companions:
                    frames = await _Progress(c.device, True).read_safe()
                    missing[id(c)] = self._missing(c, frames)
                if (
                    not any(missing.values()) and not any(s.pending for s in stamped)
                ) or loop.time() >= deadline:
                    return
                await asyncio.sleep(0.02)

        if companions or stamped:
            yield from bps.wait_for([settle])
        self._report_unrecorded(stamped)
        for c in companions:
            short = missing.get(id(c), 0)
            if short and not c.pending:
                self.shortfall[id(c)] += short
            if short and id(c) not in self._warned:
                self._warned.add(id(c))
                logger.warning(
                    "%s: %d frame(s) without a gateway stamp after %.1f s — "
                    "those shots read no live frame (logged once)",
                    c.device_name,
                    short,
                    SETTLE_S,
                )
        yield from self.streams.collect()


def _measured(
    rows: list[dict[str, Any]], keys: Mapping[str, str], iteration: int
) -> dict[str, float]:
    """Each variable's mean readback over *rows*; ``NaN`` (and a WARNING) when none is finite.

    Never the proposal: a bin without a readback is recorded as measured
    NaN and not told to the optimizer.
    """
    out = {}
    for name, key in keys.items():
        values = [
            float(r[key]) for r in rows if math.isfinite(float(r.get(key, math.nan)))
        ]
        out[name] = sum(values) / len(values) if values else math.nan
        if not values:
            logger.warning(
                "bin %d: no readback of %s in any row — measured NaN, the bin "
                "is not told to the optimizer",
                iteration,
                name,
            )
    return out


def _on_thread(pool, function, *args):
    future = pool.submit(function, *args)
    while not future.done():
        yield from bps.sleep(0.05)
    return future.result()


def optimize_plan(  # noqa: C901, PLR0915
    profiles: TriggerProfiles,
    resolver: ConfigsRepoResolver | None,
    namespace: GeecsNamespace | None,
    *,
    mock: bool = False,
    scan_mode: Any | None = None,
) -> Callable[..., Generator]:
    """Bind the optimize verb without importing the optional analysis/Xopt stack.

    *mock* connects the background telemetry's members with mock backends
    (hermetic tests), as :func:`~geecs_bluesky.plans.registry.bind_plans`
    passes it; *scan_mode* is the worker's scan-mode control, driven under
    a profile with ``use_scan_mode`` exactly as the strict binder drives it
    (``plans/scan_mode.py``).
    """

    def optimize(  # noqa: C901, PLR0912, PLR0915
        detectors: Sequence[Readable],
        *,
        optimizer_config: str,
        max_iterations: int | None = None,
        shots_per_step: int | None = None,
        trigger_profile: str | None = None,
        shot_period: float | None = None,
        required: Sequence[Readable] | None = None,
        native_image_save: bool | None = None,
        background_telemetry: bool | None = None,
        md: dict[str, Any] | None = None,
    ):
        """Optimize a configured objective through strict acquisition.

        Parameters
        ----------
        detectors : list
            The devices to record: every per-shot device (a detector, its
            ``.scalars`` view) in its own stream, every other one a
            condition read in each row.
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
        required : list, optional
            The per-shot devices a shot waits for and refires on (``None``:
            every one; ``[]``: none).  The objective's devices must be among
            them.
        native_image_save : bool, optional
            Whether the plugin-backed cameras also write their LabVIEW
            per-shot files; the experiment default when omitted.  The
            objective reads live PVA frames, never those files.
        background_telemetry : bool, optional
            Whether every other logged scalar of the experiment is read
            into every row as well, softly (the experiment default when
            omitted) — the same background every scan verb records.
        md : dict, optional
            Additional run metadata.
        """
        from .registry import (
            _refuse_twice_listed,
            resolve_required,
            background_snapshot,
            background_wrapper,
            liveness_gate,
            native_image_save_wrapper,
            require_plain_metadata,
            resolve_background_telemetry,
            resolve_native_image_save,
            refuse_listed_witness,
            run_witness,
            witness_wrapper,
        )
        from geecs_bluesky.optimization.driver import XoptDriver
        from geecs_bluesky.optimization.measurements import compile_measurements

        require_plain_metadata(md)
        if resolver is None or namespace is None:
            raise GeecsConfigurationError(
                "optimize needs a configs resolver and device namespace"
            )
        detectors = list(detectors)
        refuse_native_cameras(detectors, NOT_OPTIMIZE)
        refuse_listed_witness(detectors)
        per_shot = [d for d in detectors if is_per_shot(d)]
        _refuse_twice_listed(per_shot)
        required_devices = resolve_required(per_shot, required)
        conditions = [d for d in detectors if not is_per_shot(d)]
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
                    f"optimizer requires the device {device} — list it"
                )
            if is_per_shot(owner) and not any(
                owner_of(d) is owner for d in required_devices
            ):
                raise GeecsConfigurationError(
                    f"optimizer requires {device} required: the objective "
                    "reads it every shot"
                )
            if device in compiled.sources and (
                owner not in detectors or not (owner.native_save or owner.plugin_backed)
            ):
                raise GeecsConfigurationError(
                    f"optimizer requires images saved for {device}"
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
        announce = scan_mode if sc.use_scan_mode else None
        run_devices = run_device_names(detectors, [], sc)
        if sc.use_scan_mode:
            scan_mode_preflight(
                sc, scan_mode, acquisition="strict", run_devices=run_devices
            )
        native_files = resolve_native_image_save(native_image_save, resolver)
        snapshot = (
            background_snapshot(namespace, mock=mock)
            if resolve_background_telemetry(background_telemetry, resolver)
            else None
        )
        metadata = dict(md or {})
        metadata.update(
            plan_name="optimize",
            optimizer_config=optimizer_config,
            max_iterations=iterations,
            shots_per_step=shots,
            acquisition="strict",
            trigger_profile=trigger_profile or profiles.default,
            use_scan_mode=sc.use_scan_mode,
            shot_period=shot_period,
            native_image_save=native_files,
            background_telemetry=snapshot is not None,
        )
        witness, metadata["witness"] = yield from run_witness(
            namespace,
            acquisition="strict",
            shot_control=sc,
            readers=detectors,
            waited=True,
        )
        # Read with every row and staged (the stage resets a pre-read's
        # audit), like the background below; never waited on as a device.
        extras = [] if witness is None else [witness]
        # A plugin camera any measurement reads (its frames or a scalar):
        # its stream is datum-only until stop, so the companion carries its
        # per-shot stamp and cached scalars into the bin's join.
        read_owners = {
            id(owner_of(namespace.resolve(m.device))) for m in compiled.measurements
        }
        cameras = [
            d
            for d in per_shot
            if getattr(d, "plugin_backed", False) and id(owner_of(d)) in read_owners
        ]
        streams = DeviceStreams(per_shot, dual_write=True, companions=cameras)
        metadata.update(start_keys(streams, required_devices, witness is not None))
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
        records = _BinJoin(streams.companion_streams())
        measurement_signals = []
        record = OptimizationRecord(
            [
                "iteration",
                *[_column(OptimizationRole.BEST_MOVE, n) for n in move_targets],
                *[_column(OptimizationRole.PROPOSAL, n) for n in movables],
                *[_column(OptimizationRole.MEASURED, n) for n in movables],
                *[_column(OptimizationRole.OUTPUT, n) for n in compiled.output_names],
                *[_column(OptimizationRole.VALID_SHOTS, n) for n in cfg.measurements],
                *[
                    _column(OptimizationRole.BEST, n)
                    for n in [*movables, *cfg.vocs.objectives]
                ],
            ]
        )
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="geecs-optimize")
        settle = _Settle(streams)
        slot = OnDemand(
            sc,
            streams,
            required_devices,
            witness=witness,
            shot_period=shot_period,
            scan_mode=announce,
        )
        bins = BinCounter()
        row_extras = [bins] if announce is None else [bins, announce]

        def inner():  # noqa: C901
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
                for _ in range(shots):
                    # A checkpoint per shot: a deferred pause lands between shots.
                    yield from bps.checkpoint()
                    yield from slot.take_reading(
                        [*conditions, *measurement_signals, *row_extras, *extras]
                    )
                # Commit the acquired bin: calculation and tell must never be replayed.
                yield Msg("rewindable", None, False)
                yield from settle()
                rows = records.bin_rows(bins.name, iteration)
                frames = yield from _on_thread(pool, compiled.frames_for, rows)
                result = yield from _on_thread(
                    pool, compiled.evaluate_bin, rows, frames
                )
                measured = _measured(rows, reading_keys, iteration)
                if all(math.isfinite(v) for v in measured.values()):
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
                        _column(OptimizationRole.BEST_MOVE, n): physical_best.get(
                            n, math.nan
                        )
                        for n in move_targets
                    }
                )
                record.values.update(iteration=float(iteration))
                record.values.update(
                    {
                        _column(OptimizationRole.PROPOSAL, n): v
                        for n, v in proposal.items()
                    }
                )
                record.values.update(
                    {
                        _column(OptimizationRole.MEASURED, n): v
                        for n, v in measured.items()
                    }
                )
                record.values.update(
                    {
                        _column(OptimizationRole.OUTPUT, n): v
                        for n, v in result.outputs.items()
                    }
                )
                record.values.update(
                    {
                        _column(OptimizationRole.VALID_SHOTS, n): float(v)
                        for n, v in result.valid_shots.items()
                    }
                )
                record.values.update(
                    {
                        _column(OptimizationRole.BEST, n): best.get(n, math.nan)
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

        def prepared():  # noqa: C901
            # Read only scalar surfaces before claim. These messages also invoke
            # connect_on_demand; a camera's full describe requires prepare first.
            # A movable's readback is read in the row unless a condition
            # already carries it there; a measurement's column may come from
            # a per-shot device's stream (the in-plan join).
            keys, row_keys = set(), set()
            for obj in detectors:
                reading = yield from bps.read(getattr(obj, "scalars", obj))
                keys.update(reading)
                if not is_per_shot(obj):
                    row_keys.update(reading)
            for name, obj in movables.items():
                reading = yield from bps.read(obj)
                if len(reading) != 1:
                    raise GeecsConfigurationError(
                        f"{name}: expected one numeric readback"
                    )
                key = next(iter(reading))
                if key not in row_keys:
                    measurement_signals.append(_MeasurementReadback(obj, key, name))
                    row_keys.add(key)
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
            yield from liveness_gate(sc, [*detectors, *movables.values()])
            if snapshot is not None:
                # One more device of every row, last — after the scalar
                # pre-reads above (nothing to read before its probe).
                extras.append(snapshot)
            run = bpp.run_wrapper(
                name_failed_status(bpp.finalize_wrapper(inner(), _dump_record())),
                md=metadata,
            )
            run = witness_wrapper(run, witness, metadata["witness"])
            staged = bpp.stage_wrapper(
                streams.wrap(run), [*conditions, *extras, *movables.values()]
            )
            if snapshot is not None:
                staged = background_wrapper(
                    staged,
                    snapshot,
                    own=detectors,
                    movers=list(movables.values()),
                )
            staged = native_image_save_wrapper(staged, detectors, native_files)
            if announce is not None:
                staged = scan_mode_bracket(
                    staged,
                    announce,
                    EnterScanMode(run_devices, scan_parameter_for("optimize")),
                )
            yield from name_failed_status(run_bracket(staged, sc, TriggerState.ARMED))

        # Subscribe outside open_run to see the claim preprocessor's augmented start document.
        yield from bpp.subs_wrapper(
            bpp.finalize_wrapper(prepared(), cleanup()), records
        )

    # Every argument is annotated: the manager resolves device names where
    # the annotation says devices, and nowhere else (the metadata included).
    optimize.__signature__ = inspect.signature(optimize, eval_str=True)
    return optimize
