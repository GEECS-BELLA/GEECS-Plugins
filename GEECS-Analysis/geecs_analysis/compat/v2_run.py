"""Streaming v2 execution over explicit shot groups and a caller-owned loader.

The host resolves files, grouping and output destinations. This runner only
loads the requested group and evaluates the compiled recipe. It writes nothing.

With ``workers > 1`` the groups are analyzed by a process pool (h5py holds a
global lock, so threads would serialize the reads and the decompression)
and still yielded in declared order through a bounded window, so whatever
the host accumulates sees the same sequence — and therefore computes the
same numbers — as the serial loop.
"""

from __future__ import annotations

import logging
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Mapping

from geecs_analysis.compat.v2 import V2Recipe, analyze_v2
from geecs_analysis.pipeline import bind_inputs

if TYPE_CHECKING:
    import numpy as np
    from geecs_data_utils.frames import ShotMeta
    from geecs_analysis.compat.v2_average import _Sum
    from geecs_analysis.measurement import Measurement

#: The multiprocessing start method of the worker pool. The hosts are
#: threaded service processes (a FastAPI app, the task queue), which a fork
#: would copy mid-lock; a spawned worker starts clean and rebuilds the step
#: registry by importing it.
START_METHOD = "spawn"


@dataclass(frozen=True)
class ShotGroup:
    """One shot or bin, retaining all member rows for legacy scalar propagation."""

    key: int
    shots: tuple[int, ...]

    def __post_init__(self) -> None:
        """Own a nonempty, unique sequence of positive shot numbers."""
        shots = tuple(self.shots)
        if not shots or any(type(shot) is not int or shot < 1 for shot in shots):
            raise ValueError("A group requires positive integer shot numbers")
        if len(set(shots)) != len(shots):
            raise ValueError("A group cannot repeat a shot number")
        object.__setattr__(self, "shots", shots)


@dataclass(frozen=True)
class LoadFailure:
    """A source failed to load a member; the bin may still have usable members."""

    shot: int
    message: str


@dataclass(frozen=True)
class UnitResult:
    """One explicit outcome, including failed loads and analysis failure.

    ``group.shots`` is the legacy scalar-write membership. ``loaded_shots`` is
    the actual contribution set, which can be smaller after source failures.
    Neither set is silently inferred from the other. The caller decides where
    to log failures and how to persist/display successful measurements.
    """

    group: ShotGroup
    loaded_shots: tuple[int, ...]
    measurement: Measurement | None
    load_failures: tuple[LoadFailure, ...] = ()
    error: str | None = None


Loader = Callable[[int], "np.ndarray"]


@contextmanager
def _opened(load: Loader) -> Iterator[Loader]:
    """Enter a loader that holds a per-run resource, once for the whole run.

    A plain callable is used as is. A loader that is also a context manager
    (a source keeping one stack handle open) is entered once and whatever
    ``__enter__`` returns — itself, or a per-shot callable — reads the shots;
    it is exited when the run completes, fails or is closed.
    """
    if hasattr(load, "__enter__") and hasattr(load, "__exit__"):
        with load as entered:  # type: ignore[attr-defined]
            yield entered if callable(entered) else load
    else:
        yield load


def _run_group(
    recipe: V2Recipe,
    group: ShotGroup,
    load: Loader,
    average_before_analysis: bool,
    bound: Mapping[str, object],
    metadata: Mapping[int, ShotMeta],
) -> UnitResult:
    """Load one group's members in declared order and evaluate the recipe."""
    import numpy as np
    from geecs_data_utils.frames import ShotMeta

    from geecs_analysis.compat.v2_average import _Sum

    loaded = []
    failures = []
    # Per bin, each frame is folded into a running sum as it arrives (plain
    # mean, not nanmean: a NaN sample propagates), so the group holds one
    # native frame and one accumulator however many shots it has. Per shot,
    # the single frame is kept.
    total: _Sum | None = None
    first = native = error = None
    for shot in group.shots:
        try:
            data = load(shot)
            if not isinstance(data, np.ndarray):
                raise TypeError("Source must return a native ndarray")
        except Exception as exc:
            failures.append(LoadFailure(shot, str(exc)))
            continue
        loaded.append(shot)
        if error is not None:
            pass
        elif not average_before_analysis:
            # A streaming source may reuse its read buffer on the next
            # call. Retain this shot's native precision and values now.
            first = data.copy()
        elif total is None:
            native = data.dtype
            total = _Sum(data, skip_nan=False, dtype=_mean_dtype(native))
        elif data.shape != total.total.shape:
            # The stacked mean refused ragged members. Later members are
            # still read so the bin's loads and failures are all reported.
            error = (
                f"Incompatible raw shapes in group: "
                f"{total.total.shape} and {data.shape}"
            )
            total = None
        elif _mean_dtype(np.result_type(native, data.dtype)) != total.total.dtype:
            # The stack would sum every member in the promoted dtype; the
            # members already folded were rounded in the narrower one, so
            # no fold reproduces it. A mix that keeps the accumulator's
            # dtype (integer samples, all summed in float64) folds exactly.
            error = f"Incompatible raw dtypes in group: {native} and {data.dtype}"
            total = None
        else:
            # The result dtype follows the promoted stack (float16 members
            # joined by float32 ones average to float32, not float16).
            native = np.result_type(native, data.dtype)
            total.add(data)
        data = None
    measurement = None
    raw = None
    if error is None and not loaded:
        error = "No loadable inputs in group"
    elif error is None:
        try:
            shot = None
            if average_before_analysis:
                raw = total.mean()
                if native == np.float16:
                    raw = raw.astype(native)
            else:
                raw = first
                number = group.shots[0]
                shot = metadata.get(number, ShotMeta(recipe.device, number))
            measurement = analyze_v2(raw, recipe, shot=shot, inputs=bound)
        except Exception as exc:
            error = str(exc)
    # Release native arrays before the outcome travels. Only the owned
    # Measurement and lightweight outcome survive while the sink handles it.
    total = first = raw = None
    return UnitResult(group, tuple(loaded), measurement, tuple(failures), error)


def _mean_dtype(native: np.dtype) -> np.dtype:
    """The intermediate dtype ``np.mean`` sums samples of ``native`` in.

    Integer and bool samples sum in float64, float16 in float32 (the mean is
    cast back to float16), every other dtype in itself — so a float32 trace
    averages in float32 exactly as the stacked mean did.
    """
    import numpy as np

    if np.issubdtype(native, np.integer) or native == np.bool_:
        return np.dtype(np.float64)
    if native == np.float16:
        return np.dtype(np.float32)
    return native


def run_units(
    recipe: V2Recipe,
    groups: Iterable[ShotGroup],
    load: Loader,
    *,
    average_before_analysis: bool = False,
    inputs: Mapping[str, object] | None = None,
    shot_metadata: Mapping[int, ShotMeta] | None = None,
    workers: int = 1,
) -> Iterator[UnitResult]:
    """Yield ordered outcomes, holding at most one raw frame at a time.

    Per-shot mode requires single-member groups. Per-bin mode averages native
    arrays before v2 scaling/processing, preserving numpy's legacy dtype and
    mean (not nanmean) semantics: each member is folded into a running sum as
    it loads, so a bin of any size holds one frame and one accumulator, and
    the quotient is the stacked ``np.mean`` bit for bit. Bad loads are excluded; the original bin's
    full scalar-write membership survives. Incompatible raw shapes or analysis
    failures yield an explicit unsuccessful outcome and later groups continue.

    Required frame bindings (and a measure's service, e.g. the FROG
    retriever) are snapshotted before the first load; a pooled run sends
    them to each worker once, so a service must pickle. The loader
    belongs to the source host; no paths, config reads, writes or renderer
    state live here. A loader that is also a context manager is entered once
    per run (and once per worker), for a source that keeps one stack handle.
    Loading is sequential in declared member order so the floating-point
    reduction order is reproducible.

    ``workers <= 1`` runs the serial loop in this process; no pool exists.
    ``workers > 1`` analyzes the groups in a ``spawn`` process pool: the
    recipe, the bound inputs and the loader travel to each worker once, by
    pickle, each worker reads and analyzes its own groups, and the outcomes
    are yielded in declared group order through a window of at most twice
    ``workers`` groups in flight. Log records the workers emit are handed
    to the loggers of this process (at this process's root level or above).
    Closing the iterator early shuts the pool down; a worker's per-unit
    failures are outcomes, exactly as in the serial loop.
    """
    from geecs_data_utils.frames import ShotMeta

    bound = bind_inputs(recipe.analysis.steps, inputs, measure=recipe.analysis.measure)
    metadata = dict(shot_metadata or {})
    if any(
        not isinstance(identity, ShotMeta) or identity.shot_number != number
        for number, identity in metadata.items()
    ):
        raise ValueError("Shot metadata must match its shot-number key")
    if int(workers) > 1:
        return _run_pooled(
            recipe,
            groups,
            load,
            average_before_analysis,
            dict(bound),
            metadata,
            int(workers),
        )
    return _run_serial(recipe, groups, load, average_before_analysis, bound, metadata)


def _checked(group: ShotGroup, average_before_analysis: bool) -> ShotGroup:
    if not average_before_analysis and len(group.shots) != 1:
        raise ValueError("Per-shot execution requires single-member groups")
    return group


def _run_serial(
    recipe: V2Recipe,
    groups: Iterable[ShotGroup],
    load: Loader,
    average_before_analysis: bool,
    bound: Mapping[str, object],
    metadata: Mapping[int, ShotMeta],
) -> Iterator[UnitResult]:
    with _opened(load) as read:
        for group in groups:
            yield _run_group(
                recipe,
                _checked(group, average_before_analysis),
                read,
                average_before_analysis,
                bound,
                metadata,
            )


# ----------------------------------------------------------------- the pool


class _ForwardToLoggers(logging.Handler):
    """Hand a worker's record to this process's logger of the same name."""

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102 - Handler API
        logger = logging.getLogger(record.name)
        if logger.isEnabledFor(record.levelno):
            logger.handle(record)


#: One worker process's state, set by :func:`_worker_init`.
_WORKER: dict[str, Any] = {}


def _exit_with_parent() -> None:
    """End this worker the moment the process that owns the pool is gone.

    A pool worker blocks on its task queue, and every worker holds a writer
    end of that queue itself, so a parent killed without running its
    ``finally`` (SIGTERM or SIGKILL to a task runner or a detached analysis
    process) never delivers end-of-file: the workers would live on under
    init, each holding its stack handle. A daemon thread waits on the
    parent's sentinel and terminates the worker when it fires; the
    operating system closes the worker's files.
    """
    import multiprocessing
    import signal
    import threading
    from multiprocessing.connection import wait

    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def watch() -> None:
        wait([parent.sentinel])
        # The default SIGTERM action ends the process at once, as os._exit
        # would (the core imports no os): no queue flush, no atexit.
        signal.raise_signal(signal.SIGTERM)

    threading.Thread(target=watch, name="exit-with-parent", daemon=True).start()


def _worker_init(
    queue: Any,
    level: int,
    recipe: V2Recipe,
    load: Loader,
    inputs: Mapping[str, object],
    average_before_analysis: bool,
) -> None:
    """Configure one spawned worker: logging, the registry, the opened loader."""
    import atexit
    from logging.handlers import QueueHandler

    # A spawned interpreter has an empty registry; the builtins register on
    # import, and unpickling the recipe imports only the specs it carries.
    # Imported before the forwarding handler exists: the records those
    # modules emit at import time are start-up lines the parent already
    # logged once, and would otherwise reach the host log once per worker.
    import geecs_analysis.measures  # noqa: F401
    import geecs_analysis.steps  # noqa: F401

    root = logging.getLogger()
    root.addHandler(QueueHandler(queue))
    root.setLevel(level)
    _exit_with_parent()
    opened = _opened(load)
    read = opened.__enter__()
    atexit.register(opened.__exit__, None, None, None)
    _WORKER.update(
        recipe=recipe,
        read=read,
        average=average_before_analysis,
        bound=bind_inputs(
            recipe.analysis.steps, inputs, measure=recipe.analysis.measure
        ),
    )


def _worker_run(group: ShotGroup, metadata: Mapping[int, ShotMeta]) -> UnitResult:
    """Analyze one group in the worker holding the run's state."""
    state = _WORKER
    return _run_group(
        state["recipe"],
        group,
        state["read"],
        state["average"],
        state["bound"],
        metadata,
    )


def _run_pooled(
    recipe: V2Recipe,
    groups: Iterable[ShotGroup],
    load: Loader,
    average_before_analysis: bool,
    inputs: dict[str, object],
    metadata: Mapping[int, ShotMeta],
    workers: int,
) -> Iterator[UnitResult]:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from logging.handlers import QueueListener

    context = multiprocessing.get_context(START_METHOD)
    queue = context.Queue()
    listener = QueueListener(queue, _ForwardToLoggers(), respect_handler_level=False)
    listener.start()
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_worker_init,
        initargs=(
            queue,
            logging.getLogger().getEffectiveLevel(),
            recipe,
            load,
            inputs,
            average_before_analysis,
        ),
    )
    pending: deque = deque()
    window = 2 * workers
    try:
        for group in groups:
            group = _checked(group, average_before_analysis)
            local = {n: metadata[n] for n in group.shots if n in metadata}
            pending.append(executor.submit(_worker_run, group, local))
            if len(pending) >= window:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
    finally:
        # Groups not yet started are dropped; the ones in flight finish so
        # the workers exit through their atexit handlers (the loader's
        # __exit__) rather than being killed with a stack handle open.
        executor.shutdown(wait=True, cancel_futures=True)
        listener.stop()
        queue.close()
        queue.join_thread()
