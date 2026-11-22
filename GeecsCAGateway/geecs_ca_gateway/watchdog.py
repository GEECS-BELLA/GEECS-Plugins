"""Monitor-delivery watchdog: notice when caproto stops delivering monitors.

caproto's asyncio server delivers every monitor update, the value on
subscribe included, through ONE context-wide task (``subscription_queue_loop``)
draining ONE unbounded queue and handing each update to the subscribed
circuits with an awaited ``put`` on each circuit's bounded queue. While that
loop is parked, every channel stays connected and every get still answers
(gets run on each circuit's own command loop), but no client receives a
monitor event and the unbounded queue grows without limit. Nothing is logged.

The watchdog reads the stall from inside the process: the context's queue is
replaced by a :class:`CountingQueue`, which counts the updates taken from it.
Delivery has stalled when the queue holds updates and none has been taken for
:data:`STALL_LIMIT_S`. An empty queue is idle, never a stall, so a quiet
gateway (or one with no subscribers) cannot trip it; a busy healthy gateway
takes updates continuously, so its count advances between checks.

On a stall the gateway logs one ERROR block (:func:`stall_report`: the stall,
every circuit's queue state, the circuit the delivery loop is waiting on, and
the stack of every asyncio task), then exits with :data:`WATCHDOG_EXIT_CODE`
so the service manager restarts it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from types import CodeType, FrameType
from typing import Any

logger = logging.getLogger(__name__)

#: Seconds the context queue may hold updates with none taken before the
#: watchdog acts. A healthy loop waits on one circuit at most as long as that
#: circuit takes to send one batch: caproto caps a batch's latency at 1 s
#: (``CAPROTO_SERVER_MAX_LATENCY_SEC``) and the busiest batches seen in
#: service took about 1.2 s. 30 s is 25 times that, short against the
#: minutes a stalled gateway leaves every client without monitors (and the
#: about 270 MB/h the unbounded queue grows by while it does).
STALL_LIMIT_S = 30.0

#: Seconds between two reads of the queue. Short against the limit, so a stall
#: is acted on within ``STALL_LIMIT_S + CHECK_PERIOD_S``; a read is two
#: attribute lookups, so its cost is nil.
CHECK_PERIOD_S = 1.0

#: Exit status after a stall. Any status but the restart request's 86 is a
#: failure to systemd, so ``Restart=on-failure`` relaunches the unit and the
#: journal records the exit as a fault rather than a requested restart.
WATCHDOG_EXIT_CODE = 87

#: Seconds the orderly shutdown after a stall may take before the process is
#: ended outright with :data:`WATCHDOG_EXIT_CODE` (the evidence is already
#: logged by then). Shutdown cancels the server's tasks and closes the device
#: clients, normally well under a second.
SHUTDOWN_GRACE_S = 30.0

#: Task names listed per stack group in the report; the rest are counted.
_NAMES_PER_GROUP = 6


class MonitorDeliveryStalledError(RuntimeError):
    """caproto's monitor delivery made no progress for the stall limit.

    Parameters
    ----------
    stalled_s : float
        How long the context queue held updates with none taken.
    depth : int
        The queue depth when the watchdog acted.
    """

    def __init__(self, stalled_s: float, depth: int) -> None:
        super().__init__(
            f"CA monitor delivery stalled for {stalled_s:.1f} s "
            f"({depth} update(s) queued)"
        )
        self.stalled_s = stalled_s
        self.depth = depth


class CountingQueue(asyncio.Queue):
    """An ``asyncio.Queue`` that counts the items taken from it.

    ``gets`` advances on every item removed, by ``get`` or ``get_nowait``;
    it is the delivery loop's progress, read by :class:`DeliveryWatchdog`.
    """

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize)
        self.gets = 0

    def _get(self) -> Any:  # Any: the items are caproto's update tuples
        self.gets += 1
        return super()._get()


def install_counting_queue(context: Any) -> CountingQueue:  # Any: caproto's Context
    """Replace a caproto context's subscription queue with a counting one.

    Must run before ``context.run()``: channels register the queue object
    when a client subscribes, so a later swap would split the traffic.

    Parameters
    ----------
    context : caproto.asyncio.server.Context
        The server context, constructed and not yet running.

    Returns
    -------
    CountingQueue
        The queue now installed as ``context.subscription_queue``.
    """
    if context.subscription_queue.qsize():
        raise RuntimeError("the context's subscription queue is already in use")
    queue = CountingQueue()
    context.subscription_queue = queue
    return queue


class DeliveryWatchdog:
    """Decide, from periodic reads of the queue, whether delivery has stalled.

    Delivery has made progress at a read when the queue is empty or an item
    was taken since the previous read. It has stalled once no progress has
    been seen for ``stall_limit_s``. The watchdog starts as if progress had
    just been seen, so startup is never a stall.

    Parameters
    ----------
    queue : CountingQueue
        The context's subscription queue.
    stall_limit_s : float
        Seconds without progress, with updates queued, before :meth:`watch`
        returns.
    clock : callable, optional
        Monotonic seconds; ``time.monotonic`` by default.
    """

    def __init__(
        self,
        queue: CountingQueue,
        *,
        stall_limit_s: float = STALL_LIMIT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.queue = queue
        self.stall_limit_s = stall_limit_s
        self._clock = clock
        self._last_gets = queue.gets
        self._last_progress = clock()

    def stalled_for(self) -> float:
        """Read the queue; return the seconds since progress was last seen."""
        now = self._clock()
        gets = self.queue.gets
        if self.queue.qsize() == 0 or gets != self._last_gets:
            self._last_progress = now
        self._last_gets = gets
        return now - self._last_progress

    async def watch(self, check_period_s: float = CHECK_PERIOD_S) -> float:
        """Read the queue every ``check_period_s`` until delivery stalls.

        Returns
        -------
        float
            How long delivery had stalled when the limit was passed.
        """
        while True:
            await asyncio.sleep(check_period_s)
            stalled = self.stalled_for()
            if stalled >= self.stall_limit_s:
                return stalled


def arm_hard_exit(
    code: int,
    grace_s: float = SHUTDOWN_GRACE_S,
    exit_fn: Callable[[int], object] = os._exit,
) -> threading.Timer:
    """End the process with ``code`` after ``grace_s`` unless it has exited.

    A daemon timer thread, so it fires even if the event loop never returns
    and dies with the process when the orderly shutdown finishes first.

    Parameters
    ----------
    code : int
        The exit status.
    grace_s : float
        Seconds the orderly shutdown may take.
    exit_fn : callable
        Called with ``code`` when the grace runs out; ``os._exit`` by default.

    Returns
    -------
    threading.Timer
        The started timer.
    """

    def _expire() -> None:
        logger.error(
            "shutdown after the delivery stall did not finish in %.0f s; "
            "exiting with status %d",
            grace_s,
            code,
        )
        for handler in logging.getLogger().handlers:
            handler.flush()
        exit_fn(code)

    timer = threading.Timer(grace_s, _expire)
    timer.daemon = True
    timer.start()
    return timer


# ----------------------------------------------------------------------
# The evidence
# ----------------------------------------------------------------------


def _await_chain(task: asyncio.Task) -> list[FrameType]:
    """Return the frames of a task's await chain, outermost first.

    ``Task.get_stack`` gives only the task's own coroutine frame; following
    ``cr_await`` reaches the innermost awaiting coroutine, where a parked
    ``await`` is visible.
    """
    frames: list[FrameType] = []
    coro: Any = task.get_coro()
    while coro is not None:
        frame = getattr(coro, "cr_frame", None) or getattr(coro, "gi_frame", None)
        if frame is None:
            break
        frames.append(frame)
        coro = getattr(coro, "cr_await", None) or getattr(coro, "gi_yieldfrom", None)
    return frames


def _short_path(code: CodeType) -> str:
    """A code object's file, from its package directory down."""
    parts = code.co_filename.replace("\\", "/").split("/")
    return "/".join(parts[-3:])


def _format_frames(frames: Iterable[FrameType]) -> list[str]:
    return [
        f"{_short_path(f.f_code)}:{f.f_lineno} in {f.f_code.co_name}" for f in frames
    ]


def _waited_circuit(tasks: Iterable[asyncio.Task]) -> Any:  # Any: a caproto circuit
    """The circuit the context's delivery loop is handing an update to, if any."""
    for task in tasks:
        for frame in _await_chain(task):
            if frame.f_code.co_name == "_subscription_queue_send":
                return frame.f_locals.get("circuit")
    return None


def _address(circuit: Any) -> str:  # Any: a caproto circuit
    try:
        host, port = circuit.circuit.address[:2]
        return f"{host}:{port}"
    except (AttributeError, TypeError, ValueError):
        return repr(circuit)


def _circuit_line(circuit: Any) -> str:  # Any: a caproto circuit
    """One line of a circuit's delivery state."""
    try:
        queue = circuit.subscription_queue
        depth, cap = queue.qsize(), queue.maxsize
        full = " FULL" if cap and depth >= cap else ""
        events_on = circuit.events_on.is_set()
        n_subs = sum(len(subs) for subs in circuit.subscriptions.values())
        sub_task = getattr(circuit, "_sub_task", None)
        if sub_task is None:
            task_state = "none"
        elif sub_task.cancelled():
            task_state = "cancelled"
        elif sub_task.done():
            task_state = "done"
        else:
            task_state = "running"
        return (
            f"{_address(circuit)} connected={circuit.connected} "
            f"queue={depth}/{cap}{full} events_on={events_on} "
            f"subscriptions={n_subs} delivery_task={task_state}"
        )
    except (AttributeError, TypeError, ValueError) as exc:
        return f"{_address(circuit)} (state unreadable: {exc!r})"


def task_stacks(tasks: Iterable[asyncio.Task]) -> list[str]:
    """Every task's await chain, tasks with the same chain grouped together."""
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for task in tasks:
        chain = tuple(_format_frames(_await_chain(task))) or ("(no frames)",)
        groups[chain].append(task.get_name())
    lines: list[str] = []
    for chain, names in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        shown = ", ".join(sorted(names)[:_NAMES_PER_GROUP])
        more = len(names) - _NAMES_PER_GROUP
        suffix = f" and {more} more" if more > 0 else ""
        lines.append(f"  [{len(names)} task(s)] {shown}{suffix}")
        lines.extend(f"      {line}" for line in chain)
    return lines


def stall_report(
    context: Any,  # Any: caproto's Context
    queue: CountingQueue,
    stalled_s: float,
    tasks: Iterable[asyncio.Task] | None = None,
) -> str:
    """The ERROR block logged before the gateway exits on a stall.

    Parameters
    ----------
    context : caproto.asyncio.server.Context
        The running server context (its ``circuits``).
    queue : CountingQueue
        The context's subscription queue.
    stalled_s : float
        How long delivery had stalled.
    tasks : iterable of asyncio.Task, optional
        The tasks to dump; every task of the running loop by default.

    Returns
    -------
    str
        A multi-line report.
    """
    tasks = list(asyncio.all_tasks() if tasks is None else tasks)
    circuits = list(getattr(context, "circuits", ()))
    waited = _waited_circuit(tasks)
    lines = [
        f"CA monitor delivery STALLED: the context subscription queue has held "
        f"updates for {stalled_s:.1f} s with none taken (depth {queue.qsize()}, "
        f"{queue.gets} taken since start). Every client is without monitors; "
        f"gets still answer. Exiting with status {WATCHDOG_EXIT_CODE} for a "
        f"service restart.",
        "delivery loop is waiting on circuit: "
        + (_circuit_line(waited) if waited is not None else "(not found)"),
        f"circuits ({len(circuits)}):",
    ]
    lines.extend(f"  {_circuit_line(c)}" for c in circuits)
    lines.append(f"asyncio tasks ({len(tasks)}), grouped by await chain:")
    lines.extend(task_stacks(tasks))
    return "\n".join(lines)
