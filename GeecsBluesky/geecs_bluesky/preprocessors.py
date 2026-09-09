"""RunEngine preprocessors — the native hooks GEECS occupies (issue #807).

Each function here has the ``bluesky.preprocessors`` shape (``plan → plan``)
so it can be installed once on the RunEngine (``RE.preprocessors.append``)
and applies to **every** plan, stock or not, with no per-plan code.

Phase 1: :func:`connect_on_demand`.  Later phases add the ``md["geecs"]``
preamble/finalize preprocessor (``Planning/native_bluesky/00_overview.md``).
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from functools import partial
from typing import Any

from bluesky.preprocessors import plan_mutator
from bluesky.utils import Msg
from ophyd_async.plan_stubs import ensure_connected

logger = logging.getLogger(__name__)

#: Messages whose object must be connected before the RunEngine processes them.
#: ``stage`` covers the stock plans (they stage detectors and motors); the rest
#: cover plans and stubs that touch a device without staging it first.
TOUCH_COMMANDS: frozenset[str] = frozenset(
    {
        "stage",
        "set",
        "trigger",
        "read",
        "describe",
        "configure",
        "locate",
        "kickoff",
        "complete",
        "collect",
        "monitor",
        "prepare",
    }
)

#: Default per-connect budget, matching the existing preamble's batch timeout.
DEFAULT_CONNECT_TIMEOUT = 20.0


def is_namespace_object(obj: Any) -> bool:
    """Whether *obj* is a namespace device or one of its children (walks ``.parent``)."""
    seen = 0
    while obj is not None and seen < 16:
        if getattr(obj, "_geecs_namespace_member", False):
            return True
        obj = getattr(obj, "parent", None)
        seen += 1
    return False


def is_connected(obj: Any) -> bool:
    """Whether an ophyd-async device already holds a successful connection.

    Real mode: a finished, error-free connect task.  Mock mode: a
    ``DeviceMock`` is installed (mock connects are never cached by
    ophyd-async, and reconnecting would rebuild the mock backends and drop
    any test callbacks registered on them).
    """
    if getattr(obj, "_mock", None) is not None:
        return True
    task = getattr(obj, "_connect_task", None)
    return bool(task is not None and task.done() and task.exception() is None)


def connect_on_demand(
    plan: Generator[Msg, Any, Any],
    *,
    mock: bool = False,
    timeout: float = DEFAULT_CONNECT_TIMEOUT,
    predicate: Any = is_namespace_object,
) -> Generator[Msg, Any, Any]:
    """Connect namespace devices the first time a plan touches them.

    A :func:`~bluesky.preprocessors.plan_mutator`: before the first message
    in *plan* whose object satisfies *predicate* (default: the object, or an
    ancestor, is a :class:`~geecs_bluesky.devices.geecs_device.GeecsDevice`),
    it yields ``ensure_connected(obj)``.  The connect is **message-level** —
    the same ``ensure_connected`` stub the request preamble uses — so the
    standing rule that a *blocking* connect inside the RE loop deadlocks is
    respected: nothing here blocks the loop.  ``ensure_connected`` is
    idempotent (cached connect task), so a device touched by later plans
    costs one no-op await.  Connected devices stay connected.

    Only the touched object's subtree connects: staging ``U_S1H.current``
    connects that Movable, not every variable of ``U_S1H``.
    """
    seen: set[int] = set()

    def _insert_connect(msg: Msg) -> tuple[Any, Any]:
        obj = msg.obj
        if (
            obj is None
            or msg.command not in TOUCH_COMMANDS
            or id(obj) in seen
            or not predicate(obj)
        ):
            return None, None
        seen.add(id(obj))
        if is_connected(obj):
            return None, None

        def _connect_then_forward() -> Generator[Msg, Any, Any]:
            logger.debug(
                "connect_on_demand: connecting %s before %s", obj.name, msg.command
            )
            yield from ensure_connected(obj, mock=mock, timeout=timeout)
            return (yield msg)

        return _connect_then_forward(), None

    return (yield from plan_mutator(plan, _insert_connect))


def install_connect_on_demand(
    run_engine: Any, *, mock: bool = False, timeout: float = DEFAULT_CONNECT_TIMEOUT
) -> None:
    """Append :func:`connect_on_demand` to ``run_engine.preprocessors`` once."""
    for existing in run_engine.preprocessors:
        if getattr(existing, "func", None) is connect_on_demand:
            return
    run_engine.preprocessors.append(
        partial(connect_on_demand, mock=mock, timeout=timeout)
    )
