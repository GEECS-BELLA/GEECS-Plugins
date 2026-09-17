"""RunEngine preprocessors — the native hooks GEECS occupies (issue #807).

Each function here has the ``bluesky.preprocessors`` shape (``plan → plan``)
so it can be installed once on the RunEngine (``RE.preprocessors.append``)
and applies to **every** plan, stock or not, with no per-plan code.

Two live here — :func:`connect_on_demand` (connect a namespace device the
first time a plan touches it) and :func:`scalar_headers` (the legacy
``Device Variable`` header map into the start document).  The scan-number
claim is beside the claim itself,
:func:`geecs_bluesky.plans.claim_scan.claim_scan_preprocessor`.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from functools import partial
from typing import Any

from bluesky.preprocessors import msg_mutator, plan_mutator
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
    in *plan* that touches an object satisfying *predicate* (default: the
    object, or an ancestor, is a namespace device), it yields
    ``ensure_connected(obj)``.  The connect is **message-level** — the same
    ``ensure_connected`` stub the request preamble uses — so the standing
    rule that a *blocking* connect inside the RE loop deadlocks is
    respected: nothing here blocks the loop.  ``ensure_connected`` is
    idempotent (cached connect task), so a device touched by later plans
    costs one no-op await.  Connected devices stay connected.

    Two message shapes carry devices.  Most name one in ``msg.obj``
    (``stage``, ``set``, ``trigger``, ``read`` …).  ``declare_stream`` names
    several in ``msg.args`` with ``obj=None`` — and the RunEngine
    *describes* them right there, before any ``read``; the stock plans and
    ``SupplementalData`` both declare streams, so it is handled too.

    What connects is the touched object's subtree.  Note the stock plans
    stage the **root** ancestor of every device (``stage_wrapper``), so a
    scan over ``U_S1H.Current`` connects all of ``U_S1H``'s served
    children; ``bps.mv(U_S1H.Current, …)`` alone connects only ``Current``.
    """
    seen: set[int] = set()

    def _needs_connect(obj: Any) -> bool:
        if obj is None or id(obj) in seen or not predicate(obj):
            return False
        seen.add(id(obj))
        return not is_connected(obj)

    def _insert_connect(msg: Msg) -> tuple[Any, Any]:
        if msg.command == "declare_stream":
            targets = [d for d in msg.args if _needs_connect(d)]
        elif msg.command in TOUCH_COMMANDS:
            targets = [msg.obj] if _needs_connect(msg.obj) else []
        else:
            return None, None
        if not targets:
            return None, None

        def _connect_then_forward() -> Generator[Msg, Any, Any]:
            logger.debug(
                "connect_on_demand: connecting %s before %s",
                ", ".join(t.name for t in targets),
                msg.command,
            )
            yield from ensure_connected(*targets, mock=mock, timeout=timeout)
            return (yield msg)

        return _connect_then_forward(), None

    return (yield from plan_mutator(plan, _insert_connect))


def scalar_headers(plan: Generator[Msg, Any, Any]) -> Generator[Msg, Any, Any]:
    """Put ``geecs_scalar_headers`` into every start document.

    The s-file needs the legacy ``Device Variable`` header of every event
    column (``geecs_data_utils.tiled_export``, and the scan browser's
    display names): ``safe_name`` mangling is irreversible, so each device
    carries ``_column_headers`` (event key → header) and this
    :func:`~bluesky.preprocessors.msg_mutator` merges them into the run's
    metadata.  Which devices: the ones the plan **staged** before
    ``open_run`` — the stock plans stage every detector and motor of the
    primary stream (their root ancestors — so the walk covers every
    descendant: a settable child's header, a detector's ``scalars`` view).
    A header map the plan already carries is kept.
    """
    staged: list[Any] = []

    def _walk(obj: Any):
        yield obj
        children = getattr(obj, "children", None)
        if callable(children):
            for _, child in children():
                yield from _walk(child)

    def _mutate(msg: Msg) -> Msg:
        if msg.command == "stage":
            staged.append(msg.obj)
            return msg
        if msg.command != "open_run":
            return msg
        headers: dict[str, str] = {}
        for root in staged:
            for obj in _walk(root):
                headers.update(getattr(obj, "_column_headers", None) or {})
        staged.clear()
        if "geecs_scalar_headers" in msg.kwargs:
            return msg
        return Msg(
            "open_run",
            msg.obj,
            *msg.args,
            run=msg.run,
            **msg.kwargs,
            geecs_scalar_headers=headers,
        )

    return (yield from msg_mutator(plan, _mutate))


def install_connect_on_demand(
    run_engine: Any, *, mock: bool = False, timeout: float = DEFAULT_CONNECT_TIMEOUT
) -> None:
    """Install :func:`connect_on_demand` as the **outermost** RunEngine preprocessor.

    The RunEngine composes ``preprocessors`` in list order, first-appended
    innermost — so a preprocessor appended *later* (``SupplementalData``,
    the phase-2 preamble) injects messages that an earlier-appended
    ``connect_on_demand`` never sees, and a baseline read of an unconnected
    device fails.  This therefore removes any existing instance and
    re-appends itself last; call it again after installing anything else.
    """
    run_engine.preprocessors[:] = [
        p
        for p in run_engine.preprocessors
        if getattr(p, "func", None) is not connect_on_demand
    ]
    run_engine.preprocessors.append(
        partial(connect_on_demand, mock=mock, timeout=timeout)
    )
