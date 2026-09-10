"""RunEngine preprocessors — the native hooks GEECS occupies (issue #807).

Each function here has the ``bluesky.preprocessors`` shape (``plan → plan``)
so it can be installed once on the RunEngine (``RE.preprocessors.append``)
and applies to **every** plan, stock or not, with no per-plan code.

Phase 1: :func:`connect_on_demand`.  Phase 2: :func:`geecs_preamble`, the
GEECS scan preamble and finalize chain keyed on ``md["geecs"]``
(``Planning/native_bluesky/02_preamble_preprocessor.md``).

**Install order matters.**  ``RE.preprocessors`` compose in list order,
first-appended innermost, so :func:`connect_on_demand` must be re-installed
**last** after :func:`install_geecs_preamble` — otherwise the preamble's own
connects and reads never pass through it.  ``install_geecs_preamble`` does
that for you.
"""

from __future__ import annotations

import logging
from collections import ChainMap
from collections.abc import Generator
from functools import partial
from typing import Any

import bluesky.preprocessors as bpp
from bluesky.preprocessors import plan_mutator
from bluesky.utils import Msg
from ophyd_async.plan_stubs import ensure_connected

from geecs_bluesky.exceptions import GeecsConfigurationError

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


# ---------------------------------------------------------------------------
# Phase 2 — the GEECS scan preamble as a preprocessor
# ---------------------------------------------------------------------------

#: Run-metadata key carrying the ScanRequest a run should be prepared for.
GEECS_MD_KEY = "geecs"


def geecs_preamble(
    plan: Generator[Msg, Any, Any],
    *,
    session: Any = None,
    resolver: Any = None,
    namespace: Any = None,
    md_key: str = GEECS_MD_KEY,
) -> Generator[Msg, Any, Any]:
    """Give any plan the GEECS scan preamble, keyed on ``md["geecs"]``.

    A plan that opens its run with ``md={"geecs": <ScanRequest as a dict>}``
    gets, before the run opens: authoritative validation, name resolution,
    the shot controller, the unserved/CONNECTED preflights, action-slot
    compilation, the capture toggle, the connects the preamble itself owns,
    the **scan-number claim**, the ScanInfo write, native-save configuration
    — and the GEECS run metadata injected into its start document.  On the
    way out, in the funnel's nesting order: save-off (innermost, so saving
    stops while the trigger is still stopped), disarm, closeout actions,
    disconnect.

    All of it is :mod:`geecs_bluesky.plans.preamble`'s code, the same the
    funnel plan runs — this only supplies the seam and the ordering, so a
    stock ``bluesky.plans`` verb behaves exactly like a submitted
    ``ScanRequest``::

        RE(bp.list_grid_scan([cam], U_S1H.current, pts, md={"geecs": request}))

    The devices come from the **namespace**, not from fresh per-scan
    construction: the objects the preamble configures must be the ones the
    plan was handed (`preamble.namespace_detectors`).

    Note
    ----
    The request must ride in the **plan's** ``md=``.  RunEngine per-call
    metadata (``RE(plan, geecs=...)``) is merged at ``_open_run`` and never
    enters a message, so a preprocessor cannot see it: the start document
    would carry a ``geecs`` block while the preamble never ran.
    """
    from geecs_bluesky.plans.preamble import (
        _disconnect_plan,
        prepare_step_scan,
        resolve_request,
    )
    from geecs_bluesky.plans.run_wrapper import (
        _save_cleanup_plan,
        claim_scan_number,
        claimed_scan_metadata,
        save_enable_plan,
    )
    from geecs_bluesky.plans.scan_request_plan import _worker_session

    sess = session if session is not None else _worker_session
    created: list = []
    state: dict[str, Any] = {}
    forwarded: set[int] = set()
    armed_groups: set[Any] = set()

    def _cleanup() -> Generator[Msg, Any, Any]:
        """The funnel's finalize chain, innermost first, for whatever ran."""
        if state.get("saving"):
            yield from _save_cleanup_plan(state["saving"])
        controller = state.get("controller")
        if controller is not None:
            yield from controller.disarm()
        closeout = state.get("closeout")
        if closeout is not None:
            yield from closeout()
        if created:
            yield from _disconnect_plan(created)

    def _fire_before_wait(msg: Msg) -> tuple[Any, Any]:
        """Insert the shot between arming the waiters and waiting on them.

        ``trigger_and_read`` issues ``trigger`` for every detector under one
        group and then waits on that group.  Strict shot control needs the
        fire to land in exactly that gap — the detectors have baselined their
        ``acq_timestamp`` and are waiting, so a shot fired now cannot be
        missed — which is why ``bps.trigger_and_read`` alone cannot express
        it (see ``plans/single_shot.py``).  Doing it here means a **stock**
        plan needs no ``per_shot`` hook at all.
        """
        fire = state.get("fire")
        if fire is None:
            return None, None
        if msg.command == "trigger" and id(msg.obj) in state["detectors"]:
            group = msg.kwargs.get("group")
            if group is not None:
                armed_groups.add(group)
            return None, None
        if msg.command != "wait":
            return None, None
        group = msg.kwargs.get("group")
        if group not in armed_groups:
            return None, None
        armed_groups.discard(group)

        def _fire_then_wait() -> Generator[Msg, Any, Any]:
            yield from fire()
            return (yield msg)

        return _fire_then_wait(), None

    def _prepare(msg: Msg) -> tuple[Any, Any]:
        fired = _fire_before_wait(msg)
        if fired != (None, None):
            return fired
        if msg.command != "open_run" or id(msg) in forwarded:
            return None, None
        request = msg.kwargs.get(md_key)
        if request is None:
            return None, None
        if sess is None:
            raise GeecsConfigurationError(
                "geecs_preamble has no session: install one with "
                "set_plan_session(...) at worker startup, or pass session=..."
            )
        res = resolver
        if res is None:
            from geecs_bluesky.config_resolver import ConfigsRepoResolver

            res = ConfigsRepoResolver(sess.experiment)

        def _run_preamble() -> Generator[Msg, Any, Any]:
            resolved = resolve_request(sess, res, request)
            if not resolved.strict:
                raise GeecsConfigurationError(
                    "stock plans run strict shot control only: free-run is "
                    "retired (GEECS-Plugins#807). Set acquisition='strict'."
                )
            if resolved.controller is None:
                raise GeecsConfigurationError(
                    "strict shot control requires a trigger_profile — the plan "
                    "fires every shot, so there must be a shot-control device"
                )
            resolved.controller.require_strict_single_shot()
            prepared = yield from prepare_step_scan(
                sess, res, resolved, created, namespace=namespace
            )
            state["controller"] = prepared.controller
            state["closeout"] = prepared.closeout

            # The claim: every failure above it burns no scan number.
            scan_number, scan_folder = claim_scan_number(sess.experiment)
            saving = sess.configure_claimed_scan(
                scan_number=scan_number,
                scan_folder=scan_folder,
                detectors=prepared.detectors,
                motor=prepared.motor_arg,
                positions=prepared.spec.positions,
                shots_per_step=prepared.request.capture.shots_per_step,
                description=prepared.request.description,
                scan_info_overrides=prepared.spec.scan_info,
            )
            state["saving"] = saving

            md = claimed_scan_metadata(
                experiment=sess.experiment,
                scan_number=scan_number,
                scan_folder=scan_folder,
                saving_detectors=saving,
                devices=prepared.detectors,
                extra_md={
                    "description": prepared.request.description,
                    **prepared.spec.md,
                },
            )
            if prepared.setup is not None:
                yield from prepared.setup()

            # Order is load-bearing (Gate-2 save windowing).  arm_single_shot
            # drives the box to ARMED — single-shot source, so the free run
            # is HALTED — and waits for quiescence.  Only then is it safe to
            # turn saving on: a camera writes one file per shot it sees, so
            # enabling saving while edges are still passing writes orphan
            # frames with no event row.  Save-off is the innermost finalize
            # for the mirror reason: it runs before disarm lets edges back.
            yield from prepared.controller.arm_single_shot(prepared.detectors)
            if saving:
                yield from save_enable_plan(saving)
            state["fire"] = prepared.controller.fire_shot
            state["detectors"] = {id(d) for d in prepared.detectors}

            # Injected md wins over the plan's own, as inject_md_wrapper does.
            new = msg._replace(kwargs=ChainMap(md, msg.kwargs))
            forwarded.add(id(new))
            logger.info(
                "geecs_preamble: scan %s prepared (%d detectors) for %s",
                scan_number,
                len(prepared.detectors),
                msg.kwargs.get("plan_name", "?"),
            )
            return (yield new)

        return _run_preamble(), None

    return (yield from bpp.finalize_wrapper(plan_mutator(plan, _prepare), _cleanup()))


def install_geecs_preamble(
    run_engine: Any,
    *,
    session: Any = None,
    resolver: Any = None,
    namespace: Any = None,
    mock: bool = False,
) -> None:
    """Install :func:`geecs_preamble`, keeping :func:`connect_on_demand` outermost.

    Idempotent, and it re-appends ``connect_on_demand`` afterwards so the
    preamble's own connects and reads still pass through it (the RunEngine
    composes preprocessors first-appended-innermost).
    """
    had_connect = any(
        getattr(p, "func", None) is connect_on_demand for p in run_engine.preprocessors
    )
    run_engine.preprocessors[:] = [
        p
        for p in run_engine.preprocessors
        if getattr(p, "func", None) not in (geecs_preamble, connect_on_demand)
    ]
    run_engine.preprocessors.append(
        partial(geecs_preamble, session=session, resolver=resolver, namespace=namespace)
    )
    if had_connect:
        install_connect_on_demand(run_engine, mock=mock)
