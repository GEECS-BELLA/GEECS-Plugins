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
import math
from collections import ChainMap
from collections.abc import Generator
from functools import partial
from typing import Any

import bluesky.plan_stubs as bps
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


def _positions_match(value: Any, wanted: Any) -> bool:
    """Whether a commanded position is the declared one (floats compared loosely)."""
    try:
        return math.isclose(float(value), float(wanted), rel_tol=1e-9, abs_tol=1e-9)
    except (TypeError, ValueError):
        return value == wanted


def _check_plan_covers_save_set(msg: Msg, prepared: Any) -> None:
    """Refuse a plan that does not read every device the save set records.

    On the funnel the read set **is** the save set — one list built once and
    handed to the plan.  The stock-plan door decouples them: the preamble
    turns native saving on for the save set's devices while the plan reads
    whatever the caller passed to ``bp.count``/``bp.list_scan``.  A device
    that is saved but not read writes one file per shot with no
    ``acq_timestamp`` row to join it to — the same orphan-frame failure the
    Gate-2 save windowing exists to prevent, arriving from the other side.

    Every stock ``bluesky.plans`` verb records ``md["detectors"]``, so the
    check is available at ``open_run``, before the claim.
    """
    listed = msg.kwargs.get("detectors")
    if not isinstance(listed, (list, tuple)):
        logger.warning(
            "geecs_preamble: the plan declares no detectors in its metadata, so "
            "the save set cannot be checked against what the plan reads"
        )
        return
    read = {str(name) for name in listed}
    # The telemetry tail is not the plan's to read — the preprocessor
    # injects it into the event (see _read_telemetry_into_event).
    telemetry = {d.name for d in prepared.telemetry_readables}
    saved = {d.name for d in prepared.detectors} - telemetry
    missing = sorted(saved - read)
    if missing:
        raise GeecsConfigurationError(
            f"the save set records {missing}, which this plan does not read. "
            "Native saving would write files with no event row to join them "
            "to. Pass the save set's devices to the plan — "
            "`resolver.resolve_save_set(name)` names them — or drop them "
            "from the save set."
        )


def _check_plan_geometry(msg: Msg, prepared: Any) -> None:
    """Refuse a plan whose shape disagrees with the request it is recorded as.

    ScanInfo, the start document and the Tiled catalog take the number of
    steps and shots from the **request**; the rows actually recorded come
    from the stock plan's own arguments.  ``bp.count(num=2)`` under a
    request declaring ``shots_per_step: 5`` would write a scan record that
    its own data contradicts, and nothing downstream could tell.
    """
    declared = prepared.spec.n_shots
    actual = msg.kwargs.get("num_points")
    if actual is None or declared == actual:
        return
    raise GeecsConfigurationError(
        f"the request declares {declared} recorded shots "
        f"(steps x shots_per_step) but the plan takes {actual} points. "
        "ScanInfo and the catalog record the request's numbers, so they must "
        "agree — note a stock step scan records one shot per point, so "
        "shots_per_step > 1 needs the per-step hook (GEECS-Plugins#807 "
        "phase 3), not a stock plan."
    )


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
    plan was handed (`GeecsNamespace.select`).

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
        save_control_only_off_plan,
        save_enable_plan,
    )
    from geecs_bluesky.plan_session import get_plan_session
    from geecs_bluesky.plans.single_shot import fire_and_await_shot
    from geecs_bluesky.plans.step_scan import geecs_execution_md, normalize_motors
    from geecs_bluesky.scan_log import scan_log

    sess = session if session is not None else get_plan_session()
    created: list = []
    state: dict[str, Any] = {}
    # The forwarded messages themselves, not their ids: a Msg is dropped
    # once the RunEngine has processed it and CPython reuses the address,
    # so an id set can make a later open_run in a multi-run plan look
    # already-forwarded and silently skip its preamble.
    forwarded: list[Msg] = []
    armed_groups: dict[Any, list] = {}

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
        # The namespace devices are long-lived nouns: what this run
        # configured on them (saving mode, save path, asset definitions,
        # shot-ID origin) must not survive into the next plan, GEECS or not.
        # Only if a preamble actually ran — a plan without the key is
        # untouched on the way out as well as on the way in.
        if namespace is not None and state:
            namespace.reset_run_configuration()
        log_ctx = state.pop("scan_log", None)
        if log_ctx is not None:
            log_ctx.__exit__(None, None, None)

    def _fire_before_wait(msg: Msg) -> tuple[Any, Any]:
        """Insert the shot between arming the waiters and waiting on them.

        ``trigger_and_read`` issues ``trigger`` for every detector under one
        group and then waits on that group.  Strict shot control needs the
        fire to land in exactly that gap — the detectors have baselined their
        ``acq_timestamp`` and are waiting, so a shot fired now cannot be
        missed — which is why ``bps.trigger_and_read`` alone cannot express
        it.  Doing it here means a **stock** plan needs no ``per_shot`` hook
        at all.

        The fire, the bounded refire on a dropped frame and the device-down
        gating all come from
        :func:`~geecs_bluesky.plans.single_shot.fire_and_await_shot`, the
        same implementation ``geecs_single_shot`` uses: the seam is
        inserted here, but it is not re-derived here.
        """
        fire = state.get("fire")
        if fire is None:
            return None, None
        if msg.command == "trigger":
            group = msg.kwargs.get("group")
            if group is not None:
                armed_groups.setdefault(group, []).append(msg.obj)
            return None, None
        if msg.command != "wait":
            return None, None
        group = msg.kwargs.get("group")
        triggered = armed_groups.pop(group, None)
        if triggered is None:
            return None, None
        if not any(id(obj) in state["detectors"] for obj in triggered):
            # Some other group of the plan's own — not the shot.
            return None, None
        return (
            fire_and_await_shot(triggered, fire, pending_wait=msg),
            None,
        )

    def _check_commanded_position(msg: Msg) -> None:
        """Refuse a move to a position the run never declared.

        ScanInfo, the start document and the Tiled catalog all record the
        **request's** positions, while a stock plan moves to whatever its
        own arguments say.  Equal point counts hide a value mismatch, so
        every commanded position is checked against the declared list: a
        scan whose data does not match its own record is worse than a scan
        that stops.
        """
        declared = state.get("declared_positions")
        if not declared:
            return
        axis = state["movable_index"].get(id(msg.obj))
        if axis is None:
            return
        wanted = [row[axis] for row in declared]
        value = msg.args[0] if msg.args else None
        if any(_positions_match(value, w) for w in wanted):
            return
        raise GeecsConfigurationError(
            f"{getattr(msg.obj, 'name', msg.obj)}: the plan moves to {value!r}, "
            f"which is not among the positions this run declared ({wanted!r}). "
            "ScanInfo and the catalog record the request's positions, so the "
            "plan's points and the request's axes must be the same list."
        )

    def _read_telemetry_into_event(msg: Msg) -> tuple[Any, Any]:
        """Add the telemetry columns to the event the plan is about to save.

        Background telemetry is a **soft tier of extra columns in the
        ``primary`` stream, one value per row** — not a separate baseline
        stream (``Planning/native_bluesky/02_preamble_preprocessor.md``).
        The funnel gets that by appending the group to the plan's read set;
        a stock plan reads only the devices it was handed, so the reads are
        inserted here, between the plan's last ``read`` and its ``save``.
        The event is still one row, and the schema is unchanged.
        """
        telemetry = state.get("telemetry")
        if not telemetry or msg.command != "save" or state.get("in_telemetry"):
            return None, None

        def _read_then_save() -> Generator[Msg, Any, Any]:
            # plan_mutator re-processes the message this generator yields,
            # so the forwarded save must not be mutated again.
            state["in_telemetry"] = True
            try:
                for device in telemetry:
                    yield from bps.read(device)
                return (yield msg)
            finally:
                state["in_telemetry"] = False

        return _read_then_save(), None

    def _prepare(msg: Msg) -> tuple[Any, Any]:
        fired = _fire_before_wait(msg)
        if fired != (None, None):
            return fired
        telemetry_read = _read_telemetry_into_event(msg)
        if telemetry_read != (None, None):
            return telemetry_read
        if msg.command == "set":
            _check_commanded_position(msg)
            return None, None
        if msg.command != "open_run" or any(msg is f for f in forwarded):
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

        if namespace is None:
            raise GeecsConfigurationError(
                "geecs_preamble needs a device namespace: the objects it "
                "configures must be the ones the plan was handed. Install it "
                "with namespace=..., or set QS_DEVICE_NAMESPACE back on."
            )

        def _run_preamble() -> Generator[Msg, Any, Any]:
            resolved = resolve_request(sess, res, request)
            if not resolved.strict:
                raise GeecsConfigurationError(
                    "stock plans run strict shot control only: free-run is "
                    "retired (GEECS-Plugins#807). Set acquisition='strict'."
                )
            # The controller-present and single-shot-capable checks are
            # prepare_step_scan's (plans/preamble.py); not repeated here.
            prepared = yield from prepare_step_scan(
                sess, res, resolved, created, namespace=namespace
            )
            state["controller"] = prepared.controller
            state["closeout"] = prepared.closeout
            _check_plan_covers_save_set(msg, prepared)
            _check_plan_geometry(msg, prepared)

            # The claim: every failure above it burns no scan number.
            scan_number, scan_folder = claim_scan_number(sess.experiment)
            # The per-scan log handler, and the flush of everything buffered
            # before the claim — the funnel does this around its run
            # (scan_request_plan.py), and /triage reads the result.
            log_ctx = scan_log(scan_number, scan_folder)
            log_ctx.__enter__()
            state["scan_log"] = log_ctx
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

            # The scan motors carry `_column_headers` too, and the
            # Tiled→s-file exporter needs them to put the legacy
            # "Device Variable" header back on the axis column.
            movables = normalize_motors(prepared.motor_arg)
            scalar_devices = list(prepared.detectors) + movables
            md = claimed_scan_metadata(
                experiment=sess.experiment,
                scan_number=scan_number,
                scan_folder=scan_folder,
                saving_detectors=saving,
                devices=scalar_devices,
                extra_md={
                    "description": prepared.request.description,
                    **geecs_execution_md(
                        motors=movables,
                        positions=prepared.spec.positions,
                        shots_per_step=prepared.request.capture.shots_per_step,
                        fires_own_shots=True,
                    ),
                    **prepared.spec.md,
                },
            )
            if prepared.setup is not None:
                yield from prepared.setup()
            # Capture-owned cameras: a save flag left on out-of-band must
            # not keep writing to a stale path. Eager — turning saving OFF
            # needs no trigger windowing, unlike turning it on.
            yield from save_control_only_off_plan(scalar_devices)

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
            state["telemetry"] = list(prepared.telemetry_readables)
            state["fire"] = prepared.controller.fire_shot
            state["detectors"] = {id(d) for d in prepared.detectors}
            state["movable_index"] = {id(m): i for i, m in enumerate(movables)}
            state["declared_positions"] = [
                row if isinstance(row, (list, tuple)) else (row,)
                for row in prepared.spec.positions
                if row is not None
            ]

            # Injected md wins over the plan's own, as inject_md_wrapper
            # does.  The raw request itself is dropped: the resolved picture
            # is already in the metadata, and an unresolved copy of it in
            # every start document is a key no consumer declares and the
            # funnel's door never emits.
            passthrough = {k: v for k, v in msg.kwargs.items() if k != md_key}
            new = msg._replace(kwargs=ChainMap(md, passthrough))
            forwarded.append(new)
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
    existing = next(
        (
            p
            for p in run_engine.preprocessors
            if getattr(p, "func", None) is connect_on_demand
        ),
        None,
    )
    # Re-appending must preserve how connect_on_demand was configured — a
    # mock worker installs it with mock=True, and re-deriving the kwargs
    # here would quietly send a mock worker at real Channel Access.
    connect_kwargs = dict(getattr(existing, "keywords", None) or {})
    connect_kwargs.setdefault("mock", mock)
    run_engine.preprocessors[:] = [
        p
        for p in run_engine.preprocessors
        if getattr(p, "func", None) not in (geecs_preamble, connect_on_demand)
    ]
    run_engine.preprocessors.append(
        partial(geecs_preamble, session=session, resolver=resolver, namespace=namespace)
    )
    if existing is not None:
        install_connect_on_demand(run_engine, **connect_kwargs)
