"""Register Count, Sweep, Optimize and utility plans over the worker namespace.

The acquisition binder supplies the strict and fly hooks, the liveness
gate, the trigger bracket, the device streams and shot throttling. Count uses
stock Bluesky count; Sweep delegates traversal to scan_nd. Moving stock
verbs remain implementation details, never public queue entries.
"""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from inspect import Parameter
from types import SimpleNamespace
from typing import Any

import bluesky.plan_stubs as bps
import bluesky.plans as bp
import bluesky.preprocessors as bpp
from bluesky.protocols import Readable, Triggerable
from bluesky.utils import Msg
from geecs_schemas.trigger_profile import TriggerState

from geecs_bluesky.devices.background import BackgroundSnapshot
from geecs_bluesky.devices.ca._view import owner_of
from geecs_bluesky.devices.ca.liveness import stale_text
from geecs_bluesky.devices.detector import GeecsDetector
from geecs_bluesky.devices.native import (
    refuse_native_cameras,
)
from geecs_bluesky.devices.native_setpoint import (
    native_set_tolerances,
)
from geecs_bluesky.devices.shot_control import ShotControl
from geecs_bluesky.shot_gate import ShotGate
from geecs_bluesky.devices.witness import Witness, witness_is_down
from geecs_bluesky.exceptions import GeecsConfigurationError, GeecsDeviceDownError
from geecs_bluesky.native_scope import NOT_SCAN_MODE
from geecs_bluesky.plan_names import (
    ACQUISITION_MODES,
    GEECS_PLAN_NAMES,
    NON_SCAN_PLAN_NAMES,
    acquisition_refusal,
)
from geecs_bluesky.plans.apply_setup import SettableFactory, apply_setup_plan
from geecs_bluesky.plans.calibration import (
    check_shot_sync_plan,
    measure_shot_offsets_plan,
)
from geecs_bluesky.plans.scan_mode import (
    EnterScanMode,
    paused_device_names,
    entered_device_names,
    in_scan_mode,
    release_scan_mode_plan,
    reset_shot_stamps,
    run_device_names,
    scan_mode_bracket,
    scan_mode_preflight,
    scan_parameter_for,
)
from geecs_bluesky.plans.fly import BOX_NOT_DRIVEN, DEFAULT_TICK_PERIOD_S, bind_fly
from geecs_bluesky.plans.on_demand import (
    OnDemand,
    on_demand_per_shot,
    on_demand_per_step,
    start_keys,
)
from geecs_bluesky.plans.shot_gate import open_gate
from geecs_bluesky.plans.streams import DeviceStreams, is_per_shot
from geecs_bluesky.plans.strict import (
    down_devices,
    geecs_name,
    mover_roots,
    needed_device_names,
    name_failed_status,
    run_bracket,
    stale_devices,
)
from geecs_bluesky.plans.sweep import sweep_movers, sweep_plan
from geecs_bluesky.utils import safe_name

logger = logging.getLogger(__name__)

_HOOKS = ("per_step", "per_shot")


class TriggerProfiles:
    """The experiment's trigger profiles, each as a :class:`ShotControl`.

    Built once at worker start from the configs repo; a bound plan resolves
    its ``trigger_profile`` argument here.  The devices carry the namespace
    marker so ``connect_on_demand`` connects them on first use.

    Parameters
    ----------
    shot_controls :
        Profile name → device.
    default :
        The profile used when a plan names none (the experiment defaults'
        ``trigger_profile``); ``None`` makes the argument mandatory.
    """

    def __init__(
        self, shot_controls: Mapping[str, ShotControl], *, default: str | None = None
    ) -> None:
        self._by_name = dict(shot_controls)
        for device in self._by_name.values():
            device._geecs_namespace_member = True
        if default is not None and default not in self._by_name:
            raise GeecsConfigurationError(
                f"default trigger profile {default!r} is not one of the loaded "
                f"profiles ({', '.join(sorted(self._by_name)) or 'none'})"
            )
        self.default = default

    @classmethod
    def from_resolver(cls, resolver: Any, *, experiment: str) -> TriggerProfiles:
        """Load every trigger profile the resolver lists; skip the unusable ones loudly."""
        shot_controls: dict[str, ShotControl] = {}
        for name in resolver.list_trigger_profiles():
            try:
                profile = resolver.resolve_trigger_profile(name)
                shot_controls[name] = ShotControl.from_profile(
                    profile, experiment=experiment, name=safe_name(name)
                )
            except Exception as exc:  # a profile that names no device, bad YAML …  # noqa: BLE001 — bad config skipped, logged
                logger.warning(
                    "trigger profile %r not loaded: %s: %s",
                    name,
                    type(exc).__name__,
                    exc,
                )
        try:
            defaults = resolver.resolve_experiment_defaults()
        except Exception as exc:  # no configs root, unreadable defaults file  # noqa: BLE001 — bad config skipped, logged
            logger.warning("experiment defaults not loaded: %s", exc)
            defaults = None
        default = getattr(defaults, "trigger_profile", None)
        if default is not None and default not in shot_controls:
            logger.warning(
                "experiment default trigger profile %r is not loadable; "
                "plans must name one explicitly",
                default,
            )
            default = None
        return cls(shot_controls, default=default)

    @property
    def names(self) -> list[str]:
        """The loaded profile names, sorted."""
        return sorted(self._by_name)

    def __iter__(self) -> Iterator[ShotControl]:
        """Iterate the devices."""
        return iter(self._by_name.values())

    def resolve(self, name: str | None) -> ShotControl:
        """The device for *name*, or the default; loud when neither exists."""
        if name is None:
            if self.default is None:
                raise GeecsConfigurationError(
                    "no trigger_profile given and the experiment has no default "
                    f"(experiment_defaults.yaml); choose one of: {', '.join(self.names)}"
                )
            name = self.default
        try:
            return self._by_name[name]
        except KeyError:
            raise GeecsConfigurationError(
                f"unknown trigger profile {name!r}; choose one of: "
                f"{', '.join(self.names) or 'none loaded'}"
            ) from None


def liveness_gate(
    shot_control: Any,
    devices: Sequence[Any],
    *,
    optional: Sequence[Any] = (),
):
    """Plan: refuse the run if a device it needs is reported down; return the optional ones that are.

    The verdict is :func:`~geecs_bluesky.plans.strict.down_devices` over
    *devices* and the trigger box's device(s)
    (:attr:`~geecs_bluesky.devices.shot_control.ShotControl.liveness_signals`):
    the gateway's ``CONNECTED`` (fail-open) for a GEECS device, its IOC for
    a native camera, its readbacks' severity for a native setpoint device.
    Runs before the bracket's first move and before ``open_run``: nothing
    is driven and nothing is claimed for a run this refuses.

    Parameters
    ----------
    shot_control :
        The resolved trigger profile; its device(s) down always refuse.
    devices :
        Every device the run lists.
    optional :
        The listed devices the run can go without — in a bound scan verb,
        every one not ``required``.  One reported down is no refusal: its
        GEECS name is returned and the caller leaves it out of the run.

    Returns
    -------
    list of str
        The GEECS names of the *optional* devices reported down.

    Raises
    ------
    GeecsDeviceDownError
        Naming every device reported down that is not *optional*.
    """
    signals = getattr(shot_control, "liveness_signals", {})
    down = yield from down_devices(devices, signals)
    spare = {geecs_name(d) for d in optional} - set(signals)
    needed = [name for name in down if name not in spare]
    if needed:
        names = ", ".join(needed)
        raise GeecsDeviceDownError(
            f"{names} DISCONNECTED (the gateway's CONNECTED, a native "
            "camera's IOC not answering, or a native setpoint device's readback "
            "INVALID) — needed by the run (required, moved, or the trigger "
            "profile's box), so it was refused before the trigger box was "
            "driven (nothing claimed, no scan folder). Check the device(s), "
            "then resubmit.",
            device_name=needed[0],
        )
    return [name for name in down if name in spare]


def drop_down_devices(
    shot_control: Any,
    detectors: Sequence[Any],
    needed: Collection[str],
    movers: Sequence[Any] = (),
):
    """Plan: the run's detectors minus the ones reported down that it can go without.

    The one rule of a bound scan verb's start: :func:`liveness_gate` over
    the detectors and the devices the run moves (*movers*), every detector
    whose GEECS name is not in *needed*
    (:func:`~geecs_bluesky.plans.strict.needed_device_names`) optional.  A
    needed device down — required, moved, or the box — refuses the run; any
    other listed device down is left out of it — no stream, no columns, not
    waited on for scan mode, and not read as background telemetry either
    (the caller keeps the full list as the background's ``own``) — WARNED
    by name and recorded in the start document as ``devices_dropped``.  It
    is back in the next run that finds it up: the preset is never changed.

    Returns
    -------
    tuple
        ``(kept detectors, dropped GEECS names)``.
    """
    judged = list(detectors)
    for root in mover_roots(movers):
        if all(root is not d for d in judged):
            judged.append(root)
    optional = [d for d in detectors if geecs_name(d) not in needed]
    dropped = yield from liveness_gate(shot_control, judged, optional=optional)
    if not dropped:
        return list(detectors), []
    kept = [d for d in detectors if geecs_name(d) not in dropped]
    if not kept:
        raise GeecsDeviceDownError(
            f"{', '.join(dropped)} DISCONNECTED — every device the run lists "
            "is down, so it was refused before the trigger box was driven "
            "(nothing claimed, no scan folder). Check the device(s), then "
            "resubmit.",
            device_name=dropped[0],
        )
    logger.warning("%s", dropped_text(dropped))
    return kept, dropped


def dropped_text(dropped: Sequence[str]) -> str:
    """The warning for the devices a run left out because they were down: at bind and in ``scan.log``."""
    return (
        f"{', '.join(dropped)} DISCONNECTED — not required, so left out of this "
        "run (no stream, no columns); listed again the next run that finds it up"
    )


def set_tolerance_gate(movers: Sequence[Any]):
    """Plan: refuse the run when a native settable it moves has no set tolerance; else the start keys recording them.

    Runs where the liveness gate does — before the bracket's first move and
    before ``open_run`` — through
    :func:`~geecs_bluesky.devices.native_setpoint.native_set_tolerances`
    (the IOC's set-tolerance record of each, read now): a refusal claims
    nothing.  Returns ``{"set_tolerances": …}`` for the start document
    (``EVENT_SCHEMA.md``), or ``{}`` when the run moves no native settable.
    Each set reads its record again.

    Raises
    ------
    GeecsNoSetToleranceError
        A settable whose record gives no tolerance (never connected, no
        answer, INVALID, zero, negative, not finite).
    """
    record = yield from native_set_tolerances(movers)
    return {"set_tolerances": record} if record else {}


#: The witness record's note when no reader waits for the shot (the row is read
#: right after the fire's put, before the next edge reaches the counter).
UNWAITED_NOTE = "no triggered device: the row may record the previous shot's count"


def run_witness(
    settables: Any,
    *,
    acquisition: str,
    shot_control: Any,
    readers: Sequence[Any] = (),
    waited: bool | None = None,
):
    """Plan: the witness this run records, and its start-document record (``witness``).

    The namespace's ``main``-domain witness, read with every strict row and
    never waited on.  ``(None, None)`` when the namespace has no witness;
    ``None`` with a ``reason`` when it has one but this run does not read
    it: none for ``main``, a fly run (the soft tick clocks it), a box-less profile
    (nothing fired for it to count) or a witness that does not answer
    within 5 s (``down``: WARNING-logged, the run proceeds).  With no
    triggered device among *readers*, nothing waits for the shot, so a row
    can read the count before the edge: WARNING-logged and noted
    (:data:`UNWAITED_NOTE`), still read.  *waited* says it outright (a
    shot-table run always waits: on its required devices, else on the
    witness itself).
    """
    finder = getattr(settables, "witnesses", None)
    witnesses = finder() if callable(finder) else {}
    if not witnesses:
        return None, None
    witness = witnesses.get("main")
    if witness is None:
        reason = "no witness for domain main"
        return None, {"device": None, "domain": "main", "read": False, "reason": reason}
    record = {"device": witness._geecs_device_name, "domain": "main", "read": False}
    reason = None
    if acquisition == "fly":
        reason = "a fly run is clocked by the soft tick"
    elif getattr(shot_control, "boxless", False):
        reason = "a box-less profile fires nothing for the witness to count"
    elif (yield from witness_is_down(witness)):
        reason = "down"
    if reason is not None:
        record = {**record, "reason": reason}
        _warn_at_start(record)
        return None, record
    record["read"] = True
    if waited is None:
        waited = any(isinstance(d, Triggerable) for d in readers)
    if not waited:
        record["note"] = UNWAITED_NOTE
    _warn_at_start(record)
    return witness, record


def _warn_at_start(record: dict[str, Any] | None) -> None:
    """Log the WARNINGs a witness record carries: a down witness, the record's note."""
    if not record:
        return
    if record.get("reason") == "down":
        logger.warning(
            "%s: the witness is down — the run proceeds without it", record["device"]
        )
    if record.get("note"):
        logger.warning("%s: %s", record["device"], record["note"])


def refuse_listed_witness(devices: Sequence[Any]) -> None:
    """Refuse a witness named among a run's devices: every strict row records it already."""
    listed = [d for d in devices if isinstance(owner_of(d), Witness)]
    if listed:
        raise GeecsConfigurationError(
            f"{', '.join(owner_of(d)._geecs_device_name for d in listed)} is the "
            "run's witness, recorded in every strict row automatically — drop it "
            "from the run's devices"
        )


def warn_after_open(plan: Any, warnings: Sequence[str]) -> Any:
    """Log *warnings* once more right after ``open_run``: the start document opens ``scan.log``.

    A WARNING logged while the plan binds lands in the worker's journal
    alone; this repeats it where the operator reads the run.  *plan*
    unchanged when there are none.
    """
    if not warnings:
        return (yield from plan)
    seen: list[Msg] = []

    def _proc(msg: Msg) -> tuple[Any, Any]:
        if msg.command != "open_run" or any(m is msg for m in seen):
            return None, None
        seen.append(msg)

        def _open_then_warn():
            uid = yield msg
            for text in warnings:
                logger.warning("%s", text)
            return uid

        return _open_then_warn(), None

    return (yield from bpp.plan_mutator(plan, _proc))


def witness_wrapper(
    plan: Any, witness: Witness | None, record: dict[str, Any] | None = None
) -> Any:
    """Bracket the run's ``scan.log`` with the witness's lines.

    Right after ``open_run`` (the start document opens ``scan.log``), the
    run-start WARNINGs of *record* (:func:`run_witness`'s, logged at bind,
    before the file exists) once more; right before ``close_run`` the
    witness's run summary.  *plan* unchanged when there is neither.
    """
    if witness is None and not record:
        return (yield from plan)
    seen: list[Msg] = []

    def _proc(msg: Msg) -> tuple[Any, Any]:
        if any(m is msg for m in seen):
            return None, None
        if msg.command == "open_run":
            seen.append(msg)

            def _open_then_warn():
                uid = yield msg
                _warn_at_start(record)
                return uid

            return _open_then_warn(), None
        if msg.command != "close_run" or witness is None:
            return None, None
        seen.append(msg)

        def _summary_then_close():
            witness.summarize()
            return (yield msg)

        return _summary_then_close(), None

    return (yield from bpp.plan_mutator(plan, _proc))


def _experiment_default(
    requested: bool | None,
    resolver: Any | None,
    field: str,
    *,
    on_failure: str,
    default: bool = True,
) -> bool:
    """*requested*, else ``ExperimentDefaults.<field>``, else *default*.

    Read at every run, not at bind time, so an edit to
    ``experiment_defaults.yaml`` reaches the next scan without the worker
    reopening its environment.  A resolver that cannot read the defaults
    (no configs root, bad YAML, an older resolver without the method), or
    no defaults file, gives *default*: on for a switch every scan had
    before it existed, off for an opt-in one — *on_failure* says which,
    in the journal, once per run.
    """
    if requested is not None:
        return bool(requested)
    if resolver is None:
        return default
    try:
        defaults = resolver.resolve_experiment_defaults()
    except Exception as exc:  # no configs root, unreadable defaults file  # noqa: BLE001 — bad config skipped, logged
        logger.warning(
            "experiment defaults not read (%s: %s); %s",
            type(exc).__name__,
            exc,
            on_failure,
        )
        return default
    return bool(getattr(defaults, field, default))


def resolve_native_image_save(requested: bool | None, resolver: Any | None) -> bool:
    """The run's LabVIEW-files switch: request, else experiment default, else on."""
    return _experiment_default(
        requested, resolver, "native_image_save", on_failure="native saving stays on"
    )


def resolve_background_telemetry(requested: bool | None, resolver: Any | None) -> bool:
    """The run's background-telemetry switch: the request, else the experiment default, else on.

    On, every logged scalar of the experiment outside the run's own devices
    is read into every row, softly
    (:class:`~geecs_bluesky.devices.background.BackgroundSnapshot`); off,
    the run records its own devices alone.
    """
    return _experiment_default(
        requested,
        resolver,
        "background_telemetry",
        on_failure="background telemetry stays on",
    )


def background_snapshot(
    settables: Any, *, mock: bool = False
) -> BackgroundSnapshot | None:
    """The run's background telemetry over *settables* (the namespace), or ``None``.

    ``None`` when *settables* carries no telemetry set (a hermetic test's
    mapping, no namespace) or the set is empty: the run then records its
    own devices alone and its start document says
    ``background_telemetry: False``.
    """
    telemetry = getattr(settables, "telemetry", None)
    candidates = list(telemetry()) if callable(telemetry) else []
    return BackgroundSnapshot(candidates, mock=mock) if candidates else None


def background_wrapper(
    plan: Any,
    snapshot: BackgroundSnapshot,
    *,
    own: Sequence[Any] = (),
    movers: Sequence[Any] = (),
) -> Any:
    """Probe the background set right before ``open_run``; record the result in the start document.

    *own* is the run's own reader list (its detectors) and *movers* the motors it moves — the bound plan knows both
    before the run (the sweep resolves its axes when it is built, optimize
    holds its movables), so
    :meth:`~geecs_bluesky.devices.background.BackgroundSnapshot.probe`
    decides the membership in one place.  As a message the RunEngine
    awaits, inside the plan: after the plan staged its devices (a mover's
    device is read from the RunEngine's stage) and before the claim, so a
    slow probe costs the run nothing but its bounded budget.  The devices
    the probe dropped ride in the start document as ``background_dropped``
    (GEECS device names); a probe that failed outright (never a member's
    failure, which is its own drop) adds ``background_probe_error``.
    """
    opened = False

    def _proc(msg: Msg) -> tuple[Any, Any]:
        nonlocal opened
        if msg.command != "open_run" or opened:
            return None, None
        opened = True

        def _probe_then_open():
            yield from bps.wait_for(
                [functools.partial(snapshot.probe, own=list(own), movers=list(movers))]
            )
            record: dict[str, Any] = {"background_dropped": snapshot.dropped}
            if snapshot.probe_error:
                record["background_probe_error"] = snapshot.probe_error
            return (
                yield Msg(
                    "open_run", msg.obj, *msg.args, run=msg.run, **msg.kwargs, **record
                )
            )

        return _probe_then_open(), None

    return (yield from bpp.plan_mutator(plan, _proc))


def native_image_save_wrapper(
    plan: Any, detectors: Sequence[Any], enabled: bool
) -> Any:
    """Set the strict plugin-backed cameras' LabVIEW saving for this run; restore after.

    The run-level native-image-save switch.  It reaches the run's
    **strict full detectors** that are plugin-backed and nothing else:

    - a device without a file plugin — a LabVIEW-native camera, a
      proprietary-format DAQ — has no other record, so its native saving
      is never touched, whatever *enabled* says;
    - a ``.scalars`` view is skipped: the view leaves its owner's data
      logics unprepared (``GeecsDetectorScalars``), so the owner writes
      nothing either way;
    - a native saver without a plugin keeps its files as its only
      record, so this switch never reaches it.

    Restored to the construction default — on, the dual-write — by a
    ``finalize_wrapper``, success, abort or stop alike.  ``RE.halt()``
    skips finalizers by bluesky contract and leaves the switch where the
    run put it; every scan verb sets it for itself, so a halt costs
    nothing but the next run's journal line.

    Parameters
    ----------
    plan :
        The bound plan (staging inside it).
    detectors :
        The run's detectors (views are ignored).
    enabled :
        ``False`` = the plugin's stack is those cameras' only record this run.
    """
    cameras: list[GeecsDetector] = []
    kept: list[GeecsDetector] = []
    seen: set[int] = set()
    for d in detectors:
        if id(d) in seen or not isinstance(d, GeecsDetector) or not d.native_save:
            continue  # a view, a scalar-only device, a motor
        seen.add(id(d))
        if d.plugin_backed:
            cameras.append(d)
        else:
            kept.append(d)
    kept_names = ", ".join(c._geecs_device_name for c in kept)
    if not cameras:
        if not enabled and kept:
            logger.info(
                "native_image_save=False reaches no camera here: %s save through "
                "LabVIEW only (no file plugin)",
                kept_names,
            )
        return (yield from plan)
    for cam in cameras:
        cam.native_image_save = enabled
    logger.info(
        "native saving %s this run for %s%s",
        "on" if enabled else "off (file-plugin stacks only)",
        ", ".join(c._geecs_device_name for c in cameras),
        f"; kept on, no file plugin: {kept_names}" if kept and not enabled else "",
    )

    def restore():
        for cam in cameras:
            cam.native_image_save = True  # the construction default
        yield from bps.null()

    return (yield from bpp.finalize_wrapper(plan, restore()))


def refuse_native_run(detectors: Sequence[Any], *, shot_control: Any) -> None:
    """Refuse a scan verb's run shape a native camera cannot take part in yet.

    Called before anything moves and before the claim: a native camera
    takes part in a strict run under a box-fired profile, required or not,
    beside gateway plugin cameras or not (its own stream).  Refused: a
    scan-mode profile (LabVIEW-only).  The
    cabling check and the run lock hold at the camera's prepare.

    Raises
    ------
    GeecsConfigurationError
        Naming the native camera(s) and the reason.
    """
    if shot_control.use_scan_mode:
        refuse_native_cameras(detectors, NOT_SCAN_MODE)


def resolve_required(per_shot: Sequence[Any], required: Any) -> list[Any]:
    """The listed per-shot devices *required* names (``None``: all of them; ``[]``: none).

    Matched by owner, so ``X`` and ``X.scalars`` name one device.

    Raises
    ------
    GeecsConfigurationError
        A name that is a condition (no shot stamp) or not among the run's
        devices.
    """
    if required is None:
        return list(per_shot)
    out: list[Any] = []
    for entry in required:
        owner = owner_of(entry)
        name = getattr(owner, "name", str(entry))
        listed = next((d for d in per_shot if owner_of(d) is owner), None)
        if listed is None and not is_per_shot(entry):
            raise GeecsConfigurationError(
                f"{name}: required, but it has no shot stamp — a condition, read "
                "in every row; only a per-shot device (a detector or a native "
                "camera) can be required"
            )
        if listed is None:
            raise GeecsConfigurationError(
                f"{name}: required, but not among the run's devices — list it"
            )
        if all(listed is not d for d in out):
            out.append(listed)
    return out


def _refuse_twice_listed(per_shot: Sequence[Any]) -> None:
    owners: set[int] = set()
    for d in per_shot:
        if id(owner_of(d)) in owners:
            raise GeecsConfigurationError(
                f"{owner_of(d).name}: listed twice (a device and its scalars "
                "view are one device, one stream)"
            )
        owners.add(id(owner_of(d)))


#: A fly run's stand-in for the trigger box: no box, no scan mode, nothing to judge.
_NO_BOX = SimpleNamespace(use_scan_mode=False, boxless=False, liveness_signals={})


def gate_for_run(shot_control: Any, *, fly: bool) -> ShotGate | None:
    """The gate a run holds for: its profile's, none in a fly run.

    A fly run drives no box, and a box-less profile has no gate
    (:attr:`ShotControl.shot_gate`): neither issues the SINGLESHOT put the
    gate governs.
    """
    if fly:
        return None
    return getattr(shot_control, "shot_gate", None)


def strict_plan(  # noqa: C901, PLR0915
    stock: Callable[..., Any],
    profiles: TriggerProfiles,
    *,
    resolver: Any | None = None,
    settables: Any | None = None,
    mock: bool = False,
    scan_mode: Any | None = None,
) -> Callable[..., Any]:
    """Bind the strict hook into one stock plan; keep its name and signature.

    Parameters
    ----------
    stock :
        A ``bluesky.plans`` verb exposing ``per_step`` or ``per_shot``.  The
        background's movers come from the ``sweep`` payload (optimize's
        binder passes its movables): a moving stock verb (``scan``,
        ``grid_scan``) bound here with a background set would read the
        motor's device whole beside the row's own motor read and fail on
        the duplicate key — the roster never registers one (``plan_names``),
        and this is one more reason why.
    profiles :
        The trigger profiles a ``trigger_profile`` argument resolves against.
    resolver :
        The configs-repo resolver whose ``resolve_experiment_defaults`` the
        ``native_image_save`` and ``background_telemetry`` defaults are read
        from at every run; ``None`` (or an unreadable file) leaves both on.
    settables :
        The worker namespace: its telemetry set is the run's background
        (:func:`background_snapshot`).  A mapping without one (hermetic
        tests) means no background.
    mock :
        Connect a background member that is not yet connected with a mock
        backend (hermetic tests).
    scan_mode :
        The worker's :class:`~geecs_bluesky.devices.scan_mode.ScanModeControl`,
        driven when the resolved profile carries ``use_scan_mode``
        (``plans/scan_mode.py``: the preflight, the bracket inside the
        trigger bracket, the announce in the hook, the row's
        ``announced_shot``); ``None`` refuses such a profile.  A resolved
        profile naming a ``shot_gate`` likewise makes every SINGLESHOT of a
        strict run wait for its permission
        (:mod:`~geecs_bluesky.plans.shot_gate`; connected before the box is
        driven and before the claim).

    Returns
    -------
    callable
        A generator function with the stock signature minus the hook plus
        ``trigger_profile`` (and ``shots_per_step`` for the scan verbs).
    """
    signature = inspect.signature(stock)
    hooks = [h for h in _HOOKS if h in signature.parameters]
    if len(hooks) != 1:
        raise ValueError(f"{stock.__name__} does not expose exactly one strict hook")
    hook = hooks[0]

    def plan(*args: Any, **kwargs: Any):  # noqa: C901, PLR0912, PLR0915
        require_plain_metadata(kwargs.get("md"))
        trigger_profile = kwargs.pop("trigger_profile", None)
        shots_per_step = int(kwargs.pop("shots_per_step", 1))
        acquisition = str(kwargs.pop("acquisition", "strict") or "strict")
        required_arg = kwargs.pop("required", None)
        shot_period = kwargs.pop("shot_period", None)
        native_files = resolve_native_image_save(
            kwargs.pop("native_image_save", None), resolver
        )
        background = resolve_background_telemetry(
            kwargs.pop("background_telemetry", None), resolver
        )
        refused = acquisition_refusal(acquisition)
        if refused is not None:
            raise GeecsConfigurationError(refused)
        if shot_period is not None:
            shot_period = float(shot_period)
            if shot_period <= 0:
                raise GeecsConfigurationError(
                    f"shot_period must be positive seconds, got {shot_period}"
                )
        fly = acquisition == "fly"
        # A fly run drives no box, so it resolves no default profile: only a
        # profile named for it is checked, and refused if it is scan mode.
        if fly:
            if (
                trigger_profile is not None
                and profiles.resolve(trigger_profile).use_scan_mode
            ):
                raise GeecsConfigurationError(
                    f"trigger profile {trigger_profile!r} runs in GEECS scan "
                    "mode, which needs a box and a per-shot announce: a fly run "
                    "drives neither (name no profile, one without "
                    "use_scan_mode, or use acquisition='strict')"
                )
            shot_control: Any = _NO_BOX
        else:
            shot_control = profiles.resolve(trigger_profile)
        profile_key = (
            trigger_profile if trigger_profile is not None else profiles.default
        )
        if fly and shot_period is None:
            shot_period = DEFAULT_TICK_PERIOD_S
        bound_args = signature.bind_partial(*args, **kwargs).arguments
        detectors = list(bound_args.get("detectors") or ())
        refuse_native_run(detectors, shot_control=shot_control)
        refuse_listed_witness(detectors)
        per_shot = [d for d in detectors if is_per_shot(d)]
        _refuse_twice_listed(per_shot)
        required = resolve_required(per_shot, required_arg)
        payload = bound_args.get("sweep")
        movers = (
            sweep_movers(payload, settables)
            if payload is not None and settables is not None
            else []
        )
        # Before the box is driven and before the claim: a needed device
        # (required, moved, the box) down refuses; any other listed device
        # down is left out.
        needed = needed_device_names(required, movers, None if fly else shot_control)
        listed_detectors = detectors
        detectors, dropped = yield from drop_down_devices(
            None if fly else shot_control, detectors, needed, movers
        )
        per_shot = [d for d in detectors if is_per_shot(d)]
        # A condition or a mover is never waited on: a frozen loop (connected,
        # publishing nothing) would go unnoticed, so it is WARNED, never refused.
        stale = yield from stale_devices(
            [*(d for d in detectors if not is_per_shot(d)), *mover_roots(movers)]
        )
        start_warnings: list[str] = []
        if dropped:
            # drop_down_devices WARNED it at bind, before scan.log exists:
            # again after open_run, so the scan's own log says it too.
            start_warnings.append(dropped_text(dropped))
        if stale:
            start_warnings.append(
                f"{stale_text(stale)} — connected, but its LabVIEW loop has not "
                "updated: recorded anyway, its values may be stale (check the device)"
            )
            logger.warning("%s", start_warnings[-1])
        # Scan mode (#1084): the profile's flag selects it; refuse what it
        # cannot carry before the box is driven, then thread the control
        # through the hook (the announce before every fire, the row's
        # column) and bracket the run with the entry and the exit.
        announce = scan_mode if shot_control.use_scan_mode else None
        run_devices = run_device_names(detectors, shot_control)
        if shot_control.use_scan_mode:
            scan_mode_preflight(shot_control, scan_mode, run_devices=run_devices)
            run_devices = entered_device_names(
                scan_mode, run_devices, shot_control, needed=needed
            )
        md = dict(kwargs.pop("md", None) or {})
        # The key the plan resolved (the configs-repo file stem), not the
        # profile's own name field — so the start document replays.
        # A fly run drives no profile: none is recorded.
        md["trigger_profile"] = None if fly else profile_key
        md["use_scan_mode"] = shot_control.use_scan_mode
        md["shots_per_step"] = shots_per_step
        md["acquisition"] = acquisition
        md["native_image_save"] = native_files
        md["devices_dropped"] = dropped
        md["devices_stale"] = [name for name, _ in stale]
        # The scan-mode bracket names the devices stamped by their
        # systimestamp; any other run takes every stamp from acq_timestamp
        # (a halted scan-mode run's choice undone here).
        md["shot_stamp_systimestamp"] = []
        yield from reset_shot_stamps(per_shot)
        gate = gate_for_run(shot_control, fly=fly)
        if gate is not None:
            md["shot_gate"] = gate.describe()
        snapshot = background_snapshot(settables, mock=mock) if background else None
        md["background_telemetry"] = snapshot is not None
        witness, md["witness"] = yield from run_witness(
            settables,
            acquisition=acquisition,
            shot_control=shot_control,
            readers=detectors,
            waited=True if acquisition == "strict" else None,
        )
        tick = None
        if shot_period is not None:
            md["shot_period"] = shot_period
        if fly:
            if hook == "per_shot" and _has_delay(bound_args.get("delay", 0.0)):
                raise GeecsConfigurationError(
                    f"delay={bound_args.get('delay')!r} spaces strict shots: a fly "
                    "count is spaced by its tick (drop delay; set shot_period)"
                )
            streams = DeviceStreams(per_shot)
            tick, keys, kwargs[hook] = bind_fly(
                streams,
                required,
                hook=hook,
                shots_per_step=shots_per_step,
                period=float(shot_period),
            )
            md.update(keys)
        else:
            streams = DeviceStreams(per_shot)
            run = OnDemand(
                shot_control,
                streams,
                required,
                witness=witness,
                shot_period=shot_period,
                scan_mode=announce,
                gate=gate,
            )
            md.update(start_keys(streams, required, witnessed=witness is not None))
            kwargs[hook] = (
                on_demand_per_step(
                    run, shots_per_step=shots_per_step, scan_mode=announce
                )
                if hook == "per_step"
                else on_demand_per_shot(run, scan_mode=announce)
            )
        # The rows' readers: the conditions (every per-shot device is in its
        # own stream), then the witness and the background, neither
        # triggered nor waited on.
        rows = [d for d in detectors if not is_per_shot(d)]
        extras = [d for d in (witness, tick, snapshot) if d is not None]
        # The stock key keeps naming every device the run records.
        md["detectors"] = [getattr(d, "name", str(d)) for d in [*detectors, *extras]]
        listed = list(bound_args.get("detectors") or ())
        if extras or dropped or len(rows) != len(listed):
            bound = signature.bind_partial(*args, **kwargs)
            bound.arguments["detectors"] = [*rows, *extras]
            args, kwargs = bound.args, dict(bound.kwargs)
        inner = stock(*args, md=md, **kwargs)
        inner = streams.wrap(inner)
        if tick is not None:
            inner = tick.wrap(inner)
        inner = witness_wrapper(inner, witness, md["witness"])
        inner = warn_after_open(inner, start_warnings)
        if snapshot is not None:
            # The listed devices, dropped ones included: a device left out
            # because it is down is not read as background telemetry either.
            inner = background_wrapper(
                inner, snapshot, own=listed_detectors, movers=movers
            )
        inner = native_image_save_wrapper(inner, detectors, native_files)
        if announce is not None:
            # Inside the trigger bracket: ARMED, then ScanStart (a refused
            # entry claims nothing), the run, ScanEnd, then STANDBY.
            inner = scan_mode_bracket(
                inner,
                announce,
                EnterScanMode(
                    run_devices,
                    scan_parameter_for(stock.__name__),
                    paused=paused_device_names(detectors),
                    optional=tuple(d for d in run_devices if d not in needed),
                ),
                stamped=in_scan_mode(per_shot, run_devices),
            )
        if fly:
            # Nothing on the box: no state, no fire (its devices were judged
            # at the start, drop_down_devices).
            inner = bpp.inject_md_wrapper(
                inner, (yield from set_tolerance_gate(movers))
            )
            logger.info("%s", BOX_NOT_DRIVEN)
            return (yield from name_failed_status(inner))
        # Before the first move, before the claim.
        yield from open_gate(gate)
        inner = bpp.inject_md_wrapper(inner, (yield from set_tolerance_gate(movers)))
        # A failure outside the stock plan's run_wrapper (the bracket's own
        # move) is named here; one inside it is named by the hook, before
        # run_wrapper writes the stop document.
        return (
            yield from name_failed_status(
                run_bracket(inner, shot_control, TriggerState.ARMED)
            )
        )

    # The stock ``*args`` annotations are informational (``list_scan`` even
    # annotates each element as a ``(motor, points)`` tuple while taking
    # them flat) and the manager validates a queue item against them, so a
    # bound plan drops them: a vararg is whatever the manager resolves.
    # Every other argument is annotated: the manager turns each device and
    # plan name inside an unannotated one into the object, metadata included.
    parameters = [
        p.replace(annotation=Parameter.empty)
        if p.kind is Parameter.VAR_POSITIONAL
        else p.replace(annotation=_ARGUMENT_ANNOTATIONS[p.name])
        if p.annotation is Parameter.empty and p.name in _ARGUMENT_ANNOTATIONS
        else p
        for p in signature.parameters.values()
        if p.name not in _HOOKS
    ]
    bare = [
        p.name
        for p in parameters
        if p.annotation is Parameter.empty and p.kind is not Parameter.VAR_POSITIONAL
    ]
    if bare:
        raise ValueError(
            f"{stock.__name__}: no annotation for {', '.join(bare)} — the manager "
            "would convert every device and plan name inside it into the object"
        )
    parameters.append(
        Parameter(
            "trigger_profile",
            Parameter.KEYWORD_ONLY,
            default=None,
            annotation=str | None,
        )
    )
    if hook == "per_step":
        parameters.append(
            Parameter(
                "shots_per_step", Parameter.KEYWORD_ONLY, default=1, annotation=int
            )
        )
    parameters.append(
        Parameter(
            "acquisition", Parameter.KEYWORD_ONLY, default="strict", annotation=str
        )
    )
    parameters.append(
        Parameter(
            "required",
            Parameter.KEYWORD_ONLY,
            default=None,
            annotation=DEVICE_LIST | None,
        )
    )
    parameters.append(
        Parameter(
            "shot_period",
            Parameter.KEYWORD_ONLY,
            default=None,
            annotation=float | None,
        )
    )
    parameters.append(
        Parameter(
            "native_image_save",
            Parameter.KEYWORD_ONLY,
            default=None,
            annotation=bool | None,
        )
    )
    parameters.append(
        Parameter(
            "background_telemetry",
            Parameter.KEYWORD_ONLY,
            default=None,
            annotation=bool | None,
        )
    )
    plan.__signature__ = signature.replace(parameters=parameters)  # type: ignore[attr-defined]
    plan.__name__ = plan.__qualname__ = stock.__name__
    plan.__doc__ = _geecs_doc(stock, hook)
    return plan


#: A device-list argument as the manager reads it: names in, devices out,
#: and only there.
DEVICE_LIST = Sequence[Readable]

#: The annotation a bound stock plan's argument gets when the stock plan
#: gives it none.
_ARGUMENT_ANNOTATIONS: dict[str, Any] = {
    "detectors": DEVICE_LIST,
    "md": dict[str, Any] | None,
}


def require_plain_metadata(md: Any) -> None:
    """Refuse run metadata holding anything but plain data, naming where it sits.

    The start document carries text, numbers, booleans, ``None``, and lists
    and string-keyed dicts of those.  Anything else — a device, a plan —
    cannot be copied or recorded by the RunEngine, which would fail the run
    after its scan number is claimed; refusing here costs no claim and no
    move.

    Raises
    ------
    GeecsConfigurationError
        Naming the first offending entry (``md['geecs']['preset']``) and
        its type.
    """

    def walk(value: Any, where: str) -> None:
        if value is None or isinstance(value, (str, bool, int, float)):
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise GeecsConfigurationError(
                        f"{where} has the key {key!r} ({type(key).__name__}): "
                        "run metadata keys are text"
                    )
                walk(item, f"{where}[{key!r}]")
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{where}[{index}]")
            return
        raise GeecsConfigurationError(
            f"{where} is a {type(value).__name__}, not plain data: run metadata "
            "carries text, numbers, lists and dicts only"
        )

    if md is not None and not isinstance(md, Mapping):
        raise GeecsConfigurationError(
            f"md is a {type(md).__name__}: run metadata is a dict"
        )
    walk(md, "md")


def _has_delay(delay: Any) -> bool:
    """Whether a ``count`` ``delay`` argument would make the repeat loop sleep."""
    if delay is None:
        return False
    if isinstance(delay, (int, float)):
        return delay > 0
    try:
        return any(float(d) > 0 for d in delay)
    except TypeError:
        return True  # an unknown shape: refuse rather than idle


def _geecs_doc(stock: Callable[..., Any], hook: str) -> str:
    extra = (
        "    trigger_profile : str, optional\n"
        "        Trigger profile driving the box for this scan (the experiment\n"
        "        default when omitted). The run is bracketed ARMED → STANDBY.\n"
        "        A profile with use_scan_mode runs the scan in GEECS scan mode\n"
        "        (strict only): the devices are told the scan and every shot,\n"
        "        as under Master Control, and each row carries announced_shot.\n"
    )
    if hook == "per_step":
        extra += (
            "    shots_per_step : int, optional\n"
            "        Shots recorded at every position (default 1).\n"
        )
    extra += (
        "    acquisition : {'strict', 'fly'}, optional\n"
        "        'strict' (default) fires the box once per row (the retired\n"
        "        'gated' is refused by name).  'fly' never drives the box: a\n"
        "        software tick every shot_period (default 1 s) is the shot, a\n"
        "        Software-source native camera is pressed at each tick and\n"
        "        every other device free-runs; a scan-mode profile is refused.\n"
        "    required : list of devices, optional\n"
        "        The per-shot devices (detectors, their scalars views, native\n"
        "        cameras) whose record makes a shot complete: waited on and\n"
        "        refired for.  None (default) = every per-shot device; [] =\n"
        "        none.  Strict: every per-shot device streams into its own\n"
        "        '<name>_stream' and 'primary' is the shot table.  A device\n"
        "        with no stamp is a condition, read in every row, and cannot\n"
        "        be required.\n"
        "    shot_period : float, optional\n"
        "        Strict: seconds between fires (a deliberate rep-rate\n"
        "        throttle); None fires as fast as the shot allows.  Fly: the\n"
        "        tick period.  A fly count refuses a nonzero delay.\n"
        "    native_image_save : bool, optional\n"
        "        Whether the plugin-backed cameras also write their LabVIEW\n"
        "        per-shot files (PNGs) beside the plugin's stack; the experiment\n"
        "        default (experiment_defaults.yaml) when omitted.  A device\n"
        "        without a file plugin always keeps its native files.\n"
        "    background_telemetry : bool, optional\n"
        "        Whether every other logged scalar of the experiment (the DB's\n"
        "        get='yes' variables of the devices not in this run) is read\n"
        "        into every row as well, softly: a device that does not answer\n"
        "        when the run starts is left out of it, never failing it; the\n"
        "        experiment default (experiment_defaults.yaml) when omitted.\n"
    )
    return (
        f"GEECS {stock.__name__}: the stock plan with the trigger box driven "
        f"by the worker — fired between trigger and wait on every shot "
        f"(strict), or not driven at all (fly).\n\n"
        f"{inspect.getdoc(stock) or ''}\n\n"
        f"    Other Parameters\n    ----------------\n{extra}"
    )


@functools.wraps(bps.mv)
def _mv_named(*args: Any, **kwargs: Any):
    """The stock ``mv`` stub, with a refused move's cause in the failure.

    The manager's report of a refused manual move then names the cause,
    not ``<AsyncStatus …>``.
    """
    return (yield from name_failed_status(bps.mv(*args, **kwargs)))


def bind_plans(
    profiles: TriggerProfiles,
    *,
    resolver: Any | None = None,
    settables: SettableFactory | None = None,
    mock: bool = False,
    scan_mode: Any | None = None,
) -> dict[str, Callable[..., Any]]:
    """Every name in :data:`GEECS_PLAN_NAMES` → the plan the worker registers.

    The scan verbs come back bound through :func:`strict_plan`; ``mv`` is
    the stock stub (a manual move as a queue item, nothing strict about it)
    with its failure named (:func:`_mv_named`);
    ``apply_setup`` is
    :func:`~geecs_bluesky.plans.apply_setup.apply_setup_plan` over
    *resolver* and *settables* (the namespace).  *settables* is also where every scan
    verb's background telemetry comes from (:func:`background_snapshot`);
    *mock* connects its members with mock backends (hermetic tests).
    *scan_mode* is the worker's
    :class:`~geecs_bluesky.devices.scan_mode.ScanModeControl`: the scan
    verbs drive it under a profile with ``use_scan_mode`` and
    ``release_scan_mode`` ends it on demand; ``None`` (a hermetic worker)
    refuses both.  A trigger profile's ``shot_gate`` is held for before
    every SINGLESHOT by every plan that fires one — the scan verbs and
    ``measure_shot_offsets``.
    """
    bound: dict[str, Callable[..., Any]] = {}
    for name in GEECS_PLAN_NAMES:
        if name == "mv":
            bound[name] = _mv_named
        elif name == "sweep":
            bound[name] = strict_plan(
                sweep_plan(settables),
                profiles,
                resolver=resolver,
                settables=settables,
                mock=mock,
                scan_mode=scan_mode,
            )
        elif name == "optimize":
            from .optimize import optimize_plan

            bound[name] = optimize_plan(
                profiles, resolver, settables, mock=mock, scan_mode=scan_mode
            )
        elif name == "apply_setup":
            bound[name] = apply_setup_plan(resolver, settables)
        elif name == "measure_shot_offsets":
            bound[name] = measure_shot_offsets_plan(profiles, resolver)
        elif name == "check_shot_sync":
            bound[name] = check_shot_sync_plan(profiles)
        elif name == "release_scan_mode":
            bound[name] = release_scan_mode_plan(scan_mode)
        else:
            assert name not in NON_SCAN_PLAN_NAMES
            bound[name] = strict_plan(
                getattr(bp, name),
                profiles,
                resolver=resolver,
                settables=settables,
                mock=mock,
                scan_mode=scan_mode,
            )
    return bound


__all__ = [
    "ACQUISITION_MODES",
    "DEVICE_LIST",
    "TriggerProfiles",
    "background_snapshot",
    "background_wrapper",
    "bind_plans",
    "gate_for_run",
    "drop_down_devices",
    "liveness_gate",
    "set_tolerance_gate",
    "native_image_save_wrapper",
    "refuse_native_run",
    "require_plain_metadata",
    "resolve_background_telemetry",
    "resolve_native_image_save",
    "run_witness",
    "strict_plan",
]
