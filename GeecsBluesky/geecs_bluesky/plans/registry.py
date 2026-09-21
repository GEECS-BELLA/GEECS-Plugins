"""Register Count, Sweep, Optimize and utility plans over the worker namespace.

The acquisition binder supplies strict/gated hooks, the liveness gate,
trigger bracket, non-essential streaming and shot throttling. Count uses
stock Bluesky count; Sweep delegates traversal to scan_nd. Moving stock
verbs remain implementation details, never public queue entries.
"""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from inspect import Parameter
from typing import Any

import bluesky.plan_stubs as bps
import bluesky.plans as bp
import bluesky.preprocessors as bpp
from geecs_schemas.trigger_profile import TriggerState

from geecs_bluesky.devices.ca.liveness import read_disconnected
from geecs_bluesky.devices.detector import GeecsDetector
from geecs_bluesky.devices.shot_control import ShotControl
from geecs_bluesky.exceptions import GeecsConfigurationError, GeecsDeviceDownError
from geecs_bluesky.plan_names import (
    ACQUISITION_MODES,
    GEECS_PLAN_NAMES,
    NON_SCAN_PLAN_NAMES,
)
from geecs_bluesky.plans.action_compiler import SettableFactory, run_action_plan
from geecs_bluesky.plans.calibration import (
    check_shot_sync_plan,
    measure_shot_offsets_plan,
)
from geecs_bluesky.plans.gated import (
    gated_per_shot,
    gated_per_step,
    non_essential_wrapper,
    refuse_native_essentials,
    run_bracket,
    shot_clock,
)
from geecs_bluesky.plans.strict import (
    geecs_name,
    geecs_per_shot,
    geecs_per_step,
    name_failed_status,
)
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
            except Exception as exc:  # a profile that names no device, bad YAML …
                logger.warning(
                    "trigger profile %r not loaded: %s: %s",
                    name,
                    type(exc).__name__,
                    exc,
                )
        try:
            defaults = resolver.resolve_experiment_defaults()
        except Exception as exc:  # no configs root, unreadable defaults file
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


def liveness_gate(shot_control: Any, devices: Sequence[Any]):
    """Plan: refuse the run if the gateway reports any of its devices Disconnected.

    One ``CONNECTED`` read each for the trigger box's device(s)
    (:attr:`~geecs_bluesky.devices.shot_control.ShotControl.liveness_signals`)
    and every device in *devices* that carries a ``connected_status``
    signal (a detector, its scalars view, a scalar-only device); a device
    without one is not judged.  The verdict rule is the shared
    :func:`~geecs_bluesky.devices.ca.liveness.read_disconnected` —
    fail-open, only the exact ``Disconnected`` string counts.

    Runs before the bracket's first move and before ``open_run``: nothing
    is driven and nothing is claimed for a run this refuses.

    Raises
    ------
    GeecsDeviceDownError
        Naming every device reported down.
    """
    signals: dict[str, Any] = dict(getattr(shot_control, "liveness_signals", {}))
    for obj in devices:
        signal = getattr(obj, "connected_status", None)
        if signal is None:
            continue
        signals.setdefault(geecs_name(obj), signal)
    # An unreadable CONNECTED before a run is abnormal (the gateway serves
    # it for every DB device) and cost the connect timeout: say so.
    down = yield from read_disconnected(signals, unreadable_level=logging.WARNING)
    if down:
        names = ", ".join(down)
        raise GeecsDeviceDownError(
            f"the gateway reports {names} DISCONNECTED — the run was refused "
            "before the trigger box was driven (nothing claimed, no scan "
            "folder). Check the GEECS device(s), then resubmit.",
            device_name=down[0],
        )


def resolve_native_image_save(requested: bool | None, resolver: Any | None) -> bool:
    """The run's LabVIEW-files switch: the request, else the experiment default, else on.

    Read at every run, not at bind time, so an edit to
    ``experiment_defaults.yaml`` reaches the next scan without the worker
    reopening its environment.  Fail-open to *on*: a resolver that cannot
    read the defaults (no configs root, bad YAML, an older resolver without
    the method) keeps the dual-write — the state every scan had before the
    switch existed — and says so once in the journal.
    """
    if requested is not None:
        return bool(requested)
    if resolver is None:
        return True
    try:
        defaults = resolver.resolve_experiment_defaults()
    except Exception as exc:  # no configs root, unreadable defaults file
        logger.warning(
            "experiment defaults not read (%s: %s); native saving stays on",
            type(exc).__name__,
            exc,
        )
        return True
    return bool(getattr(defaults, "native_image_save", True))


def native_image_save_wrapper(plan: Any, devices: Sequence[Any], enabled: bool) -> Any:
    """Set the plugin-backed cameras' LabVIEW saving for this run; restore after.

    The run-level switch of PNG retirement (#738).  It reaches the cameras
    whose frames the file plugin captures (``plugin_backed``) and nothing
    else: a device without a plugin — a LabVIEW-native camera, a
    proprietary-format DAQ — has no other record, so its native saving is
    never touched, whatever *enabled* says.  A ``.scalars`` view counts as
    its owner (the owner's data logics are what a prepare consults).  The
    previous values are restored in a ``finalize_wrapper``, success or not,
    so the namespace's long-lived detectors carry their construction
    default into the next run.

    Parameters
    ----------
    plan :
        The bound plan (staging inside it).
    devices :
        The run's detectors and non-essential devices.
    enabled :
        ``False`` = the plugin's stack is those cameras' only record this run.
    """
    cameras: list[GeecsDetector] = []
    kept: list[GeecsDetector] = []
    seen: set[int] = set()
    for d in devices:
        owner = getattr(d, "_owner", d)
        if id(owner) in seen or not isinstance(owner, GeecsDetector):
            continue
        seen.add(id(owner))
        if not owner.native_save:
            continue
        if owner.plugin_backed:
            cameras.append(owner)
        else:
            kept.append(owner)
    kept_names = ", ".join(c._geecs_device_name for c in kept)
    if not cameras:
        if not enabled and kept:
            logger.info(
                "native_image_save=False reaches no camera here: %s save through "
                "LabVIEW only (no file plugin)",
                kept_names,
            )
        return (yield from plan)
    previous = [(cam, cam.native_image_save) for cam in cameras]
    for cam in cameras:
        cam.native_image_save = enabled
    logger.info(
        "native saving %s this run for %s%s",
        "on" if enabled else "off (file-plugin stacks only)",
        ", ".join(c._geecs_device_name for c in cameras),
        f"; kept on, no file plugin: {kept_names}" if kept and not enabled else "",
    )

    def restore():
        for cam, was in previous:
            cam.native_image_save = was
        yield from bps.null()

    return (yield from bpp.finalize_wrapper(plan, restore()))


def strict_plan(
    stock: Callable[..., Any],
    profiles: TriggerProfiles,
    *,
    resolver: Any | None = None,
) -> Callable[..., Any]:
    """Bind the strict hook into one stock plan; keep its name and signature.

    Parameters
    ----------
    stock :
        A ``bluesky.plans`` verb exposing ``per_step`` or ``per_shot``.
    profiles :
        The trigger profiles a ``trigger_profile`` argument resolves against.
    resolver :
        The configs-repo resolver whose ``resolve_experiment_defaults`` the
        ``native_image_save`` default is read from at every run; ``None``
        (or an unreadable file) leaves native saving on.

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

    def plan(*args: Any, **kwargs: Any):
        trigger_profile = kwargs.pop("trigger_profile", None)
        shots_per_step = int(kwargs.pop("shots_per_step", 1))
        acquisition = str(kwargs.pop("acquisition", "strict") or "strict")
        non_essential = list(kwargs.pop("non_essential", None) or ())
        shot_period = kwargs.pop("shot_period", None)
        native_files = resolve_native_image_save(
            kwargs.pop("native_image_save", None), resolver
        )
        if acquisition not in ACQUISITION_MODES:
            raise GeecsConfigurationError(
                f"acquisition={acquisition!r} is not one of {ACQUISITION_MODES}"
            )
        if shot_period is not None:
            shot_period = float(shot_period)
            if shot_period <= 0:
                raise GeecsConfigurationError(
                    f"shot_period must be positive seconds, got {shot_period}"
                )
            if acquisition == "gated":
                raise GeecsConfigurationError(
                    "shot_period is a strict-mode throttle: a gated batch runs "
                    "at the box's rate (drop shot_period or use acquisition='strict')"
                )
        shot_control = profiles.resolve(trigger_profile)
        profile_key = (
            trigger_profile if trigger_profile is not None else profiles.default
        )
        bound_args = signature.bind_partial(*args, **kwargs).arguments
        detectors = list(bound_args.get("detectors") or ())
        # Compared by OWNER: ``X.scalars`` essential with ``X`` non-essential
        # is the same camera twice — its one acquire logic would be in fly
        # mode for the stream while the view expects the strict stamp wait.
        owners = {id(getattr(d, "_owner", d)) for d in detectors}
        both = [d for d in non_essential if id(getattr(d, "_owner", d)) in owners]
        if both:
            names = ", ".join(getattr(d, "name", str(d)) for d in both)
            raise GeecsConfigurationError(
                f"{names}: listed both as a detector (or its scalars view) and as "
                "non-essential — a device is waited on every shot or streamed "
                "for the run, not both"
            )
        md = dict(kwargs.pop("md", None) or {})
        # The key the plan resolved (the configs-repo file stem), not the
        # profile's own name field — so the start document replays.
        md["trigger_profile"] = profile_key
        md["shots_per_step"] = shots_per_step
        md["acquisition"] = acquisition
        md["non_essential"] = [getattr(d, "name", str(d)) for d in non_essential]
        md["native_image_save"] = native_files
        if shot_period is not None:
            md["shot_period"] = shot_period
        if acquisition == "gated":
            refuse_native_essentials(detectors)  # before the claim, before any move
            clock, clock_name = shot_clock(detectors)
            md["shot_clock"] = clock_name
            # The row COLUMN as well as the device: the s-file writer and the
            # offline re-export need the column the sampler writes, and
            # deriving it from the device name would put a second copy of the
            # naming contract in a package that cannot import it.
            md["shot_clock_column"] = clock.name
            if hook == "per_step":
                kwargs[hook] = gated_per_step(
                    shot_control, shots_per_step=shots_per_step
                )
            else:
                num = bound_args.get("num", 1)
                if num is None:
                    raise GeecsConfigurationError(
                        "a gated count needs a finite num (the batch size)"
                    )
                delay = bound_args.get("delay", 0.0)
                if _has_delay(delay):
                    # The stock repeat loop sleeps `delay` after every one of
                    # its `num` iterations, and the gated hook does its work
                    # on the first only — the run would idle (num-1)×delay
                    # after the batch.  A delay between shots is a strict
                    # notion; the batch runs at the box's rate.
                    raise GeecsConfigurationError(
                        f"delay={delay!r} is a strict-mode spacing between shots: a "
                        "gated count is one batch at the box's rate (drop delay "
                        "or use acquisition='strict')"
                    )
                kwargs[hook] = gated_per_shot(shot_control, quota=int(num))
        elif hook == "per_step":
            kwargs[hook] = geecs_per_step(
                shot_control, shots_per_step=shots_per_step, shot_period=shot_period
            )
        else:
            kwargs[hook] = geecs_per_shot(shot_control, shot_period=shot_period)
        inner = non_essential_wrapper(stock(*args, md=md, **kwargs), non_essential)
        inner = native_image_save_wrapper(
            inner, [*detectors, *non_essential], native_files
        )
        opening = TriggerState.OFF if acquisition == "gated" else TriggerState.ARMED
        # Before the first move, before the claim (#852).
        yield from liveness_gate(shot_control, [*detectors, *non_essential])
        # A failure outside the stock plan's run_wrapper (the bracket's own
        # move, a non-essential prepare) is named here; one inside it is
        # named by the hook, before run_wrapper writes the stop document.
        return (
            yield from name_failed_status(run_bracket(inner, shot_control, opening))
        )

    # The stock ``*args`` annotations are informational (``list_scan`` even
    # annotates each element as a ``(motor, points)`` tuple while taking
    # them flat) and the manager validates a queue item against them, so a
    # bound plan drops them: a vararg is whatever the manager resolves.
    parameters = [
        p.replace(annotation=Parameter.empty)
        if p.kind is Parameter.VAR_POSITIONAL
        else p
        for p in signature.parameters.values()
        if p.name not in _HOOKS
    ]
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
    # Devices, like ``detectors``: the manager resolves names in any argument.
    parameters.append(Parameter("non_essential", Parameter.KEYWORD_ONLY, default=None))
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
    plan.__signature__ = signature.replace(parameters=parameters)  # type: ignore[attr-defined]
    plan.__name__ = plan.__qualname__ = stock.__name__
    plan.__doc__ = _geecs_doc(stock, hook)
    return plan


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
    )
    if hook == "per_step":
        extra += (
            "    shots_per_step : int, optional\n"
            "        Shots recorded at every position (default 1): strict single\n"
            "        shots, or one gated batch of that size.\n"
        )
    extra += (
        "    acquisition : {'strict', 'gated'}, optional\n"
        "        'strict' (default) fires the box once per row; 'gated' lets it\n"
        "        free-run while the plugin-backed cameras count a batch — the\n"
        "        run is bracketed OFF → STANDBY, frames go to 'primary' as datums\n"
        "        and one 'shots' event per shot carries everything else.\n"
        "    non_essential : list of devices, optional\n"
        "        Plugin-backed detectors streamed for the run's duration, each in\n"
        "        its own '<name>_stream'; never waited on.\n"
        "    shot_period : float, optional\n"
        "        Strict only: seconds between fires (a deliberate rep-rate\n"
        "        throttle); None fires as fast as the shot allows.  A gated\n"
        "        count refuses a nonzero delay for the same reason.\n"
        "    native_image_save : bool, optional\n"
        "        Whether the plugin-backed cameras also write their LabVIEW\n"
        "        per-shot files (PNGs) beside the plugin's stack; the experiment\n"
        "        default (experiment_defaults.yaml) when omitted.  A device\n"
        "        without a file plugin always keeps its native files.\n"
    )
    return (
        f"GEECS {stock.__name__}: the stock plan with the trigger box driven "
        f"by the worker — fired between trigger and wait on every shot "
        f"(strict), or free-running through a counted batch (gated).\n\n"
        f"{inspect.getdoc(stock) or ''}\n\n"
        f"    Other Parameters\n    ----------------\n{extra}"
    )


@functools.wraps(bps.mv)
def _mv_named(*args: Any, **kwargs: Any):
    """The stock ``mv`` stub with a failure's name (#868): the manager's report of a refused manual move reads its cause, not ``<AsyncStatus …>``."""
    return (yield from name_failed_status(bps.mv(*args, **kwargs)))


def bind_plans(
    profiles: TriggerProfiles,
    *,
    resolver: Any | None = None,
    settables: SettableFactory | None = None,
) -> dict[str, Callable[..., Any]]:
    """Every name in :data:`GEECS_PLAN_NAMES` → the plan the worker registers.

    The scan verbs come back bound through :func:`strict_plan`; ``mv`` is
    the stock stub (a manual move as a queue item, nothing strict about it)
    with its failure named (:func:`_mv_named`);
    ``run_action`` is :func:`run_action_plan` over *resolver* and
    *settables* (the namespace).
    """
    bound: dict[str, Callable[..., Any]] = {}
    for name in GEECS_PLAN_NAMES:
        if name == "mv":
            bound[name] = _mv_named
        elif name == "sweep":
            from .sweep import sweep_plan

            bound[name] = strict_plan(
                sweep_plan(settables), profiles, resolver=resolver
            )
        elif name == "optimize":
            from .optimize import optimize_plan

            bound[name] = optimize_plan(profiles, resolver, settables)
        elif name == "run_action":
            bound[name] = run_action_plan(resolver, settables)
        elif name == "measure_shot_offsets":
            bound[name] = measure_shot_offsets_plan(profiles, resolver)
        elif name == "check_shot_sync":
            bound[name] = check_shot_sync_plan(profiles)
        else:
            assert name not in NON_SCAN_PLAN_NAMES
            bound[name] = strict_plan(getattr(bp, name), profiles, resolver=resolver)
    return bound


__all__ = [
    "ACQUISITION_MODES",
    "TriggerProfiles",
    "bind_plans",
    "liveness_gate",
    "native_image_save_wrapper",
    "resolve_native_image_save",
    "strict_plan",
]
