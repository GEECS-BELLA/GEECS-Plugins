"""The registration table: stock plan names with the GEECS ``take_reading`` pre-bound.

The worker registers the stock ``bluesky.plans`` verbs under their own
names (``count``, ``scan``, ``list_scan`` …, :data:`~geecs_bluesky.plan_names.GEECS_PLAN_NAMES`)
with the one GEECS difference bound in: the ``per_step`` / ``per_shot``
hook — strict (:mod:`geecs_bluesky.plans.strict`: the trigger box fired
between trigger and wait, one row per shot) or gated
(:mod:`geecs_bluesky.plans.gated`: the box free-runs while the
plugin-backed cameras count a batch; one datum per camera per step and
one ``shots`` event per shot).  A queue item naming ``scan`` with
namespace devices therefore runs a complete GEECS scan; the plan's
arguments are the scan's one description (plan of record §4.D, §10.5).

What a bound plan keeps and what it adds
----------------------------------------
The stock parameters are kept verbatim — positions, ``num``,
``snake_axes``, ``md`` — minus the ``per_step`` / ``per_shot`` hook (a
callable no queue item can carry).  Keyword-only GEECS parameters are
appended, because each is a fact of *this* scan and belongs in its
description rather than in a side channel:

- ``trigger_profile`` — which trigger profile drives the box (the
  experiment's default when omitted).  The bound plan brackets the run
  through that profile's :class:`ShotControl`: ``ARMED → … → STANDBY``
  strict, ``OFF → … → STANDBY`` gated.
- ``shots_per_step`` (scan verbs only) — rows per position; ``count``'s
  ``num`` already is the shot count.
- ``acquisition`` — ``"strict"`` (default) or ``"gated"``: which
  ``take_reading`` the hook binds (``08_gated_batch.md`` §4.1).
- ``non_essential`` — detectors streamed for the run's duration in their
  own streams, never waited on (§4.3); names resolve like ``detectors``.
- ``shot_period`` — the strict rep-rate throttle, seconds between fires
  (GEECS-Plugins#840); refused with ``gated`` (the box free-runs there).

All ride in the start document beside the stock ``plan_args``
(``non_essential`` as the devices' names, plus ``shot_clock`` — the
device whose stamp counted a gated run's shots).

Which stock plans
-----------------
Every ``bluesky.plans`` verb exposing the hook that a queue item can
express: the deprecated aliases (``relative_scan`` for ``rel_scan`` …) and
``scan_nd`` (a ``Cycler`` argument) are left out.
:data:`~geecs_bluesky.plan_names.GEECS_PLAN_NAMES` pins the list for the
import-light readers; ``tests/test_plan_registry.py`` asserts the two agree.

The non-scan queue items
------------------------
``mv`` is the stock stub — a manual move as a queue item.  ``run_action``
(:func:`~geecs_bluesky.plans.action_compiler.run_action_plan`) runs a named
plan from the experiment's action library (``actions.yaml``): the steps compile to plain stubs
(:mod:`geecs_bluesky.plans.action_compiler`) over the device namespace,
which hands out each ``(device, variable)`` as the settable child or the
readable signal it already is — no run is opened, so nothing is claimed
and no file is written.  Neither takes a detector list, so a preset cannot
name them.

``measure_shot_offsets`` and ``check_shot_sync``
(:mod:`geecs_bluesky.plans.calibration`) are the two once-run shot-offset
plans of plan of record §4.F: the calibration that measures each device's
edge-to-stamp latency, and the preflight that says whether the stored
measurement still holds.  They *do* take a detector list, but no positions
— they open no run, claim no scan number and write no scan data, so a
preset (which describes a scan) still cannot express them.  Both drive the
trigger box OFF and cost at least the longest device timeout in the set,
which is exactly why they are queue items and never steps inside a scan
(§11.2).
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Iterator, Mapping
from inspect import Parameter
from typing import Any

import bluesky.plan_stubs as bps
import bluesky.plans as bp
from geecs_schemas.trigger_profile import TriggerState

from geecs_bluesky.devices.shot_control import ShotControl
from geecs_bluesky.exceptions import GeecsConfigurationError
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
from geecs_bluesky.plans.strict import geecs_per_shot, geecs_per_step
from geecs_bluesky.utils import safe_name

logger = logging.getLogger(__name__)

#: Stock plans with the hook that are **not** registered: deprecated aliases
#: of the ``rel_*`` verbs, and ``scan_nd`` whose ``cycler`` no queue item
#: can carry.
EXCLUDED_STOCK_PLANS: frozenset[str] = frozenset(
    {
        "scan_nd",
        "inner_product_scan",
        "outer_product_scan",
        "relative_inner_product_scan",
        "relative_outer_product_scan",
        "relative_scan",
        "relative_list_scan",
        "relative_log_scan",
        "relative_spiral",
        "relative_spiral_fermat",
    }
)

_HOOKS = ("per_step", "per_shot")


def stock_plans_with_hook() -> dict[str, str]:
    """Every ``bluesky.plans`` verb exposing ``per_step`` or ``per_shot`` → its hook."""
    found: dict[str, str] = {}
    for name in dir(bp):
        obj = getattr(bp, name)
        if name.startswith("_") or not inspect.isfunction(obj):
            continue
        try:
            params = inspect.signature(obj).parameters
        except (TypeError, ValueError):
            continue
        for hook in _HOOKS:
            if hook in params:
                found[name] = hook
    return found


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


def strict_plan(
    stock: Callable[..., Any], profiles: TriggerProfiles
) -> Callable[..., Any]:
    """Bind the strict hook into one stock plan; keep its name and signature.

    Parameters
    ----------
    stock :
        A ``bluesky.plans`` verb exposing ``per_step`` or ``per_shot``.
    profiles :
        The trigger profiles a ``trigger_profile`` argument resolves against.

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
        opening = TriggerState.OFF if acquisition == "gated" else TriggerState.ARMED
        return (yield from run_bracket(inner, shot_control, opening))

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
    )
    return (
        f"GEECS {stock.__name__}: the stock plan with the trigger box driven "
        f"by the worker — fired between trigger and wait on every shot "
        f"(strict), or free-running through a counted batch (gated).\n\n"
        f"{inspect.getdoc(stock) or ''}\n\n"
        f"    Other Parameters\n    ----------------\n{extra}"
    )


def bind_plans(
    profiles: TriggerProfiles,
    *,
    resolver: Any | None = None,
    settables: SettableFactory | None = None,
) -> dict[str, Callable[..., Any]]:
    """Every name in :data:`GEECS_PLAN_NAMES` → the plan the worker registers.

    The scan verbs come back bound through :func:`strict_plan`; ``mv`` is
    the stock stub (a manual move as a queue item, nothing strict about it);
    ``run_action`` is :func:`run_action_plan` over *resolver* and
    *settables* (the namespace).
    """
    bound: dict[str, Callable[..., Any]] = {}
    for name in GEECS_PLAN_NAMES:
        if name == "mv":
            bound[name] = bps.mv
        elif name == "run_action":
            bound[name] = run_action_plan(resolver, settables)
        elif name == "measure_shot_offsets":
            bound[name] = measure_shot_offsets_plan(profiles, resolver)
        elif name == "check_shot_sync":
            bound[name] = check_shot_sync_plan(profiles)
        else:
            assert name not in NON_SCAN_PLAN_NAMES
            bound[name] = strict_plan(getattr(bp, name), profiles)
    return bound


__all__ = [
    "ACQUISITION_MODES",
    "EXCLUDED_STOCK_PLANS",
    "TriggerProfiles",
    "bind_plans",
    "stock_plans_with_hook",
    "strict_plan",
]
