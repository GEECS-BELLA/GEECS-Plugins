"""The registration table: stock plan names with the strict ``take_reading`` pre-bound.

The worker registers the stock ``bluesky.plans`` verbs under their own
names (``count``, ``scan``, ``list_scan`` …, :data:`~geecs_bluesky.plan_names.GEECS_PLAN_NAMES`)
with the one GEECS difference bound in: the strict ``per_step`` /
``per_shot`` that fires the trigger box between trigger and wait
(:mod:`geecs_bluesky.plans.strict`).  A queue item naming ``scan`` with
namespace devices therefore runs a complete strict GEECS scan; the plan's
arguments are the scan's one description (plan of record §4.D, §10.5).

What a bound plan keeps and what it adds
----------------------------------------
The stock parameters are kept verbatim — positions, ``num``,
``snake_axes``, ``md`` — minus the ``per_step`` / ``per_shot`` hook (a
callable no queue item can carry).  Two keyword-only GEECS parameters are
appended, because both are facts of *this* scan and belong in its
description rather than in a side channel:

- ``trigger_profile`` — which trigger profile drives the box (the
  experiment's default when omitted).  The bound plan brackets the run
  ``ARMED → … → STANDBY`` through that profile's :class:`ShotControl`.
- ``shots_per_step`` (scan verbs only) — rows per position; ``count``'s
  ``num`` already is the shot count.

Both ride in the start document (``trigger_profile``,
``shots_per_step``) beside the stock ``plan_args``.

Which stock plans
-----------------
Every ``bluesky.plans`` verb exposing the hook that a queue item can
express: the deprecated aliases (``relative_scan`` for ``rel_scan`` …) and
``scan_nd`` (a ``Cycler`` argument) are left out.
:data:`~geecs_bluesky.plan_names.GEECS_PLAN_NAMES` pins the list for the
import-light readers; ``tests/test_plan_registry.py`` asserts the two agree.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Iterator, Mapping
from inspect import Parameter
from typing import Any

import bluesky.plan_stubs as bps
import bluesky.plans as bp
import bluesky.preprocessors as bpp
from geecs_schemas.trigger_profile import TriggerState

from geecs_bluesky.devices.shot_control import ShotControl
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.plan_names import GEECS_PLAN_NAMES
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
        shot_control = profiles.resolve(trigger_profile)
        profile_key = (
            trigger_profile if trigger_profile is not None else profiles.default
        )
        if hook == "per_step":
            kwargs[hook] = geecs_per_step(shot_control, shots_per_step=shots_per_step)
        else:
            kwargs[hook] = geecs_per_shot(shot_control)
        md = dict(kwargs.pop("md", None) or {})
        # The key the plan resolved (the configs-repo file stem), not the
        # profile's own name field — so the start document replays.
        md["trigger_profile"] = profile_key
        md["shots_per_step"] = shots_per_step
        yield from bps.mv(shot_control, TriggerState.ARMED.value)

        def standby():
            yield from bps.mv(shot_control, TriggerState.STANDBY.value)

        return (
            yield from bpp.finalize_wrapper(stock(*args, md=md, **kwargs), standby())
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
    plan.__signature__ = signature.replace(parameters=parameters)  # type: ignore[attr-defined]
    plan.__name__ = plan.__qualname__ = stock.__name__
    plan.__doc__ = _geecs_doc(stock, hook)
    return plan


def _geecs_doc(stock: Callable[..., Any], hook: str) -> str:
    extra = (
        "    trigger_profile : str, optional\n"
        "        Trigger profile driving the box for this scan (the experiment\n"
        "        default when omitted). The run is bracketed ARMED → STANDBY.\n"
    )
    if hook == "per_step":
        extra += (
            "    shots_per_step : int, optional\n"
            "        Strict single shots recorded at every position (default 1).\n"
        )
    return (
        f"GEECS strict {stock.__name__}: the stock plan with the trigger box "
        f"fired between trigger and wait on every shot.\n\n"
        f"{inspect.getdoc(stock) or ''}\n\n"
        f"    Other Parameters\n    ----------------\n{extra}"
    )


def bind_strict_plans(profiles: TriggerProfiles) -> dict[str, Callable[..., Any]]:
    """Every name in :data:`GEECS_PLAN_NAMES` → the plan the worker registers.

    The scan verbs come back bound through :func:`strict_plan`; ``mv`` is
    the stock stub (a manual move as a queue item, nothing strict about it).
    """
    bound: dict[str, Callable[..., Any]] = {}
    for name in GEECS_PLAN_NAMES:
        if name == "mv":
            bound[name] = bps.mv
        else:
            bound[name] = strict_plan(getattr(bp, name), profiles)
    return bound


__all__ = [
    "EXCLUDED_STOCK_PLANS",
    "TriggerProfiles",
    "bind_strict_plans",
    "stock_plans_with_hook",
    "strict_plan",
]
