"""Expand a preset into the queue item it stands for — client-side, import-light.

A :class:`geecs_schemas.Preset` is a saved scan: the device group plus the
plan call.  The worker registers stock plans over namespace devices by
**name** (:data:`~geecs_bluesky.plan_names.GEECS_PLAN_NAMES`), so
submission is a translation of names, nothing more (plan of record §4.D):

- each device of the group becomes its namespace binding —
  ``UC_Amp4_IR_input``, or ``UC_Amp4_IR_input.scalars`` when
  ``save_images`` is off (the scalars-only view every namespace device
  carries: on a detector the shot wait without the files, on a
  scalar-only device what the device reads);
- each scan-variable string in ``plan.args`` / ``plan.kwargs`` — a
  ``Device:Variable`` pair or a scan-variable catalog name — becomes the
  namespace's Movable child, ``U_S1H.current``;
- ``trigger_profile`` and ``background`` ride as the bound plan's keyword
  argument and the run metadata; the preset name and the submission
  record ride in ``md["geecs"]`` as provenance.

The manager resolves the names against the worker namespace at submission
but does **not** refuse an unknown one (bluesky-queueserver 0.0.25 passes
an unresolved string through to the plan), so the pre-submit preflight
checks every reference against the manager's device tree
(:func:`~geecs_bluesky.qs_client.submit_preflight.run_submit_preflight`) —
the typo fails at preflight, not at queue-front.  Pseudo scan variables
(``kind: pseudo``) have no namespace noun yet (phase 3): expanding one is
refused here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.plan_names import GEECS_PLAN_NAMES
from geecs_bluesky.utils import device_reference


@dataclass(frozen=True)
class QueueItem:
    """A stock plan call ready for ``QueueClient.submit_plan``."""

    name: str
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)


def scan_variable_reference(
    target: str, catalog: Mapping[str, Any] | None = None
) -> str:
    """The queue-item spelling of a scan variable.

    *target* is ``"Device:Variable"`` or a name from the experiment's
    scan-variable catalog (``catalog``: name → ``ScanVariableSpec``); a
    plain device name passes through as the device itself.

    Raises
    ------
    GeecsConfigurationError
        A catalog pseudo variable (no namespace noun until phase 3).
    """
    spec = (catalog or {}).get(target)
    if spec is not None:
        if getattr(spec, "kind", None) == "pseudo":
            raise GeecsConfigurationError(
                f"scan variable {target!r} is a pseudo variable — pseudo axes are "
                "not scannable through the namespace yet (phase 3)"
            )
        target = str(spec.target)
    device, sep, variable = target.partition(":")
    return device_reference(device, variable if sep else None)


def expand_preset(
    preset: Any,
    *,
    catalog: Mapping[str, Any] | None = None,
    md: Mapping[str, Any] | None = None,
) -> QueueItem:
    """The queue item a preset submits.

    Parameters
    ----------
    preset :
        A :class:`geecs_schemas.Preset` (with a ``plan``).
    catalog :
        The experiment's scan-variable catalog (name → spec), for catalog
        names inside the plan arguments.
    md :
        Extra run metadata (the submission record under ``geecs``).

    Raises
    ------
    GeecsConfigurationError
        No plan call, a plan name the worker does not register, or a
        pseudo scan variable.
    """
    plan = preset.plan
    if plan is None:
        raise GeecsConfigurationError(
            f"preset {preset.name!r} is a device group with no plan call — add a "
            "plan (name/args/kwargs) before submitting it"
        )
    if plan.name not in GEECS_PLAN_NAMES:
        raise GeecsConfigurationError(
            f"preset {preset.name!r} names plan {plan.name!r}; the worker "
            f"registers: {', '.join(GEECS_PLAN_NAMES)}"
        )
    detectors = [
        device_reference(d.device)
        if d.save_images
        else device_reference(d.device) + ".scalars"
        for d in preset.devices
    ]
    args = [_resolve(a, catalog) for a in plan.args]
    kwargs = {k: _resolve(v, catalog) for k, v in plan.kwargs.items()}
    if preset.trigger_profile is not None:
        kwargs.setdefault("trigger_profile", preset.trigger_profile)
    run_md: dict[str, Any] = dict(kwargs.pop("md", None) or {})
    run_md.update(md or {})
    run_md.setdefault("description", preset.description)
    run_md.setdefault("background", preset.background)
    geecs = dict(run_md.get("geecs") or {})
    geecs.setdefault("preset", preset.name)
    run_md["geecs"] = geecs
    kwargs["md"] = run_md
    return QueueItem(name=plan.name, args=[detectors, *args], kwargs=kwargs)


def device_references(item: QueueItem) -> list[str]:
    """Every device reference the item names, in order: the detectors, then string arguments.

    A string argument counts when it is spelled like a namespace reference
    (an identifier, optionally dotted); anything else is a plain value.
    The GEECS keyword arguments (``trigger_profile``, ``md``) never name a
    device.
    """
    refs: list[str] = []
    detectors = item.args[0] if item.args and isinstance(item.args[0], list) else []
    kwargs = [v for k, v in item.kwargs.items() if k not in ("trigger_profile", "md")]
    for value in [*detectors, *item.args[1:], *kwargs]:
        if isinstance(value, str) and _REFERENCE.match(value):
            refs.append(value)
    return refs


_REFERENCE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$|^[A-Za-z_][A-Za-z0-9_]*$"
)


def _resolve(value: Any, catalog: Mapping[str, Any] | None) -> Any:
    """Turn a scan-variable string into its namespace reference; pass the rest through."""
    if isinstance(value, str) and (":" in value or (catalog and value in catalog)):
        return scan_variable_reference(value, catalog)
    return value


__all__ = ["QueueItem", "device_references", "expand_preset", "scan_variable_reference"]
