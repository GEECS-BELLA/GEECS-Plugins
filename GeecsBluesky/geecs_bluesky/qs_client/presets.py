"""Expand a preset into the queue item it stands for — client-side, import-light.

A :class:`geecs_schemas.Preset` is a saved scan: the device group plus the
plan call.  The worker registers stock plans over namespace devices by
**name** (:data:`~geecs_bluesky.plan_names.GEECS_PLAN_NAMES`), so
submission is a translation of names, nothing more (plan of record §4.D):

- each device of the group becomes its namespace binding —
  ``UC_Amp4_IR_input``, or ``UC_Amp4_IR_input.scalars`` when
  ``save_images`` is off (the scalars-only view every namespace device
  carries: on a detector the shot wait without the files, on a
  scalar-only device what the device reads); an ``essential: false``
  device goes to the bound plan's ``non_essential`` list instead (phase
  2, ``08_gated_batch.md`` §4.6 — streamed for the run, never waited on;
  it needs its frames, so ``save_images: false`` there is refused);
- ``acquisition`` (``strict`` / ``gated``) and ``shot_period`` ride in
  ``plan.kwargs`` like ``shots_per_step`` does;
- each scan-variable string in ``plan.args`` / ``plan.kwargs`` — a
  ``Device:Variable`` pair or a scan-variable catalog name — becomes the
  namespace's Movable child, ``U_S1H.current``;
- ``trigger_profile`` and ``background`` ride as the bound plan's keyword
  argument and the run metadata; the preset name and the submission
  record ride in ``md["geecs"]`` as provenance.

The manager resolves the names against the worker namespace at submission
but does **not** refuse an unknown one (bluesky-queueserver 0.0.25 passes
an unresolved string through to the plan), so the expansion records every
reference it created (:attr:`QueueItem.references`: the detectors and the
resolved scan variables — never a literal string argument such as an enum
value) and the pre-submit preflight checks exactly those against the
manager's device tree
(:func:`~geecs_bluesky.qs_client.submit_preflight.run_submit_preflight`) —
the typo fails at preflight, not at queue-front.  Pseudo scan variables
(``kind: pseudo``) have no namespace noun yet (phase 3): expanding one is
refused here, and so is a preset whose plan is not a scan verb (``mv``
and ``run_action`` are queue items of their own — ``submit_plan("mv", …)``,
``submit_plan("run_action", ["name"])`` — never a preset).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.plan_names import GEECS_PLAN_NAMES, NON_SCAN_PLAN_NAMES
from geecs_bluesky.utils import device_reference


#: The acquisition modes a preset's plan call may name (the bound plans'
#: ``acquisition`` keyword, ``plans.registry.ACQUISITION_MODES`` — spelled
#: here too so this module stays import-light).
ACQUISITION_MODES: tuple[str, ...] = ("strict", "gated")

#: The plans a preset may name: the scan verbs.  ``mv`` and ``run_action``
#: are queue items of their own (``submit_plan("mv", ["U_S1H.current",
#: 0.0])``, ``submit_plan("run_action", ["Amp4_DUMP_HP"])``), never a
#: preset — neither takes a detector list.
PRESET_PLAN_NAMES: tuple[str, ...] = tuple(
    n for n in GEECS_PLAN_NAMES if n not in NON_SCAN_PLAN_NAMES
)


@dataclass(frozen=True)
class QueueItem:
    """A stock plan call ready for ``QueueClient.submit_plan``.

    ``references`` are the device references the expansion created (the
    detector bindings and the resolved scan variables) — what the preflight
    checks against the manager's device tree.  A literal string argument
    (an enum value in a ``list_scan`` point list, say) is never one.
    """

    name: str
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    references: list[str] = field(default_factory=list)


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
        No plan call, a plan that is not a scan verb the worker registers,
        or a pseudo scan variable.
    """
    plan = preset.plan
    if plan is None:
        raise GeecsConfigurationError(
            f"preset {preset.name!r} is a device group with no plan call — add a "
            "plan (name/args/kwargs) before submitting it"
        )
    if plan.name not in PRESET_PLAN_NAMES:
        raise GeecsConfigurationError(
            f"preset {preset.name!r} names plan {plan.name!r}; a preset runs a "
            f"scan verb: {', '.join(PRESET_PLAN_NAMES)}"
            + (
                f" ({plan.name!r} is a queue item of its own: "
                f"submit_plan({plan.name!r}, …))"
                if plan.name in NON_SCAN_PLAN_NAMES
                else ""
            )
        )
    detectors: list[str] = []
    non_essential: list[str] = []
    for d in preset.devices:
        essential = getattr(d, "essential", True)
        if essential:
            detectors.append(
                device_reference(d.device)
                if d.save_images
                else device_reference(d.device) + ".scalars"
            )
        elif not d.save_images:
            raise GeecsConfigurationError(
                f"preset {preset.name!r}: {d.device!r} is non-essential with "
                "save_images off — a non-essential device is its frame stream "
                "(a scalars-only device cannot fly); make it essential or save "
                "its images"
            )
        else:
            non_essential.append(device_reference(d.device))
    references: list[str] = [*detectors, *non_essential]
    args = [_resolve(a, catalog, references) for a in plan.args]
    kwargs = {k: _resolve(v, catalog, references) for k, v in plan.kwargs.items()}
    acquisition = kwargs.get("acquisition", "strict")
    if acquisition not in ACQUISITION_MODES:
        raise GeecsConfigurationError(
            f"preset {preset.name!r}: acquisition={acquisition!r} is not one of "
            f"{ACQUISITION_MODES}"
        )
    if non_essential:
        kwargs["non_essential"] = non_essential
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
    return QueueItem(
        name=plan.name, args=[detectors, *args], kwargs=kwargs, references=references
    )


def _resolve(
    value: Any, catalog: Mapping[str, Any] | None, references: list[str]
) -> Any:
    """Turn a scan-variable string into its namespace reference; pass the rest through.

    A string is a scan variable when it is a ``Device:Variable`` pair or a
    catalog name; the reference it becomes is recorded in *references*.
    Anything else — a number, a point list, a literal such as ``"on"`` — is
    the plan's own value.
    """
    if isinstance(value, str) and (":" in value or (catalog and value in catalog)):
        reference = scan_variable_reference(value, catalog)
        references.append(reference)
        return reference
    return value


__all__ = [
    "ACQUISITION_MODES",
    "PRESET_PLAN_NAMES",
    "QueueItem",
    "expand_preset",
    "scan_variable_reference",
]
