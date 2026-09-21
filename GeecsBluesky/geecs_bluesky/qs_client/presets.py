"""Expand a preset into the queue item it stands for — client-side, import-light.

A :class:`geecs_schemas.Preset` is a saved scan: the device group plus the
plan call.  The worker registers stock plans over namespace devices by
**name** (:data:`~geecs_bluesky.plan_names.GEECS_PLAN_NAMES`), so
submission is a translation of names, nothing more:

- each device of the group becomes its namespace binding —
  ``UC_Amp4_IR_input``, or ``UC_Amp4_IR_input.scalars`` when
  ``save_images`` is off (the scalars-only view every namespace device
  carries: on a detector the shot wait without the files, on a
  scalar-only device what the device reads); an ``essential: false``
  device goes to the bound plan's ``non_essential`` list instead (phase
  2 — streamed for the run, never waited on;
  it needs its frames, so ``save_images: false`` there is refused);
- ``acquisition`` (``strict`` / ``gated``) and ``shot_period`` ride in
  ``plan.kwargs`` like ``shots_per_step`` does;
- each scan-variable string in ``plan.args`` / ``plan.kwargs`` — a
  ``Device:Variable`` pair or a scan-variable catalog name — becomes the
  namespace's Movable child, ``U_S1H.current``;
- ``trigger_profile`` and ``background`` ride as the bound plan's keyword
  argument and the run metadata; so does ``native_image_save`` — only
  when the preset sets it, since unset means the experiment default the
  worker reads at every scan (PNG retirement, #738), and a copy of it in
  ``plan.kwargs`` is refused (the preset field is the one source of
  truth); the preset name and the submission record ride in
  ``md["geecs"]`` as provenance.

The manager resolves the names against the worker namespace at submission
but does **not** refuse an unknown one (bluesky-queueserver 0.0.25 passes
an unresolved string through to the plan), so the expansion records every
reference it created (:attr:`QueueItem.references`: the detectors and the
resolved scan variables — never a literal string argument such as an enum
value) and the pre-submit preflight checks exactly those against the
manager's device tree
(:func:`~geecs_bluesky.qs_client.submit_preflight.run_submit_preflight`) —
the typo fails at preflight, not at queue-front.  A pseudo scan variable
(``kind: pseudo``) is a namespace noun of its own under its catalog name
(``GeecsNamespace.add_pseudos``), so it expands to that binding —
``ALine_e_beam_angle_offset_x`` — and the preflight checks it like any
device.  A preset whose plan is not a scan verb is refused (``mv`` and
``run_action`` are queue items of their own — ``submit_plan("mv", …)``,
``submit_plan("run_action", ["name"])`` — never a preset).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from geecs_schemas import Preset
    from geecs_bluesky.config_resolver import ConfigsRepoResolver

from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.plan_names import (
    ACQUISITION_MODES,
    GEECS_PLAN_NAMES,
    NON_SCAN_PLAN_NAMES,
)
from geecs_bluesky.utils import device_reference, identifier_name


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
    ``devices`` contains the final device group, including optimizer-required
    devices, for gateway liveness checks.
    """

    name: str
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    references: list[str] = field(default_factory=list)
    devices: tuple[str, ...] = ()


def scan_variable_reference(
    target: str, catalog: Mapping[str, Any] | None = None
) -> str:
    """The queue-item spelling of a scan variable.

    *target* is ``"Device:Variable"`` or a name from the experiment's
    scan-variable catalog (``catalog``: name → ``ScanVariableSpec``); a
    plain device name passes through as the device itself.  A catalog
    pseudo is its own namespace noun: the reference is the catalog name
    as an identifier (the namespace's :func:`identifier_name` rule).
    """
    spec = (catalog or {}).get(target)
    if spec is not None:
        if getattr(spec, "kind", None) == "pseudo":
            return identifier_name(target)
        target = str(spec.target)
    device, sep, variable = target.partition(":")
    return device_reference(device, variable if sep else None)


def merge_required_devices(preset: Preset, required_devices: frozenset[str]) -> Preset:
    """Copy a preset with optimizer-required devices saved and essential."""
    from geecs_schemas import PresetDevice

    if preset.plan is None or preset.plan.name != "optimize":
        return preset
    required = {name.casefold(): name for name in required_devices}
    devices = [
        d.model_copy(update={"essential": True, "save_images": True})
        if d.device.casefold() in required
        else d
        for d in preset.devices
    ]
    present = {d.device.casefold() for d in devices}
    devices.extend(
        PresetDevice(device=name, essential=True, save_images=True)
        for key, name in sorted(required.items())
        if key not in present
    )
    return preset.model_copy(update={"devices": devices})


def prepare_optimizer_preset(preset: Preset, resolver: ConfigsRepoResolver) -> Preset:
    """Resolve optimizer defaults and merge its devices before client preflight."""
    from geecs_schemas import optimizer_required_devices
    from geecs_schemas.optimizer_config import DiagnosticMeasurement

    if preset.plan is None or preset.plan.name != "optimize":
        return preset
    kwargs = dict(preset.plan.kwargs)
    name = kwargs.get("optimizer_config")
    if not isinstance(name, str) or not name:
        raise GeecsConfigurationError("choose an optimizer config")
    cfg = resolver.resolve_optimizer_config(name)
    diagnostic_devices = {
        m.diagnostic: resolver.diagnostic_device(m.diagnostic)
        for m in cfg.measurements.values()
        if isinstance(m, DiagnosticMeasurement)
    }
    preset = merge_required_devices(
        preset, optimizer_required_devices(cfg, diagnostic_devices)
    )
    if kwargs.pop("acquisition", "strict") != "strict":
        raise GeecsConfigurationError("optimization requires strict acquisition")
    kwargs.setdefault("shots_per_step", cfg.run.shots_per_step)
    kwargs.setdefault("max_iterations", cfg.run.max_iterations)
    for key in ("shots_per_step", "max_iterations"):
        value = kwargs[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise GeecsConfigurationError(f"{key} must be a positive integer")
    return preset.model_copy(
        update={"plan": preset.plan.model_copy(update={"kwargs": kwargs})}
    )


def expand_preset(
    preset: Any,
    *,
    catalog: Mapping[str, Any] | None = None,
    md: Mapping[str, Any] | None = None,
    required_devices: frozenset[str] = frozenset(),
    resolver: ConfigsRepoResolver | None = None,
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
    resolver : ConfigsRepoResolver, optional
        Required for optimize presets; resolves defaults and required devices.

    Raises
    ------
    GeecsConfigurationError
        No plan call, or a plan that is not a scan verb the worker registers.
    """
    if preset.plan is not None and preset.plan.name == "optimize":
        if resolver is None:
            raise GeecsConfigurationError(
                "an optimize preset requires a configs resolver"
            )
        preset = prepare_optimizer_preset(preset, resolver)
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
    devices = merge_required_devices(preset, required_devices).devices
    for d in devices:
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
    if plan.name == "sweep":
        from geecs_schemas import Sweep
        from pydantic import ValidationError

        if args:
            raise GeecsConfigurationError("sweep takes its trajectory in kwargs.sweep")
        try:
            payload = Sweep.model_validate(kwargs.get("sweep"))
            for axis in payload.axis_references():
                if (
                    ":" in axis.axis
                    or (catalog and axis.axis in catalog)
                    or "." not in axis.axis
                ):
                    axis.axis = scan_variable_reference(axis.axis, catalog)
                references.append(axis.axis)
            # Two catalog entries may resolve to one axis.
            kwargs["sweep"] = Sweep.model_validate(payload.model_dump()).model_dump(
                mode="json"
            )
        except ValidationError as exc:
            raise GeecsConfigurationError(
                f"preset {preset.name!r}: invalid sweep: {exc}"
            ) from exc
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
    # The preset field is the one source of truth for the LabVIEW-files
    # switch: a copy in plan.kwargs would silently win a setdefault (Codex
    # review of #944), so it is refused rather than merged.
    if "native_image_save" in kwargs:
        raise GeecsConfigurationError(
            f"preset {preset.name!r}: native_image_save is a preset field, not a "
            "plan keyword — set it at the top level of the preset (the scanner's "
            "'LabVIEW files' control) and drop it from plan.kwargs"
        )
    if preset.native_image_save is not None:
        kwargs["native_image_save"] = preset.native_image_save
    run_md: dict[str, Any] = dict(kwargs.pop("md", None) or {})
    run_md.update(md or {})
    run_md.setdefault("description", preset.description)
    run_md.setdefault("background", preset.background)
    geecs = dict(run_md.get("geecs") or {})
    geecs.setdefault("preset", preset.name)
    run_md["geecs"] = geecs
    kwargs["md"] = run_md
    return QueueItem(
        name=plan.name,
        args=[detectors, *args],
        kwargs=kwargs,
        references=references,
        devices=tuple(d.device for d in devices),
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
