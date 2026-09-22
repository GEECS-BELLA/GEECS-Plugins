"""Shared utilities for GeecsBluesky — import-light (no device family)."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Callable

from geecs_core.pv_naming import normalize_component


def safe_name(s: str) -> str:
    """Convert an arbitrary string to a valid Python/ophyd-async identifier.

    Delegates to the shared naming contract
    (:func:`geecs_core.pv_naming.normalize_component` — runs of
    non-alphanumeric characters collapse to one underscore, lowercase), so a
    GEECS name mangles identically whether it becomes a PV component or an
    event-column component.  Returns ``"var"`` for strings that reduce to
    empty (an identifier, unlike a PV component, must be non-empty).
    """
    return normalize_component(s) or "var"


def identifier_name(geecs_name: str) -> str:
    """The **namespace binding** for a GEECS device name.

    The GEECS spelling is kept when it is a Python identifier (``U_S1H``) —
    what operators see in GEECS and will type into a plan argument; anything
    else goes through :func:`safe_name`.  Only the binding keeps GEECS case:
    ophyd device names, child attributes and hence event-column keys are
    :func:`safe_name` (lowercase), as ``EVENT_SCHEMA.md`` requires
    (``u_s1h-current-position``, ``uc_amp4_ir_input-meancounts``).

    Shared by the worker's namespace (the binding) and the client seam (the
    plan-argument spelling a preset expands to), so the two cannot drift.
    """
    return geecs_name if geecs_name.isidentifier() else safe_name(geecs_name)


#: The public names of a namespace device that a settable child may not
#: shadow — the ophyd-async ``Device`` / ``StandardDetector`` protocol
#: surface plus ``GeecsDetector``'s own.  A GEECS variable whose
#: :func:`safe_name` lands on one of these binds with a trailing
#: underscore (the Amp4 camera's settable enum ``trigger`` → ``trigger_``).
#: Frozen here so the client seam can spell a reference without importing
#: the device family; ``tests/test_namespace.py`` pins it to
#: ``dir(GeecsDetector)``.
RESERVED_DEVICE_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "abandon_step",
        "add_config_signals",
        "add_detector_logics",
        "add_readables",
        "children",
        "collect_asset_docs",
        "complete",
        "connect",
        "count_zeroed",
        "describe",
        "describe_collect",
        "describe_configuration",
        "discard_uncollected",
        "events_to_kickoff",
        "get_index",
        "has_file_plugin",
        "get_trigger_deadtime",
        "hints",
        "kickoff",
        "last_acq_timestamp",
        "log",
        "mark_abandoned",
        "missed_shot",
        "name",
        "native_image_save",
        "parent",
        "plugin_backed",
        "prepare",
        "read",
        "read_configuration",
        "rewind_to_step_baseline",
        "set_name",
        "stage",
        "step_baseline",
        "trigger",
        "truncate_to_quota",
        "unstage",
        "zero_count",
    }
)


def settable_attribute(variable: str) -> str:
    """The attribute a settable variable binds to on its namespace device.

    The variable's :func:`safe_name`, with a trailing underscore when that
    would shadow a device method or start with one (``trigger`` →
    ``trigger_``).  The **one** rule, shared by the namespace (the binding)
    and the client seam (the reference a preset expands to), so
    ``"UC_Amp4_IR_input:trigger"`` spells ``UC_Amp4_IR_input.trigger_`` on
    both sides.
    """
    attr = safe_name(variable)  # lowercase: event keys follow EVENT_SCHEMA.md
    if attr.startswith("_") or attr in RESERVED_DEVICE_ATTRIBUTES:
        return attr.lstrip("_") + "_"
    return attr


def device_reference(device: str, variable: str | None = None) -> str:
    """The queue-item spelling of a namespace device or one of its settables.

    ``"U_S1H"`` for a device, ``"U_S1H.current"`` for its ``Current``
    child — the dotted sub-device form the RE Manager resolves against the
    worker namespace at submission (``profile_ops._get_nspace_object``).
    The child attribute is :func:`settable_attribute`, the namespace's own
    rule; the pre-submit preflight then checks the reference against the
    manager's device tree (the manager itself passes an unknown name
    through to the plan).
    """
    base = identifier_name(device)
    return base if variable is None else f"{base}.{settable_attribute(variable)}"


def resolve_annotations(
    plan: Callable[..., Any], annotations: Mapping[str, Any]
) -> Callable[..., Any]:
    """Give *plan* a ``__signature__`` carrying resolved annotation objects.

    The queueserver manager builds a pydantic model from a registered plan's
    signature at submission, evaluating the annotations **in its own
    namespace**.  A plan defined in a module using ``from __future__ import
    annotations`` hands it strings instead of objects, and anything that is
    not a plain builtin then fails with "`Model` is not fully defined; you
    should define `Sequence`" — at ``queue add``, on hardware, with every
    unit test green (GEECS-Plugins#861).  The stock ``bluesky.plans`` verbs
    are immune only because that module does not postpone its annotations.

    So every GEECS-defined registered plan passes through here with a
    mapping of parameter name → the resolved object.  A parameter the
    mapping does not name loses its annotation entirely, which the manager
    accepts (it is what ``strict_plan`` already does for the stock varargs:
    an unannotated argument is whatever the manager resolves).

    Parameters
    ----------
    plan :
        The generator function being registered.
    annotations :
        Parameter name → annotation object (not a string).

    Returns
    -------
    callable
        *plan*, with ``__signature__`` set.
    """
    signature = inspect.signature(plan)
    unmapped = [n for n in signature.parameters if n not in annotations]
    if unmapped:
        raise ValueError(
            f"{plan.__name__}: no resolved annotation given for "
            f"{', '.join(unmapped)} — add each to the mapping (map to None to "
            "drop the annotation deliberately), or the manager loses its "
            "validation and device-name conversion for that argument"
        )
    parameters = [
        parameter.replace(
            annotation=(
                inspect.Parameter.empty
                if annotations[name] is None
                else annotations[name]
            )
        )
        for name, parameter in signature.parameters.items()
    ]
    plan.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=parameters, return_annotation=inspect.Signature.empty
    )
    return plan
