"""Shared utilities for GeecsBluesky — import-light (no device family)."""

from __future__ import annotations

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
        "describe",
        "describe_collect",
        "describe_configuration",
        "discard_uncollected",
        "events_to_kickoff",
        "get_index",
        "get_trigger_deadtime",
        "hints",
        "kickoff",
        "last_acq_timestamp",
        "log",
        "mark_abandoned",
        "missed_shot",
        "name",
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
