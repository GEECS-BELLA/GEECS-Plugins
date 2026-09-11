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


def device_reference(device: str, variable: str | None = None) -> str:
    """The queue-item spelling of a namespace device or one of its settables.

    ``"U_S1H"`` for a device, ``"U_S1H.current"`` for its ``Current``
    child — the dotted sub-device form the RE Manager resolves against the
    worker namespace at submission (``profile_ops._get_nspace_object``).
    The child attribute is the variable's :func:`safe_name`; the namespace
    appends a trailing underscore only when that collides with a detector
    method (``trigger`` → ``trigger_``), a case the pre-submit preflight
    then refuses against the manager's device tree (the manager itself
    passes an unknown name through to the plan).
    """
    base = identifier_name(device)
    return base if variable is None else f"{base}.{safe_name(variable)}"
