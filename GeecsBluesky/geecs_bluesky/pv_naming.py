"""Shared GEECS-name → EPICS Channel Access PV-name policy.

This is the naming **contract** between the caproto gateway (which publishes
GEECS device variables as PVs) and the CA-backed ophyd-async devices (which
consume them).  Both sides must agree on it exactly, so it lives here — the one
module both import — rather than being duplicated where it could drift.

GEECS raw names are close to EPICS PV naming, which uses ``:`` as its hierarchy
separator, but LabVIEW tolerates characters CA does not.  Within a single name
*component* only ``[A-Za-z0-9_]`` is kept; every other run of characters maps to
one ``_``.  The most important case is the **dot**: EPICS reads ``.`` as the
record/field separator, so ``Trigger.Source`` must become ``Trigger_Source`` or
clients would parse ``.Source`` as a field.

The mapping is deliberately **lossy** (``Trigger.Source`` and ``Trigger Source``
both collapse to ``Trigger_Source``); the gateway keeps the authoritative
``geecs_var ↔ PV`` manifest, so never reverse-engineer a GEECS name from a PV.
"""

from __future__ import annotations

import re

# Any run of characters outside the CA-safe component set becomes one underscore.
_INVALID = re.compile(r"[^A-Za-z0-9_]+")


def normalize_component(name: str) -> str:
    """Return a Channel-Access-safe version of a single name component.

    Maps every run of characters outside ``[A-Za-z0-9_]`` (spaces, dots, dashes,
    parentheses, …) to a single underscore and strips leading/trailing
    underscores.  Does not touch ``:`` — that is the reserved namespace separator
    applied by :func:`pv_name`, never inside a component.

    Parameters
    ----------
    name : str
        Raw GEECS device name or variable name.

    Returns
    -------
    str
        A CA-safe name component.
    """
    return _INVALID.sub("_", name.strip()).strip("_")


def pv_name(*parts: str) -> str:
    """Join namespace *parts* into a full PV name, normalizing each component.

    Falsy parts (``None``/``""``) are skipped, so an absent experiment prefix
    simply drops out.  The ``:`` separator is reserved and applied here.

    Parameters
    ----------
    *parts : str
        Ordered namespace components, e.g. ``experiment, device, variable``.

    Returns
    -------
    str
        The full ``a:b:c`` PV name.
    """
    return ":".join(normalize_component(part) for part in parts if part)
