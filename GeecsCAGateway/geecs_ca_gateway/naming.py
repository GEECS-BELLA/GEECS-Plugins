"""GEECS name → EPICS Channel Access PV name mapping.

GEECS raw names are already ``deviceName:variable`` — nearly 1:1 with EPICS PV
naming, which uses ``:`` as its hierarchy separator.  The only routine fix-up is
whitespace: GEECS variable names may contain spaces, which CA names may not.
Units live in the GEECS attributes database, not the name, so no unit-stripping
is attempted here.
"""

from __future__ import annotations

import re

_WHITESPACE = re.compile(r"\s+")


def normalize_pv_component(name: str) -> str:
    """Return a Channel-Access-safe version of a single name component.

    Collapses internal whitespace runs to single underscores and strips leading
    and trailing whitespace.  Does not touch ``:`` — GEECS already uses it as the
    device/variable separator, matching EPICS hierarchy convention.

    Parameters
    ----------
    name : str
        Raw GEECS device name or variable name.

    Returns
    -------
    str
        A whitespace-free name safe to use as a CA PV component.
    """
    return _WHITESPACE.sub("_", name.strip())


def pv_name(prefix: str, suffix: str) -> str:
    """Join a device prefix and a variable suffix into a full PV name.

    Parameters
    ----------
    prefix : str
        Device-level PV prefix (normalized here).
    suffix : str
        Variable-level PV suffix (assumed already normalized).

    Returns
    -------
    str
        The full ``prefix:suffix`` PV name.
    """
    return f"{normalize_pv_component(prefix)}:{suffix}"
