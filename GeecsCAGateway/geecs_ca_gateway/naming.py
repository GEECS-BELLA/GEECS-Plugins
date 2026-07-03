"""GEECS name → EPICS Channel Access PV name mapping.

GEECS raw names are close to EPICS PV naming, which uses ``:`` as its hierarchy
separator — but LabVIEW tolerates characters CA does not.  Within a single name
*component* only ``[A-Za-z0-9_]`` is kept; every other character is mapped to
``_``.  The most important case is the **dot**: EPICS reads ``.`` as the
record/field separator, so ``Trigger.Source`` must become ``Trigger_Source`` or
clients would parse ``.Source`` as a field.

The mapping is deliberately **lossy** (``Trigger.Source`` and ``Trigger Source``
both collapse to ``Trigger_Source``).  Do not reverse-engineer GEECS names from
PV strings — the gateway keeps the authoritative ``geecs_var ↔ PV`` map and
publishes a manifest.  Collisions are caught at pvdb-build time.

Full PV names (``[Experiment:]Device:Variable``) are assembled by
:meth:`geecs_ca_gateway.config.DeviceSpec.pv_name_for`; this module only
normalizes individual name components.
"""

from __future__ import annotations

import re

# Any run of characters outside the CA-safe component set becomes one underscore.
_INVALID = re.compile(r"[^A-Za-z0-9_]+")


def normalize_pv_component(name: str) -> str:
    """Return a Channel-Access-safe version of a single name component.

    Maps every run of characters outside ``[A-Za-z0-9_]`` (spaces, dots, dashes,
    parentheses, …) to a single underscore and strips leading/trailing
    underscores.  Does not touch ``:`` at the caller level — it is the reserved
    device/variable separator applied by
    :meth:`geecs_ca_gateway.config.DeviceSpec.pv_name_for`, never inside a
    component.

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
