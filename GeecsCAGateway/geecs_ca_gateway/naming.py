"""GEECS name → EPICS Channel Access PV name mapping.

The normalization policy is shared with the CA-backed ophyd-async devices that
consume these PVs — it lives in :mod:`geecs_bluesky.pv_naming` (the one module
both the gateway *producer* and the device *consumer* import) so the two can
never drift.  This module re-exports it under the gateway's local name.

Full PV names (``[Experiment:]Device:Variable``) are assembled by
:meth:`geecs_ca_gateway.config.DeviceSpec.pv_name_for`; the shared policy only
normalizes individual name components (runs of non-``[A-Za-z0-9_]`` → ``_`` — the
dot is critical: EPICS reads ``.`` as the record/field separator, so
``Trigger.Source`` → ``Trigger_Source``).  The mapping is deliberately lossy;
the gateway keeps the authoritative ``geecs_var ↔ PV`` manifest.
"""

from __future__ import annotations

from geecs_bluesky.pv_naming import normalize_component as normalize_pv_component

__all__ = ["normalize_pv_component"]
