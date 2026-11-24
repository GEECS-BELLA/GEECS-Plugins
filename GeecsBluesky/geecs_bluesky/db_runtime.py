"""DB-integration runtime: what the GEECS experiment DB says a device records.

Turns the GEECS experiment DB into the providers the device namespace
builds from (``geecs_bluesky.namespace``): the **device types**,
:class:`GeecsDbDeviceTypes`.  The gateway's served set is
:func:`geecs_core.db.served_set.served_variables`, read directly by the
namespace's roster.

The **subscribed scalars** rule (``get='yes'`` — what every device reads
into its rows and what the run's background telemetry carries) is
:class:`geecs_core.db.scalar_policy.GeecsDbScalarPolicy`:
the PVA gateway's file plugin writes the same list as per-frame attributes
and depends on GEECS-Core alone, so the rule has one home there.

The **set-side** (DB scan start/end writes) is intentionally disabled: the
boundary writes would race the shot controller / TriggerProfile on the DG645,
so the reserved schema fields stay inert.  Everything here is a pure function
except the failure-tolerant ``GeecsDb`` touchpoints — a scan must never
abort because the DB blipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GeecsDbDeviceTypes:
    """Batch ``{device: devicetype}`` for one experiment — failure-tolerant.

    Wraps :meth:`GeecsDb.get_experiment_device_types` (one connection),
    cached on first use.  A DB failure degrades to an empty mapping with one
    warning, and that degradation is **not** neutral.  Two consumers read
    the map: :func:`~geecs_bluesky.namespace.looks_triggerable` falls back
    to the trigger-variable name rule when it has no devicetype, so both of
    its devicetype lists stop applying — a trigger *source* (DG645) can
    register as an acquirer, and a frame consumer with no trigger variable
    of its own (``magspecstitcher``) stops being
    triggered, which switches its native saving off.  And
    :func:`~geecs_bluesky.namespace.capture_streams` falls back to the
    one-image guess, so a device whose declared streams differ from it
    silently regresses — the FROG arms its plugin on ``SpatialImage``
    (never pushed: an arm timeout on every ``prepare``) and a MagSpec camera
    drops ``ImageInterp``.  An empty map is a degraded run to investigate,
    not a safe default.

    Parameters
    ----------
    experiment : str
        GEECS experiment name.
    enabled_only : bool
        Restrict to devices enabled in the experiment (default true).
    db : type, optional
        The ``GeecsDb`` class (injectable for tests); imported lazily by
        default.
    """

    experiment: str
    enabled_only: bool = True
    db: object | None = None
    _types: Optional[dict[str, str]] = field(default=None, init=False)

    def _geecs_db(self) -> object:
        if self.db is not None:
            return self.db
        from geecs_core.db.geecs_db import GeecsDb

        self.db = GeecsDb
        return GeecsDb

    def by_device(self) -> dict[str, str]:
        """Return ``{device: devicetype}`` (cached; empty on DB failure)."""
        if self._types is None:
            try:
                self._types = self._geecs_db().get_experiment_device_types(
                    self.experiment, enabled_only=self.enabled_only
                )
            except Exception:  # noqa: BLE001 — DB read fails open
                logger.warning(
                    "Could not read devicetypes for experiment %r; every "
                    "device falls back to the trigger-variable name rule, "
                    "which misclassifies trigger sources and frame consumers "
                    "(#934), and every plugin-backed camera falls back to the "
                    "one-image capture guess instead of its declared streams "
                    "(the FROG then arms on SpatialImage and times out) — "
                    "treat this run as degraded",
                    self.experiment,
                    exc_info=True,
                )
                self._types = {}
        return self._types
