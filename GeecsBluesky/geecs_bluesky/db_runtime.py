"""DB-integration runtime: what the GEECS experiment DB says a device records.

Turns the GEECS experiment DB's per-experiment variable policy
(``expt_device_variable``) into the providers the device namespace builds
from (``geecs_bluesky.namespace``):

1. **Served-set resolution** — :class:`GeecsDbServedSetProvider` (the
   gateway serves ``get='yes'`` union settable variables of enabled
   devices; anything else has no PV).
2. **Device types** — :class:`GeecsDbDeviceTypes`.

The **subscribed scalars** rule (``get='yes'`` — what every device reads
into its rows and what the run's baseline telemetry carries) is
:class:`geecs_core.db.scalar_policy.GeecsDbScalarPolicy` since 2026-09-12:
the PVA gateway's file plugin writes the same list as per-frame attributes
and depends on GEECS-Core alone, so the rule has one home there.

The **set-side** (DB scan start/end writes) is intentionally disabled: the
boundary writes would race the shot controller / TriggerProfile on the DG645,
so the reserved schema fields stay inert.  Everything here is a pure function
except the failure-tolerant ``GeecsDb`` touchpoints — a scan must never
abort because the DB blipped.  Design rationale: ``GeecsBluesky/CLAUDE.md``
(M3c).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional


logger = logging.getLogger(__name__)


#: Variables the gateway synthesizes for EVERY device, independent of the
#: DB ``get`` flags: the timestamp ladder (``DeviceSpec.timestamp_vars`` in
#: GeecsCAGateway's ``config.py`` — ``acq_timestamp`` preferred,
#: ``systimestamp`` fallback, both always subscribed) and the per-device
#: ``CONNECTED`` status PV (``gateway.py``). These are served even though no
#: ``expt_device_variable`` row exists, so the unserved-variables check must
#: treat them as always-served (field regression 2026-07-16: the optimizer's
#: auto-provisioned ``acq_timestamp`` request drew a false "not served"
#: dialog threatening to drop the whole device).
GATEWAY_SYNTHESIZED_VARIABLES = frozenset(
    {"acq_timestamp", "systimestamp", "CONNECTED"}
)


@dataclass
class GeecsDbServedSetProvider:
    """The gateway's served variable set, per device — failure-tolerant.

    The gateway serves each enabled device's ``get='yes'`` variables plus its
    settable *control surface* (``GeecsCAGateway/DEPLOYMENT.md``,
    "subscribed_only semantics") — a variable outside that union has no PV
    at all, so a detector signal on it can never connect.  This provider
    computes that union from two batched DB queries, cached on first use.

    Failure semantics differ from
    :class:`~geecs_core.db.scalar_policy.GeecsDbScalarPolicy` deliberately:
    the served set drives a *check*, so a DB failure must read as "unknown"
    (``None`` — the check is skipped with one warning), never as "empty"
    (which would condemn every variable as unserved and dialog the operator
    for a DB blip).  A scan never aborts because the DB was unreachable.

    Parameters
    ----------
    experiment : str
        GEECS experiment name.
    enabled_only : bool
        Restrict to devices enabled in the experiment (default true) —
        matching the gateway's own config builder.
    db : type, optional
        The ``GeecsDb`` class (injectable for tests); imported lazily by
        default so this module has no hard dependency on the ``ca`` DB stack.
    """

    experiment: str
    enabled_only: bool = True
    db: object | None = None
    _served: Optional[dict[str, set[str]]] = field(default=None, init=False)
    _attempted: bool = field(default=False, init=False)

    def _geecs_db(self) -> object:
        if self.db is not None:
            return self.db
        from geecs_core.db.geecs_db import GeecsDb

        self.db = GeecsDb
        return GeecsDb

    def served_by_device(self) -> dict[str, set[str]] | None:
        """Return ``{device: {served variables}}``, or ``None`` on DB failure.

        Returns
        -------
        dict or None
            The gateway's served set (``get='yes'`` union settable) for
            every enabled device, or ``None`` when the DB could not be read
            — callers must then skip the unserved-variables check (degrade
            to pass with a warning, never block a scan on a DB blip).
        """
        if not self._attempted:
            self._attempted = True
            try:
                db = self._geecs_db()
                subscribed = db.get_subscribed_variables(
                    self.experiment, enabled_only=self.enabled_only
                )
                metadata = db.get_experiment_device_variables(
                    self.experiment, enabled_only=self.enabled_only
                )
            except Exception:
                logger.warning(
                    "Could not read the gateway served set for experiment %r; "
                    "the unserved-variables pre-flight check will be skipped",
                    self.experiment,
                    exc_info=True,
                )
                return None
            served: dict[str, set[str]] = {
                device: set(variables) for device, variables in subscribed.items()
            }
            for device, rows in metadata.items():
                settable = {
                    row["name"] for row in rows if bool(row.get("settable", False))
                }
                if settable:
                    served.setdefault(device, set()).update(settable)
            self._served = served
        return self._served


@dataclass
class GeecsDbDeviceTypes:
    """Batch ``{device: devicetype}`` for one experiment — failure-tolerant.

    Wraps :meth:`GeecsDb.get_experiment_device_types` (one connection),
    cached on first use.  A DB failure degrades to an empty mapping with one
    warning — consumers treating "unknown devicetype" as "not
    capture-eligible" therefore fail open: native image saving is never
    switched off because the DB blipped.

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
            except Exception:
                logger.warning(
                    "Could not read devicetypes for experiment %r; no device "
                    "is capture-eligible this run (native saving unaffected)",
                    self.experiment,
                    exc_info=True,
                )
                self._types = {}
        return self._types
