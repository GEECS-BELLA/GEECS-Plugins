"""DB-integration runtime (M3c): the get-side two-tier recording model, wired.

Turns the GEECS experiment DB's per-experiment variable policy
(``expt_device_variable``) into the two **get-side** runtime capabilities the
schema describes (the ``SaveSetEntry`` runtime contract in
:mod:`geecs_schemas.save_set`):

1. **db_scalars resolution** (Tier 1 recorded scalars) —
   :func:`resolve_entry_scalars`.
2. **Background telemetry selection** (Tier 2) —
   :func:`select_telemetry_variables`; the soft read lives in
   :class:`~geecs_bluesky.devices.ca.telemetry.CaTelemetryReadable`.

The **set-side** (DB scan start/end writes) is intentionally disabled: the
boundary writes would race the shot controller / TriggerProfile on the DG645,
so the reserved schema fields stay inert.  Everything here is a pure function
except :class:`GeecsDbScalarPolicy`, the one failure-tolerant place touching
``GeecsDb`` — a scan must never abort because the DB blipped.  Design
rationale: ``GeecsBluesky/CLAUDE.md`` (M3c).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from geecs_schemas import SaveSet

logger = logging.getLogger(__name__)


@runtime_checkable
class ScalarPolicyProvider(Protocol):
    """Supplies per-device DB variable policy for one experiment.

    The seam between the pure resolution logic and
    :class:`~geecs_ca_gateway.db.geecs_db.GeecsDb`.  Every method returns an
    empty result rather than raising when the DB is unavailable or a device
    is uncurated.  This covers only the **get-side** (subscribed ``get='yes'``
    variables + all-variables queries); the set-side is disabled (see the
    module docstring).
    """

    def get_variables(self, device: str) -> list[str]:
        """Return the device's ``get='yes'`` variables (may be empty)."""
        ...

    def all_variables(self, device: str) -> list[str]:
        """Return every variable the experiment tracks for *device* (may be empty)."""
        ...

    def subscribed_by_device(self) -> dict[str, list[str]]:
        """Return ``{device: [get='yes' vars]}`` for the whole experiment."""
        ...


@dataclass
class GeecsDbScalarPolicy:
    """DB-backed :class:`ScalarPolicyProvider`, one batched query per kind.

    Wraps :class:`~geecs_ca_gateway.db.geecs_db.GeecsDb` for one experiment,
    caching each of its two get-side whole-experiment queries on first use.
    Every query is wrapped so a DB failure (off the lab network, a missing
    table, an uncurated experiment) degrades to empty policy with a single
    warning — a scan must never abort because the DB was briefly unreachable.

    Parameters
    ----------
    experiment : str
        GEECS experiment name.
    enabled_only : bool
        Restrict to devices enabled in the experiment (default true).
    db : type, optional
        The ``GeecsDb`` class (injectable for tests); imported lazily by
        default so this module has no hard dependency on the ``ca`` DB stack.
    """

    experiment: str
    enabled_only: bool = True
    db: object | None = None
    _subscribed: Optional[dict[str, list[str]]] = field(default=None, init=False)
    _all: Optional[dict[str, list[str]]] = field(default=None, init=False)

    def _geecs_db(self) -> object:
        if self.db is not None:
            return self.db
        from geecs_ca_gateway.db.geecs_db import GeecsDb

        self.db = GeecsDb
        return GeecsDb

    def subscribed_by_device(self) -> dict[str, list[str]]:
        """Return ``{device: [get='yes' vars]}`` (cached; empty on DB failure)."""
        if self._subscribed is None:
            try:
                self._subscribed = self._geecs_db().get_subscribed_variables(
                    self.experiment, enabled_only=self.enabled_only
                )
            except Exception:
                logger.warning(
                    "Could not read get='yes' variables for experiment %r; "
                    "db_scalars and background telemetry will use no DB rows",
                    self.experiment,
                    exc_info=True,
                )
                self._subscribed = {}
        return self._subscribed

    def _all_by_device(self) -> dict[str, list[str]]:
        if self._all is None:
            try:
                self._all = self._geecs_db().get_all_experiment_variables(
                    self.experiment, enabled_only=self.enabled_only
                )
            except Exception:
                logger.warning(
                    "Could not read all variables for experiment %r; "
                    "all_scalars entries will fall back to get='yes'/explicit",
                    self.experiment,
                    exc_info=True,
                )
                self._all = {}
        return self._all

    def get_variables(self, device: str) -> list[str]:
        """Return *device*'s ``get='yes'`` variables (empty if uncurated)."""
        return list(self.subscribed_by_device().get(device, []))

    def all_variables(self, device: str) -> list[str]:
        """Return every tracked variable for *device* (empty if uncurated)."""
        return list(self._all_by_device().get(device, []))


def resolve_entry_scalars(
    device: str,
    explicit: list[str],
    *,
    db_scalars: bool,
    all_scalars: bool,
    provider: ScalarPolicyProvider | None,
) -> list[str]:
    """Resolve the recorded scalar list for one save-set entry.

    ``db_scalars=False`` → only the explicit list; ``db_scalars=True`` → the
    DB ``get='yes'`` variables ∪ the explicit list (``all_scalars=True``
    widens the DB side to every DB variable).  Order is stable: DB variables
    first, then any explicit variable not already present.  With no
    *provider* (no DB access) only the explicit list is recorded — the DB
    tier is strictly additive.

    Parameters
    ----------
    device : str
        GEECS device name (for the DB lookup).
    explicit : list of str
        The entry's explicit ``scalars`` list.
    db_scalars : bool
        The entry's ``db_scalars`` flag.
    all_scalars : bool
        The entry's ``all_scalars`` flag.
    provider : ScalarPolicyProvider or None
        Where DB variables come from; ``None`` means no DB contribution.

    Returns
    -------
    list of str
        The resolved recorded scalar list, in stable order.
    """
    if not db_scalars:
        return list(explicit)
    db_vars: list[str] = []
    if provider is not None:
        db_vars = (
            provider.all_variables(device)
            if all_scalars
            else provider.get_variables(device)
        )
    ordered: list[str] = []
    seen: set[str] = set()
    for var in list(db_vars) + list(explicit):
        if var not in seen:
            seen.add(var)
            ordered.append(var)
    return ordered


def select_telemetry_variables(
    save_set: SaveSet | None,
    subscribed_by_device: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Select the Tier-2 background-telemetry ``{device: [variables]}``.

    Every experiment device with a ``get='yes'`` variable that is **not** in
    the save set becomes telemetry.  A device already in the save set is
    excluded wholesale (its data is Tier-1, with guarantees) — telemetry never
    duplicates a required device's columns.

    Parameters
    ----------
    save_set : SaveSet or None
        The scan's save set (its entry devices are the Tier-1 set); ``None``
        means no required devices, so every subscribed device is telemetry.
    subscribed_by_device : dict
        ``{device: [get='yes' variables]}`` for the whole experiment.

    Returns
    -------
    dict
        ``{device: [variables]}`` for telemetry, save-set devices removed,
        empty-variable devices dropped.
    """
    required = set()
    if save_set is not None:
        required = {entry.device for entry in save_set.entries}
    selected: dict[str, list[str]] = {}
    for device, variables in subscribed_by_device.items():
        if device in required:
            continue
        variables = [v for v in variables if v]
        if variables:
            selected[device] = list(variables)
    return selected
