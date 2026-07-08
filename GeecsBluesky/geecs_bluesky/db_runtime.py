"""DB-integration runtime (M3c): the two-tier recording model, wired.

This module turns the GEECS experiment database's per-experiment
device-variable policy (MySQL table ``expt_device_variable``) into the three
runtime capabilities the schema already describes (see the ``SaveSetEntry``
runtime contract in :mod:`geecs_schemas.save_set`, and
:class:`~geecs_schemas.experiment_defaults.ExperimentDefaults`):

1. **db_scalars resolution** (Tier 1 recorded scalars).  For a save-set entry
   with ``db_scalars=True`` (the default), the scalars recorded for the device
   are its DB ``get='yes'`` variables **unioned** with the entry's explicit
   ``scalars`` list; ``all_scalars=True`` unions *every* DB variable for the
   device instead of just the ``get='yes'`` subset; ``db_scalars=False`` (what
   the legacy converter pins) records only the explicit ``scalars``.

2. **Scan start/end DB writes** (participants-only).  For devices
   **participating** in the scan (save-set devices + scan-variable devices —
   not every experiment device), the ``set='yes'`` rows' ``startvalue`` is
   written at scan start and ``endvalue`` at scan end, with the entry's
   ``at_scan_start`` / ``at_scan_end`` overrides layered on top (a value
   replaces the DB value, an explicit ``None`` suppresses the write, absence
   keeps the DB value).  ``save`` / ``localsavingpath`` are always skipped
   (native saving is owned by the run discipline).

3. **Background telemetry tier** (Tier 2).  Every live experiment device with
   a ``get='yes'`` variable that is *not* in the save set is recorded as
   best-effort snapshot columns — read-only, never waited on, a dead device
   dropped with a log line.  Selection of which ``(device, variables)`` become
   telemetry is pure here; the soft read lives in
   :class:`~geecs_bluesky.devices.ca.telemetry.CaTelemetryReadable`.

Everything in this module is a **pure function** except the thin
:class:`GeecsDbScalarPolicy` provider, which is the one place that touches
:class:`~geecs_ca_gateway.db.geecs_db.GeecsDb`.  The provider caches its three
queries per experiment and **tolerates a missing/incompletely-curated DB**
(the maintainer's known curation caveat): a lookup failure logs and yields
empty policy rather than aborting the scan.  Keeping the policy separable is
what makes the resolution logic testable with no MySQL access.

The engine never talks to the DB at *write* time: the DB only supplies which
variables and what values; the writes themselves ride the same CA ``:SP``
setpoint path that action plans use (see
:func:`~geecs_bluesky.scan_request_runner.make_boundary_write_plan`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from geecs_schemas import SaveSet

logger = logging.getLogger(__name__)

#: Variables that are never DB-driven scan-boundary writes — native saving is
#: owned by the run discipline, not per-device ``expt_device_variable`` rows.
SUPPRESSED_BOUNDARY_VARIABLES = frozenset({"save", "localsavingpath"})


@dataclass(frozen=True)
class BoundaryWrite:
    """One resolved scan-boundary setpoint write.

    Attributes
    ----------
    device : str
        GEECS device name.
    variable : str
        Variable to write (a ``set='yes'`` row's variable).
    value : str
        The wire-string value to send (the DB startvalue/endvalue, or an
        override).
    source : str
        Provenance tag: ``"db"`` (from the row) or ``"override"`` (replaced by
        an entry's ``at_scan_start`` / ``at_scan_end``).
    """

    device: str
    variable: str
    value: str
    source: str = "db"


@runtime_checkable
class ScalarPolicyProvider(Protocol):
    """Supplies per-device DB variable policy for one experiment.

    The seam between the pure resolution logic and
    :class:`~geecs_ca_gateway.db.geecs_db.GeecsDb`.  Every method returns an
    empty result rather than raising when the DB is unavailable or a device
    is uncurated.
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

    def boundary_writes(self, device: str) -> list[dict]:
        """Return the device's ``set='yes'`` start/end rows (may be empty).

        Each row is ``{"variable", "startvalue", "endvalue"}`` with string or
        ``None`` values (as :meth:`GeecsDb.get_scan_boundary_writes` returns).
        """
        ...


@dataclass
class GeecsDbScalarPolicy:
    """DB-backed :class:`ScalarPolicyProvider`, one batched query per kind.

    Wraps :class:`~geecs_ca_gateway.db.geecs_db.GeecsDb` for one experiment,
    caching each of its three whole-experiment queries on first use.  Every
    query is wrapped so a DB failure (off the lab network, a missing table, an
    uncurated experiment) degrades to empty policy with a single warning — a
    scan must never abort because the DB was briefly unreachable.

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
    _boundary: Optional[dict[str, list[dict]]] = field(default=None, init=False)

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

    def _boundary_by_device(self) -> dict[str, list[dict]]:
        if self._boundary is None:
            try:
                self._boundary = self._geecs_db().get_scan_boundary_writes(
                    self.experiment, enabled_only=self.enabled_only
                )
            except Exception:
                logger.warning(
                    "Could not read set='yes' scan boundary writes for "
                    "experiment %r; no DB-driven start/end writes will be applied",
                    self.experiment,
                    exc_info=True,
                )
                self._boundary = {}
        return self._boundary

    def get_variables(self, device: str) -> list[str]:
        """Return *device*'s ``get='yes'`` variables (empty if uncurated)."""
        return list(self.subscribed_by_device().get(device, []))

    def all_variables(self, device: str) -> list[str]:
        """Return every tracked variable for *device* (empty if uncurated)."""
        return list(self._all_by_device().get(device, []))

    def boundary_writes(self, device: str) -> list[dict]:
        """Return *device*'s ``set='yes'`` start/end rows (empty if none)."""
        return list(self._boundary_by_device().get(device, []))


def resolve_entry_scalars(
    device: str,
    explicit: list[str],
    *,
    db_scalars: bool,
    all_scalars: bool,
    provider: ScalarPolicyProvider | None,
) -> list[str]:
    """Resolve the recorded scalar list for one save-set entry.

    The rules (from the ``SaveSetEntry`` docstring):

    - ``db_scalars=False`` → only the explicit list (the legacy converter pins
      this, preserving each converted element's exact behavior).
    - ``db_scalars=True``, ``all_scalars=False`` → the DB ``get='yes'``
      variables ∪ the explicit list.
    - ``db_scalars=True``, ``all_scalars=True`` → every DB variable for the
      device ∪ the explicit list.

    The union preserves order: the DB variables first (in DB order), then any
    explicit variable not already present.  When no *provider* is available (no
    DB access, the GUI-bridge path) the DB contribution is empty and only the
    explicit list is recorded — the same shape M3b produced, so the DB tier is
    strictly additive.

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


def resolve_boundary_writes(
    device: str,
    rows: list[dict],
    *,
    which: str,
    overrides: dict[str, Optional[str]],
) -> list[BoundaryWrite]:
    """Resolve one device's start or end setpoint writes (DB + overrides).

    Applies the three override cases documented on ``SaveSetEntry``:

    - a variable absent from *overrides* uses the DB value (the row's
      ``startvalue`` / ``endvalue``);
    - a variable mapped to a non-``None`` value replaces the DB value;
    - a variable mapped to ``None`` suppresses the write entirely.

    A DB row whose value is null and which no override replaces contributes no
    write (there is nothing to send).  ``save`` / ``localsavingpath`` rows are
    always dropped.  Overrides naming a variable with no DB row are honored as
    forced writes (``source="override"``) so an entry can add a boundary write
    the DB did not curate — except a ``None`` override for such a variable,
    which is a no-op.

    Parameters
    ----------
    device : str
        GEECS device name.
    rows : list of dict
        The device's ``set='yes'`` rows
        (``{"variable", "startvalue", "endvalue"}``).
    which : str
        ``"start"`` or ``"end"`` — selects the row column.
    overrides : dict
        The entry's ``at_scan_start`` or ``at_scan_end`` mapping.

    Returns
    -------
    list of BoundaryWrite
        Ordered writes to apply (DB row order; forced overrides appended).
    """
    if which not in ("start", "end"):
        raise ValueError(f"which={which!r} must be 'start' or 'end'")
    column = "startvalue" if which == "start" else "endvalue"
    overrides = overrides or {}
    writes: list[BoundaryWrite] = []
    handled: set[str] = set()
    for row in rows:
        variable = row["variable"]
        if variable in SUPPRESSED_BOUNDARY_VARIABLES:
            continue
        handled.add(variable)
        if variable in overrides:
            override = overrides[variable]
            if override is None:
                logger.debug(
                    "scan %s write for %s:%s suppressed by override",
                    which,
                    device,
                    variable,
                )
                continue
            writes.append(BoundaryWrite(device, variable, str(override), "override"))
            continue
        db_value = row.get(column)
        if db_value is None:
            continue
        writes.append(BoundaryWrite(device, variable, str(db_value), "db"))
    # Overrides for variables the DB did not curate: honor a value as a forced
    # write; a None override for an uncurated variable is simply a no-op.
    for variable, override in overrides.items():
        if variable in handled or variable in SUPPRESSED_BOUNDARY_VARIABLES:
            continue
        if override is None:
            continue
        writes.append(BoundaryWrite(device, variable, str(override), "override"))
    return writes


def collect_scan_boundary_writes(
    participants: dict[str, dict[str, dict[str, Optional[str]]]],
    provider: ScalarPolicyProvider | None,
) -> tuple[list[BoundaryWrite], list[BoundaryWrite]]:
    """Resolve start and end writes for every participating device.

    Participants are the devices taking part in the scan (save-set devices +
    scan-variable devices) — never every experiment device (the maintainer's
    explicit decision).  Each maps to its ``at_scan_start`` / ``at_scan_end``
    override dicts.

    Parameters
    ----------
    participants : dict
        ``{device: {"at_scan_start": {...}, "at_scan_end": {...}}}``.  A
        device with no overrides still participates (empty dicts).
    provider : ScalarPolicyProvider or None
        Supplies the ``set='yes'`` rows; ``None`` means no DB writes at all
        (only forced overrides survive).

    Returns
    -------
    tuple
        ``(start_writes, end_writes)`` in device then row order.
    """
    start: list[BoundaryWrite] = []
    end: list[BoundaryWrite] = []
    for device in participants:
        overrides = participants[device]
        rows = provider.boundary_writes(device) if provider is not None else []
        start.extend(
            resolve_boundary_writes(
                device,
                rows,
                which="start",
                overrides=overrides.get("at_scan_start", {}),
            )
        )
        end.extend(
            resolve_boundary_writes(
                device,
                rows,
                which="end",
                overrides=overrides.get("at_scan_end", {}),
            )
        )
    return start, end


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
