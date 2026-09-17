"""The subscribed-scalars rule: what the experiment DB says a device records.

``expt_device_variable`` records, per device instance in an experiment,
which variables are logged on every shot (``get='yes'``).  That list is
the one rule for *what a device's row carries*, shared by every consumer
that builds a row from it: GeecsBluesky's device namespace (each device's
event columns and the run's baseline telemetry), GeecsPvaGateway's file
plugin (the per-frame scalar attributes it writes beside a camera's
frames, so a gated row and a strict row carry the same columns for that
device — ``Planning/native_bluesky/08_gated_batch.md`` §4.4), and the
gateways' served set.  It lived in ``geecs_bluesky.db_runtime`` until
2026-09-12 and moved here (GEECS-Core 0.6.0) beside
:mod:`geecs_core.db.variable_types` — the PVA gateway depends on GEECS-Core
alone, and a second copy of the rule in the gateway is exactly the drift
the "same columns" promise cannot survive.

Failure semantics: every query is wrapped so a DB failure (off the lab
network, a missing table, an uncurated experiment) degrades to **empty
policy** with a single warning — a scan must never abort because the DB
was briefly unreachable.  (The gateways' *served set* has the opposite
semantics — a failure reads as "unknown", never "empty" — and stays in
``geecs_bluesky.db_runtime`` with the check it drives.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class ScalarPolicyProvider(Protocol):
    """Supplies per-device DB variable policy for one experiment.

    The seam between the pure resolution logic and
    :class:`~geecs_core.db.geecs_db.GeecsDb`.  Every method returns an
    empty result rather than raising when the DB is unavailable or a device
    is uncurated.  This covers only the **get-side** (subscribed ``get='yes'``
    variables + all-variables queries).
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

    Wraps :class:`~geecs_core.db.geecs_db.GeecsDb` for one experiment,
    caching each of its two get-side whole-experiment queries on first use.

    Parameters
    ----------
    experiment : str
        GEECS experiment name.
    enabled_only : bool
        Restrict to devices enabled in the experiment (default true).
    db : type, optional
        The ``GeecsDb`` class (injectable for tests); imported lazily by
        default so importing this module never touches the MySQL driver.
    """

    experiment: str
    enabled_only: bool = True
    db: object | None = None
    _subscribed: Optional[dict[str, list[str]]] = field(default=None, init=False)
    _all: Optional[dict[str, list[str]]] = field(default=None, init=False)

    def _geecs_db(self) -> object:
        if self.db is not None:
            return self.db
        from geecs_core.db.geecs_db import GeecsDb

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
