"""The served set: which of a device's variables the CA gateway serves.

A variable is served when it is on the device's ``get='yes'`` list or is
settable, and its effective vartype is scalar. A get-list name matches the
row spelled the same, else the first row equal to it ignoring case (case
carries no meaning in GEECS), so one get-list entry never serves two
case-only twins (they would collide on one PV).
The first qualifying row per exact name wins over the DB's duplicate rows. The CA
gateway, the archiver and the Bluesky namespace all read the rule from here;
each keeps its own failure handling and post-filters.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from geecs_core.db.variable_types import effective_vartype, is_scalar_vartype

if TYPE_CHECKING:
    from geecs_core.db.geecs_db import VariableMeta


@dataclass(frozen=True)
class ServedVariable:
    """One served variable: its DB row and why it is served."""

    row: VariableMeta
    monitored: bool
    settable: bool

    @property
    def name(self) -> str:
        """The variable name as the DB spells it."""
        return self.row["name"]


def served_variables(
    var_map: Mapping[str, Sequence[VariableMeta]],
    sub_map: Mapping[str, Iterable[str]] | None,
    *,
    devices: Iterable[str] | None = None,
    include_settable: bool = True,
) -> dict[str, list[ServedVariable]]:
    """Apply the served-set rule to already-fetched DB data.

    Parameters
    ----------
    var_map : mapping
        ``{device: [row, ...]}`` (``GeecsDb.get_experiment_device_variables``).
    sub_map : mapping or None
        ``{device: [variable, ...]}`` of ``get='yes'`` variables
        (``GeecsDb.get_subscribed_variables``); ``None`` serves every
        variable, each counted as monitored.
    devices : iterable of str, optional
        The devices to consider, in output order — the endpoint filter
        (``GeecsDb.get_experiment_devices`` keys). Defaults to *var_map*'s.
    include_settable : bool
        Serve settable variables that are not on the get-list (default true).

    Returns
    -------
    dict
        ``{device: [ServedVariable, ...]}`` in DB row order; a device that
        serves nothing is absent.
    """
    out: dict[str, list[ServedVariable]] = {}
    for device in var_map.keys() if devices is None else devices:
        rows = var_map.get(device, ())
        monitored: set[str] | None = None
        if sub_map is not None:
            spelled = {str(row["name"]) for row in rows}
            first_by_lower: dict[str, str] = {}
            for row in rows:
                first_by_lower.setdefault(str(row["name"]).lower(), str(row["name"]))
            monitored = {
                v if v in spelled else first_by_lower.get(v.lower(), v)
                for v in map(str, sub_map.get(device, ()))
            }
        seen: set[str] = set()
        served: list[ServedVariable] = []
        for row in rows:
            name = str(row["name"])
            settable = bool(row.get("settable", False))
            is_monitored = monitored is None or name in monitored
            if not (is_monitored or (include_settable and settable)):
                continue
            if name in seen:
                continue
            seen.add(name)
            vartype = effective_vartype(row.get("variabletype"), row.get("choices"))
            if is_scalar_vartype(vartype):
                served.append(ServedVariable(row, is_monitored, settable))
        if served:
            out[device] = served
    return out
