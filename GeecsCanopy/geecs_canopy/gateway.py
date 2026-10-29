"""TRANSITIONAL: the CA gateway's served set for the LabVIEW devices, read from the GEECS DB.

The one place the GEECS DB enters Canopy (``elementType=gateway``): the
gateway's own read-only queries and ``geecs_core.db.served_set``, named
with ``geecs_core.pv_naming``, plus the timestamp ladder and ``connected``
it serves on every device (its derived channels are not listed). Deleted together with the CA gateway when
the last LabVIEW device retires; no other module of this package may
import ``GeecsDb`` (the ``canopy-db-is-transitional`` import contract).

The same read gives the device module its DB devices (:meth:`GatewaySource.resolve`):
a role per served variable, named by its GEECS variable name; the family
word is the DB device type.

Credentials come from ``GEECS_CANOPY_DB_URL``
(``mysql://<user>:<password>@<host>:<port>/<database>``, a read-only user)
in the container's secrets file, never from a client ``config.ini``.
"""

from __future__ import annotations

from urllib.parse import unquote, urlsplit

from geecs_core.db import geecs_db
from geecs_core.db.geecs_db import GeecsDb
from geecs_core.db.served_set import ServedVariable, served_variables
from geecs_core.db.variable_types import (
    TIMESTAMP_LADDER,
    effective_vartype,
    enum_choices,
)
from geecs_core.pv_naming import (
    DEVICE_STATUS_VARIABLE,
    device_status_pv,
    pv_name,
    setpoint_pv,
)
from geecs_schemas import Role

from geecs_canopy.native import Device, Element, Resolved, SourceError

#: The environment variable carrying the DB URL (a secret: never an ``extras`` key).
DB_URL_ENV = "GEECS_CANOPY_DB_URL"


def credentials_from_url(url: str) -> dict[str, object]:
    """``mysql://user:password@host:port/database`` → GeecsDb's connection parameters."""
    parts = urlsplit(url)
    database = parts.path.lstrip("/")
    if parts.scheme != "mysql" or not (parts.hostname and parts.username and database):
        raise ValueError(f"{DB_URL_ENV} is not mysql://user:pass@host:port/database")
    return {
        "host": parts.hostname,
        "port": parts.port or 3306,
        "database": database,
        "user": unquote(parts.username),
        "password": unquote(parts.password or ""),
    }


def _role(var: ServedVariable, prefix: str, readback: str) -> Role:
    """A served variable as a role: settable numeric → setpoint (the DB min/max), a choice → enum, text, else read-only."""
    row, settable = var.row, var.settable
    vartype = effective_vartype(row.get("variabletype"), row.get("choices"))
    fields: dict = {"name": var.name, "suffix": readback[len(prefix) :]}
    fields |= {"label": row.get("alias") or var.name, "units": row.get("units") or None}
    if settable:
        fields |= {"writable": True, "write": setpoint_pv(readback)[len(prefix) :]}
    if vartype == "choice" and enum_choices(row.get("choices")) is None:
        vartype = "string"  # beyond DBR_ENUM's limits: the gateway serves it as text
    if vartype == "choice":  # options are read from the PV, as the gateway serves them
        fields["type"] = "enum"
    else:
        fields["type"] = "setpoint" if settable else "readback"
        if vartype in ("string", "path"):
            fields["text"] = "char_array" if vartype == "path" else "string"
        low, high = row.get("min"), row.get("max")
        if settable and vartype == "numeric" and None not in (low, high) and low < high:
            fields["limits"] = (low, high)  # the span the gateway serves as DRVL/DRVH
    # A GEECS variable name is no RoleName (dots, spaces): built, not validated.
    return Role.model_construct(**fields)


class GatewaySource:
    """The gateway's devices: the served-set rule over the batched DB queries."""

    def __init__(self, experiment: str, db_url: str) -> None:
        self.experiment = experiment
        geecs_db.use_credentials(credentials_from_url(db_url))

    def _served(self) -> tuple[dict[str, list[ServedVariable]], dict[str, str]]:
        expt = self.experiment
        try:
            endpoints = GeecsDb.get_experiment_devices(expt)
            var_map = GeecsDb.get_experiment_device_variables(expt)
            sub_map = GeecsDb.get_subscribed_variables(expt)
            types = GeecsDb.get_experiment_device_types(expt)
        # The DB is the one upstream: any failure is unavailable.
        except Exception as exc:
            raise SourceError(f"GEECS DB unavailable: {exc}") from exc
        return served_variables(var_map, sub_map, devices=endpoints), types

    def devices(self) -> list[Device]:
        """Every enabled device of the experiment the gateway serves, with its PVs."""
        expt = self.experiment
        served_map, types = self._served()
        devices = []
        for name, served in served_map.items():
            elements = []
            for var in served:
                rb = pv_name(expt, name, var.name)
                pvs = (rb, setpoint_pv(rb)) if var.settable else (rb,)
                elements.append(Element(f"{name}:{var.name}", pvs))
            # The gateway serves the timestamp ladder and `connected` on every device.
            own = {var.name for var in served}
            stamp = {
                t: pv_name(expt, name, t) for t in TIMESTAMP_LADDER if t not in own
            }
            elements += [Element(f"{name}:{t}", (pv,)) for t, pv in stamp.items()]
            status = Element(f"{name}:connected", (device_status_pv(expt, name),))
            tag = types.get(name, "")
            devices.append(Device(name, "gateway", tag, (*elements, status)))
        return devices

    def resolve(self) -> list[Resolved]:
        """Every served device as the module's device: its variables as roles, ``connected`` last; no run lock."""
        expt = self.experiment
        served_map, types = self._served()
        out = []
        for name, served in served_map.items():
            prefix = pv_name(expt, name) + ":"
            roles = [_role(v, prefix, pv_name(expt, name, v.name)) for v in served]
            status = device_status_pv(expt, name)[len(prefix) :]
            # every DB device's `connected` PV (0 Disconnected, 1 Connected): its state
            roles.append(Role(name=DEVICE_STATUS_VARIABLE, suffix=status, type="enum", label="connected"))  # fmt: skip
            writable = frozenset(r.name for r in roles if r.writable)
            family = types.get(name, "")
            out.append(Resolved(name, family, prefix, (), tuple(roles), (), None, writable, "gateway"))  # fmt: skip
        return out
