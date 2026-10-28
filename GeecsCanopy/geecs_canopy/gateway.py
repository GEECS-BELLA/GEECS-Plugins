"""TRANSITIONAL: the CA gateway's served set for the LabVIEW devices, read from the GEECS DB.

The one place the GEECS DB enters Canopy (``elementType=gateway``): the
gateway's own read-only queries and ``geecs_core.db.served_set``, named
with ``geecs_core.pv_naming``, plus the timestamp ladder and ``connected``
it serves on every device (its derived channels are not listed). Deleted together with the CA gateway when
the last LabVIEW device retires; no other module of this package may
import ``GeecsDb`` (the ``canopy-db-is-transitional`` import contract).

Credentials come from ``GEECS_CANOPY_DB_URL``
(``mysql://<user>:<password>@<host>:<port>/<database>``, a read-only user)
in the container's secrets file, never from a client ``config.ini``.
"""

from __future__ import annotations

from urllib.parse import unquote, urlsplit

from geecs_core.db import geecs_db
from geecs_core.db.geecs_db import GeecsDb
from geecs_core.db.served_set import served_variables
from geecs_core.db.variable_types import TIMESTAMP_LADDER
from geecs_core.pv_naming import device_status_pv, pv_name, setpoint_pv

from geecs_canopy.native import Device, Element, SourceError

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


class GatewaySource:
    """The gateway's devices: the served-set rule over the batched DB queries."""

    def __init__(self, experiment: str, db_url: str) -> None:
        self.experiment = experiment
        geecs_db.use_credentials(credentials_from_url(db_url))

    def devices(self) -> list[Device]:
        """Every enabled device of the experiment the gateway serves, with its PVs."""
        expt = self.experiment
        try:
            endpoints = GeecsDb.get_experiment_devices(expt)
            var_map = GeecsDb.get_experiment_device_variables(expt)
            sub_map = GeecsDb.get_subscribed_variables(expt)
            types = GeecsDb.get_experiment_device_types(expt)
        # The DB is the one upstream: any failure is unavailable.
        except Exception as exc:
            raise SourceError(f"GEECS DB unavailable: {exc}") from exc
        devices = []
        served_map = served_variables(var_map, sub_map, devices=endpoints)
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
