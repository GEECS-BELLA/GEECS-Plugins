"""TRANSITIONAL: the CA gateway's served set for the LabVIEW devices, read from the GEECS DB.

The one place the GEECS DB enters Canopy (``elementType=gateway``): the
gateway's own read-only queries and ``geecs_core.db.served_set``, named
with ``geecs_core.pv_naming``. Deleted together with the CA gateway when
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
from geecs_core.pv_naming import device_status_pv, pv_name, setpoint_pv

from geecs_canopy.native import Device, Element, SourceError

#: The environment variable carrying the DB URL (a secret: never an ``extras`` key).
DB_URL_ENV = "GEECS_CANOPY_DB_URL"


def credentials_from_url(url: str) -> dict[str, object]:
    """``mysql://user:password@host:port/database`` → GeecsDb's connection parameters."""
    parts = urlsplit(url)
    database = parts.path.lstrip("/")
    if parts.scheme != "mysql" or not (parts.hostname and parts.username and database):
        raise ValueError(
            f"{DB_URL_ENV} is not mysql://user:password@host:port/database"
        )
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
        for name, served in served_variables(
            var_map, sub_map, devices=endpoints
        ).items():
            elements = []
            for var in served:
                readback = pv_name(expt, name, var.name)
                pvs = (readback, setpoint_pv(readback)) if var.settable else (readback,)
                elements.append(Element(f"{name}:{var.name}", pvs))
            elements.append(
                Element(f"{name}:connected", (device_status_pv(expt, name),))
            )
            devices.append(
                Device(name, "gateway", types.get(name, ""), tuple(elements))
            )
        return devices
