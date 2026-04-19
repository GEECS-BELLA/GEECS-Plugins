"""Lightweight GEECS MySQL database client.

Reads connection credentials from the standard GEECS configuration files —
the same sources used by geecs-python-api — so no extra setup is needed
on machines that already have GEECS installed.

Credential discovery order
--------------------------
1. Look for ``~/.config/geecs_python_api/config.ini`` and read the
   ``[Paths] geecs_data`` key to find the user-data directory.
2. Open ``{geecs_data}/Configurations.INI`` for ``[Database]`` credentials.

This mirrors what ``geecs_python_api.controls.interface.geecs_database``
does, but without dragging in the rest of that package.
"""

from __future__ import annotations

import configparser
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Cached connection parameters so we only parse config files once.
_credentials: Optional[dict] = None


def _find_credentials() -> dict:
    """Return DB connection params parsed from GEECS config files."""
    global _credentials
    if _credentials is not None:
        return _credentials

    # Step 1: resolve user-data directory.
    user_cfg_path = Path.home() / ".config" / "geecs_python_api" / "config.ini"
    if not user_cfg_path.exists():
        raise FileNotFoundError(
            f"GEECS user config not found at {user_cfg_path}. "
            "Ensure geecs_python_api is configured on this machine."
        )

    user_cfg = configparser.ConfigParser()
    user_cfg.read(user_cfg_path)

    try:
        geecs_data = user_cfg["Paths"]["geecs_data"]
    except KeyError as exc:
        raise KeyError(f"[Paths] geecs_data key missing from {user_cfg_path}") from exc

    # Step 2: read Configurations.INI for DB credentials.
    db_ini_path = Path(os.path.expandvars(geecs_data)) / "Configurations.INI"
    if not db_ini_path.exists():
        raise FileNotFoundError(f"GEECS database config not found at {db_ini_path}")

    db_cfg = configparser.ConfigParser()
    db_cfg.read(db_ini_path)

    try:
        section = db_cfg["Database"]
        _credentials = {
            "host": section["ipaddress"],
            "port": int(section.get("port", 3306)),
            "database": section["name"],
            "user": section["user"],
            "password": section["password"],
        }
    except KeyError as exc:
        raise KeyError(
            f"Missing key in [Database] section of {db_ini_path}: {exc}"
        ) from exc

    logger.debug("GEECS DB credentials loaded from %s", db_ini_path)
    return _credentials


class GeecsDb:
    """Namespace for GEECS database queries.

    All methods are class-level; no instance needed::

        host, port = GeecsDb.find_device("U_ESP_JetXYZ")
    """

    @classmethod
    def find_device(cls, device_name: str) -> tuple[str, int]:
        """Return ``(ip_address, port)`` for *device_name*.

        Parameters
        ----------
        device_name:
            GEECS device name exactly as it appears in the database
            (e.g. ``"U_ESP_JetXYZ"``).

        Raises
        ------
        RuntimeError
            If the device is not found in the database.
        """
        try:
            import mysql.connector
        except ImportError as exc:
            raise ImportError(
                "mysql-connector-python is required for DB lookups. "
                "Install with: pip install mysql-connector-python"
            ) from exc

        creds = _find_credentials()
        conn = mysql.connector.connect(**creds)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT ipaddress, commport FROM device WHERE name = %s",
                (device_name,),
            )
            row = cur.fetchone()
        finally:
            conn.close()

        if row is None:
            raise RuntimeError(f"Device '{device_name}' not found in GEECS database.")

        ip, port_str = row
        return ip.strip(), int(port_str)

    @classmethod
    def list_devices(cls, experiment: Optional[str] = None) -> list[str]:
        """Return all device names, optionally filtered by experiment.

        Parameters
        ----------
        experiment:
            If given, only return devices belonging to this experiment
            (e.g. ``"Undulator"``).
        """
        try:
            import mysql.connector
        except ImportError as exc:
            raise ImportError(
                "mysql-connector-python is required for DB lookups."
            ) from exc

        creds = _find_credentials()
        conn = mysql.connector.connect(**creds)
        try:
            cur = conn.cursor()
            if experiment is not None:
                cur.execute(
                    "SELECT DISTINCT ed.device FROM expt_device ed "
                    "JOIN expt e ON e.name = ed.expt "
                    "WHERE e.name = %s ORDER BY ed.device",
                    (experiment,),
                )
            else:
                cur.execute("SELECT name FROM device ORDER BY name")
            rows = cur.fetchall()
        finally:
            conn.close()

        return [r[0] for r in rows]

    @classmethod
    def get_device_variables(cls, device_name: str) -> list[dict]:
        """Return variable metadata for *device_name*.

        Each entry is a dict with keys: ``name``, ``units``, ``min``, ``max``,
        ``settable`` (bool).
        """
        try:
            import mysql.connector
        except ImportError as exc:
            raise ImportError(
                "mysql-connector-python is required for DB lookups."
            ) from exc

        creds = _find_credentials()
        conn = mysql.connector.connect(**creds)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT dtv.name, dtv.units, dtv.min, dtv.max, dtv.`set` "
                "FROM devicetype_variable dtv "
                "JOIN device d ON d.devicetype = dtv.devicetype "
                "WHERE d.name = %s ORDER BY dtv.name",
                (device_name,),
            )
            rows = cur.fetchall()
        finally:
            conn.close()

        return [
            {
                "name": r[0],
                "units": r[1] or "",
                "min": float(r[2]) if r[2] is not None else None,
                "max": float(r[3]) if r[3] is not None else None,
                "settable": (r[4] or "no").lower() == "yes",
            }
            for r in rows
        ]
