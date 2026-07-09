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

from pydantic import ValidationError

from geecs_ca_gateway.alarms import AlarmLimits
from geecs_ca_gateway.exceptions import GeecsDeviceNotFoundError

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


def _connect_mysql(mysql_connector):
    """Open a MySQL connection using the pure-Python connector implementation.

    The mysql-connector-python 9.x C extension has crashed silently on Windows in
    the legacy API layer.  GeecsBluesky runs on the same lab machines, so use the
    pure implementation here too.
    """
    return mysql_connector.connect(**_find_credentials(), use_pure=True)


def _num(value: object) -> Optional[float]:
    """Coerce a DB value to float, or ``None`` if it isn't numeric."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _variable_row_to_meta(row: tuple) -> dict:
    """Map a ``devicetype_variable`` (+ ``choice``) row onto the metadata dict.

    Row order: name, units, min, max, set, variabletype, choices, tolerance —
    the shared SELECT column order of :meth:`GeecsDb.get_device_variables` and
    :meth:`GeecsDb.get_experiment_device_variables`.
    """
    return {
        "name": row[0],
        "units": row[1] or "",
        "min": _num(row[2]),
        "max": _num(row[3]),
        "settable": (row[4] or "no").lower() == "yes",
        "variabletype": (row[5] or "").strip().lower() or None,
        "choices": row[6],
        "tolerance": _num(row[7]),
    }


def _alarm_row_to_limits(row: tuple) -> AlarmLimits:
    """Map a ``ca_alarm_limits`` row onto an :class:`AlarmLimits` model.

    Row order is the SELECT order in :meth:`GeecsDb.get_ca_alarm_limits`, after
    experiment/device/variable.
    """
    return AlarmLimits(
        lolo=_num(row[0]),
        low=_num(row[1]),
        high=_num(row[2]),
        hihi=_num(row[3]),
        lolo_severity=row[4] or "MAJOR",
        low_severity=row[5] or "MINOR",
        high_severity=row[6] or "MINOR",
        hihi_severity=row[7] or "MAJOR",
        hysteresis=_num(row[8]),
        description=row[9] or "",
    )


def _is_missing_table_error(exc: Exception) -> bool:
    """Return whether *exc* is MySQL's ER_NO_SUCH_TABLE rollout case."""
    return getattr(exc, "errno", None) == 1146


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

        conn = _connect_mysql(mysql.connector)
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
            raise GeecsDeviceNotFoundError(device_name)

        ip, port_str = row
        return ip.strip(), int(port_str)

    @classmethod
    def get_device_type(cls, device_name: str) -> str:
        """Return the GEECS database device type for *device_name*.

        Parameters
        ----------
        device_name:
            GEECS device name exactly as it appears in the database
            (e.g. ``"UC_TopView"``).

        Raises
        ------
        GeecsDeviceNotFoundError
            If the device is not found in the database.
        """
        try:
            import mysql.connector
        except ImportError as exc:
            raise ImportError(
                "mysql-connector-python is required for DB lookups. "
                "Install with: pip install mysql-connector-python"
            ) from exc

        conn = _connect_mysql(mysql.connector)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT devicetype FROM device WHERE name = %s",
                (device_name,),
            )
            row = cur.fetchone()
        finally:
            conn.close()

        if row is None:
            raise GeecsDeviceNotFoundError(device_name)

        return str(row[0]).strip()

    @classmethod
    def list_devices(
        cls, experiment: Optional[str] = None, *, enabled_only: bool = False
    ) -> list[str]:
        """Return all device names, optionally filtered by experiment.

        Parameters
        ----------
        experiment:
            If given, only return devices belonging to this experiment
            (e.g. ``"Undulator"``).
        enabled_only:
            If true (only meaningful with ``experiment``), return only devices
            whose ``expt_device.enabled`` field is ``"yes"``.  A device may
            belong to an experiment but be disabled.
        """
        try:
            import mysql.connector
        except ImportError as exc:
            raise ImportError(
                "mysql-connector-python is required for DB lookups."
            ) from exc

        conn = _connect_mysql(mysql.connector)
        try:
            cur = conn.cursor()
            if experiment is not None:
                query = (
                    "SELECT DISTINCT ed.device FROM expt_device ed "
                    "JOIN expt e ON e.name = ed.expt "
                    "WHERE e.name = %s"
                )
                if enabled_only:
                    query += " AND LOWER(ed.enabled) = 'yes'"
                query += " ORDER BY ed.device"
                cur.execute(query, (experiment,))
            else:
                cur.execute("SELECT name FROM device ORDER BY name")
            rows = cur.fetchall()
        finally:
            conn.close()

        return [r[0] for r in rows]

    @classmethod
    def get_subscribed_variables(
        cls, experiment: str, *, enabled_only: bool = True
    ) -> dict:
        """Return ``{device: [variablename, ...]}`` for ``get='yes'`` variables.

        ``expt_device_variable`` records, per device *instance* in an experiment,
        which variables are logged on every shot (``get='yes'``).  That is the
        experiment's meaningful monitoring subset — far smaller than every
        device-type variable — so it makes a sensible default set of PVs to serve.

        (The table's ``set``/``startvalue``/``endvalue`` fields describe scan
        start/end actions and are unrelated to whether a PV is writable, which
        comes from ``devicetype_variable``.)

        Parameters
        ----------
        experiment:
            GEECS experiment name.
        enabled_only:
            Restrict to devices enabled in the experiment (default true).

        Returns
        -------
        dict
            Device name → ordered list of subscribed variable names.  Devices
            with no ``get`` variables are absent.
        """
        try:
            import mysql.connector
        except ImportError as exc:
            raise ImportError(
                "mysql-connector-python is required for DB lookups."
            ) from exc

        conn = _connect_mysql(mysql.connector)
        try:
            cur = conn.cursor()
            query = (
                "SELECT ed.device, edv.variablename "
                "FROM expt_device_variable edv "
                "JOIN expt_device ed ON ed.id = edv.expt_device_id "
                "WHERE ed.expt = %s AND edv.get = 'yes'"
            )
            if enabled_only:
                query += " AND LOWER(ed.enabled) = 'yes'"
            query += " ORDER BY ed.device, edv.variablename"
            cur.execute(query, (experiment,))
            rows = cur.fetchall()
        finally:
            conn.close()

        result: dict = {}
        for device, variablename in rows:
            result.setdefault(device, []).append(variablename)
        return result

    @classmethod
    def get_all_experiment_variables(
        cls, experiment: str, *, enabled_only: bool = True
    ) -> dict[str, list[str]]:
        """Return ``{device: [variablename, ...]}`` for every ``expt_device_variable`` row.

        The ``all_scalars`` counterpart of :meth:`get_subscribed_variables`:
        every variable the experiment tracks for a device instance, not just
        the ``get='yes'`` subset.  Order is stable (device, then variable)
        and duplicates are collapsed.

        Parameters
        ----------
        experiment:
            GEECS experiment name.
        enabled_only:
            Restrict to devices enabled in the experiment (default true).

        Returns
        -------
        dict
            Device name → ordered list of variable names.  Devices with no
            rows are absent.
        """
        try:
            import mysql.connector
        except ImportError as exc:
            raise ImportError(
                "mysql-connector-python is required for DB lookups."
            ) from exc

        conn = _connect_mysql(mysql.connector)
        try:
            cur = conn.cursor()
            query = (
                "SELECT DISTINCT ed.device, edv.variablename "
                "FROM expt_device_variable edv "
                "JOIN expt_device ed ON ed.id = edv.expt_device_id "
                "WHERE ed.expt = %s"
            )
            if enabled_only:
                query += " AND LOWER(ed.enabled) = 'yes'"
            query += " ORDER BY ed.device, edv.variablename"
            cur.execute(query, (experiment,))
            rows = cur.fetchall()
        finally:
            conn.close()

        result: dict[str, list[str]] = {}
        for device, variablename in rows:
            bucket = result.setdefault(device, [])
            if variablename not in bucket:
                bucket.append(variablename)
        return result

    @classmethod
    def get_scan_boundary_writes(
        cls, experiment: str, *, enabled_only: bool = True
    ) -> dict[str, list[dict]]:
        """Return the ``set='yes'`` scan start/end writes per device.

        ``expt_device_variable`` rows with ``set='yes'`` name the variables
        Master Control writes at scan boundaries — the row's ``startvalue`` at
        scan start and ``endvalue`` at scan end.  The values are returned as
        raw wire strings (or ``None`` when the DB column is null).  Row order
        is preserved (the DB's ``id`` order) so writes replay in a stable
        sequence.

        .. note::

           **Reserved / not currently consumed by the engine.**  This is a
           read-only library query; the GeecsBluesky engine does *not* apply
           these DB set-side boundary writes in the current version (the
           set-side is intentionally disabled — triggering is owned by the
           trigger profile / shot controller and camera saving by the
           scanner's save-windowing).  The method is kept for inspection and
           for a possible future DB scan-write feature.

        Parameters
        ----------
        experiment:
            GEECS experiment name.
        enabled_only:
            Restrict to devices enabled in the experiment (default true).

        Returns
        -------
        dict
            Device name → ordered list of
            ``{"variable": str, "startvalue": str|None, "endvalue": str|None}``
            dicts, one per ``set='yes'`` row.  Devices with no such rows are
            absent.
        """
        try:
            import mysql.connector
        except ImportError as exc:
            raise ImportError(
                "mysql-connector-python is required for DB lookups."
            ) from exc

        conn = _connect_mysql(mysql.connector)
        try:
            cur = conn.cursor()
            query = (
                "SELECT ed.device, edv.variablename, edv.startvalue, edv.endvalue "
                "FROM expt_device_variable edv "
                "JOIN expt_device ed ON ed.id = edv.expt_device_id "
                "WHERE ed.expt = %s AND edv.`set` = 'yes'"
            )
            if enabled_only:
                query += " AND LOWER(ed.enabled) = 'yes'"
            query += " ORDER BY ed.device, edv.id"
            cur.execute(query, (experiment,))
            rows = cur.fetchall()
        finally:
            conn.close()

        result: dict[str, list[dict]] = {}
        for device, variablename, startvalue, endvalue in rows:
            result.setdefault(device, []).append(
                {
                    "variable": variablename,
                    "startvalue": None if startvalue is None else str(startvalue),
                    "endvalue": None if endvalue is None else str(endvalue),
                }
            )
        return result

    @classmethod
    def get_device_variables(cls, device_name: str) -> list[dict]:
        """Return variable metadata for *device_name*.

        Each entry is a dict with keys: ``name``, ``units``, ``min``, ``max``,
        ``settable`` (bool), ``variabletype`` (``"numeric"``, ``"choice"``,
        ``"string"``, ``"path"``, ``"image"``, ``"1darray"``, …), ``choices``
        (comma-separated option string from the ``choice`` table for ``choice``
        variables, else ``None``), and ``tolerance`` (numeric, or ``None``).
        """
        try:
            import mysql.connector
        except ImportError as exc:
            raise ImportError(
                "mysql-connector-python is required for DB lookups."
            ) from exc

        conn = _connect_mysql(mysql.connector)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT dtv.name, dtv.units, dtv.min, dtv.max, dtv.`set`, "
                "dtv.variabletype, c.choices, dtv.tolerance "
                "FROM devicetype_variable dtv "
                "JOIN device d ON d.devicetype = dtv.devicetype "
                "LEFT JOIN choice c ON c.id = dtv.choice_id "
                "WHERE d.name = %s ORDER BY dtv.name",
                (device_name,),
            )
            rows = cur.fetchall()
        finally:
            conn.close()

        return [_variable_row_to_meta(r) for r in rows]

    @classmethod
    def get_experiment_devices(
        cls, experiment: str, *, enabled_only: bool = True
    ) -> dict[str, tuple[str, int]]:
        """Return ``{device: (ip_address, port)}`` for an experiment — one query.

        The batch counterpart of :meth:`find_device`: endpoints for every device
        in *experiment* in a single connection, so building a whole-experiment
        gateway config doesn't open one MySQL connection per device.

        Parameters
        ----------
        experiment:
            GEECS experiment name.
        enabled_only:
            Restrict to devices enabled in the experiment (default true).
        """
        try:
            import mysql.connector
        except ImportError as exc:
            raise ImportError(
                "mysql-connector-python is required for DB lookups."
            ) from exc

        conn = _connect_mysql(mysql.connector)
        try:
            cur = conn.cursor()
            query = (
                "SELECT DISTINCT d.name, d.ipaddress, d.commport "
                "FROM expt_device ed "
                "JOIN device d ON d.name = ed.device "
                "WHERE ed.expt = %s"
            )
            if enabled_only:
                query += " AND LOWER(ed.enabled) = 'yes'"
            query += " ORDER BY d.name"
            cur.execute(query, (experiment,))
            rows = cur.fetchall()
        finally:
            conn.close()

        return {name: (ip.strip(), int(port)) for name, ip, port in rows}

    @classmethod
    def get_experiment_device_variables(
        cls, experiment: str, *, enabled_only: bool = True
    ) -> dict[str, list[dict]]:
        """Return ``{device: [variable metadata, ...]}`` for an experiment.

        The batch counterpart of :meth:`get_device_variables`: metadata for
        every device in *experiment* in a single query, with the same per-row
        dict shape.  Duplicate ``devicetype_variable`` rows (a known DB quirk)
        are passed through — spec building dedupes by name.

        Parameters
        ----------
        experiment:
            GEECS experiment name.
        enabled_only:
            Restrict to devices enabled in the experiment (default true).
        """
        try:
            import mysql.connector
        except ImportError as exc:
            raise ImportError(
                "mysql-connector-python is required for DB lookups."
            ) from exc

        conn = _connect_mysql(mysql.connector)
        try:
            cur = conn.cursor()
            query = (
                "SELECT d.name, dtv.name, dtv.units, dtv.min, dtv.max, dtv.`set`, "
                "dtv.variabletype, c.choices, dtv.tolerance "
                "FROM (SELECT DISTINCT ed.device FROM expt_device ed "
                "      WHERE ed.expt = %s{enabled}) sel "
                "JOIN device d ON d.name = sel.device "
                "JOIN devicetype_variable dtv ON dtv.devicetype = d.devicetype "
                "LEFT JOIN choice c ON c.id = dtv.choice_id "
                "ORDER BY d.name, dtv.name"
            ).format(enabled=" AND LOWER(ed.enabled) = 'yes'" if enabled_only else "")
            cur.execute(query, (experiment,))
            rows = cur.fetchall()
        finally:
            conn.close()

        result: dict[str, list[dict]] = {}
        for row in rows:
            result.setdefault(row[0], []).append(_variable_row_to_meta(row[1:]))
        return result

    @classmethod
    def get_ca_alarm_limits(cls, experiment: str) -> dict[tuple[str, str], AlarmLimits]:
        """Return enabled curated CA alarm limits for *experiment*.

        The ``ca_alarm_limits`` table is an optional overlay.  If the table is
        absent during rollout, or if the lookup otherwise fails, the gateway
        starts with no value alarms rather than failing the whole IOC.  Invalid
        rows are skipped with a warning.

        Parameters
        ----------
        experiment : str
            GEECS experiment name.

        Returns
        -------
        dict
            ``(device, variable)`` → validated alarm limits.
        """
        try:
            import mysql.connector
        except ImportError as exc:
            raise ImportError(
                "mysql-connector-python is required for DB lookups."
            ) from exc

        try:
            conn = _connect_mysql(mysql.connector)
        except Exception:
            logger.warning(
                "ca_alarm_limits lookup failed before query; starting without "
                "curated value alarms",
                exc_info=True,
            )
            return {}
        try:
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT experiment, device, `variable`, "
                    "lolo, low, high, hihi, "
                    "lolo_severity, low_severity, high_severity, hihi_severity, "
                    "hysteresis, description "
                    "FROM ca_alarm_limits "
                    "WHERE experiment = %s AND enabled = TRUE "
                    "ORDER BY device, variable",
                    (experiment,),
                )
                rows = cur.fetchall()
            except Exception as exc:
                if _is_missing_table_error(exc):
                    logger.info(
                        "ca_alarm_limits table is absent; starting without "
                        "curated value alarms"
                    )
                else:
                    logger.warning(
                        "ca_alarm_limits lookup failed; starting without "
                        "curated value alarms",
                        exc_info=True,
                    )
                return {}
        finally:
            conn.close()

        result: dict[tuple[str, str], AlarmLimits] = {}
        for row in rows:
            _experiment, device, variable = row[:3]
            try:
                result[(device, variable)] = _alarm_row_to_limits(row[3:])
            except ValidationError:
                logger.warning(
                    "skipping invalid ca_alarm_limits row for %s/%s in %s",
                    device,
                    variable,
                    experiment,
                    exc_info=True,
                )
        return result
