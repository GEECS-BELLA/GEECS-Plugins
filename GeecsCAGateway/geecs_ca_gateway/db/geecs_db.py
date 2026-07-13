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
    """Map a ``devicetype_variable`` or ``variable`` (+ ``choice``) row onto the metadata dict.

    Row order: name, units, min, max, set, variabletype, choices, tolerance,
    description — the shared SELECT column order (after the leading id/link
    column) of :meth:`GeecsDb.get_device_variables` and
    :meth:`GeecsDb.get_experiment_device_variables`.  Both the type-default
    table (``devicetype_variable``) and the per-instance table (``variable``)
    carry the same columns, so one mapper serves both.  ``description`` exists
    only on ``variable`` — the type query selects ``NULL`` in that slot, so a
    description is a purely per-instance fact (which the wholesale inheritance
    already implies).
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
        "description": (row[8] or "").strip() if len(row) > 8 else "",
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


def _merge_variable_rows(
    type_rows: list[tuple], instance_rows: list[tuple]
) -> list[dict]:
    """Resolve the GEECS capability inheritance chain for one device.

    ``devicetype_variable`` rows define type defaults which a device instance
    inherits *unless* a row exists for that device+variable in the ``variable``
    table — and when one exists it is used **wholesale**: every field comes
    from the instance row, with no field-level fallback.  An instance row with
    NULL limits means that instance has *no* limits, even if the type row has
    them.  The link is explicit: ``variable.devicetype_variable_id →
    devicetype_variable.id``.

    Parameters
    ----------
    type_rows:
        ``(id, name, units, min, max, set, variabletype, choices, tolerance)``
        rows from ``devicetype_variable``.
    instance_rows:
        ``(devicetype_variable_id, name, units, min, max, set, variabletype,
        choices, tolerance)`` rows from ``variable`` for the same device.

    Returns
    -------
    list[dict]
        Metadata dicts (:func:`_variable_row_to_meta` shape) in type-row
        order, with overridden entries replaced in place; instance-only
        variables are appended at the end.

    Notes
    -----
    An instance row whose ``devicetype_variable_id`` is NULL or points at no
    type row of this device defines an *instance-only* variable.  It has no
    explicit link to key on, so it is keyed by ``name`` instead: a same-named
    type row is replaced (instance rows are ground truth) rather than served
    alongside it — two entries normalizing to one PV name would otherwise
    collide at gateway startup.
    """
    linked: dict[object, tuple] = {}
    unlinked: list[tuple] = []
    for row in instance_rows:
        if row[0] is None:
            unlinked.append(row)
        else:
            linked[row[0]] = row

    result: list[dict] = []
    used_links: set = set()
    for row in type_rows:
        instance = linked.get(row[0])
        if instance is not None:
            used_links.add(row[0])
            result.append(_variable_row_to_meta(instance[1:]))
        else:
            result.append(_variable_row_to_meta(row[1:]))

    # Dangling links (type row absent) behave like instance-only variables.
    unlinked.extend(row for link, row in linked.items() if link not in used_links)
    for row in unlinked:
        meta = _variable_row_to_meta(row[1:])
        replaced = False
        for i, existing in enumerate(result):
            if existing["name"] == meta["name"]:
                result[i] = meta
                replaced = True
        if not replaced:
            result.append(meta)
    return result


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

        Metadata resolves the capability inheritance chain: type defaults come
        from ``devicetype_variable``; a per-instance row in ``variable``
        replaces its type row **wholesale** (see :func:`_merge_variable_rows`).
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
                "SELECT dtv.id, dtv.name, dtv.units, dtv.min, dtv.max, dtv.`set`, "
                "dtv.variabletype, c.choices, dtv.tolerance, NULL "
                "FROM devicetype_variable dtv "
                "JOIN device d ON d.devicetype = dtv.devicetype "
                "LEFT JOIN choice c ON c.id = dtv.choice_id "
                "WHERE d.name = %s ORDER BY dtv.name",
                (device_name,),
            )
            type_rows = cur.fetchall()
            cur.execute(
                "SELECT v.devicetype_variable_id, v.name, v.units, v.min, v.max, "
                "v.`set`, v.variabletype, c.choices, v.tolerance, v.description "
                "FROM variable v "
                "LEFT JOIN choice c ON c.id = v.choice_id "
                "WHERE v.device = %s ORDER BY v.name",
                (device_name,),
            )
            instance_rows = cur.fetchall()
        finally:
            conn.close()

        return _merge_variable_rows(type_rows, instance_rows)

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
        every device in *experiment* in two batched queries (type defaults +
        per-instance overrides), with the same per-row dict shape and the same
        wholesale inheritance-chain resolution (see
        :func:`_merge_variable_rows`).  Duplicate ``devicetype_variable`` rows
        (a known DB quirk) are passed through — spec building dedupes by name.

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

        enabled = " AND LOWER(ed.enabled) = 'yes'" if enabled_only else ""
        conn = _connect_mysql(mysql.connector)
        try:
            cur = conn.cursor()
            type_query = (
                "SELECT d.name, dtv.id, dtv.name, dtv.units, dtv.min, dtv.max, "
                "dtv.`set`, dtv.variabletype, c.choices, dtv.tolerance, NULL "
                "FROM (SELECT DISTINCT ed.device FROM expt_device ed "
                "      WHERE ed.expt = %s{enabled}) sel "
                "JOIN device d ON d.name = sel.device "
                "JOIN devicetype_variable dtv ON dtv.devicetype = d.devicetype "
                "LEFT JOIN choice c ON c.id = dtv.choice_id "
                "ORDER BY d.name, dtv.name"
            ).format(enabled=enabled)
            cur.execute(type_query, (experiment,))
            type_rows = cur.fetchall()
            instance_query = (
                "SELECT v.device, v.devicetype_variable_id, v.name, v.units, "
                "v.min, v.max, v.`set`, v.variabletype, c.choices, v.tolerance, "
                "v.description "
                "FROM (SELECT DISTINCT ed.device FROM expt_device ed "
                "      WHERE ed.expt = %s{enabled}) sel "
                "JOIN variable v ON v.device = sel.device "
                "LEFT JOIN choice c ON c.id = v.choice_id "
                "ORDER BY v.device, v.name"
            ).format(enabled=enabled)
            cur.execute(instance_query, (experiment,))
            instance_rows = cur.fetchall()
        finally:
            conn.close()

        type_by_device: dict[str, list[tuple]] = {}
        for row in type_rows:
            type_by_device.setdefault(row[0], []).append(row[1:])
        instance_by_device: dict[str, list[tuple]] = {}
        for row in instance_rows:
            instance_by_device.setdefault(row[0], []).append(row[1:])

        return {
            device: _merge_variable_rows(
                type_by_device.get(device, []), instance_by_device.get(device, [])
            )
            for device in sorted(type_by_device.keys() | instance_by_device.keys())
        }

    @classmethod
    def get_fk_orphan_variables(cls, *, sample_limit: int = 20) -> dict:
        """Return ``expt_device_variable`` rows whose ``expt_device`` is gone.

        Such rows are left behind because DB deletions do not cascade (see
        :mod:`geecs_ca_gateway.audit`).  A LEFT JOIN with ``ed.id IS NULL``
        finds them.  Read-only — this only reports.

        These orphans are **not experiment-scoped** (their owning
        ``expt_device`` row, which carried the experiment name, is gone), so the
        query is global.

        Parameters
        ----------
        sample_limit:
            Maximum number of orphan rows to return in the ``sample`` list for
            eyeballing (the ``count`` is always the full total).

        Returns
        -------
        dict
            ``{"count": int, "sample": [{"id", "expt_device_id",
            "variablename"}, ...]}``.
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
                "SELECT COUNT(*) FROM expt_device_variable edv "
                "LEFT JOIN expt_device ed ON ed.id = edv.expt_device_id "
                "WHERE ed.id IS NULL"
            )
            count_row = cur.fetchone()
            count = int(count_row[0]) if count_row else 0
            cur.execute(
                "SELECT edv.id, edv.expt_device_id, edv.variablename "
                "FROM expt_device_variable edv "
                "LEFT JOIN expt_device ed ON ed.id = edv.expt_device_id "
                "WHERE ed.id IS NULL "
                "ORDER BY edv.id LIMIT %s",
                (int(sample_limit),),
            )
            sample_rows = cur.fetchall()
        finally:
            conn.close()

        sample = [
            {"id": row[0], "expt_device_id": row[1], "variablename": row[2]}
            for row in sample_rows
        ]
        return {"count": count, "sample": sample}

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
