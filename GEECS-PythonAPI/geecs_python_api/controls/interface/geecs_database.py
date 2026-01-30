"""GEECS database utilities: configuration discovery, MySQL access helpers, and experiment metadata retrieval."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import configparser
import mysql.connector
import tkinter as tk
from tkinter import filedialog

from geecs_python_api.controls.interface.geecs_database_models import (
    Device,
    Experiment,
    Variable,
)

try:
    # Optional import for type checking; adjust if ExpDict moves.
    from geecs_python_api.controls.api_defs import ExpDict
except Exception:  # pragma: no cover
    ExpDict = Dict[str, Dict[str, Any]]  # fallback typing if package unavailable

logger = logging.getLogger(__name__)


# --- Path & config discovery -------------------------------------------------
def find_user_data_directory_relative(
    start_path: Union[str, os.PathLike[str]] = ".",
) -> Optional[str]:
    """Walk upward from start_path to find a 'user data' directory and return a relative path if found."""
    current_path = os.path.abspath(start_path)
    original_path = current_path
    root = os.path.abspath(os.sep)

    while current_path != root:
        check_path = os.path.join(current_path, "user data")
        if os.path.isdir(check_path):
            rel = os.path.relpath(check_path, original_path)
            logger.debug("found 'user data' at %s (relative %s)", check_path, rel)
            return rel
        current_path = os.path.dirname(current_path)

    logger.debug("'user data' directory not found from %s", start_path)
    return None


def load_config() -> Optional[configparser.ConfigParser]:
    """Load ~.config.geecs_python_api.config.ini if present and return a ConfigParser, else None."""
    config_path = os.path.expanduser("~/.config/geecs_python_api/config.ini")
    if os.path.exists(config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        logger.debug("loaded config from %s", config_path)
        return config
    logger.debug("config not found at %s", config_path)
    return None


def find_database() -> Tuple[
    Optional[str], Optional[str], Optional[str], Optional[str]
]:
    """Resolve DB (name, ip, user, password) from INI in 'user data' or fallback config, prompting if needed."""
    default_path: Optional[str] = find_user_data_directory_relative()
    if default_path is None:
        cfg = load_config()
        if cfg and "Paths" in cfg and "geecs_data" in cfg["Paths"]:
            default_path = cfg["Paths"]["geecs_data"]
            logger.info("Using GEECS data path from config: %s", default_path)
        else:
            logger.error("Configuration file not found or the path is not set.")
            raise FileNotFoundError(
                "Configuration file not found or the path is not set."
            )

    default_name = "Configurations.INI"
    db_name = db_ip = db_user = db_pwd = None

    ini_candidate = os.path.join(default_path, default_name)
    if not os.path.isfile(ini_candidate):
        # Lazily create a Tk context for file dialog if needed
        try:
            root = tk.Tk()
            root.withdraw()
        except Exception:  # pragma: no cover
            root = None
        path_cfg = filedialog.askopenfilename(
            filetypes=[("INI Files", "*.INI"), ("All Files", "*.*")],
            initialdir=default_path,
            initialfile=default_name,
            title="Choose a configuration file:",
        )
        if root:
            root.update_idletasks()
            root.destroy()
    else:
        path_cfg = ini_candidate

    logger.debug("configuration path selected: %s", path_cfg)
    if path_cfg:
        try:
            config = configparser.ConfigParser()
            config.read(path_cfg)
            db_name = config["Database"]["name"]
            db_ip = config["Database"]["ipaddress"]
            db_user = config["Database"]["user"]
            db_pwd = config["Database"]["password"]
            logger.info("database config loaded for %s@%s", db_user, db_ip)
        except Exception:
            logger.exception("failed to parse database configuration at %s", path_cfg)

    return db_name, db_ip, db_user, db_pwd


# --- Database access ---------------------------------------------------------
class GeecsDatabase:
    """Static helpers for reading experiment metadata from the GEECS MySQL database."""

    try:
        name, ipv4, username, password = find_database()
    except FileNotFoundError:
        logger.warning("No GEECS user data defined; skipping database initialization")
        name = ipv4 = username = password = None

    # Cache for loaded experiments (Pydantic models)
    # Key is tuple: (exp_name, enabled_devices_only, subscribed_vars_only)
    _experiment_cache: Dict[tuple, Experiment] = {}

    @staticmethod
    def _get_db():
        """Open and return a MySQL connection using class credentials."""
        db = mysql.connector.connect(
            host=GeecsDatabase.ipv4,
            user=GeecsDatabase.username,
            password=GeecsDatabase.password,
            database=GeecsDatabase.name,
        )
        return db

    @staticmethod
    def _close_db(db, db_cursor) -> None:
        """Close cursor/connection, ignoring secondary errors."""
        try:
            if db_cursor:
                db_cursor.close()
        except Exception:
            logger.debug("ignored error while closing cursor", exc_info=True)
        if db:
            try:
                db.close()
            except Exception:
                logger.debug("ignored error while closing connection", exc_info=True)

    @staticmethod
    def collect_exp_info(
        exp_name: str = "Undulator",
    ) -> Dict[str, Union[ExpDict, Dict[str, Path], Path, int, str]]:
        """Collect experiment devices, GUIs, data path, and MC port for exp_name."""
        if GeecsDatabase.name is None:
            try:
                (
                    GeecsDatabase.name,
                    GeecsDatabase.ipv4,
                    GeecsDatabase.username,
                    GeecsDatabase.password,
                ) = find_database()
            except FileNotFoundError:
                logger.error("No GEECS user data defined; database not initialized")
                GeecsDatabase.name = GeecsDatabase.ipv4 = GeecsDatabase.username = (
                    GeecsDatabase.password
                ) = None
                raise AttributeError("Geecs Database not set properly")

        db = GeecsDatabase._get_db()
        db_cursor = db.cursor(dictionary=True)

        try:
            exp_devs = GeecsDatabase._find_exp_variables(db_cursor, exp_name)
            exp_guis = GeecsDatabase._find_exp_guis(db_cursor, exp_name)
            exp_path = GeecsDatabase._find_exp_data_path(db_cursor, exp_name)
            mc_port = GeecsDatabase._find_mc_port(db_cursor, exp_name)
        finally:
            GeecsDatabase._close_db(db, db_cursor)

        exp_info: Dict[str, Any] = {
            "name": exp_name,
            "devices": exp_devs,
            "GUIs": exp_guis,
            "data_path": exp_path,
            "MC_port": mc_port,
        }
        return exp_info

    # --- Query helpers (private) --------------------------------------------
    @staticmethod
    def _find_exp_variables(db_cursor, exp_name: str = "Undulator") -> ExpDict:
        """Return dict mapping device -> variable -> attributes for exp_name."""
        cmd_str = """
            SELECT * FROM
                (
                    SELECT devicename, variablename, MAX(precedence_sourcetable) AS precedence_sourcetable
                    FROM
                    (
                        (SELECT `name` AS variablename, device AS devicename, '2_variable' AS precedence_sourcetable FROM variable)
                        UNION
                        (SELECT devicetype_variable.name AS variablename, device.name AS devicename, '1_devicetype_variable' AS precedence_sourcetable
                         FROM devicetype_variable JOIN device ON devicetype_variable.devicetype = device.devicetype)
                    ) AS variable_device_from_both_tables
                    GROUP BY devicename, variablename
                ) AS max_precedence
                LEFT JOIN
                (
                    (SELECT variable.name AS variablename, variable.device AS devicename, '2_variable' AS precedence_sourcetable,
                            defaultvalue, `min`, `max`, stepsize, units, choice_id, tolerance, alias, default_experiment, GUIexe_default
                     FROM variable JOIN device ON variable.device = device.name)
                    UNION
                    (SELECT devicetype_variable.name AS variablename, device.name AS devicename, '1_devicetype_variable' AS precedence_sourcetable,
                            defaultvalue, `min`, `max`, stepsize, units, choice_id, tolerance, alias, default_experiment, GUIexe_default
                     FROM devicetype_variable JOIN device ON devicetype_variable.devicetype = device.devicetype)
                ) AS variable_device_parameters_from_both_tables
                USING (variablename, devicename, precedence_sourcetable)
                LEFT JOIN (SELECT id AS choice_id, choices FROM choice) AS datatype USING (choice_id)
            WHERE default_experiment = %s;
        """
        db_cursor.execute(cmd_str, (exp_name,))
        rows = db_cursor.fetchall()

        exp_vars: ExpDict = {}
        while rows:
            row = rows.pop()
            dev = row["devicename"]
            var = row["variablename"]
            if dev in exp_vars:
                exp_vars[dev][var] = row
            else:
                exp_vars[dev] = {var: row}
        return exp_vars

    @staticmethod
    def _find_exp_guis(
        db_cursor,
        exp_name: str = "Undulator",
        git_base: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Path]:
        """Return dict mapping GUI display name -> executable path for exp_name."""
        base = (
            Path(r"C:\GEECS\Developers Version\builds\Interface builds")
            if git_base is None
            else Path(git_base)
        )
        db_cursor.execute(
            "SELECT `name` , `path` FROM commongui WHERE experiment = %s;", (exp_name,)
        )
        rows = db_cursor.fetchall()

        exp_guis: Dict[str, Path] = {}
        while rows:
            row = rows.pop()
            path: Path = base / row["path"][1:]
            exp_guis[row["name"]] = path
        return exp_guis

    @staticmethod
    def _find_exp_data_path(db_cursor, exp_name: str = "Undulator") -> Path:
        """Return Path to experiment’s data root directory for exp_name."""
        cmd_str = f"SELECT RootPath FROM {GeecsDatabase.name}.expt WHERE name = %s;"
        db_cursor.execute(cmd_str, (exp_name,))
        db_result = db_cursor.fetchone()
        data_path: Path = Path(db_result.popitem()[1])
        return data_path

    @staticmethod
    def _find_mc_port(db_cursor, exp_name: str = "Undulator") -> int:
        """Return MC UDP local port for exp_name."""
        cmd_str = (
            f"SELECT MCUDPLocalPortSlow FROM {GeecsDatabase.name}.expt WHERE name = %s;"
        )
        db_cursor.execute(cmd_str, (exp_name,))
        db_result = db_cursor.fetchone()
        return int(db_result["MCUDPLocalPortSlow"])

    # --- Public helpers ------------------------------------------------------
    @staticmethod
    def find_device(dev_name: str = "") -> Tuple[str, int]:
        """Return (ip, port) for device by name, or ('', 0) if not found/error."""
        db_cursor = db = None
        dev_ip: str = ""
        dev_port: int = 0

        try:
            selectors = ["ipaddress", "commport"]
            db = GeecsDatabase._get_db()
            db_cursor = db.cursor()
            db_cursor.execute(
                f"SELECT {','.join(selectors)} FROM {GeecsDatabase.name}.device WHERE name=%s;",
                (dev_name,),
            )
            db_result = db_cursor.fetchone()
            if db_result:
                dev_ip = db_result[0]
                dev_port = int(db_result[1])
            else:
                logger.warning("device %s not found", dev_name)
        except Exception:
            logger.exception('failed in find_device("%s")', dev_name)
        finally:
            GeecsDatabase._close_db(db, db_cursor)

        return dev_ip, dev_port

    @staticmethod
    def find_device_type(dev_name: str = "") -> Optional[str]:
        """Return device type string for device by name, or None if not found/error."""
        db_cursor = db = None
        dev_type: Optional[str] = None

        try:
            selectors = ["devicetype"]
            db = GeecsDatabase._get_db()
            db_cursor = db.cursor()
            db_cursor.execute(
                f"SELECT {','.join(selectors)} FROM {GeecsDatabase.name}.device WHERE name=%s;",
                (dev_name,),
            )
            db_result = db_cursor.fetchone()
            if db_result:
                dev_type = str(db_result[0])
            else:
                logger.warning("device %s not found", dev_name)
        except Exception:
            logger.exception('failed in find_device_type("%s")', dev_name)
        finally:
            GeecsDatabase._close_db(db, db_cursor)

        return dev_type

    @staticmethod
    def search_dict(
        haystack: Dict[str, Any], needle: str, path: str = "/"
    ) -> List[Tuple[str, str]]:
        """Recursively search dict values containing needle (case-insensitive) and return list of (keypath, value)."""
        results: List[Tuple[str, str]] = []
        for k, v in haystack.items():
            if v is None:
                continue
            subpath = f"{path}{k}/"
            if isinstance(v, dict):
                results.extend(GeecsDatabase.search_dict(v, needle, subpath))
            else:
                try:
                    val_str = str(v)
                    if needle.lower() in val_str.lower():
                        results.append((path + k, val_str))
                except Exception:
                    logger.debug(
                        "skipping non-stringable value at %s", path + k, exc_info=True
                    )
        return results

    @staticmethod
    def _write_default_value(
        db_cursor, new_default: str, dev_name: str, var_name: str
    ) -> bool:
        """Update the defaultvalue for (device,var) and return True if exactly one row updated."""
        # Note: table identifiers cannot be parameterized; ensure GeecsDatabase.name is trusted.
        cmd_str = (
            f"UPDATE {GeecsDatabase.name}.variable SET defaultvalue=%s "
            "WHERE device=%s AND name=%s;"
        )
        db_cursor.execute(cmd_str, (new_default, dev_name, var_name))
        db_cursor.connection.commit()
        return db_cursor.rowcount == 1

    # --- New Pydantic-based API (efficient single-query loading) -------------
    @staticmethod
    def load_experiment(
        exp_name: str = "Undulator",
        use_cache: bool = True,
        enabled_devices_only: bool = True,
        subscribed_vars_only: bool = True,
    ) -> Experiment:
        """Load complete experiment configuration as a Pydantic model.

        This method performs a single efficient SQL query to load all devices
        and their variables, including device types. The result is cached
        for subsequent calls.

        Args:
            exp_name: Name of the experiment to load (default: "Undulator")
            use_cache: Whether to use cached result if available (default: True)
            enabled_devices_only: Only include devices marked as enabled in
                exp_device table (default: True). This filters out devices
                not actively used in the experiment.
            subscribed_vars_only: Only include variables marked with get='yes'
                in exp_device_variable table (default: True). This filters out
                internal variables like ComPort, IPAddress, etc.

        Returns
        -------
            Experiment model with all devices and variables populated

        Raises
        ------
            AttributeError: If database is not properly configured

        Example:
            >>> # Default: curated view (enabled devices, subscribed variables)
            >>> exp = GeecsDatabase.load_experiment("Undulator")
            >>> print(f"Found {len(exp.devices)} devices")

            >>> # Full view: all devices and variables (no filtering)
            >>> exp_full = GeecsDatabase.load_experiment(
            ...     "Undulator",
            ...     enabled_devices_only=False,
            ...     subscribed_vars_only=False
            ... )
        """
        # Cache key includes filter settings to avoid confusion
        cache_key = (exp_name, enabled_devices_only, subscribed_vars_only)

        # Check cache first
        if use_cache and cache_key in GeecsDatabase._experiment_cache:
            logger.debug(
                "Returning cached experiment: %s (filters: enabled=%s, subscribed=%s)",
                exp_name,
                enabled_devices_only,
                subscribed_vars_only,
            )
            return GeecsDatabase._experiment_cache[cache_key]

        # Ensure database is configured
        if GeecsDatabase.name is None:
            try:
                (
                    GeecsDatabase.name,
                    GeecsDatabase.ipv4,
                    GeecsDatabase.username,
                    GeecsDatabase.password,
                ) = find_database()
            except FileNotFoundError:
                logger.error("No GEECS user data defined; database not initialized")
                raise AttributeError("GEECS Database not set properly")

        db = GeecsDatabase._get_db()
        db_cursor = db.cursor(dictionary=True)

        try:
            # Build query based on filter settings
            if enabled_devices_only or subscribed_vars_only:
                # Filtered query using exp_device and exp_device_variable tables
                cmd_str = GeecsDatabase._build_filtered_query(
                    enabled_devices_only, subscribed_vars_only
                )
            else:
                # Unfiltered query - all variables
                cmd_str = """
                    SELECT
                        var_data.*,
                        device.devicetype,
                        device.description,
                        device.ipaddress,
                        device.commport
                    FROM
                        (
                            SELECT devicename, variablename, MAX(precedence_sourcetable) AS precedence_sourcetable
                            FROM
                            (
                                (SELECT `name` AS variablename, device AS devicename, '2_variable' AS precedence_sourcetable FROM variable)
                                UNION
                                (SELECT devicetype_variable.name AS variablename, device.name AS devicename, '1_devicetype_variable' AS precedence_sourcetable
                                 FROM devicetype_variable JOIN device ON devicetype_variable.devicetype = device.devicetype)
                            ) AS variable_device_from_both_tables
                            GROUP BY devicename, variablename
                        ) AS max_precedence
                        LEFT JOIN
                        (
                            (SELECT variable.name AS variablename, variable.device AS devicename, '2_variable' AS precedence_sourcetable,
                                    defaultvalue, `min`, `max`, stepsize, units, choice_id, tolerance, alias, `set`, default_experiment, GUIexe_default
                             FROM variable JOIN device ON variable.device = device.name)
                            UNION
                            (SELECT devicetype_variable.name AS variablename, device.name AS devicename, '1_devicetype_variable' AS precedence_sourcetable,
                                    defaultvalue, `min`, `max`, stepsize, units, choice_id, tolerance, alias, `set`, default_experiment, GUIexe_default
                             FROM devicetype_variable JOIN device ON devicetype_variable.devicetype = device.devicetype)
                        ) AS var_data
                        USING (variablename, devicename, precedence_sourcetable)
                        LEFT JOIN (SELECT id AS choice_id, choices FROM choice) AS datatype USING (choice_id)
                        LEFT JOIN device ON var_data.devicename = device.name
                    WHERE var_data.default_experiment = %s;
                """

            db_cursor.execute(cmd_str, (exp_name,))
            rows = db_cursor.fetchall()

            # Build devices dictionary
            devices: Dict[str, Device] = {}
            for row in rows:
                dev_name = row["devicename"]
                var_name = row["variablename"]

                # Create device if not seen yet
                if dev_name not in devices:
                    devices[dev_name] = Device(
                        name=dev_name,
                        description=row.get("description"),
                        device_type=row.get("devicetype"),
                        ipaddress=row.get("ipaddress"),
                        commport=row.get("commport"),
                    )

                # Add variable to device
                variable = Variable.from_db_row(row)
                devices[dev_name].variables[var_name] = variable

            # Get experiment metadata (data path, MC port)
            data_path = GeecsDatabase._find_exp_data_path(db_cursor, exp_name)
            mc_port = GeecsDatabase._find_mc_port(db_cursor, exp_name)

            # Create experiment model
            experiment = Experiment(
                name=exp_name,
                devices=devices,
                data_path=data_path,
                mc_port=mc_port,
            )

            # Cache the result
            GeecsDatabase._experiment_cache[cache_key] = experiment

            filter_desc = ""
            if enabled_devices_only or subscribed_vars_only:
                filter_desc = f" (filtered: enabled_devices={enabled_devices_only}, subscribed_vars={subscribed_vars_only})"

            logger.info(
                "Loaded experiment '%s': %d devices, %d total variables%s",
                exp_name,
                len(devices),
                sum(len(d.variables) for d in devices.values()),
                filter_desc,
            )

            return experiment

        finally:
            GeecsDatabase._close_db(db, db_cursor)

    @staticmethod
    def _build_filtered_query(
        enabled_devices_only: bool, subscribed_vars_only: bool
    ) -> str:
        """Build SQL query with optional filtering by expt_device and expt_device_variable.

        Args:
            enabled_devices_only: Filter to devices where expt_device.enabled = 'yes'
            subscribed_vars_only: Filter to variables where expt_device_variable.get = 'yes'

        Returns
        -------
            SQL query string (uses %s placeholder for exp_name)
        """
        # Base query with expt_device join for filtering
        query = """
            SELECT
                var_data.*,
                device.devicetype,
                device.description,
                device.ipaddress,
                device.commport
            FROM
                expt_device ed
                JOIN device ON ed.device = device.name
                JOIN (
                    SELECT devicename, variablename, MAX(precedence_sourcetable) AS precedence_sourcetable
                    FROM
                    (
                        (SELECT `name` AS variablename, device AS devicename, '2_variable' AS precedence_sourcetable FROM variable)
                        UNION
                        (SELECT devicetype_variable.name AS variablename, device.name AS devicename, '1_devicetype_variable' AS precedence_sourcetable
                         FROM devicetype_variable JOIN device ON devicetype_variable.devicetype = device.devicetype)
                    ) AS variable_device_from_both_tables
                    GROUP BY devicename, variablename
                ) AS max_precedence ON max_precedence.devicename = ed.device
                LEFT JOIN
                (
                    (SELECT variable.name AS variablename, variable.device AS devicename, '2_variable' AS precedence_sourcetable,
                            defaultvalue, `min`, `max`, stepsize, units, choice_id, tolerance, alias, `set`, default_experiment, GUIexe_default
                     FROM variable JOIN device ON variable.device = device.name)
                    UNION
                    (SELECT devicetype_variable.name AS variablename, device.name AS devicename, '1_devicetype_variable' AS precedence_sourcetable,
                            defaultvalue, `min`, `max`, stepsize, units, choice_id, tolerance, alias, `set`, default_experiment, GUIexe_default
                     FROM devicetype_variable JOIN device ON devicetype_variable.devicetype = device.devicetype)
                ) AS var_data
                USING (variablename, devicename, precedence_sourcetable)
                LEFT JOIN (SELECT id AS choice_id, choices FROM choice) AS datatype USING (choice_id)
        """

        # Add expt_device_variable join if filtering by subscribed variables
        if subscribed_vars_only:
            query += """
                JOIN expt_device_variable edv ON edv.expt_device_id = ed.id
                    AND edv.variablename = var_data.variablename
            """

        # WHERE clause
        query += """
            WHERE ed.expt = %s
        """

        if enabled_devices_only:
            query += " AND ed.enabled = 'yes'"

        if subscribed_vars_only:
            query += " AND edv.get = 'yes'"

        return query

    @staticmethod
    def clear_experiment_cache(exp_name: Optional[str] = None) -> None:
        """Clear cached experiment data.

        Args:
            exp_name: Specific experiment to clear, or None to clear all

        Returns
        -------
            None
        """
        if exp_name is None:
            GeecsDatabase._experiment_cache.clear()
            logger.debug("Cleared all experiment cache")
        else:
            # Clear all cache entries for this experiment (any filter combination)
            keys_to_remove = [
                k for k in GeecsDatabase._experiment_cache if k[0] == exp_name
            ]
            for key in keys_to_remove:
                del GeecsDatabase._experiment_cache[key]
            logger.debug(
                "Cleared cache for experiment: %s (%d entries)",
                exp_name,
                len(keys_to_remove),
            )

    @staticmethod
    def find_exp_devices(exp_name: str = "Undulator") -> List[str]:
        """Return list of device names for an experiment.

        This is a convenience method that uses load_experiment() internally.

        Args:
            exp_name: Name of the experiment

        Returns
        -------
            List of device names
        """
        experiment = GeecsDatabase.load_experiment(exp_name)
        return experiment.list_device_names()

    @staticmethod
    def find_exp_variables(
        exp_name: str = "Undulator", device_name: str = ""
    ) -> Dict[str, Dict[str, Any]]:
        """Return variables for a specific device in an experiment.

        This method returns data in the legacy format (nested dicts) for
        backward compatibility. For new code, prefer using load_experiment()
        and accessing device.variables directly.

        Args:
            exp_name: Name of the experiment
            device_name: Name of the device

        Returns
        -------
            Dict mapping variable name -> variable attributes dict
        """
        experiment = GeecsDatabase.load_experiment(exp_name)
        device = experiment.get_device(device_name)

        if device is None:
            logger.warning(
                "Device %s not found in experiment %s", device_name, exp_name
            )
            return {}

        # Convert to legacy format
        result: Dict[str, Dict[str, Any]] = {}
        for var_name, variable in device.variables.items():
            result[var_name] = variable.raw_data

        return result


# --- Demo / CLI entrypoint ---------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    logger.info("Name: %s", GeecsDatabase.name)
    logger.info("IP: %s", GeecsDatabase.ipv4)
    logger.info("User: %s", GeecsDatabase.username)
    # Avoid logging plaintext passwords in real usage; this is for parity with prior behavior.
    logger.info("Password: %s", GeecsDatabase.password)

    try:
        exp_info = GeecsDatabase.collect_exp_info()
        logger.info("Collected exp info for %s", exp_info.get("name"))
    except Exception:
        logger.exception("Failed to collect experiment info")

    try:
        device_ip, device_port = GeecsDatabase.find_device("U_ESP_JetXYZ")
        if device_ip:
            logger.info("Device: %s, %s", device_ip, device_port)
        else:
            logger.info("Device not found")
    except Exception:
        logger.exception("Failed during device lookup")
