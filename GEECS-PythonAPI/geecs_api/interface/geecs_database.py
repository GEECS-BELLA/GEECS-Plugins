import os
from typing import Any
import configparser
import mysql.connector
from typing import Union, Optional
from geecs_api.api_defs import ExpDict, SysPath
from geecs_api.interface.geecs_errors import api_error
import tkinter as tk
from tkinter import filedialog

# Pop-ups initialization
tk_root = tk.Tk()
tk_root.withdraw()


def find_database():
    default_path = r'C:\GEECS\user data'
    default_name = 'Configurations.INI'

    db_name = db_ip = db_user = db_pwd = ''
    if not os.path.isfile(os.path.join(default_path, default_name)):
        path_cfg = filedialog.askopenfilename(filetypes=[('INI Files', '*.INI'), ('All Files', '*.*')],
                                              initialdir=default_path,
                                              initialfile=default_name,
                                              title='Choose a configuration file:')
    else:
        path_cfg = os.path.join(default_path, default_name)

    if path_cfg:
        try:
            config = configparser.ConfigParser()
            config.read(path_cfg)

            db_name = config['Database']['name']
            db_ip = config['Database']['ipaddress']
            db_user = config['Database']['user']
            db_pwd = config['Database']['password']

        except Exception:
            pass

    return db_name, db_ip, db_user, db_pwd


class GeecsDatabase:
    name, ipv4, username, password = find_database()

    @staticmethod
    def _get_db():
        db = mysql.connector.connect(
            host=GeecsDatabase.ipv4,
            user=GeecsDatabase.username,
            password=GeecsDatabase.password,
            database=GeecsDatabase.name)
        return db

    @staticmethod
    def _close_db(db, db_cursor):
        try:
            db_cursor.close()
        except Exception:
            pass

        if db:
            try:
                db.close()
            except Exception:
                pass

    @staticmethod
    def collect_exp_info(exp_name: str = 'Undulator')\
            -> dict[str, Union[ExpDict, dict[str, SysPath], SysPath, int]]:
        db = GeecsDatabase._get_db()
        db_cursor = db.cursor(dictionary=True)

        exp_devs = GeecsDatabase._find_exp_variables(db_cursor, exp_name)
        exp_guis = GeecsDatabase._find_exp_guis(db_cursor, exp_name)
        exp_path = GeecsDatabase._find_exp_data_path(db_cursor, exp_name)
        mc_port = GeecsDatabase._find_mc_port(db_cursor, exp_name)

        exp_info: dict[str, Any] = {'name': exp_name,
                                    'devices': exp_devs,
                                    'GUIs': exp_guis,
                                    'data_path': exp_path,
                                    'MC_port': mc_port}

        GeecsDatabase._close_db(db, db_cursor)
        return exp_info

    @staticmethod
    def _find_exp_variables(db_cursor, exp_name: str = 'Undulator') -> ExpDict:
        """ Dictionary of (key) devices with (values) dictionary of (key) variables and (values) attributes. """

        cmd_str = """
            SELECT * FROM
                -- subquery that returns devicename, variablename, and source table where the defaultvalue,
                -- min, max, etc. should come from
                (
                    SELECT devicename, variablename, MAX(precedence_sourcetable) AS precedence_sourcetable
                    FROM
                    (
                        (
                        SELECT `name` AS variablename, device AS devicename,
                        '2_variable' AS precedence_sourcetable FROM variable
                        )
                    UNION
                        (
                        SELECT devicetype_variable.name AS variablename, device.name AS devicename,
                        '1_devicetype_variable' AS precedence_sourcetable
                        FROM devicetype_variable JOIN device ON devicetype_variable.devicetype = device.devicetype
                        )
                    ) AS variable_device_from_both_tables GROUP BY devicename, variablename
                ) AS max_precedence
                -- subquery containing defaultvalue, min, max, etc from both tables. the correct version,
                -- by precedence, is selected through the join.
                LEFT JOIN
                (
                    (
                    SELECT variable.name AS variablename, variable.device AS devicename,
                    '2_variable' AS precedence_sourcetable, defaultvalue, `min`, `max`, stepsize, units,
                    choice_id, tolerance, alias, default_experiment, GUIexe_default 
                    FROM variable JOIN device ON variable.device = device.name -- to pull default_experiment
                    )
                UNION
                    (
                    SELECT devicetype_variable.name AS variablename, device.name AS devicename,
                    '1_devicetype_variable' AS precedence_sourcetable, defaultvalue, `min`, `max`, stepsize, units,
                    choice_id, tolerance, alias, default_experiment, GUIexe_default 
                    FROM devicetype_variable JOIN device ON devicetype_variable.devicetype = device.devicetype
                    )
                ) AS variable_device_parameters_from_both_tables 
                USING (variablename, devicename, precedence_sourcetable) 
                -- Get datatype
                LEFT JOIN (SELECT id AS choice_id, choices FROM choice) AS datatype USING (choice_id)
                -- Now filter for device, experiment
            WHERE default_experiment = %s;
        """
        db_cursor.execute(cmd_str, (exp_name,))
        rows = db_cursor.fetchall()

        exp_vars: ExpDict = {}
        while rows:
            row = rows.pop()
            if row['devicename'] in exp_vars:
                exp_vars[row['devicename']][row['variablename']] = row
            else:
                exp_vars[row['devicename']] = {row['variablename']: row}

        return exp_vars

    @staticmethod
    def _find_exp_guis(db_cursor, exp_name: str = 'Undulator',
                       git_base: Optional[Union[str, os.PathLike]] = None) -> dict[str, SysPath]:
        """ Dictionary of (key) descriptive names with (values) executable paths. """

        if git_base is None:
            git_base = r'C:\GEECS\Developers Version\builds\Interface builds'

        cmd_str = 'SELECT `name` , `path` FROM commongui WHERE experiment = %s;'
        db_cursor.execute(cmd_str, (exp_name,))
        rows = db_cursor.fetchall()

        exp_guis: dict[str, SysPath] = {}
        while rows:
            row = rows.pop()
            path: SysPath = os.path.join(git_base, row['path'][1:])
            exp_guis[row['name']] = path

        return exp_guis

    @staticmethod
    def _find_exp_data_path(db_cursor, exp_name: str = 'Undulator') -> SysPath:
        """ Path to experiment's data root directory. """

        cmd_str = f'SELECT RootPath FROM {GeecsDatabase.name}.expt WHERE name = %s;'

        db_cursor.execute(cmd_str, (exp_name,))
        db_result = db_cursor.fetchone()
        data_path: SysPath = os.path.realpath(db_result.popitem()[1])

        return data_path

    @staticmethod
    def _find_mc_port(db_cursor, exp_name: str = 'Undulator') -> int:
        """ Dictionary of (key) devices with (values) dictionary of (key) variables and (values) attributes. """

        cmd_str = f'SELECT MCUDPLocalPortSlow FROM {GeecsDatabase.name}.expt WHERE name = %s;'
        db_cursor.execute(cmd_str, (exp_name,))
        db_result = db_cursor.fetchone()
        mc_port = int(db_result['MCUDPLocalPortSlow'])

        return mc_port

    @staticmethod
    def find_device(dev_name=''):
        db_cursor = db = None
        dev_ip: str = ''
        dev_port: int = 0

        try:
            selectors = ["ipaddress", "commport"]

            db = GeecsDatabase._get_db()
            db_cursor = db.cursor()
            db_cursor.execute(f'SELECT {",".join(selectors)} FROM {GeecsDatabase.name}.device WHERE name=%s;',
                              (dev_name,))
            db_result = db_cursor.fetchone()
            dev_ip = db_result[0]
            dev_port = int(db_result[1])

        except Exception as ex:
            api_error.error(str(ex), f'GeecsDatabase class, static method "find_device({dev_name})"')

        GeecsDatabase._close_db(db, db_cursor)
        return dev_ip, dev_port

    @staticmethod
    def search_dict(haystack: dict, needle: str, path="/") -> list[tuple[str, str]]:
        search_results = []
        for k, v in haystack.items():
            if v is None:
                continue
            elif isinstance(v, dict):
                GeecsDatabase.search_dict(v, needle, path + k + '/')
            elif needle in v.lower():
                search_results.append((path + k, v))

        return search_results

    @staticmethod
    def _write_default_value(db_cursor, new_default: str, dev_name: str, var_name: str) -> bool:
        """ Updates the default value of a variable. """

        cmd_str = f'UPDATE {db_cursor.connection.escape_string(GeecsDatabase.name)}.variable SET defaultvalue=%s ' \
                  f'WHERE device=%s AND name=%s;'
        db_cursor.execute(cmd_str, (new_default, dev_name, var_name))
        db_cursor.connection.commit()

        return db_cursor.rowcount == 1


if __name__ == '__main__':
    print('Name:\t\t' + GeecsDatabase.name)
    print('IP:\t\t\t' + GeecsDatabase.ipv4)
    print('User:\t\t' + GeecsDatabase.username)
    print('Password:\t' + GeecsDatabase.password)

    api_error.clear()

    _exp_info = GeecsDatabase.collect_exp_info()
    device_ip, device_port = GeecsDatabase.find_device('U_ESP_JetXYZ')
    print(api_error)

    if device_ip:
        print('Device:\t' + device_ip + f', {device_port}')
    else:
        print('Device not found')
