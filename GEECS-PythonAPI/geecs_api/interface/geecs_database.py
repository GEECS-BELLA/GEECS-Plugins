import os
import configparser
import mysql.connector
from geecs_api.interface.geecs_errors import *
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
    def find_device(dev_name=''):
        db_cursor = None
        dev_ip = ''
        dev_port = 0
        err = ErrorAPI()

        try:
            db = mysql.connector.connect(
                host=GeecsDatabase.ipv4,
                user=GeecsDatabase.username,
                password=GeecsDatabase.password)

            selectors = ["ipaddress", "commport"]

            db_cursor = db.cursor()
            db_cursor.execute(f'SELECT {",".join(selectors)} FROM {GeecsDatabase.name}.device WHERE name=%s;',
                              (dev_name,))
            db_result = db_cursor.fetchone()
            dev_ip = db_result[0]
            dev_port = int(db_result[1])

        except Exception as ex:
            err = ErrorAPI(str(ex), 'GeecsDatabase class, static method "find_device"')

        finally:
            try:
                db_cursor.close()
            except Exception:
                pass

        return dev_ip, dev_port, err


if __name__ == '__main__':
    print('Name:\n\t' + GeecsDatabase.name)
    print('IP:\n\t' + GeecsDatabase.ipv4)
    print('User:\n\t' + GeecsDatabase.username)
    print('Password:\n\t' + GeecsDatabase.password)

    device_ip, device_port, error = GeecsDatabase.find_device('U_ESP_JetXYZ')
    print(error)

    if device_ip:
        print('Device:\n\t' + device_ip + f', {device_port}')
    else:
        print('Device not found')
