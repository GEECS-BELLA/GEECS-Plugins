import os
import configparser
import mysql.connector
from geecs_errors import *
import tkinter as tk
from tkinter import filedialog

# Pop-ups initialization
tk_root = tk.Tk()
tk_root.withdraw()


def find_database():
    default_path = r'C:\GEECS\user data'
    default_name = 'Configurations.INI'

    if not os.path.isfile(os.path.join(default_path, default_name)):
        path_cfg = filedialog.askopenfilename(filetypes=[('INI Files', '*.INI'), ('All Files', '*.*')],
                                              initialdir=default_path,
                                              initialfile=default_name,
                                              title='Choose a configuration file:')
    else:
        path_cfg = os.path.join(default_path, default_name)

    if not path_cfg:
        return False

    try:
        config = configparser.ConfigParser()
        config.read(path_cfg)

        db_ip = config['Database']['ipaddress']
        db_user = config['Database']['user']
        db_pwd = config['Database']['password']
    except ErrorAPI():
        db_ip = db_user = db_pwd = ''

    return db_ip, db_user, db_pwd


class GeecsDatabase:
    database_ip, database_user, database_pwd = find_database()

    @staticmethod
    def find_device(dev_name=''):
        try:
            db = mysql.connector.connect(
                host=GeecsDatabase.database_ip,
                user=GeecsDatabase.database_user,
                password=GeecsDatabase.database_pwd)

            selectors = ["ipaddress", "commport"]
            selector_string = ",".join(selectors)

            db_cursor = db.cursor()
            db_name = 'loasis'
            select_stmt = "SELECT " + selector_string\
                          + " FROM " + db_name + ".device where name="\
                          + '"' + dev_name + '"' + ";"
            db_cursor.execute(select_stmt)
            db_result = list(db_cursor.fetchall()[0])

            dev_ip = db_result[0]
            dev_port = int(db_result[1])

        except ErrorAPI('Device not found in database!'):
            dev_ip = dev_port = ''
            pass

        return dev_ip, dev_port


if __name__ == '__main__':
    print(find_database())
