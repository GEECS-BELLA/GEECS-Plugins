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

        db_name = config['Database']['name']
        db_ip = config['Database']['ipaddress']
        db_user = config['Database']['user']
        db_pwd = config['Database']['password']
    except ErrorAPI():
        db_name = db_ip = db_user = db_pwd = ''

    return db_name, db_ip, db_user, db_pwd


class GeecsDatabase:
    database_name, database_ip, database_user, database_pwd = find_database()

    @staticmethod
    def find_device(dev_name=''):
        try:
            db = mysql.connector.connect(
                host=GeecsDatabase.database_ip,
                user=GeecsDatabase.database_user,
                password=GeecsDatabase.database_pwd)

            selectors = ["ipaddress", "commport"]

            with db.cursor() as db_cursor:
                db_cursor.execute(f'SELECT {",".join(selectors)} FROM device WHERE name=%s;', (dev_name,))
                db_result = db_cursor.fetchone()
                dev_ip = db_result[0]
                dev_port = int(db_result[1])

        except Exception as ex:
            print(ex)
            dev_ip = ''
            dev_port = 0

        finally:
            db_cursor.close()

        return dev_ip, dev_port


if __name__ == '__main__':
    print('Name:\n\t' + GeecsDatabase.database_name)
    print('IP:\n\t' + GeecsDatabase.database_ip)
    print('User:\n\t' + GeecsDatabase.database_user)
    print('Password:\n\t' + GeecsDatabase.database_pwd)

    device_ip, device_port = GeecsDatabase.find_device('U_ESP_JetXYZ')
    if device_ip:
        print('Device:\n\t' + device_ip + f', {device_port}')
    else:
        print('Device not found')

    MCAST_GRP = '234.5.6.8'
    MCAST_PORT = 58432
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, MULTICAST_TTL)
    sock.sendto("preset>>HTU_Amp4>>192.168.7.227".encode(), (MCAST_GRP, MCAST_PORT))
    sock.close()
