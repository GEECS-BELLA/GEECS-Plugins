import os
import configparser
import mysql.connector
from geecs_errors import *


def find_database():
    found = False
    while not found:
        os.chdir('../..')
        found = os.path.exists('user data')

    try:
        os.chdir('user data')
        config = configparser.ConfigParser()
        config.read('Configurations.INI')

        db_ip = config['Database']['ipaddress']
        db_user = config['Database']['user']
        db_pwd = config['Database']['password']
    except ErrorAPI():
        db_ip = db_user = db_pwd = 'boo'

    return db_ip, db_user, db_pwd


class GeecsDatabase:
    database_ip, database_user, database_pwd = find_database()

    @staticmethod
    def lookup_database(dev_name=''):
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
            dev_tcp_port = int(db_result[1])

        except ErrorAPI('Device not found in database!'):
            dev_ip = dev_tcp_port = ''
            pass

        return dev_ip, dev_tcp_port


if __name__ == '__main__':
    print(find_database())
