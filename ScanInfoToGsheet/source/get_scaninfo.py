from configparser import ConfigParser
import pandas as pd
from decimal import Decimal

from getanalysisdata import analysisdata

class get_scaninfo:
    def __init__(self, dir_date, para_txt, n_scan):
        self.dir_date = dir_date
        self.para_txt = para_txt
        self.n_scan = n_scan

    def get_info(self):
    '''Get info from ScanInfoScan***.ini as a dictionary'''
    
    #read configuration file
    scan_3d = '{0:03}'.format(self.n_scan)
    file_config = self.dir_date + '\\scans\\Scan' + scan_3d + '\\ScanInfoScan' + scan_3d + '.ini'
    config = ConfigParser()
    config_read = config.read(file_config)

    #Strip "", normalize numbers (get rid of unnecessary decimals)
    for i in config['Scan Info'].keys():
        config['Scan Info'][i] = config['Scan Info'][i].strip('""')
        try:
            config['Scan Info'][i] = str(Decimal(config['Scan Info'][i]).normalize())
        except:
            None
    return dict(config.items('Scan Info'))


def main():
    dir_date = 'Z:\\data\\Undulator\\Y2020\\08-Aug\\20_0811'
    para_txt = 'Jet_X,Jet_Y,Jet_Z,Pressure,separation'
    n_scan = 10
    info_scan = get_scaninfo(dir_date, para_txt, 10)
    a = info_scan.get_info()
    
if __name__ == '__main__':
    main()
