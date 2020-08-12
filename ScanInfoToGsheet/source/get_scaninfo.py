from configparser import ConfigParser
import pandas as pd
from decimal import Decimal

from getanalysisdata import analysisdata

class scaninfo_row:
    def __init__(self, dir_date, exp_paras, n_scan):
        self.dir_date = dir_date
        self.exp_paras = exp_paras
        self.n_scan = n_scan
        self.analysis = analysisdata(dir_date, n_scan) #get additional infomation from analysis file

        #get scan infomation
        self.info_vals = self.get_info_list()

        #get experimental variables
        exp_vals = self.get_exp_vals()
        #combine scan info and experimental variables
        self.scaninfo_row = self.info_vals+exp_vals

    def get_info_list(self):
        '''Modify data getting from get_info, and return a list of values associated with info_keys.'''
    
        info_keys = ['scan no', 'scanstartinfo', 'scan parameter', 'start', 'end', 'step size', 'shots per step',
               'scanendinfo']
        
        #get an empty list with scan nomber on the first cell
        info_vals = [self.n_scan] + ['-'] * (len(info_keys) - 1)
        
        #get scan info, fill the scan info into the list
        infodict = self.get_info()
        for i in range(len(info_keys)-1):
            if info_keys[i+1] in infodict:
                info_vals[i+1] = infodict[info_keys[i+1]]
                        
        #for no scan, get total shot number
        if info_vals[2]=='Shotnumber':
            info_vals[2] = 'No Scan'
            _, info_vals[6] = self.analysis.get_start_end_val('Shotnumber')
        else:
            #For scan, get alias of scan parameter if exists
            try: info_vals[2] = self.analysis.get_par_alias(info_vals[2])
            except: None
            
            #For old MC version, append start&end values
            if 'start'=='-':
                info_vals[4], info_vals[5] = self.analysis.get_start_end_val(values[2])
                _, totalshot = self.analysis.get_start_end_val('Shotnumber')
                #info_vals[6] = str(totalshot)+'/step' #get total shot number
                
        return info_vals

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

    def get_exp_vals(self):
        '''get additional experimental parameters'''
        exp_vals = []
        for i in range(len(self.exp_paras)):
            exp_val = self.analysis.get_val(self.exp_paras[i])
            # say 'scan' if this is a scan parameter
            if self.exp_paras[i] in self.info_vals[2]:
                exp_val = 'scan'
            exp_vals = exp_vals + [exp_val]
        return exp_vals


def main():
    dir_date = 'Z:\\data\\Undulator\\Y2020\\08-Aug\\20_0811'
    para_txt = 'Jet_X,Jet_Y,Jet_Z,Pressure,separation'

    exp_paras = para_txt.replace(", ", ",").split(",") if para_txt else None
    #get scan info
    scaninfo = scaninfo_row(dir_date, exp_paras, 9)
    print(scaninfo.scaninfo_row)
    
if __name__ == '__main__':
    main()
