"""
Get scan infomation
"""

from configparser import ConfigParser
import pandas as pd
import os
import re
import glob

from functions.analysisdata import AnalysisData
from functions.time import PT_timestr

class ScanInfo:
    def __init__(self, dir_date, exp_paras):
        """
        dir_date: path of date directory
        exp_paras: list of experimental parameters to get
        """
        self.dir_date = dir_date
        self.exp_paras = exp_paras        
        
    def get_scaninfo(self, n_scan):
        """
        Get a list of scan info (one full row of scan infomation)
        """
        #get scan infomation
        info_vals = self.get_info_vals(n_scan)
        #get experimental variables
        exp_vals = self.get_exp_vals(n_scan, info_vals)        
        #combine scan info and experimental variables
        return info_vals+exp_vals                
        
    def get_scaninfo_all(self):
        """
        Get a dataframe consisting all scan infos of the day
        the columns consist of a list of info keys + experiemnt values
        """
        #alias of info_keys in the module get_info_vals
        info_keys_alias = ['scan', 'shots', 'start time',' startinfo', 'scan parameter', 'start', 'end', 'step size', 'shot/step','endinfo']
        
        #fill in data
        data = pd.DataFrame(columns=info_keys_alias+self.exp_paras)
        last_scan = self.get_last_scannumber()
        for i in range(last_scan):
            data.loc[i] = self.get_scaninfo(i+1)
        return data
        

    def get_info_vals(self, n_scan):
        """
        Get values from saninfo ini file, modify/add missing components, then insert total shot number
        and time. 
        Return: a list with values associated with info_keys
        """
        #following names has to match the key names in the Scaninfo txt file
        info_keys = ['scan no', 'scanstartinfo', 'scan parameter', 'start', 'end', 'step size', 'shots per step','scanendinfo']
        
        #Create an empty list with scan number on the first cell
        info_vals = [n_scan] + ['-'] * (len(info_keys) - 1)
        
        #Get scan info, fill the scan info into the list
        infodict = self.get_info(n_scan)
        for i in range(len(info_keys)-1):
            if info_keys[i+1] in infodict:
                info_vals[i+1] = infodict[info_keys[i+1]]
        
        #Modify or to fill up the missing components in the list
        #get additional infomation from analysis file
        analysis = AnalysisData(self.dir_date, n_scan) 
        
        #for no scan, say 'No Scan'
        if info_vals[2]=='Shotnumber':
            info_vals[2] = 'No Scan'
        else:
            #For scan, get alias of scan parameter if exists
            if analysis.get_par_alias(info_vals[2]):
                info_vals[2] = analysis.get_par_alias(info_vals[2])
            
            #For old MC version, append start&end values
            if 'start'=='-':
                info_vals[4], info_vals[5] = analysis.get_start_end_val(values[2])
                
        #Insert total shot number into second index of the list
        _, shots = analysis.get_start_end_val('Shotnumber')
        info_vals.insert(1, shots)
        
        #Insert time into third index of the list
        #get the time of the first shot
        timestamp,_ = analysis.get_start_end_val('Timestamp')
        
        #convert to pacific time
        try:
            time_str = PT_timestr(float(timestamp), "%H:%M")
        except:
            time_str = timestamp
        info_vals.insert(2, time_str)
                
        return info_vals
        

    def get_info(self, n_scan):
        '''Load scan info from ScanInfoScan***.ini as a dictionary'''
        
        #read configuration file
        scan_3d = '{0:03}'.format(n_scan)
        file_config = self.dir_date + '\\scans\\Scan' + scan_3d + '\\ScanInfoScan' + scan_3d + '.ini'
        config = ConfigParser()
        config_read = config.read(file_config)

        #Strip "", get rid of unnecessary decimals
        for i in config['Scan Info'].keys():
            config['Scan Info'][i] = config['Scan Info'][i].strip('""')            
            try:
                val = float(config['Scan Info'][i])
                if val.is_integer():
                    config['Scan Info'][i] = str(int(val))
                else:
                    config['Scan Info'][i] = str(round(val,3))                
            except:
                None
        return dict(config.items('Scan Info'))

    def get_exp_vals(self, n_scan, info_vals):
        '''
        get a list of experimental parameters associated with a list 'exp_paras', from analysis\s*.txt
        '''
        analysis = AnalysisData(self.dir_date, n_scan)
        exp_vals = []
        for i in range(len(self.exp_paras)):
            #get the value of the first shot
            exp_val,_ = analysis.get_start_end_val(self.exp_paras[i])
            # say 'scan' if this is the scan parameter
            if self.exp_paras[i] in info_vals[4]:
                exp_val = 'scan'
            exp_vals = exp_vals + [exp_val]
        return exp_vals
    
    def get_last_scannumber(self):
        '''Get the last finished scan number of the day by looking at s*info.txt'''

        path = self.dir_date + '\\analysis'
        if not os.path.isdir(path):
            return 0
        else:
            # get last scan info file name
            files = glob.glob(path + '\\s*info.txt')
            # regexp. find number in the file name
            n_scans = len(files)
            return n_scans

def main():
    dir_date = 'Z:\\data\\Undulator\\Y2020\\09-Sep\\20_0922'
    para_txt = 'Jet_X,Jet_Y,Jet_Z,Pressure,separation'

    exp_paras = para_txt.replace(", ", ",").split(",") if para_txt else None
    #get scan info
    scaninfo = scaninfo_row(dir_date, exp_paras, 19)
    print(scaninfo.scaninfo_row)
    
if __name__ == '__main__':
    main()
