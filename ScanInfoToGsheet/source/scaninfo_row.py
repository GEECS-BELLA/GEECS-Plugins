from configparser import ConfigParser
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime
from pytz import timezone


from analysisdata import analysisdata

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
        '''Get values from saninfo ini file, modify/add missing components, then insert total shot number
        and time. Return a list (len=10) with values associated with info_keys in getscaninfo.df_scaninfo.  '''
        
        #following names has to match the key names in the Scaninfo txt file
        info_keys = ['scan no', 'scanstartinfo', 'scan parameter', 'start', 'end', 'step size', 'shots per step',
               'scanendinfo']
        
        #Create an empty list with scan number on the first cell
        info_vals = [self.n_scan] + ['-'] * (len(info_keys) - 1)
        
        #Get scan info, fill the scan info into the list
        infodict = self.get_info()
        for i in range(len(info_keys)-1):
            if info_keys[i+1] in infodict:
                info_vals[i+1] = infodict[info_keys[i+1]]
        
        #Modify or to fill up the missing components in the list
        #for no scan, say 'No Scan'
        if info_vals[2]=='Shotnumber':
            info_vals[2] = 'No Scan'
        else:
            #For scan, get alias of scan parameter if exists
            if self.analysis.get_par_alias(info_vals[2]):
                info_vals[2] = self.analysis.get_par_alias(info_vals[2])
            
            #For old MC version, append start&end values
            if 'start'=='-':
                info_vals[4], info_vals[5] = self.analysis.get_start_end_val(values[2])
                
        #Insert total shot number into second index of the list
        _, shots = self.analysis.get_start_end_val('Shotnumber')
        info_vals.insert(1, shots)
        
        #Insert time into third index of the list
        #get the time of the first shot
        timestamp,_ = self.analysis.get_start_end_val('Timestamp')
        
        #convert to pacific time
        time_str = PT_timestr(float(timestamp), "%H:%M")
        info_vals.insert(2, time_str)
                
        return info_vals
        

    def get_info(self):
        '''Get info from ScanInfoScan***.ini as a dictionary'''
        
        #read configuration file
        scan_3d = '{0:03}'.format(self.n_scan)
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

    def get_exp_vals(self):
        '''get additional experimental parameters, return the list'''
        exp_vals = []
        for i in range(len(self.exp_paras)):
            #get the value of the first shot
            exp_val,_ = self.analysis.get_start_end_val(self.exp_paras[i])
            # say 'scan' if this is the scan parameter
            if self.exp_paras[i] in self.info_vals[4]:
                exp_val = 'scan'
            exp_vals = exp_vals + [exp_val]
        return exp_vals

def PT_timestr(lvtimestamp,strformat):
    '''Conert the labview timestamp to pacific time.
    lvtimestamp: labview timestamp (float). should be 10 digit (36...)
    strformat: format of the string to be returned. ex) "%m/%d/%Y, %H:%M:%S"'''
    lv_dt = datetime.fromtimestamp(lvtimestamp) #labview time
    utc_dt = lv_dt - relativedelta(years=66, days=1) #UTC time
    #convert to Pacific time
    ca_tz = timezone('America/Los_Angeles')
    ca_date = utc_dt.astimezone(ca_tz)

    return ca_date.strftime(strformat)
    
def main():
    dir_date = 'Z:\\data\\Undulator\\Y2020\\09-Sep\\20_0922'
    para_txt = 'Jet_X,Jet_Y,Jet_Z,Pressure,separation'

    exp_paras = para_txt.replace(", ", ",").split(",") if para_txt else None
    #get scan info
    scaninfo = scaninfo_row(dir_date, exp_paras, 19)
    print(scaninfo.scaninfo_row)
    
if __name__ == '__main__':
    main()
