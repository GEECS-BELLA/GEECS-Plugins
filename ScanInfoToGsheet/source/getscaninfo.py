import os
import re
import pandas as pd
import glob
from scaninfo_row import scaninfo_row


def df_scaninfo(dir_date, exp_paras):
    '''Create a dataframe from all the scan data'''

    #following naming can be changed, but are associated with info_keys in scaninfo_row.get_info_list function.
    info_keys = ['scan', 'shots', 'start time',' startinfo', 'scan parameter', 'start', 'end', 'step size', 'shot/step',
               'endinfo']        
    #add experimental parameters to the column names
    print(exp_paras)
    columns = info_keys+exp_paras

    # fill in data
    data = pd.DataFrame(columns=columns)
    last_scan = get_last_scannumber(dir_date)
    for i in range(last_scan):
        scaninfo = scaninfo_row(dir_date, exp_paras, i+1)
        data.loc[i] = scaninfo.scaninfo_row
    return data

def get_last_scannumber(dir_date):
    '''Get the last scan number which is already done'''

    path = dir_date + '\\analysis'
    if not os.path.isdir(path):
        return 0
    else:
        # get last scan info file name
        files = glob.glob(path + '\\s*info.txt')
        file_last = os.path.basename(files[-1])
        # regexp. find number in the file name
        n_scans = int(re.findall(r"\d+", file_last)[0])
        return n_scans
