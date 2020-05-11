from configparser import ConfigParser
import os
import re
import pandas as pd
import glob
from getanalysisdata import analysisdata


def df_scaninfo(dir_date, para_txt):
    '''Create a dataframe from all the scan data'''

    #get column names for the dataframe
    info_key = config_keys(dir_date)
    if len(info_key)==3: #old MC version
        columns = ['Scan', 'Start Info', 'Scan Parameter','Start','End','Shots']
    elif len(info_key)==8:#new MC from May 2020
        columns = ['Scan', 'Start Info', 'Scan Parameter', 'Start', 'End',
               'Step', 'Shots/step', 'End Info']
    else:
        columns = info_key
        
    #add experimental parameters to the column names    
    if para_txt:
        exp_paras = para_txt.replace(", ", ",").split(",")
        columns = columns+exp_paras

    # fill in data
    data = pd.DataFrame(columns=columns)
    last_scan = get_last_scannumber(dir_date)
    for i in range(last_scan):
        info_row = get_scaninfo_row(dir_date, para_txt, i + 1, len(columns))
        data.loc[i] = info_row
    return data

def config_keys(dir_date, last_scan=10):
    '''Get keys of a ScanInfo(config) file. Look for each scan until finding non-empty one.
    '''
    keys = None
    for i in range(last_scan):
        keys,values = get_scaninfo(dir_date, i+1)
        if keys:
            break
    return keys

def get_scaninfo_row(dir_date, para_txt, n_scan, len_col):
    '''Get scan info for a scan (n_scan). Return a text of a row to write on a table.'''

    #Get basic scan info from scaninfo file
    keys, values = get_scaninfo(dir_date, n_scan)
    if not keys:
        row_list = [n_scan] + ['-'] * (len(len_col) - 1)
    else:
        #Get data from analysis file
        analysis = analysisdata(dir_date, n_scan)
        
        # Say 'No scan' if scan parameter is 'Shotnumber'. 
        # Get alias of scan parameter if exists
        if values[2] in 'Shotnumber':
            values[2] = 'No Scan'
        elif analysis.get_par_alias(values[2]):
            values[2] = analysis.get_par_alias(values[2])

        for i in range(1,len(values)):
            #replace None with '-'
            if not values[i]:
                values[i] = '-'
            # round up a value to 3 decimal
            elif values[i][0].isdecimal():
                values[i] = str(round(float(values[i][0]), 3))
            
        #in case of old MC version, append additional info
        if len(keys)==3:
            _, n_shot = analysis.get_start_end_val('Shotnumber')
            val_start, val_end = analysis.get_start_end_val(values[2])
            values = values + [val_start, val_end, n_shot]

        #get additional experimental parameters
        exp_vals = []
        if para_txt:
            exp_paras = para_txt.replace(", ", ",").split(",")
            for i in range(len(exp_paras)):
                exp_val = analysis.get_val(exp_paras[i])
                # say 'scan' if this is a scan parameter
                if exp_paras[i] in values[2]:
                    exp_val = 'scan'
                exp_vals = exp_vals + [exp_val]

        # Make a row of the dataframe
        row_list = values + exp_vals

    return row_list


def get_scaninfo(dir_date, n_scan):
    '''Get scan number, scan parameter, scan start info from ScanInfoScan***.ini'''
    scan_3d = '{0:03}'.format(n_scan)
    file_config = dir_date + '\\scans\\Scan' + scan_3d + '\\ScanInfoScan' + scan_3d + '.ini'
    config = ConfigParser()
    config_read = config.read(file_config)
    if config_read:
        keys = list(config['Scan Info'].keys())
        values = [i.strip('"') for i in config['Scan Info'].values()]
    else:
        keys, values = None, None
    return keys, values


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


def get_val(dir_date, n_scan, par):
    "Get the parameter value of the first shot"
    file = dir_date + '\\analysis\\s' + str(n_scan) + '.txt'
    data = pd.read_csv(file, sep='\t')
    indices = [k for k, s in enumerate(list(data)) if par in s]
    if not indices or data.empty:
        return '-'
    else:
        par_full = list(data)[indices[0]]
        return round(data[par_full].iloc[0], 3)

def get_start_end_val(dir_date, n_scan, par, isalias=True):
    '''Get the value of the first shot and the last shot. Using this only for the old MC version'''

    file = dir_date + '\\analysis\\s' + str(n_scan) + '.txt'
    data = pd.read_csv(file, sep='\t')
    indices = [k for k, s in enumerate(list(data)) if par in s]
    if not indices or data.empty:
        if par=='Shotnumber':
            return par, 0, 0
        else:
            return '-', '-', '-'
    else:
        par_full = list(data)[indices[0]]
        val_first, val_end = data[par_full].iloc[0], data[par_full].iloc[-1]

    # Get Alias if exists
    if isalias:
        if 'Alias' in par_full:
            par_full = par_full.split('Alias:', 1)[1]
    return par_full, round(val_first, 3), round(val_end, 3)