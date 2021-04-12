"""
Following functions are specific to the analysis of the data saved
with BELLA control system
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import re
from numpy import unravel_index
import numpy as np
from scipy import stats
import json
from functions.data_analysis import df_outlier2none

def get_data(dir_date, nscan=None, para=None, trim_std=None):
    '''Get DataFrame 
    dir_date: directory of a date where scan data is stored (str)
    nscan: list of scan number(int)
    para_list: list of parameters(str).No need to write the full name.
    '''
    path = get_scan_path(dir_date, nscan)
    df = get_data_from_path(path, para)
    
    #parameters to consider getting rid of outliers...(don't consider scan)
    para_vals = list(df.columns)
    if 'scan' in para_vals:
        para_vals.remove('scan')
    if 'DateTime Timestamp' in para_vals:
        para_vals.remove('DateTime Timestamp')
    if 'Shotnumber' in para_vals:
        para_vals.remove('Shotnumber')
    
    #get rid of outliers
    if trim_std:
        df_new = df_outlier2none(df, std=trim_std, columns = para_vals )
    return df


def get_files_list(dirpath,f_format):
    """
    get get path of all files with f_format in the directory
    dir_date: directory path
    f_format: ex) txt
    """
    return sorted(glob.glob(dirpath+'/*.'+f_format))
    

def get_notebook_name():
    """
    Return the full path of the jupyter notebook.
    """
    import ipykernel
    import requests
    from requests.compat import urljoin
    from notebook.notebookapp import list_running_servers
    
    kernel_id = re.search('kernel-(.*).json',
                          ipykernel.connect.get_connection_file()).group(1)
    servers = list_running_servers()
    for ss in servers:
        response = requests.get(urljoin(ss['url'], 'api/sessions'),
                                params={'token': ss.get('token', '')})
        for nn in json.loads(response.text):
            if nn['kernel']['id'] == kernel_id:
                relative_path = nn['notebook']['path']
                return os.path.join(ss['notebook_dir'], relative_path)

def save_dataframe(df, name, ipynb = None):
    '''save dataframe under data/"current ipython name"/'''
    
    #get the file name of ipynb
    if ipynb == None:
        ipynb_fullpath = get_notebook_name()
        ipynb = os.path.splitext(os.path.basename(ipynb_fullpath))[0]    
    
    #Open the data folder if doesnt exist
    if not os.path.exists('data_ipynb'):
        os.makedirs('data_ipynb')
    if not os.path.exists('data_ipynb/'+ipynb):
        os.makedirs('data_ipynb/'+ipynb)
    #Save data
    df.to_pickle('data_ipynb/'+ipynb+'/'+name+'.pkl')
    print(name+' saved')
    return None

def load_dataframe(name, ipynb = None):
    """load dataframe which was saved using the function save_dataframe
    name: correspons to the name of the daframe you sppecified with save_dataframe
    ipynb: the ipynb name you are running. If None, it will be automatically aquired. (NOt working sometime).
    """
    #get the file name of ipynb
    if ipynb == None:
        ipynb_fullpath = get_notebook_name()
        ipynb = os.path.splitext(os.path.basename(ipynb_fullpath))[0]
    load_path = 'data_ipynb/'+ipynb+'/'+name+'.pkl'
    
    df = pd.read_pickle(load_path)
    print(name+' loaded')
    return df
    
def get_data_from_path(path_list, para_list = None):
    '''Get DataFrame from the file.
    path_list: a filename or list of multiple filenames. they will append all data sets.
    para_list: list of parameters (column names) you want to select from dataframe
    output: dataframe
    '''
    data_list = []
    for i in range(len(path_list)):
        data_i = pd.read_csv(path_list[i], sep='\t')
        if para_list:
            #get full name of the parameters
            para_list_full = []
            for j in para_list:
                para_full = par_full(path_list[i], j)
                if para_full:
                    para_list_full = para_list_full+[para_full]
            #If you can get all parameters, append the data of the scan
            if len(para_list_full) == len(para_list):
                data_i = data_i[para_list_full]
                data_list.append(data_i)
            else:
                print('Skip saving data from', os.path.basename(path_list[i]))
        else:
            #if there is no para_list, get all the parameters that are saved
            data_list.append(data_i)
                        
    data = pd.concat(data_list, sort=False)
    
    #rename column names to alias if exists
    for col in data.columns:
        if 'Alias:' in col:
            alias = col.split('Alias:', 1)[1]
            data = data.rename(columns={col:alias})
    return data

def get_nscan_last(dir_date):
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
    
def get_scan_path(dir_date, nscan=None):
    '''
    Get a path of the scan file s**.txt in the analysis 
    nscan: List or int of scan number. if None, creat a list of all scan text paths
    '''

    #if nscan_list=None, make a list of all scan #s
    if not nscan:
        nscan_last = get_nscan_last(dir_date)
        nscan_list = range(1, nscan_last+1)
    elif isinstance(nscan, int):
        nscan_list = [nscan]
    else:
        nscan_list = nscan
    path_list = []
    #make a list of all scan file paths
    for i in nscan_list:
        path = dir_date + '\\analysis\\s' + str(i) + '.txt'
        path_list = path_list + [path]
    return path_list
    
def par_full(file, par):
    '''get a full name of the parameter'''

    data = pd.read_csv(file, sep='\t')
    indices = [k for k, s in enumerate(list(data)) if par in s]
    if not indices or data.empty:
        print(par, 'not found in', os.path.basename(file))
        return None
    elif len(indices) > 1:
        for j in indices:
            if list(data)[j]==par:
                return list(data)[j]
        raise NameError('Please Specify the Name. Several parameters match for ',par,list( list(data)[i] for i in indices )  )
        return None
    else:
        return list(data)[indices[0]]

def show_time_xaxis():
    '''X axis changed from timestamp to day time of california when you plot a graph with time stamp on x axis.
    '''
    from datetime import datetime
    summer20_start = datetime.timestamp(datetime(2020,3,8,3,0))
    summer20_end = datetime.timestamp(datetime(2020,11,1,2,0))
    # get current axis
    ax = plt.gca()
    # get current xtick labels
    xticks = ax.get_xticks()
    if xticks[0] > summer20_start and xticks[0] <summer20_end:
        xticks = xticks - 7*60*60
    else:
        xticks = xticks - 8*60*60
    # convert all xtick labels to selected format from ms timestamp
    ax.set_xticklabels([pd.to_datetime(tm, unit='s').strftime('%Y-%m-%d\n %H:%M:%S') for tm in xticks],
     rotation=50)
    return None

def get_calib(Dict, device, axis = None):
    '''Get paramters from calibration dictionary
    device: device name except axis (drop off the X or Y in the device name)'''
    
    if device in Dict:
        #get target position
        try: target = Dict[device]['target'][axis]
        except: target = 0

        #get sign (positive or negative)
        try: sign = Dict[device]['sign'][axis]
        except: sign = 1

        #get calibration
        try: calib = Dict[device]['calib'][axis]
        except: 
            try: calib = Dict[device]['calib']
            except: calib = 1
        
        #get unit
        try: unit = ' ('+Dict[device]['unit']+')'
        except: unit = ''
        
        return target, sign, calib, unit
            
    #if 2ndmomW0, get calibration data from centroid data
    elif device.split()[1]=='2ndmomW0' or '2ndmom' or '2mdmom' or 'FWHM':
        device_centroid = device.split()[0]+' centroid'
        return get_calib(Dict, device_centroid)
    
    else:
        print('can not find a calibration for ', device)
        return None, None, None, None
    
def PT_time(lvtimestamp):
    '''
    Conert the labview timestamp to pacific time.
    lvtimestamp: labview timestamp (float). should be 10 digit (36...)
    "'''
    lv_dt = datetime.fromtimestamp(lvtimestamp) #labview time
    utc_dt = lv_dt - relativedelta(years=66, days=1) #UTC time
    #convert to Pacific time
    ca_tz = timezone('America/Los_Angeles')
    ca_date = utc_dt.astimezone(ca_tz)

    return ca_date
    

def df_calib(df, Dict):
    '''Return a new dataframe with calibrated values. Only those with calibration saved in Dict are stored in the
    output dataframe.
    df: DataFrame
    Dict: calibration dictionary. It's in a separate file
    ''' 
    #parameters of the DataFrame
    para = df.columns
    #define a new dataframe
    df_new = pd.DataFrame()
    
    #get device name and axis for the camera
    for i in para:
        #if camera device
        if i[:3]=='UC_':  
            #if add column is used, strip ..T..., and find axis
            if i.find(' T ') > -1:
                axis = i[i.find(' T ')-1]
                device = i[:i.find(' T ')-1].rstrip() #get rid of space in the end if exist
            #if the name ends with x or y
            elif i[-1]=='x' or i[-1]=='y':
                axis = i[-1]
                device=i[:-1]
            else:
                axis=''
                device = i
                print('cant identify the parameter ',i)
                
            #get the calibration and convert to the calibrated data
            target, sign, calib, unit = get_calib(Dict, device, axis)
            para_new = device[3:]+axis+unit
            df_new[para_new]=sign*(df[i]-target)*calib                
                        
        #non camera device but exists in calibration dictionary
        elif i in Dict:
            if i[:2]=='U_':
                df_new[i[2:]] = df[i]
            else:
                target, sign, calib, unit = get_calib(Dict, i)
                df_new[i+unit] = sign*(df[i]-target)*calib
                
        #if it's timestamp
        elif i=='DateTime Timestamp':            
            df_new['DateTime (Pacific Time)'] = PT_time(df[i])
        
        else:
            print('can not find a calibration data for ', para[i])
    
    if len(df_new.columns) < len(para):
        print( len(para) - len(df_new.columns), 'parameters not saved to the calibrated dataframe' )
    return df_new


    
    