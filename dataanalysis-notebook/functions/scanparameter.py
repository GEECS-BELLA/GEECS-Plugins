from configparser import ConfigParser
import pandas as pd

def get_info(dir_date, n_scan):
    '''Load scan info from ScanInfoScan***.ini as a dictionary
    taken from ScanInfoGsheet scaninfo.py
    '''

    #read configuration file
    scan_3d = '{0:03}'.format(n_scan)
    file_config = dir_date + '\\scans\\Scan' + scan_3d + '\\ScanInfoScan' + scan_3d + '.ini'
    config = ConfigParser()
    config_read = config.read(file_config)

    #Strip "", get rid of unnecessary decimals
    for i in config['Scan Info'].keys():
        config['Scan Info'][i] = config['Scan Info'][i].strip('""')            

    return dict(config.items('Scan Info'))

def get_fullkey(data, par):
    '''Get a full key name from the dataframe.
    Return None if the parameter cannot be found
    '''
    i_par = [k for k, s in enumerate(list(data)) if par in s]
    if i_par:
        return list(data)[i_par[0]]
    else:
        return None


def get_scanpara(dir_date, scan):
    """
    Get scan parameters of the scan
    return:If it is a scan, dataframe with shotnumber and scan values.
            If not a scan, None
    """
    
    #get scan paramter from info file
    info = get_info(dir_date, scan)
    scanpara = info['scan parameter']

    if scanpara=='Shotnumber':
        return None
    else:
        #read data from analysis file
        f_ana = dir_date + '\\analysis\\s' + str(scan) + '.txt'
        data = pd.read_csv(f_ana, sep='\t')

        #get full name of the parameter that includes alias
        par_full = get_fullkey(data, scanpara)

        #get a dataframe with shotnumber and a scanning parameter
        df_s = data[[par_full]]

        #if there is an alias, change the key name to the alias
        if 'Alias' in par_full:
            alias = par_full.split('Alias:', 1)[1]
            print('alias: ', alias)
            df_s = df_s.rename(columns={par_full:alias})
    
        return df_s