import pandas as pd

class analysisdata():
    '''Get data(alias of a parameter or start&end value of the parameter) of a scan from
    analysis\s**.txt file
    dir_date: Date directory where scan data are stored.
    n_scan: scan number (int)
    '''
    
    def __init__(self, dir_date, n_scan):
            self.file = dir_date + '\\analysis\\s' + str(n_scan) + '.txt'
            self.data = pd.read_csv(self.file, sep='\t')
            
    def get_val(self, par):
        "Get the parameter value of the first shot"
        
        #get index number of the experimental parameter
        if not self.idx(par) or self.data.empty:            
            return '-'
        else:
            par_full = list(self.data)[self.idx(par)]
            return str(round(self.data[par_full].iloc[0], 3))
    
    def get_start_end_val(self, par):
        '''Get a value of the first shot and the last shot. Using this only for the old MC version'''
        if not self.idx(par) or self.data.empty:
            return '-', '-'
        else:
            par_full = list(self.data)[self.idx(par)]
            val_first, val_end = self.data[par_full].iloc[0], self.data[par_full].iloc[-1]
            if par=='Shotnumber':
                return str(int(val_first)), str(int(val_end))
            else:
                return str(round(val_first, 3)), str(round(val_end, 3))
    
    def get_par_alias(self, par):
        '''Get the Alias name of the parameter if exists'''

        if self.idx(par):
            par_full = list(self.data)[self.idx(par)]
            # Get Alias if exists
            if 'Alias' in par_full:
                return par_full.split('Alias:', 1)[1]
        else:
            return par
        
    def idx(self, par):
        '''Get index of a parameter. Return None if the parameter cannot be found'''
        i_par = [k for k, s in enumerate(list(self.data)) if par in s]
        if i_par:
            return i_par[0]
        else:
            return None
        
        
