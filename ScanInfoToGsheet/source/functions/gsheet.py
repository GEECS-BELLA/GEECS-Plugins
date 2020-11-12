"""Get all the scan info of the day, write on google sheet
"""

import time
from functions.scaninfo import ScanInfo


class GSheet:
    def __init__(self, dir_date, para_txt):
        self.dir_date = dir_date
        self.exp_paras = para_txt.replace(", ", ",").split(",") if para_txt else None
        self.sheet = None
        self.n_columns = None
        self.n_scans = None
        
        #load ScanInfo class
        self.scaninfo = ScanInfo(self.dir_date, self.exp_paras)
        
    def write(self, gdr_dir, sheet_title):
        '''
        Open (Create if not exist) a google sheet in the google drive, then write down the all scans' infomation
        '''
        # get a dataframe to write
        df = self.scaninfo.get_scaninfo_all()
        # write the dataframe into a google sheet
        self.sheet = df2gsheet(sheet_title, df, gdr_dir)
        self.n_scans = df.index[-1] + 1
        self.n_columns = len(df.keys())
        return self.sheet

    def update(self):
        '''If there are new scans, these infomation will be added in the google sheet
        return: latest scan number if there is a new scan. Otherwise return None
        '''
        scan_new = self.scaninfo.get_last_scannumber()        
        if scan_new > self.n_scans:
            self.sheet[0].append_table(self.scaninfo.get_scaninfo(scan_new))
            self.n_scans = scan_new
            return self.n_scans
        else:
            return None
        
def df2gsheet(sheet_title, df, gdrive_dir):
    """
    Open (Create if not exist) a google sheet in the google drive, then write down a given dataframe
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))    
    gc = pygsheets.authorize(service_file = dir_path+'\\service_account.json')

    #if can't find an exisiting google sheet, create it
    try:
        sheet = gc.open(sheet_title)
    except pygsheets.SpreadsheetNotFound as error:
        sheet = gc.create(sheet_title, folder = gdrive_dir)

    sheet[0].set_dataframe(df, (1,1))
    return sheet


def main():
    dir_date = 'Z:\\data\\Undulator\\Y2020\\09-Sep\\20_0908'
    sheet_title = 'HTU 20_0908 ScanSummary_'
    para_txt = 'Jet_X,Jet_Y,Jet_Z,JetBlade,Pressure,separation'
    gdrive_dir = '1CIhAy9Ykh4r4Tq4FfL-msCfeJ_vTTJCVNziObS43RYc'
    isAutoUpdate = True

    gsheet = GSheet(dir_date, para_txt)
    sheet = gsheet.write(gdrive_dir, sheet_title)
    if isAutoUpdate:
        for i in range(60 * 10):  # run for 10 hour
            time.sleep(60)  # wait for 1 min
            nscan_new = gsheet.update()


if __name__ == '__main__':
    main()
