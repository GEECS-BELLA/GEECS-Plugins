import time
from getscaninfo import df_scaninfo, get_last_scannumber
from df2gsheet import df2gsheet
from scaninfo_row import scaninfo_row


class scaninfo2gsheet:
    def __init__(self, dir_date, para_txt):
        self.dir_date = dir_date
        self.exp_paras = para_txt.replace(", ", ",").split(",") if para_txt else None
        self.sheet = None
        self.n_columns = None
        self.n_scans = None

    def write(self, gdr_dir, sheet_title):
        '''
        Open (Create if not exist) a google sheet in the google drive, then write down the all scans' infomation
        '''
        # get a dataframe to write
        df = df_scaninfo(self.dir_date, self.exp_paras)
        # write the dataframe into a google sheet
        self.sheet = df2gsheet(sheet_title, df, gdr_dir)
        self.n_scans = df.index[-1] + 1
        self.n_columns = len(df.keys())
        return self.sheet

    def update(self):
        '''If there are new scans, these infomation will be added in the google sheet
        return: latest scan number if there is a new scan. Otherwise return None
        '''
        scan_new = get_last_scannumber(self.dir_date)
        if scan_new > self.n_scans:
            newinfo = scaninfo_row(self.dir_date, self.exp_paras, scan_new)
            self.sheet[0].append_table(newinfo.scaninfo_row)
            self.n_scans = scan_new
            return self.n_scans
        else:
            return None


def main():
    dir_date = 'Z:\\data\\Undulator\\Y2020\\06-Jun\\20_0622'
    sheet_title = 'test0622'
    para_txt = 'Jet_X,Jet_Y,Jet_Z,Pressure,separation'
    gdrive_dir = '0B3exNkpbT8vdREk0VzV5RVEyd1E'
    isAutoUpdate = False

    scaninfo = scaninfo2gsheet(dir_date, para_txt)
    sheet = scaninfo.write(gdrive_dir, sheet_title)
    if isAutoUpdate:
        for i in range(60 * 10):  # run for 10 hour
            time.sleep(60)  # wait for 1 min
            nscan_new = scaninfo.update()


if __name__ == '__main__':
    main()
