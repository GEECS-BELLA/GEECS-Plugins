'''
This code runs an interface that shows a list of scan infomation of the day.
Simply run this script by  clicking the file and choose 'open with python'.
Or open a cygwin or terminal then type "python ScanInfoTable.py".
Each row has a button on the left that will clipboard the scan infromation as a text.

updated on Mar 13, 2020
Fumika Isono fisono@lbl.gov
'''

from tkinter import filedialog
from tkinter.tix import*


from configparser import ConfigParser
import os
import re
import pandas as pd
import glob
import configparser

class ScanInfoTable:

    def __init__(self, root):
        self.root = root
        self.last_scan = 0
        self.isfirstshow = True
        self.exp_paras = None
        
        #load configuration file
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')    
    
        # Make a top frame and a frame with a scrolled window
        self.win_width, self.win_height = '1000', '800'
        frame_top = Frame(self.root)
        frame_top.pack(fill=BOTH, expand=True)
        frame = Frame(width=self.win_width, height=self.win_height)
        frame.pack()
        self.new_scroll_win(frame, self.win_width, self.win_height)
        
        # put a browse button
        dirbutton = Button(frame_top, text='Browse A Date Folder', command=lambda : self.load_new_date(frame)).pack(side=LEFT, anchor=W)

        #Show a directory
        self.dir_date = self.config['DEFAULT']['scandir_date']
        self.dir_show = StringVar()
        self.dir_show.set(self.dir_date)
        Label(frame_top, textvariable=self.dir_show, bg='white').pack(side=LEFT, anchor=W)
        
        #put a parameter entry
        Label(frame_top, text='Parameters').pack(side=LEFT, padx=10)
        self.entry = Entry(frame_top)
        self.entry.insert(END, self.config['DEFAULT']['parameters'])
        self.entry.pack(side=LEFT)        
        Button(frame_top, text='RUN', command=lambda:self.show_table(frame)).pack(side=LEFT, padx=10)

        #Put an update button
        Button(self.root, text='Update', command=self.update_table).pack(side=BOTTOM)
        
    def load_new_date(self, frame):
        '''Browse a date folder, then renew the window & table'''
        #browse
        self.dir_date = filedialog.askdirectory(initialdir="Z:/data/", title="Select A File")
        #update the display of the directory
        self.dir_show.set(self.dir_date)
        self.show_table(frame)
        
    def new_scroll_win(self, frame, width, height):
        '''Creat a new scroll window where table will be shown'''
        self.swin = ScrolledWindow(frame, width=width, height=height)
        self.swin.pack()
        self.win = self.swin.window
        
    def show_table(self, frame):
        '''Show a table of all the past scans' info'''        
        # clear data by rebuilding the window
        if not self.isfirstshow:
            print('reconstruct a window')
            self.win.destroy()
            self.swin.destroy()
            self.new_scroll_win(frame, self.win_width, self.win_height)
            
        #get parameters from an entry
        paras_str = self.entry.get().replace(", ",",")
        self.exp_paras = paras_str.split(",")
        self.show_labels()
        self.last_scan = self.get_last_scannumber()
        for i in range(self.last_scan):
            self.show_scaninfo_row(i + 1)
        if self.isfirstshow:
            self.isfirstshow=False
            
        #save settings in config.ini
        self.write_config()

    def update_table(self):
        '''Add new rows of scan to a table'''
        new_scan = self.get_last_scannumber()
        for i in range(self.last_scan, new_scan):
            self.show_scaninfo_row(i + 1)
        self.last_scan = new_scan

    def show_labels(self):
        '''Show labels of the infomation, first row of the table'''
        column_name = ['','Scan', 'Shot','Scan Parameter', 'Start', 'End', 'Info'] + self.exp_paras
        for j in range(len(column_name)):
            b = Label(self.win, text='%s' % column_name[j], font=("Helvetica", 10), relief=RIDGE, anchor=W).grid(row=1,column=j,sticky=NSEW)

    def show_scaninfo_row(self, n_scan):
        '''Show one row of scan info'''
        # get basic scan infos
        para, info = self.get_scaninfo(n_scan)

        #if scan info is not found
        if para==None:
            row_list = [n_scan]+ ['-']*(5+len(self.exp_paras))
            clip = ''
        else:
            para_scan, val_start, val_end = self.get_start_end_val(n_scan, para)
            
            #get total shotnumber of the scan
            aa, bb, n_shot = self.get_start_end_val(n_scan, 'Shotnumber')
            
            #if scan parameter is 'Shotnumber', it displays as 'No Scan'
            if para_scan in 'Shotnumber':
                para_scan = 'No Scan'
                val_start, val_end = '-','-'

            # get additional experimental parameters
            exp_vals = []
            exp_vals_clip = ''
            for i in range(len(self.exp_paras)):
                exp_val = self.get_val(n_scan, self.exp_paras[i])
                # say 'scan' if this is a scan parameter
                if self.exp_paras[i] in para_scan:
                    exp_val = 'scan'
                exp_vals = exp_vals + [exp_val]
                if not str(exp_val)=='-':
                    exp_vals_clip= exp_vals_clip + self.exp_paras[i] + '=' + str(exp_val) + ', '

            # Make a list of data for the table
            row_list = [n_scan, int(n_shot), para_scan, val_start, val_end, info] + exp_vals
            
            #Make a text for the clipboard
            clip = 'Scan '+str(n_scan)+': '+para_scan
            if para_scan != 'No Scan':
                clip = clip +' ('+str(val_start)+', '+str(val_end)+')'
            clip = clip +' '+info
            if exp_vals_clip:
                clip = clip + '\n'+exp_vals_clip[:-2]
            
        #clipboard button at the left of the row
        Button(self.win, text=' ', command = lambda : self.get_clipboard(clip)).grid(row=n_scan+1, column=0)        
        #fill in all data in a row
        for j in range(len(row_list)):
            Label(self.win, text='%s' % row_list[j], font=("Helvetica", 10), relief=RIDGE, anchor=W, bg='white').grid(row=n_scan + 1, column=j+1,sticky=NSEW)
        return None
    
    def get_clipboard(self,clip):
        self.win.clipboard_clear()
        self.win.clipboard_append(clip)
        return self.win.clipboard_get()
    
    def write_config(self):
        '''Save updated parameter list in config.ini file'''
        self.config['DEFAULT']['parameters'] = ",".join(self.exp_paras)
        self.config['DEFAULT']['scandir_date'] = self.dir_date
        with open('config.ini','w') as configfile:
            self.config.write(configfile)

    def get_scaninfo(self, n_scan):
        '''Get scan number, scan parameter, scan start info from ScanInfoScan***.ini'''
        scan_3d = '{0:03}'.format(n_scan)
        file_config = self.dir_date + '\\scans\\Scan' + scan_3d + '\\ScanInfoScan' + scan_3d + '.ini'
        config = ConfigParser()
        config_read = config.read(file_config)
        if config_read:
            para = config['Scan Info']['Scan Parameter'].strip('"')
            info = config['Scan Info']['ScanStartInfo'].strip('"')
        else:
            para = None
            info = None
        return para, info

    def get_last_scannumber(self):
        '''Get the last scan number which is already done'''

        path = self.dir_date + '\\analysis'
        if not os.path.isdir(path):
            return 0
        else:
            # get last scan info file name
            files = glob.glob(path + '\\s*info.txt')
            file_last = os.path.basename(files[-1])
            # regexp. find number in the file name
            n_scans = int(re.findall(r"\d+", file_last)[0])
            return n_scans

    def get_start_end_val(self, n_scan, par, isalias=True):
        '''Get the value of the first shot and the last shot'''

        file = self.dir_date + '\\analysis\\s' + str(n_scan) + '.txt'
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

    def get_val(self, n_scan, par):
        "Get the parameter value of the first shot"
        file = self.dir_date + '\\analysis\\s' + str(n_scan) + '.txt'
        data = pd.read_csv(file, sep='\t')
        indices = [k for k, s in enumerate(list(data)) if par in s]
        if not indices or data.empty:
            return '-'
        else:
            par_full = list(data)[indices[0]]
            return round(data[par_full].iloc[0], 3)

def main():

    root = Tk()
    root.title('Scan Info')
    ScanInfoTable(root)
    root.mainloop()


if __name__ == '__main__':
    main()
