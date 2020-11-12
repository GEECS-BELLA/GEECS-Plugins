from tkintertoy import Window
import os
from functions.gsheet import GSheet
import configparser
import webbrowser


class Gui(object):
    def __init__(self):
        # read configuration file
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.config = configparser.ConfigParser()
        self.config.read(self.dir_path+'\\config_parameters.ini')

        # write gui
        self.gui = Window()
        self.gui.setTitle('Export ScanInfo to Gsheet')
        self.gui.addChooseDir('dir_date', 'Scan Date Folder', width=40, initialdir='')
        #self.gui.addEntry('para_list', 'Parameters to list', width=50)
        #self.gui.set('para_list', self.config['DEFAULT']['parameters'])
        self.gui.addCombo('para_list', 'Paramters to list', None,
                          postcommand=self.para_update, width=47)
        self.gui.addCombo('gdir', 'Save location in google drive', None,
                          postcommand=self.gdir_update, width=47)
        self.gui.addText('status', width=40, height=5, prompt='Status:')
        self.gui.addCheck('auto_update', '', ['Turn On Auto Update'])
        self.gui.addButton('commands')
        self.gui.changeWidget('commands', 0, text='Export', command=self.export)
        self.gui.changeWidget('commands', 1, text='Exit')

        self.gui.plot('dir_date', row=0)
        self.gui.plot('para_list', row=1, pady=5)
        self.gui.plot('gdir', row=2, pady=5)
        self.gui.plot('status', row=3, pady=10)
        self.gui.plot('auto_update', row=4)
        self.gui.plot('commands', row=5, pady=10)

        self.exported = False
        self.proj = None  # bella project name
        self.g_name = None  # google drive folder name
        self.g_ID = None  # google drive folder id

        self.autoupdateOn = False
        self.update()

    def gdir_update(self):
        '''Update a saving location option on GUI, depending on your project'''
        # get BELLA project name
        dir_date = self.gui.get('dir_date', allValues=True)
        self.proj = get_proj_name(dir_date)

        # get google drive folder name and ID from .ini file
        cfg = configparser.ConfigParser()
        cfg.read(self.dir_path+'\\config_gdrive.ini')
        self.g_name = [cfg[i]['name'] for i in cfg.sections() if cfg[i]['proj'] == self.proj]
        self.g_ID = [cfg[i]['id'] for i in cfg.sections() if cfg[i]['proj'] == self.proj]

        # update save folder options in GUI
        self.gui.set('gdir', self.g_name, allValues=True)
        
    def para_update(self):
        """
        Update a parameter list on GUI
        """
        # get google drive folder name and ID from .ini file
        cfg = configparser.ConfigParser()
        cfg.read(self.dir_path+'\\config_parameters.ini')
        self.para = [cfg[i]['parameters'] for i in cfg.sections()]

        # update save folder options in GUI
        self.gui.set('para_list', self.para, allValues=True)
        

    def export(self):
        '''When Export button is pressed, scan info is written into a google sheet'''
        # Get settings from GUI
        dir_date = self.gui.get('dir_date', allValues=True)  # directory of scandata
        para_list = self.gui.get('para_list')  # parameters to save
        gdrive_name = self.gui.get('gdir')  # name of the google drive

        if not dir_date or not gdrive_name:
            self.gui.set('status', 'Please fill in all sections\n')
        else:
            gdrive_id = self.g_ID[self.g_name.index(gdrive_name)]  # google drive ID
            sheet_title = self.proj + ' ' + os.path.basename(dir_date) + ' ScanSummary'

            # write          
            self.scaninfo = GSheet(dir_date, para_list)
            sheet = self.scaninfo.write(gdrive_id, sheet_title)
            self.exported = True

            # add text to status window
            message = 'Gsheet \'' + sheet.title + '\' saved\n'#URL: ' + sheet.url
            self.gui.set('status', message)
            
            #open the url
            webbrowser.open(sheet.url)

            # update the config file
            #self.write_config(para_list, dir_date)

    def update(self):
        '''Update google sheet every 30 sec.'''
        if bool(self.gui.get('auto_update')) and self.exported:
            # if autoupdate is turned on, message appears on status window
            if not self.autoupdateOn:
                self.gui.set('status', '\nAuto update On...')
            self.autoupdateOn = True
            nscan_new = self.scaninfo.update()
            
            # if there is a new scan, add text to the status window
            if nscan_new:
                message = '\nScan ' + str(nscan_new) + ' updated'
                self.gui.set('status', message)
        else:
            # if autoupdate is turned off, message appears on status window
            if self.autoupdateOn:
                self.gui.set('status', '\nAuto update Off')
            self.autoupdateOn = False

        #update after 30 seconds
        self.gui.master.after(30 * 1000, self.update)

    def write_config(self, para_list, dir_date):
        '''Save updated parameter list in config.ini file'''
        self.config['DEFAULT']['parameters'] = para_list
        with open(self.dir_path+'\\config_parameters.ini', 'w') as configfile:
            self.config.write(configfile)


def get_proj_name(dir_date):
    '''Get a project name from a given directory path.
    dct_parent: keys: project directory name (3 up of dir_date)
                values: [proj] in google_dir.ini
    '''
    path_parent = os.path.normpath(dir_date).split(os.path.sep)[-4]
    dct_project = {'Thomson': 'HTT', 'Undulator': 'HTU', 'data': 'PW', 'kHzLPA': 'kHz'}
    return dct_project[path_parent]


def main():
    app = Gui()
    app.gui.waitforUser()


if __name__ == '__main__':
    main()
