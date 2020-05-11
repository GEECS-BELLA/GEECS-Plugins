# ScanInoToGsheetGUI
This GUI gets scan information/variables of all scans of a day and export it to a google sheet in a google drive.

For 'Scan Date Folder' section, please choose a folder of a scan date where scan data are stored.
    ex) Z:\\data\\Undulator\\Y2020\\01-Jan\\20_0123
    
For 'enable auto update' section, you can check/un-check anytime to run/stop the auto update. It will update the Google sheet every minute as long as you already clicked 'Export' before.
    
*If you want to add/edit a Google drive folder where Google sheets are going to be saved, open 'config_gdrive.ini' and edit it.

## If you want to run a python program and choose setting manually...
Run scaninfo2gsheet.py

scaninfo2gsheet.py does a same thing as ScanInfoToGsheetGUI.
Before running, open the file and fill in sections in main function.
dir_date: Date folder where scan data is stored.
    ex) dir_date = 'Z:\\data\\Undulator\\Y2020\\01-Jan\\20_0123'    
sheet_title: title of the Google sheet you want to create (str)
para_txt: experimental parameters to be shown in the Google sheet. Can be just part of the full parameter name (show the value of first shot of each scan) 
    ex) para_txt = 'Jet_X,Jet_Y,Jet_Z,Pressure,separation'
gdrive_dir: folder ID of a Google drive where the Google sheet is saved. ID can be found in the last part of the URL.(str)
isAutoUpdate: If you want to keep running the code to automatically update the Google sheet during the run (every minute), set isAutoUpdate=True

Run this python script in the command line. 'python ExportScanInfo.py'
Or, you can run right click the file in the folder, 'Open with > Python'.


May 8th, 2020
Fumika Isono
fisono@lbl.gov, fumika21@gmail.com
