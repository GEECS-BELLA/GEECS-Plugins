# ScanInfoToGsheetGUI
This GUI gets scan information/variables of all scans of a day and export it to a google sheet in a google drive.

## Download
Go to [release page](https://github.com/GEECS-BELLA/GEECS-Plugins/releases) in github and download the build package.
If your PC asks you about access rights, right-click --> properties, general tab --> "this file came from another computer and might be blocked help protect this computer" --> unblock

## How to run
* In 'Scan Date Folder' section, choose a folder of a scan date where scan data are stored. Ex) `Z:\\data\\Undulator\\Y2020\\01-Jan\\20_0123`. Fill in experimental parameters to list, then choose a google folder location for the gsheet to be saved. Click **Export**.
    
* In 'enable auto update' section, you can select/unselect anytime to run/stop the auto update. It will update the Google sheet every 30 seconds as long as you already clicked **Export** before.
    
* If you want to add/edit Google drive folders where Google sheets are going to be saved, open 'config_gdrive.ini' and edit it. (**scaninfo@scaninfo-275704.iam.gserviceaccount.com** needs permission to the google folder)

## If you want to run a python program and choose setting manually...
Run scaninfo2gsheet.py. (scaninfo2gsheet.py does a same thing as ScanInfoToGsheetGUI)

Before running, open the file and fill in sections in main function.

* dir_date: Date folder where scan data is stored.
    ex) `dir_date = 'Z:\\data\\Undulator\\Y2020\\01-Jan\\20_0123'`
    
* sheet_title: title of the Google sheet you want to create (str)

* para_txt: experimental parameters to be shown in the Google sheet. Can be just part of the full parameter name (show the value of first shot of each scan) 
    ex) `para_txt = 'Jet_X,Jet_Y,Jet_Z,Pressure,separation'`
    
* gdrive_dir: folder ID of a Google drive where the Google sheet is saved. ID can be found in the last part of the URL.(str)

* isAutoUpdate: If you want to keep running the code to automatically update the Google sheet during the run (every minute), set `isAutoUpdate=True`

Run this python script in the command line `python ExportScanInfo.py`. Or, you can run right click the file in the folder, 'Open with > Python'.


May 29th, 2020
Fumika Isono
fisono@lbl.gov, fumika21@gmail.com
