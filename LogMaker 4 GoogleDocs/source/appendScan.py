# This script uses the docgen library to make automated experiment logbooks at BELLA-HTT@LBNL on Google Drive.
# A version of the Google Project source code will be stored locally for convenience.
#
# by Tobias Ostermayr, last updated 08/06/2020

from __future__ import print_function
import pickle
#from wand.image import Image
import os.path
import os.path, time
import glob
from googleapiclient import errors
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from datetime import datetime
import configparser
import docgen
import sys

###################################################################################################
###############################  READ INI FILES AND PREPARE STUFF  ################################
###################################################################################################

argconfig = sys.argv[1]
argplaceholders = sys.argv[2]
argcurrentvalues = sys.argv[3]

#print(argconfig + "  " + argplaceholders + "  " + argcurrentvalues)

config = configparser.ConfigParser()
config.read(argconfig)

# DON'T TOUCH
SCOPES = "shttps://www.googleapis.com/auth/documents https://www.googleapis.com/auth/drive https://www.googleapis.com/auth/spreadsheets"
scriptconfig = configparser.ConfigParser()
scriptconfig.read('config.ini')
SCRIPT_ID = scriptconfig['DEFAULT']['script']

# GOOGLE DOCS TEMPLATES ID's TAKEN FROM THE URL's STORED IN CONFIG.INI
LOGTEMPLATE_ID = config['DEFAULT']['LogTemplateID']
TEMPLATE_ID = config['DEFAULT']['ScanTemplateID']
ENDTEMPLATE_ID = config['DEFAULT']['EndTemplateID']

# GOOGLE DOCS FOLDERS ID's TAKEN FROM THE URL's
TEMPLATEFOLDER_ID = config['DEFAULT']['TemplateFolderID']
LOGFOLDER_ID = config['DEFAULT']['LogFolderID']
SCREENSHOTFOLDER_ID = config['DEFAULT']['ScreenshotFolderID']

# GOOGLE SPREADSHEET INFO FOR OPTIONAL DATA OF LAST ROW OF THIS SPREADSHEET
spreadsheetID = config['SPREADSHEET']['SpreadsheetID']
sheetName = config['SPREADSHEET']['SheetName']
firstcolumn = config['SPREADSHEET']['FirstColumn']
lastcolumn = config['SPREADSHEET']['LastColumn']

# LOCAL PATHs TO SCAN DATA AND SCREENSHOTS FROM LABVIEW 
LOCALSCANPATH = config['DEFAULT']['databasepath']
PATHTOIMAGE = config['DEFAULT']['PathToScreenshot']

# BASE NAME FOR YOUR EXPERIMENT LOG
logname = config['DEFAULT']['logname']

# DATE & TIME
specificdate = config['DATE']['specificdate']
specificdate = specificdate.replace('"', '')
if specificdate != '0':
    today = datetime(int(specificdate.split(",")[0]),int(specificdate.split(",")[1]),int(specificdate.split(",")[2]),1,1,1)
else:
    today = datetime.now()
date = today.strftime("%m-%d-%y")

print(date)
#today = datetime(2020, 4, 1, 14, 30, 5)

#time = today.strftime("%H:%M")        

# FULL EXPERIMENT LOG FILENAME (here for example with date)
LOGFILENAME = date + " "+logname #" HTT Scanlog" #You may want to edit this    
print(LOGFILENAME)

# READ TEMPLATE KEYS AND REPLACEMENTS
#global placeholders
placeholders = configparser.ConfigParser()
placeholders.read(argplaceholders)
placeholderlist = list(placeholders.items('DEFAULT'))


# margins for screenshot
ml = int(config['SCREENSHOT']['ml'])
mt = int(config['SCREENSHOT']['mt'])
mr = int(config['SCREENSHOT']['mr'])
mb = int(config['SCREENSHOT']['mb'])
imgscale = float(config['SCREENSHOT']['imgscale']) 

# INIT search variable
search = "notinthefileforsure.,.,.><><>"


# PREPARE PATHs TO VOL1 FOLDERS FOR ECS DUMPS AND SCANINFO FILES
datescanstring = today.strftime("Y%Y/%m-%b/%y_%m%d/scans")
localscanfolder = LOCALSCANPATH + datescanstring

dateECSstring = today.strftime("Y%Y/%m-%b/%y_%m%d/ECS Live dumps")
localECSfolder = LOCALSCANPATH + dateECSstring


# Create search for "Scan XX:" to avoid multiple postings of the same scan table.
# And define path to this scanfolder
# specificscan ='0' is automatically the latest scan
# specificscan != '0' generates the table for an arbitrary scan
# Search goes as variable into appentToLog function.

specificscan = str(config['SCAN']['scannumber'])

if specificscan != '0':
    try: 
        latestScanDir = docgen.latestFileInDirectory(localscanfolder,'Scan*'+specificscan)
        search = date +'.*Scan ' + str(int(latestScanDir.split("\\Scan")[1])) + ':' 
        print('Search updated: ' + search)
        scanNo = str(int(latestScanDir.split("\\Scan")[1]))
        #print(scanNo)
    except: print('This scan does not exist bro'); sys.exit()
elif os.path.exists(localscanfolder+'/Scan001') and specificscan == '0':
    latestScanDir = docgen.latestFileInDirectory(localscanfolder,'Scan')
    search = date + '.*Scan ' + str(int(latestScanDir.split("\\Scan")[1])) + ':' 
    scanNo = str(int(latestScanDir.split("\\Scan")[1]))
    #print(scanNo)
    print('Search updated: ' + search)
else: print("No scans for today yet"); sys.exit()

#global currentvalues
# CLEAN FILE FOR STORING LATEST SCAN INFO AND VALUES
if os.path.exists(argcurrentvalues):
    #print('if executed')
    os.remove(argcurrentvalues)
currentvalues = configparser.ConfigParser()
currentvalues['DEFAULT']["MM-DD-YY"]=date
#get timestamp of ecs file for this scan
ecsfilepath = localECSfolder + '/Scan' + scanNo + '.txt'
#if os.path.exists(ecsfilepath):
currentvalues['DEFAULT']["-HHMM-"]=time.ctime(os.path.getctime(ecsfilepath)).split(" ")[4]

###################################################################################################
################  EXECUTE SCRIPTS: GENERATE AND MODIFY THE GDOCS EXPERIMENT LOG  ##################
###################################################################################################
service = docgen.establishService('script','v1')

try:
    spreadsheetreader = docgen.lastRowOfSpreadsheet(spreadsheetID,sheetName,firstcolumn,lastcolumn,service)

    counter=1
    for i in spreadsheetreader[0]:
        currentvalues['DEFAULT']["SpreadSheetColumn"+str(counter)] = i
        counter = counter + 1
except: print("Spreadsheet data not updated")

currentvalues.write(open(argcurrentvalues,'w'))
currentvalues.read(argcurrentvalues)


DOCUMENT_ID = docgen.createExperimentLog(LOGTEMPLATE_ID,TEMPLATEFOLDER_ID,LOGFOLDER_ID,LOGFILENAME,argconfig,service)
docgen.appendToLog(TEMPLATE_ID,DOCUMENT_ID,search,service)
if specificscan != '0': 
    docgen.getValueForNameKeysECS(localECSfolder,"Scan*"+specificscan,placeholderlist,currentvalues,argcurrentvalues)
    docgen.getValueForNameKeysScanFiles(latestScanDir,"ScanInfo*"+specificscan,currentvalues,argcurrentvalues)
else: 
    docgen.getValueForNameKeysECS(localECSfolder,"Scan*",placeholderlist,currentvalues,argcurrentvalues)
    docgen.getValueForNameKeysScanFiles(latestScanDir,"ScanInfo*",currentvalues,argcurrentvalues)
docgen.findAndReplace(DOCUMENT_ID,currentvalues,service)
try: 
    docgen.cropAndScaleImage(PATHTOIMAGE,ml,mt,mr,mb,imgscale)
    imageid = docgen.uploadImage(PATHTOIMAGE,SCREENSHOTFOLDER_ID)
    docgen.findAndReplaceImage(DOCUMENT_ID,imageid,"{{screenshot}}",service)
except: 
    print("no screenshot added")
    tmpconf = configparser.ConfigParser()
    tmpconf['DEFAULT']['screenshot']="no screenshot"
    docgen.findAndReplace(DOCUMENT_ID,tmpconf,service)