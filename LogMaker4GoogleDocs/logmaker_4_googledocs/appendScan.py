"""
Append a single scan to the daily Google Docs experiment log (BELLA‑HTT).

This command‑line script wires up the `docgen` helpers (Google APIs + Apps Script)
to:
1) locate either the latest scan of the day or a specific scan folder,
2) create (or find) today's experiment log from a template,
3) append the scan table template to that log (only once per scan),
4) collect current values from local sources (ScanInfo/ECS dumps and optional
   spreadsheet "last row") into an INI, and
5) perform placeholder replacement in the log (+ optionally insert a screenshot).

It is intentionally pragmatic and preserves legacy behavior (including broad
`except:` blocks and `print` logging).

Usage
-----
Run with three INI files:

    python appendScan.py parameters.ini placeholders.ini currentvalues.ini

Parameters
----------
parameters.ini
    Main configuration used by this script. Expected sections/keys (examples):
    - [DEFAULT]
        LogTemplateID, ScanTemplateID, EndTemplateID
        TemplateFolderID, LogFolderID, ScreenshotFolderID
        databasepath, PathToScreenshot, logname, skipecs
    - [DATE]
        specificdate               # "YYYY,MM,DD" or "0" for today
    - [SPREADSHEET]
        SpreadsheetID, SheetName, FirstColumn, LastColumn
    - [SCREENSHOT]
        ml, mt, mr, mb, imgscale   # margins & scale for image cropping
    - [SCAN]
        scannumber                 # "0" for latest, or a specific integer

placeholders.ini
    Placeholder keys (e.g., `{{My Key}}`) used by templates; this script just
    loads them for downstream helpers.

currentvalues.ini
    Output file this script writes/updates with values gathered from:
    - spreadsheet last row (optional),
    - ScanInfo*.ini in the scan folder,
    - ECS dump for that scan (unless SKIP_ECS), and
    - a timestamp for the scan.

Behavior notes
--------------
- “Specific scan”: If [SCAN]/scannumber != "0", the script targets that scan.
  Otherwise it finds the latest scan folder for today.
- “SKIP_ECS”: If truthy in parameters.ini, ECS lookups are skipped.
- The daily log name is `<MM-DD-YY> <logname>`. The log is created (or found)
  and then the scan template is appended only if not already present (checked
  via a search string like `.*Scan N:`).
- Values are gathered into `currentvalues.ini` and then applied via
  `docgen.findAndReplace`.
- If a screenshot path is configured, the image is cropped, uploaded to Drive,
  and inserted into the log (falling back to a “no screenshot” message if it
  fails).
"""

# This script uses the docgen library to make automated experiment logbooks at BELLA-HTT@LBNL on Google Drive.
# A version of the Google Project source code will be stored locally for convenience.
#
# by Tobias Ostermayr, last updated 01/19/2021

from __future__ import print_function

# from wand.image import Image
import os.path
import time
from googleapiclient.errors import HttpError
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

# print(argconfig + "  " + argplaceholders + "  " + argcurrentvalues)

config = configparser.ConfigParser()
config.read(argconfig)
print("hello")
# DON'T TOUCH
SCOPES = "shttps://www.googleapis.com/auth/documents https://www.googleapis.com/auth/drive https://www.googleapis.com/auth/spreadsheets"
scriptconfig = configparser.ConfigParser()
scriptconfig.read("config.ini")
SCRIPT_ID = scriptconfig["DEFAULT"]["script"]

# GOOGLE DOCS TEMPLATES ID's TAKEN FROM THE URL's STORED IN CONFIG.INI
LOGTEMPLATE_ID = config["DEFAULT"]["LogTemplateID"]
TEMPLATE_ID = config["DEFAULT"]["ScanTemplateID"]
ENDTEMPLATE_ID = config["DEFAULT"]["EndTemplateID"]

# GOOGLE DOCS FOLDERS ID's TAKEN FROM THE URL's
TEMPLATEFOLDER_ID = config["DEFAULT"]["TemplateFolderID"]
LOGFOLDER_ID = config["DEFAULT"]["LogFolderID"]
SCREENSHOTFOLDER_ID = config["DEFAULT"]["ScreenshotFolderID"]
SKIP_ECS = config["DEFAULT"]["skipecs"]

# GOOGLE SPREADSHEET INFO FOR OPTIONAL DATA OF LAST ROW OF THIS SPREADSHEET
spreadsheetID = config["SPREADSHEET"]["SpreadsheetID"]
sheetName = config["SPREADSHEET"]["SheetName"]
firstcolumn = config["SPREADSHEET"]["FirstColumn"]
lastcolumn = config["SPREADSHEET"]["LastColumn"]

# LOCAL PATHs TO SCAN DATA AND SCREENSHOTS FROM LABVIEW
LOCALSCANPATH = config["DEFAULT"]["databasepath"]
PATHTOIMAGE = config["DEFAULT"]["PathToScreenshot"]

# BASE NAME FOR YOUR EXPERIMENT LOG
logname = config["DEFAULT"]["logname"]

# DATE & TIME
specificdate = config["DATE"]["specificdate"]
specificdate = specificdate.replace('"', "")
if specificdate != "0":
    today = datetime(
        int(specificdate.split(",")[0]),
        int(specificdate.split(",")[1]),
        int(specificdate.split(",")[2]),
        1,
        1,
        1,
    )
else:
    today = datetime.now()
date = today.strftime("%m-%d-%y")

print(date)
# today = datetime(2020, 4, 1, 14, 30, 5)

# time = today.strftime("%H:%M")

# FULL EXPERIMENT LOG FILENAME (here for example with date)
LOGFILENAME = date + " " + logname  # " HTT Scanlog" #You may want to edit this
print(LOGFILENAME)

# READ TEMPLATE KEYS AND REPLACEMENTS
# global placeholders
placeholders = configparser.ConfigParser()
placeholders.read(argplaceholders)
placeholderlist = list(placeholders.items("DEFAULT"))

# margins for screenshot
ml = int(config["SCREENSHOT"]["ml"])
mt = int(config["SCREENSHOT"]["mt"])
mr = int(config["SCREENSHOT"]["mr"])
mb = int(config["SCREENSHOT"]["mb"])
imgscale = float(config["SCREENSHOT"]["imgscale"])

# INIT search variable
search = "notinthefileforsure.,.,.><><>"

# PREPARE PATHs TO VOL1 FOLDERS FOR ECS DUMPS AND SCANINFO FILES
datescanstring = today.strftime("Y%Y/%m-%b/%y_%m%d/scans")
localscanfolder = LOCALSCANPATH + datescanstring
print(localscanfolder)

dateECSstring = today.strftime("Y%Y/%m-%b/%y_%m%d/ECS Live dumps")
localECSfolder = LOCALSCANPATH + dateECSstring
print(localECSfolder)

# Create search for "Scan XX:" to avoid multiple postings of the same scan table.
# And define path to this scanfolder
# specificscan ='0' is automatically the latest scan
# specificscan != '0' generates the table for an arbitrary scan
# Search goes as variable into appentToLog function.

specificscan = str(config["SCAN"]["scannumber"])
print("specific scan is " + specificscan)
print(os.path.exists(localscanfolder))

if specificscan != "0":
    try:
        latestScanDir = docgen.latestFileInDirectory(
            localscanfolder, "Scan*" + specificscan
        )
        print("lastest scan dir" + latestScanDir)
        search = date + ".*Scan " + str(int(latestScanDir.split("\\Scan")[1])) + ":"
        print("Search updated: " + search)
        scanNo = str(int(latestScanDir.split("\\Scan")[1]))
        # print(scanNo)
    except Exception as e:
        print(f"This scan does not exist: {e}")
        sys.exit()
elif os.path.exists(localscanfolder + "/Scan001") and specificscan == "0":
    latestScanDir = docgen.latestFileInDirectory(localscanfolder, "Scan")
    print(latestScanDir)
    search = ".*Scan " + str(int(latestScanDir.split("\\Scan")[1])) + ":"
    scanNo = str(int(latestScanDir.split("\\Scan")[1]))
    print(scanNo)
    print("Search updated: " + search)
else:
    print("No scans for today yet")
    sys.exit()

# global currentvalues
# CLEAN FILE FOR STORING LATEST SCAN INFO AND VALUES
if os.path.exists(argcurrentvalues):
    # print('if executed')
    os.remove(argcurrentvalues)
currentvalues = configparser.ConfigParser()
currentvalues["DEFAULT"]["MM-DD-YY"] = date
# get timestamp of ecs file for this scan
ecsfilepath = localECSfolder + "/Scan" + scanNo + ".txt"
if os.path.exists(ecsfilepath):
    filedate = os.path.getctime(ecsfilepath)
else:
    print("local scan folder")
    intScanNo = int(scanNo)
    formated_string = f"{localscanfolder}/Scan{intScanNo:03}"
    print(formated_string)
    filedate = os.path.getctime(formated_string)

hourandminuteandsecond = datetime.fromtimestamp(filedate).strftime("%H:%M:%S")
currentvalues["DEFAULT"]["-HHMM-"] = hourandminuteandsecond

###################################################################################################
################  EXECUTE SCRIPTS: GENERATE AND MODIFY THE GDOCS EXPERIMENT LOG  ##################
###################################################################################################
service = docgen.establishService("script", "v1")

try:
    spreadsheetreader = docgen.lastRowOfSpreadsheet(
        spreadsheetID, sheetName, firstcolumn, lastcolumn, service
    )

    counter = 1
    for i in spreadsheetreader[0]:
        currentvalues["DEFAULT"]["SpreadSheetColumn" + str(counter)] = i
        counter = counter + 1
except HttpError as e:
    print(f"Spreadsheet data not updated (API): {e}")
except Exception as e:
    print(f"Spreadsheet data not updated: {e}")

currentvalues.write(open(argcurrentvalues, "w"))
currentvalues.read(argcurrentvalues)

for i in range(0, 4):
    try:
        DOCUMENT_ID = docgen.createExperimentLog(
            LOGTEMPLATE_ID,
            TEMPLATEFOLDER_ID,
            LOGFOLDER_ID,
            LOGFILENAME,
            argconfig,
            service,
        )
        break
    except HttpError as e:
        print(f"createExperimentLog failed (API), retrying: {e}")
        time.sleep(1)
    except Exception as e:
        print(f"createExperimentLog failed, retrying: {e}")
        time.sleep(1)

returnvalue = 2
for i in range(0, 4):
    try:
        returnvalue = docgen.appendToLog(TEMPLATE_ID, DOCUMENT_ID, search, service)
    except HttpError as e:
        print(f"appendToLog failed (API), retrying: {e}")
        time.sleep(1)
    except Exception as e:
        print(f"appendToLog failed, retrying: {e}")
        time.sleep(1)
    if returnvalue == 0:
        break

print("**Scanfiles found**")
if specificscan != "0":
    if SKIP_ECS:
        docgen.getValueForNameKeysScanFiles(
            latestScanDir, "ScanInfo*" + specificscan, currentvalues, argcurrentvalues
        )
    else:
        docgen.getValueForNameKeysScanFiles(
            latestScanDir, "ScanInfo*" + specificscan, currentvalues, argcurrentvalues
        )
        docgen.getValueForNameKeysECS(
            localECSfolder,
            "Scan*" + specificscan,
            placeholderlist,
            currentvalues,
            argcurrentvalues,
        )
else:
    if SKIP_ECS:
        docgen.getValueForNameKeysScanFiles(
            latestScanDir, "ScanInfo*", currentvalues, argcurrentvalues
        )
    else:
        docgen.getValueForNameKeysECS(
            localECSfolder, "Scan*", placeholderlist, currentvalues, argcurrentvalues
        )
        docgen.getValueForNameKeysScanFiles(
            latestScanDir, "ScanInfo*", currentvalues, argcurrentvalues
        )

returnvalue = 2
for i in range(0, 4):
    print("**Find and replace placeholders with current values**")
    try:
        returnvalue = docgen.findAndReplace(DOCUMENT_ID, currentvalues, service)
    except HttpError as e:
        print(f"findAndReplace failed (API), retrying: {e}")
        time.sleep(1)
    except Exception as e:
        print(f"findAndReplace failed, retrying: {e}")
        time.sleep(1)
    if returnvalue == 0:
        break

try:
    docgen.cropAndScaleImage(PATHTOIMAGE, ml, mt, mr, mb, imgscale)
    imageid = docgen.uploadImage(PATHTOIMAGE, SCREENSHOTFOLDER_ID)
    for i in range(0, 4):
        print("**Adding screenshot**")
        try:
            docgen.findAndReplaceImage(DOCUMENT_ID, imageid, "{{screenshot}}", service)
            break
        except HttpError as e:
            print(f"findAndReplaceImage failed (API), retrying: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"findAndReplaceImage failed, retrying: {e}")
            time.sleep(1)
except Exception as e:
    print(f"**No screenshot added**: {e}")
    tmpconf = configparser.ConfigParser()
    tmpconf["DEFAULT"]["screenshot"] = "no screenshot"

    returnvalue = 2
    for i in range(0, 4):
        try:
            print("...find and replace screenshot placeholder with generic sentence")
            returnvalue = docgen.findAndReplace(DOCUMENT_ID, tmpconf, service)
        except HttpError as e:
            print(f"findAndReplace (fallback) failed (API), retrying: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"findAndReplace (fallback) failed, retrying: {e}")
            time.sleep(1)
        if returnvalue == 0:
            break
