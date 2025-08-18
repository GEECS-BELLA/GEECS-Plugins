"""
Automated experiment log creation for BELLA-HTT on Google Drive.

This command-line script wires together the `docgen` helpers (Google APIs + Apps
Script) to:
1) create (or find) a daily experiment log from a template,
2) pull the last row from a spreadsheet (optional) and copy those values into a
   local INI of "current values",
3) replace placeholders in the target Google Doc with the current values.

Usage
-----
Run with three INI files:

    python appendScan.py parameters.ini placeholders.ini currentvalues.ini

Parameters
----------
parameters.ini
    Main configuration used by this script (template/folder IDs, spreadsheet
    info, local paths, margins/scales, and log name).
placeholders.ini
    Placeholder keys that may be used in document templates (read here and
    commonly paired with values via separate utilities).
currentvalues.ini
    Output INI file that this script writes/updates with current values
    (spreadsheet row, date, and anything else you add before calling
    `docgen.findAndReplace`).

Notes
-----
- This script is intentionally pragmatic and uses prints/try blocks liberally.
- The Google Apps Script project ID is read from a `config.ini` colocated with
  the `docgen` module.
- No functions are defined here; it’s a straight execution script by design.
"""

from __future__ import print_function

# from wand.image import Image
import os.path
import time
from datetime import datetime
import configparser
import docgen
import sys

from googleapiclient.errors import HttpError

###################################################################################################
###############################  READ INI FILES AND PREPARE STUFF  ################################
###################################################################################################

# --- CLI args: paths to INI files ---
argconfig = sys.argv[1]
argplaceholders = sys.argv[2]
argcurrentvalues = sys.argv[3]

# Read main config (template IDs, folders, spreadsheet info, etc.)
config = configparser.ConfigParser()
config.read(argconfig)

# OAuth scopes string (left as-is; `docgen` manages auth in practice)
SCOPES = "shttps://www.googleapis.com/auth/documents https://www.googleapis.com/auth/drive https://www.googleapis.com/auth/spreadsheets"

# Apps Script config (SCRIPT_ID) colocated with this script
scriptconfig = configparser.ConfigParser()
scriptconfig.read("config.ini")
SCRIPT_ID = scriptconfig["DEFAULT"]["script"]

# --- Google Docs template & folder IDs (from parameters.ini) ---
LOGTEMPLATE_ID = config["DEFAULT"]["LogTemplateID"]
TEMPLATE_ID = config["DEFAULT"]["ScanTemplateID"]
ENDTEMPLATE_ID = config["DEFAULT"]["EndTemplateID"]

TEMPLATEFOLDER_ID = config["DEFAULT"]["TemplateFolderID"]
LOGFOLDER_ID = config["DEFAULT"]["LogFolderID"]
SCREENSHOTFOLDER_ID = config["DEFAULT"]["ScreenshotFolderID"]

# --- Optional spreadsheet info for “last row” values ---
spreadsheetID = config["SPREADSHEET"]["SpreadsheetID"]
sheetName = config["SPREADSHEET"]["SheetName"]
firstcolumn = config["SPREADSHEET"]["FirstColumn"]
lastcolumn = config["SPREADSHEET"]["LastColumn"]

# --- Local LabVIEW paths (ECS dumps, scan files, screenshots) ---
LOCALSCANPATH = config["DEFAULT"]["databasepath"]
PATHTOIMAGE = config["DEFAULT"]["PathToScreenshot"]

# --- Base log name (will be prefixed with date) ---
logname = config["DEFAULT"]["logname"]

# --- Date selection: use a specific date or now ---
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

# Log filename date format (kept as in original)
date = today.strftime("%m-%d-%y")
print(date)

# Final log filename (e.g., "06-12-25 HTT Scanlog")
LOGFILENAME = date + " " + logname
print(LOGFILENAME)

# --- Placeholder keys (read but not used directly here) ---
placeholders = configparser.ConfigParser()
placeholders.read(argplaceholders)
placeholderlist = list(placeholders.items("DEFAULT"))

# --- Screenshot margins and scaling (used by other utilities) ---
ml = int(config["SCREENSHOT"]["ml"])
mt = int(config["SCREENSHOT"]["mt"])
mr = int(config["SCREENSHOT"]["mr"])
mb = int(config["SCREENSHOT"]["mb"])
imgscale = float(config["SCREENSHOT"]["imgscale"])

# Sentinel “search” string
search = "notinthefileforsure.,.,.><><>"

# --- Derived paths to daily folders on local storage ---
datescanstring = today.strftime("Y%Y/%m-%b/%y_%m%d/scans")
localscanfolder = LOCALSCANPATH + datescanstring

dateECSstring = today.strftime("Y%Y/%m-%b/%y_%m%d/ECS Live dumps")
localECSfolder = LOCALSCANPATH + dateECSstring

# --- Prepare the current-values INI (reset & seed with date) ---
if os.path.exists(argcurrentvalues):
    os.remove(argcurrentvalues)
currentvalues = configparser.ConfigParser()
currentvalues["DEFAULT"]["MM-DD-YY"] = date

###################################################################################################
################  EXECUTE SCRIPTS: GENERATE AND MODIFY THE GDOCS EXPERIMENT LOG  ##################
###################################################################################################

# Establish a single Script API service to reuse
service = docgen.establishService("script", "v1")

# Optionally pull “last row” from a spreadsheet to seed placeholders
try:
    spreadsheetreader = docgen.lastRowOfSpreadsheet(
        spreadsheetID, sheetName, firstcolumn, lastcolumn, service
    )

    counter = 1
    for i in spreadsheetreader[0]:
        currentvalues["DEFAULT"]["SpreadSheetColumn" + str(counter)] = i
        counter = counter + 1

except HttpError as e:
    print(f"Spreadsheet data not updated: {e}")
except Exception as e:
    # Fallback catch-all, but still logs the error
    print(f"Unexpected error reading spreadsheet: {e}")

# Write & reload the current-values file (kept as in original)
currentvalues.write(open(argcurrentvalues, "w"))
currentvalues.read(argcurrentvalues)

# Create or locate the daily experiment log (retry a few times)
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
        print(f"Failed to create experiment log (API error): {e}")
        time.sleep(1)
    except Exception as e:
        print(f"Unexpected error creating experiment log: {e}")
        time.sleep(1)

print(DOCUMENT_ID)
print(currentvalues)

# Replace placeholders in the log with the current values (retry loop)
returnvalue = 2
for i in range(0, 4):
    print("**Find and replace placeholders with current values**")
    try:
        returnvalue = docgen.findAndReplace(DOCUMENT_ID, currentvalues, service)
    except HttpError as e:
        print(f"Failed to create experiment log (API error): {e}")
        time.sleep(1)
    except Exception as e:
        print(f"Unexpected error creating experiment log: {e}")
        time.sleep(1)
    if returnvalue == 0:
        break
