"""
logmaker4googledocs: Helpers for experiment logs on Google Docs/Drive/Sheets.

This module wraps a small set of Google APIs plus a companion Apps Script to:
- create or locate a daily experiment log from a template,
- append a template block to an existing log,
- replace placeholders with current values,
- upload images and insert them into a specific table cell,
- pull small bits of data from spreadsheets.

It is currently used for automated experiment logs at BELLA Center @ LBNL.

Notes
-----
- The **second half** of the system runs as a *Google Apps Script* (deployed in
  your Google account) and is referenced here via its `SCRIPT_ID` loaded from
  `config.ini` next to this file.
- Credentials are expected in `credentials.json`; OAuth tokens are cached in
  `token.pickle` (both next to this file).
- This code is intentionally light‑touch: it prefers explicit prints and simple
  returns over elaborate exception handling. Treat it as a pragmatic utility.

Examples
--------
Create/find a daily log, then append a template block and replace placeholders:

>>> from logmaker_4_googledocs import docgen as g
>>> svc = g.establishService('script', 'v1')
>>> doc_id = g.createExperimentLog(LOG_TEMPLATE_ID, TEMPLATE_FOLDER_ID, LOG_FOLDER_ID,
...                                'Experiment Log 2025-06-12', ARG_CONFIG_PATH, svc)
>>> g.appendToLog(TEMPLATE_BLOCK_ID, doc_id, search='Scan 042', servicevar=svc)
>>> g.findAndReplace(doc_id, placeholders_cfg, servicevar=svc)

The scripts `createGdoc.py` and `appendScan.py` in this package show a more
complete flow using INI files with placeholders and current values.

Authors
-------
Tobias Ostermayr, updated 2020-08-06
"""

__version__ = "0.2"
__author__ = "Tobias Ostermayr"

# stdlib
import pickle
from PIL import Image
import os.path
import os
import mimetypes
from pathlib import Path
import glob
from googleapiclient import errors
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload
from datetime import datetime
import configparser
import decimal
import logging
from typing import Optional

# OAuth scopes required for Docs, Drive, and Sheets
SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]

logger = logging.getLogger(__name__)

# Fallback staging folder used when no per-experiment image folder is configured.
# This folder may be periodically purged; configure ImageParentFolderID in the
# experiment INI to use a persistent per-day subfolder instead.
_FALLBACK_IMAGE_FOLDER = "1O5JCAz3XF0h_spw-6kvFQOMgJHwJEvP2"

# Fixed network paths to per-experiment INI files.
# These files are updated daily by createExperimentLog() and contain LogID
# and ImageParentFolderID. All lab computers share the Z:\ drive at these paths.
_EXPERIMENT_INI_PATHS: dict = {
    "Undulator": Path(
        r"Z:\software\control-all-loasis\HTU\Active Version"
        r"\GEECS-Plugins\LogMaker4GoogleDocs\logmaker_4_googledocs\HTUparameters.ini"
    ),
    "Thomson": Path(
        r"Z:\software\control-all-loasis\HTW2020\Active Version"
        r"\GEECS-Plugins\LogMaker4GoogleDocs\logmaker_4_googledocs\HTTparameters.ini"
    ),
}


# Module-level caches — persist for the lifetime of the process.
# _service_cache: avoids re-building the OAuth service on every API call.
# _folder_id_cache: avoids a Drive list query for the per-day folder on every upload.
_service_cache: dict = {}  # key: "apiservice/apiversion" → Resource
_folder_id_cache: dict = {}  # key: (parent_folder_id, folder_name) → folder_id

# Load `config.ini` colocated with this module. Must contain DEFAULT.script (Apps Script ID).
scriptconfig = configparser.ConfigParser()
script_dir = Path(__file__).parent
config_path = script_dir / "config.ini"
scriptconfig.read(config_path)

if not scriptconfig.sections():
    logger.warning(
        "Could not load config file from %s; Apps Script calls will be unavailable.",
        config_path,
    )
else:
    logger.debug("Loaded config file: %s", config_path)

try:
    SCRIPT_ID = scriptconfig["DEFAULT"]["script"]
except KeyError:
    SCRIPT_ID = None
    logger.warning(
        "'script' key missing in %s; Apps Script calls will be unavailable.",
        config_path,
    )

# Date/time convenience strings used in file naming
today = datetime.now()
date = today.strftime("%y-%m-%d")
time = today.strftime("%H:%M")


# ========================LOCAL FUNCTIONS=============================
def format_number(num):
    """
    Convert a numeric value to a compact decimal string (trim trailing zeros).

    Parameters
    ----------
    num : float or str
        Numeric value that may contain trailing zeros when rendered as text.

    Returns
    -------
    str
        A compact decimal string with trailing zeros removed (and no trailing dot).
        Returns 'bad' if conversion to `Decimal` fails.

    Notes
    -----
    This is used to prettify values written into INI files / Google Docs.
    """
    try:
        dec = decimal.Decimal(num)
    except Exception:
        return "bad"
    tup = dec.as_tuple()
    delta = len(tup.digits) + tup.exponent
    digits = "".join(str(d) for d in tup.digits)
    if delta <= 0:
        zeros = abs(tup.exponent) - len(tup.digits)
        val = "0." + ("0" * zeros) + digits
    else:
        val = digits[:delta] + ("0" * tup.exponent) + "." + digits[delta:]
    val = val.rstrip("0")
    if val and val[-1] == ".":
        val = val[:-1]
    if tup.sign:
        return "-" + val
    return val


def latestFileInDirectory(path, pattern):
    """
    Return the most recently modified item in a directory matching a pattern.

    Parameters
    ----------
    path : str
        Directory to search.
    pattern : str
        Substring (globbed as `*{pattern}*`) used to filter entries.

    Returns
    -------
    str
        Full path to the newest matching file.

    Raises
    ------
    ValueError
        If no files match the pattern.
    """
    list_of_files = glob.glob(path + "/*" + pattern + "*")
    latest_file = max(list_of_files, key=os.path.getctime)
    logger.debug("Latest file in %s matching '%s': %s", path, pattern, latest_file)
    return latest_file


def getValueForNameKeysECS(path, pattern, keylist, currentvalues, argcurrentvalues):
    """
    Populate a config with current ECS/scan values for a set of placeholder keys.

    Parameters
    ----------
    path : str
        Directory containing the latest ECS/scan INI dump.
    pattern : str
        Filename pattern used to locate the latest ECS/scan file (e.g., "ECS Live Dump").
    keylist : list[tuple]
        Iterable of `(placeholder_key, "Device&&Parameter")` entries specifying what to copy.
    currentvalues : configparser.ConfigParser
        Destination config in which DEFAULT section will be filled.
    argcurrentvalues : str or Path
        File path where `currentvalues` will be written after update.

    Returns
    -------
    None
        Writes the updated config to `argcurrentvalues`.

    Notes
    -----
    The ECS file is expected to have multiple sections; each section contains
    keys like `"Device Name"` and the parameter names to be copied.
    """
    latestfile = latestFileInDirectory(path, pattern)
    latest = configparser.ConfigParser()
    latest.read(latestfile)

    for i in keylist:
        tmp = i[1]
        devicename = tmp.split("&&")[0]
        parameter = tmp.split("&&")[1]
        for j in latest.sections():
            try:
                devnam = latest[j]["Device Name"].replace('"', "")
            except Exception:
                devnam = None
            if devnam == devicename:
                try:
                    tmp = latest[j][parameter]
                except Exception:
                    tmp = "No value found"
                currentvalues["DEFAULT"][i[0]] = tmp.replace('"', "")
    with open(argcurrentvalues, "w") as configfile:
        currentvalues.write(configfile)


def getValueForNameKeysScanFiles(path, pattern, currentvalues, argcurrentvalues):
    """
    Append the 'Scan Info' section from the newest scan INI to current values.

    Parameters
    ----------
    path : str
        Directory to search for scan files.
    pattern : str
        Filename pattern used to locate the latest scan file (e.g., "Scan Info").
    currentvalues : configparser.ConfigParser
        Destination config whose DEFAULT section will be extended.
    argcurrentvalues : str or Path
        File path where `currentvalues` will be written after update.

    Returns
    -------
    None
        Writes the updated config to `argcurrentvalues`.

    Notes
    -----
    For each key/value from "Scan Info", the value is de‑quoted and then
    optionally compacted via `format_number`.
    """
    latestfile = latestFileInDirectory(path, pattern)
    latest = configparser.ConfigParser()
    latest.read(latestfile)

    for key, value in latest["Scan Info"].items():
        currentvalues["DEFAULT"][key] = value.replace('"', "")
        if format_number(currentvalues["DEFAULT"][key]) != "bad":
            currentvalues["DEFAULT"][key] = format_number(currentvalues["DEFAULT"][key])

    with open(argcurrentvalues, "w") as f:
        currentvalues.write(f)


def cropAndScaleImage(
    imagepath, margin_left, margin_top, margin_right, margin_bottom, scalefactor
):
    """
    Crop and scale an image in place.

    Parameters
    ----------
    imagepath : str
        Path to the image to modify (overwritten on save).
    margin_left, margin_top, margin_right, margin_bottom : int
        Crop margins in pixels.
    scalefactor : float
        Uniform scale factor applied to the cropped image.

    Returns
    -------
    None
        Saves back to `imagepath` (PNG).

    Notes
    -----
    Uses Pillow. Consider using a copy if you need to preserve the original.
    """
    img = Image.open(imagepath)
    width = int(img.size[0] - margin_right)
    height = int(img.size[1] - margin_bottom)
    logger.debug("Crop image size: %s x %s", img.size[0], img.size[1])
    logger.debug(
        "Crop margins — left=%s top=%s right=%s bottom=%s",
        int(margin_left),
        int(margin_top),
        width,
        height,
    )
    img = img.crop((int(margin_left), int(margin_top), width, height))
    img = img.resize((int(width * scalefactor), int(height * scalefactor)))
    img.save(imagepath, "PNG", quality=94)


def scale_image(image_path, target_width_in_inches=4.75, dpi=100):
    """
    Scale an image to a target width (inches) while preserving aspect ratio.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    target_width_in_inches : float, default 4.75
        Desired width.
    dpi : int, default 100
        Dots per inch used to convert inches to pixels.

    Returns
    -------
    str or None
        Path to the new image written alongside the original with suffix `_scaled`,
        or `None` if scaling fails.

    Notes
    -----
    Uses Pillow with `Image.Resampling.LANCZOS` for quality. Height is derived from
    the original aspect ratio.
    """
    try:
        target_width_pixels = int(target_width_in_inches * dpi)
        with Image.open(image_path) as img:
            aspect_ratio = img.height / img.width
            target_height_pixels = int(target_width_pixels * aspect_ratio)
            scaled_img = img.resize(
                (target_width_pixels, target_height_pixels), Image.Resampling.LANCZOS
            )
            base, ext = os.path.splitext(image_path)
            scaled_image_path = f"{base}_scaled{ext}"
            scaled_img.save(scaled_image_path)
            logger.debug("Scaled image saved to: %s", scaled_image_path)
            return scaled_image_path
    except FileNotFoundError:
        logger.error("Image file not found: %s", image_path)
    except Exception as e:
        logger.error("Error scaling image %s: %s", image_path, e)


# ==================FUNCTIONS USING GOOGLE API========================
def establishService(apiservice, apiversion):
    """
    Create an authenticated Google API client for a given service/version.

    Parameters
    ----------
    apiservice : str
        Service name (e.g., 'script', 'drive', 'docs', 'sheets').
    apiversion : str
        Version string (e.g., 'v1', 'v3').

    Returns
    -------
    googleapiclient.discovery.Resource
        Bound client for the requested API.

    Raises
    ------
    FileNotFoundError
        If `credentials.json` is missing.
    Exception
        If client creation fails.

    Notes
    -----
    - OAuth tokens are cached in `token.pickle`.
    - Required scopes are defined in `SCOPES`.
    """
    cache_key = f"{apiservice}/{apiversion}"
    base_path = Path(__file__).parent
    token_path = base_path / "token.pickle"

    if cache_key in _service_cache:
        creds, service = _service_cache[cache_key]
        if creds.valid:
            return service
        # Access token expired — refresh in-place so the long-running process
        # keeps working without re-authentication.
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                with token_path.open("wb") as token_file:
                    pickle.dump(creds, token_file)
                service = build(apiservice, apiversion, credentials=creds)
                _service_cache[cache_key] = (creds, service)
                logger.debug("Refreshed credentials for %s %s", apiservice, apiversion)
                return service
            except Exception as e:
                logger.warning(
                    "Failed to refresh credentials for %s %s: %s. Re-authenticating.",
                    apiservice,
                    apiversion,
                    e,
                )
        # Could not refresh; clear cache and fall through to full auth.
        del _service_cache[cache_key]

    logger.debug("Establishing Google API service: %s %s", apiservice, apiversion)

    creds = None
    credentials_path = base_path / "credentials.json"

    if token_path.exists():
        with token_path.open("rb") as token_file:
            creds = pickle.load(token_file)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                raise FileNotFoundError(
                    f"Credentials file not found at: {credentials_path}"
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_path), SCOPES
            )
            creds = flow.run_local_server(port=0)

        with token_path.open("wb") as token_file:
            pickle.dump(creds, token_file)

    try:
        service = build(apiservice, apiversion, credentials=creds)
        logger.debug("Google API service created: %s %s", apiservice, apiversion)
        _service_cache[cache_key] = (creds, service)
        return service
    except Exception as e:
        logger.error(
            "Error creating Google API service %s %s: %s", apiservice, apiversion, e
        )
        raise


def createExperimentLog(
    logtempID, tempfolderID, logfolderID, logfilename, argconfig, servicevar
):
    """
    Create (or find) a daily Google Doc experiment log from a template.

    Parameters
    ----------
    logtempID : str
        Google ID of the template Google Doc.
    tempfolderID : str
        Google ID of the template folder (Apps Script may use this).
    logfolderID : str
        Google ID of the target folder where the new document will live.
    logfilename : str
        Desired filename of the new document (e.g., 'Experiment Log 25-06-12').
    argconfig : str or Path
        Path to a config file (INI) where the resulting LogID will be written.
    servicevar : googleapiclient.discovery.Resource or None
        An existing 'script' API service. If `None`, this function will create one.

    Returns
    -------
    str
        Google Doc ID for the (new or existing) daily log.

    Notes
    -----
    This calls the Apps Script function `createExperimentLog` and records the
    resulting document ID in `argconfig` under `DEFAULT/LogID`.
    """
    logger.info("Creating or finding experiment log: %s", logfilename)
    API_SERVICE_NAME = "script"
    API_VERSION = "v1"
    if servicevar is None:
        service = establishService(API_SERVICE_NAME, API_VERSION)
    else:
        service = servicevar

    request = {
        "function": "createExperimentLog",
        "parameters": [logtempID, tempfolderID, logfolderID, logfilename],
        "devMode": True,
    }

    try:
        response = service.scripts().run(body=request, scriptId=SCRIPT_ID).execute()
        if "error" in response:
            error = response["error"]["details"][0]
            logger.error("createExperimentLog script error: %s", error["errorMessage"])
        else:
            documentID = response["response"]["result"]
            logger.info("Experiment log document ID: %s", documentID)
            config = configparser.ConfigParser()
            config.read(argconfig)
            config["DEFAULT"]["LogID"] = documentID
            with open(argconfig, "w") as configfile:
                config.write(configfile)
    except errors.HttpError as e:
        logger.error("HTTP error in createExperimentLog: %s", e.content)
    except Exception:
        logger.exception("Unexpected error in createExperimentLog")
    return documentID


def appendToLog(templateID, documentID, search, servicevar):
    """
    Append a template block to a Google Doc unless a search phrase already exists.

    Parameters
    ----------
    templateID : str
        Google ID of the template document (block to insert).
    documentID : str
        Google ID of the target document to modify.
    search : str or None
        If provided and found in the target document, skip insertion.
        If `None`, a sentinel string is used (i.e., always insert).
    servicevar : googleapiclient.discovery.Resource or None
        An existing 'script' API service. If `None`, a new one will be created.

    Returns
    -------
    int
        0 on success (inserted or already present), 1 on retry/error path.

    Notes
    -----
    Uses the Apps Script function `appendTemplate`.
    """
    API_SERVICE_NAME = "script"
    API_VERSION = "v1"
    if servicevar is None:
        service = establishService(API_SERVICE_NAME, API_VERSION)
    else:
        service = servicevar

    request = {
        "function": "appendTemplate",
        "parameters": [templateID, documentID],
        "devMode": True,
    }

    if search is None:
        search = "SomethingNobodyWouldEverWriteInADocentEver"

    tmp = None
    try:
        tmp = checkFileContains(documentID, search, service)
    except Exception:
        logger.warning("Failed to check document for search pattern '%s'", search)
        return 1

    if tmp is False:
        try:
            logger.info("Appending template to document %s", documentID)
            response = service.scripts().run(body=request, scriptId=SCRIPT_ID).execute()
            if "error" in response:
                error = response["error"]["details"][0]
                logger.error("appendToLog script error: %s", error["errorMessage"])
            else:
                logger.info("appendToLog result: %s", response["response"]["result"])
                return 0
        except errors.HttpError as e:
            logger.error("HTTP error in appendToLog: %s", e)
        except Exception as e:
            logger.error("Error in appendToLog: %s", e)
    elif tmp:
        logger.info("Scan already present in log %s; skipping append.", documentID)
        return 0
    else:
        logger.warning("appendToLog: unexpected state, retry needed.")
        return 1


def findAndReplace(documentID, placeholdersandvalues, servicevar):
    """
    Replace `{{placeholder}}` tokens in a Google Doc with values.

    Parameters
    ----------
    documentID : str
        Google ID of the document to modify.
    placeholdersandvalues : configparser.ConfigParser
        Config whose DEFAULT section maps placeholder keys (without `{{ }}`) to values.
    servicevar : googleapiclient.discovery.Resource or None
        An existing 'script' API service. If `None`, a new one will be created.

    Returns
    -------
    int
        0 on success, 1 if the Apps Script reported an error.

    Notes
    -----
    Uses the Apps Script function `findAndReplace`.
    """
    API_SERVICE_NAME = "script"
    API_VERSION = "v1"

    keys = list(placeholdersandvalues["DEFAULT"].keys())
    values = list(placeholdersandvalues["DEFAULT"].values())

    if servicevar is None:
        service = establishService(API_SERVICE_NAME, API_VERSION)
    else:
        service = servicevar

    request = {
        "function": "findAndReplace",
        "parameters": [documentID, keys, values],
        "devMode": False,
    }

    try:
        response = service.scripts().run(body=request, scriptId=SCRIPT_ID).execute()
        if "error" in response:
            error = response["error"]["details"][0]
            logger.error("findAndReplace script error: %s", error["errorMessage"])
            return 1
        else:
            return 0

    except Exception:
        logger.exception("Unexpected error in findAndReplace")
    except errors.HttpError as e:
        logger.error("HTTP error in findAndReplace: %s", e.content)


def findAndReplaceImage(documentID, imageID, pattern, servicevar):
    """
    Replace a text pattern in a Google Doc table cell with a Drive image.

    Parameters
    ----------
    documentID : str
        Google Doc ID to modify.
    imageID : str
        Google Drive file ID of the image to insert.
    pattern : str
        Text marker to be replaced (e.g., '{{screenshot}}').
    servicevar : googleapiclient.discovery.Resource or None
        An existing 'script' API service. If `None`, a new one will be created.

    Returns
    -------
    int or None
        0 on success; `None` if a low‑level error occurred.

    Notes
    -----
    Currently only supports replacements *inside tables*, as implemented in the
    companion Apps Script `findAndReplaceImage`.
    """
    API_SERVICE_NAME = "script"
    API_VERSION = "v1"

    if servicevar is None:
        service = establishService(API_SERVICE_NAME, API_VERSION)
    else:
        service = servicevar

    request = {
        "function": "findAndReplaceImage",
        "parameters": [documentID, imageID, pattern],
        "devMode": False,
    }

    try:
        response = service.scripts().run(body=request, scriptId=SCRIPT_ID).execute()
        if "error" in response:
            error = response["error"]["details"][0]
            logger.error("findAndReplaceImage script error: %s", error["errorMessage"])
        else:
            return 0

    except errors.HttpError as e:
        logger.error("HTTP error in findAndReplaceImage: %s", e.content)


def uploadImage(localimagepath, destinationID):
    """
    Upload a local image (PNG or GIF) to Google Drive.

    Parameters
    ----------
    localimagepath : str
        Path to the image file to upload (PNG/GIF).
    destinationID : str
        Google Drive folder ID where the image will be created.

    Returns
    -------
    str or None
        The Drive file ID of the uploaded image, or `None` on error.

    Notes
    -----
    - PNG images are scaled (width ~ 4.75 in at 100 DPI) via `scale_image`
      before upload; GIFs are uploaded as-is.
    - Uses Drive API v3.
    """
    API_SERVICE_NAME = "drive"
    API_VERSION = "v3"
    driveservice = establishService(API_SERVICE_NAME, API_VERSION)

    file_path = Path(localimagepath)
    mime_type, _ = mimetypes.guess_type(localimagepath)

    if not mime_type:
        raise ValueError(f"Unable to determine MIME type for file: {localimagepath}")

    if mime_type not in ["image/png", "image/gif"]:
        raise ValueError("Only PNG and GIF file types are supported.")

    scaled_image_path = localimagepath
    if mime_type == "image/png":
        scaled_image_path = scale_image(localimagepath)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_metadata = {
        "name": f"{timestamp}_tmp.{file_path.suffix.lstrip('.')}",
        "parents": [destinationID],
    }

    media = MediaFileUpload(scaled_image_path, mimetype=mime_type)

    try:
        file = (
            driveservice.files()
            .create(body=file_metadata, media_body=media, supportsAllDrives=True)
            .execute()
        )
        if "error" in file:
            error = file["error"]["details"][0]
            logger.error("Drive upload error: %s", error["errorMessage"])
            return None
        else:
            imageID = file["id"]
            return imageID

    except errors.HttpError as e:
        logger.error("HTTP error uploading image: %s", e.content)
        return None


def checkFileContains(fileID, search, servicevar):
    """
    Check if a Google Doc contains a search string.

    Parameters
    ----------
    fileID : str
        Google Doc ID to inspect.
    search : str
        String to search for.
    servicevar : googleapiclient.discovery.Resource or None
        An existing 'script' API service. If `None`, a new one will be created.

    Returns
    -------
    bool or None
        True if found, False if not found, or `None` if the script failed.
    """
    API_SERVICE_NAME = "script"
    API_VERSION = "v1"

    if servicevar is None:
        service = establishService(API_SERVICE_NAME, API_VERSION)
    else:
        service = servicevar

    request = {
        "function": "checkFileContains",
        "parameters": [fileID, search],
        "devMode": False,
    }

    try:
        response = service.scripts().run(body=request, scriptId=SCRIPT_ID).execute()
        if "error" in response:
            error = response["error"]["details"][0]
            logger.error("checkFileContains script error: %s", error["errorMessage"])
        else:
            return response["response"]["result"]

    except errors.HttpError as e:
        logger.error("HTTP error in checkFileContains: %s", e.content)


def lastRowOfSpreadsheet(fileID, sheetString, firstcol, lastcol, servicevar):
    """
    Return the last non-empty row (as values) from a Google Sheet range.

    Parameters
    ----------
    fileID : str
        Spreadsheet file ID.
    sheetString : str
        Sheet name (tab) within the spreadsheet.
    firstcol : str
        Left column letter (e.g., 'A').
    lastcol : str
        Right column letter (e.g., 'G').
    servicevar : googleapiclient.discovery.Resource or None
        An existing 'script' API service. If `None`, a new one will be created.

    Returns
    -------
    list[list] or None
        2D array of values representing the last row inside the requested
        column window, or `None` if the call failed.

    Notes
    -----
    This calls the Apps Script function `lastRowFromSpreadsheet` which scans
    the range `firstcol:lastcol`.
    """
    API_SERVICE_NAME = "script"
    API_VERSION = "v1"

    if servicevar is None:
        service = establishService(API_SERVICE_NAME, API_VERSION)
    else:
        service = servicevar

    request = {
        "function": "lastRowFromSpreadsheet",
        "parameters": [fileID, sheetString, firstcol, lastcol],
        "devMode": False,
    }

    try:
        response = service.scripts().run(body=request, scriptId=SCRIPT_ID).execute()
        if "error" in response:
            error = response["error"]["details"][0]
            logger.error("lastRowOfSpreadsheet script error: %s", error["errorMessage"])
        else:
            return response["response"]["result"]

    except errors.HttpError as e:
        logger.error("HTTP error in lastRowOfSpreadsheet: %s", e.content)


def insertImageToTableCell(documentID, scanNumber, row, column, imageID, servicevar):
    """
    Insert a Drive image into a specific table cell identified under a scan heading.

    Parameters
    ----------
    documentID : str
        Google Doc ID containing the table.
    scanNumber : int
        Scan number used by the Apps Script to locate the relevant section/heading.
    row : int
        Zero-based row index of the target table.
    column : int
        Zero-based column index of the target table.
    imageID : str
        Drive file ID of the image to insert.
    servicevar : googleapiclient.discovery.Resource or None
        An existing 'script' API service. If `None`, a new one will be created.

    Returns
    -------
    dict or None
        Script result payload on success, else `None`.

    Notes
    -----
    Uses Apps Script function `insertImageToTableCell`. The script is responsible
    for locating the correct table based on `scanNumber`.
    """
    API_SERVICE_NAME = "script"
    API_VERSION = "v1"

    if servicevar is None:
        service = establishService(API_SERVICE_NAME, API_VERSION)
    else:
        service = servicevar

    request = {
        "function": "insertImageToTableCell",
        "parameters": [documentID, scanNumber, row, column, imageID],
        "devMode": True,
    }

    try:
        response = service.scripts().run(body=request, scriptId=SCRIPT_ID).execute()
        if "error" in response:
            error = response["error"]["details"][0]
            logger.error(
                "insertImageToTableCell script error: %s", error["errorMessage"]
            )
        else:
            return response["response"]["result"]

    except errors.HttpError as e:
        logger.error("HTTP error in insertImageToTableCell: %s", e.content)


def get_or_create_folder(parent_folder_id: str, folder_name: str) -> str:
    """Find or create a subfolder by name inside a parent Drive folder.

    Parameters
    ----------
    parent_folder_id : str
        Google Drive ID of the parent folder.
    folder_name : str
        Name of the subfolder to find or create (e.g., '2025-06-12').

    Returns
    -------
    str
        Drive file ID of the found or created subfolder.

    Notes
    -----
    If multiple folders with the same name exist, the first result is returned.
    """
    cache_key = (parent_folder_id, folder_name)
    if cache_key in _folder_id_cache:
        return _folder_id_cache[cache_key]

    driveservice = establishService("drive", "v3")
    query = (
        f"'{parent_folder_id}' in parents"
        f" and name = '{folder_name}'"
        f" and mimeType = 'application/vnd.google-apps.folder'"
        f" and trashed = false"
    )
    results = (
        driveservice.files()
        .list(
            q=query,
            spaces="drive",
            fields="files(id, name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )
    files = results.get("files", [])
    if files:
        folder_id = files[0]["id"]
        logger.info("Found existing Drive folder '%s': %s", folder_name, folder_id)
        _folder_id_cache[cache_key] = folder_id
        return folder_id

    folder_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }
    folder = (
        driveservice.files()
        .create(body=folder_metadata, fields="id", supportsAllDrives=True)
        .execute()
    )
    folder_id = folder["id"]
    logger.info("Created Drive folder '%s': %s", folder_name, folder_id)
    _folder_id_cache[cache_key] = folder_id
    return folder_id

def get_document_id(experiment: str) -> Optional[str]:
    r"""Return the current Google Doc LogID for an experiment.

    Reads the LogID from the experiment's config INI at its fixed network path.
    This works on any lab computer with Z:\ access regardless of where this
    package is installed.

    Parameters
    ----------
    experiment : str
        Experiment name (e.g. ``'Undulator'``, ``'Thomson'``).

    Returns
    -------
    str or None
        The Google Doc ID, or ``None`` if the INI is missing or has no LogID.
    """
    ini_path = _EXPERIMENT_INI_PATHS.get(experiment)
    if ini_path is None:
        logger.warning("No INI path configured for experiment '%s'.", experiment)
        return None
    if not ini_path.exists():
        logger.warning("INI not found for experiment '%s' at %s.", experiment, ini_path)
        return None
    cfg = configparser.ConfigParser()
    cfg.read(ini_path)
    doc_id = cfg["DEFAULT"].get("logid")
    if not doc_id:
        logger.warning(
            "LogID not set in %s — run createExperimentLog() first.", ini_path
        )
        return None
    return doc_id


def append_link_to_scan(documentID, scan_number, label, url, servicevar):
    """Append a hyperlink paragraph inside the "Additional diagnostics:" cell for a scan.

    Parameters
    ----------
    documentID : str
        Google Doc ID containing the scan entry.
    scan_number : int
        Scan number used to locate the section heading.
    label : str
        Visible link text (e.g. ``"UC_GaiaMode: summary.png"``).
    url : str
        URL the hyperlink points to (e.g. a Drive view URL).
    servicevar : googleapiclient.discovery.Resource or None
        An existing ``'script'`` API service. If ``None``, a new one is created.

    Returns
    -------
    str or None
        ``'success'`` on success, an error description string, or ``None`` on HTTP error.
    """
    API_SERVICE_NAME = "script"
    API_VERSION = "v1"

    if servicevar is None:
        service = establishService(API_SERVICE_NAME, API_VERSION)
    else:
        service = servicevar

    request = {
        "function": "appendLinkToScan",
        "parameters": [documentID, scan_number, label, url],
        "devMode": True,
    }

    try:
        response = service.scripts().run(body=request, scriptId=SCRIPT_ID).execute()
        if "error" in response:
            error = response["error"]["details"][0]
            logger.error("appendLinkToScan script error: %s", error["errorMessage"])
            return error["errorMessage"]
        else:
            return response["response"]["result"]

    except errors.HttpError as e:
        logger.error("HTTP error in append_link_to_scan: %s", e.content)
        return None


def upload_display_files_and_link(
    scan_number, analyzer_id, display_files, document_id=None, experiment="Undulator"
):
    """Upload display files to Drive and append hyperlinks to the scan log entry.

    For each file in *display_files*, uploads it to the per-day Drive folder and
    calls ``appendLinkToScan`` to add a clickable link inside the
    ``"Additional diagnostics:"`` cell of the scan entry.

    Parameters
    ----------
    scan_number : int
        Scan number used to locate the section heading in the Google Doc.
    analyzer_id : str
        Analyzer identifier; used as a label prefix (e.g. ``"UC_GaiaMode"``).
    display_files : list of str
        Local paths to the files to upload.
    document_id : str, optional
        Google Doc ID. If ``None``, the ID is read from the experiment INI.
    experiment : str, default ``'Undulator'``
        Experiment key used to pick the INI file.

    Returns
    -------
    int
        Number of files successfully uploaded and linked.
    """
    experiment_mapping = _EXPERIMENT_INI_PATHS
    experiment_config = configparser.ConfigParser()
    config_file = experiment_mapping.get(experiment)
    if config_file:
        config_path = config_file
        experiment_config.read(config_path)
        if document_id is None and experiment_config.has_option("DEFAULT", "logid"):
            document_id = experiment_config["DEFAULT"]["logid"]

    if document_id is None:
        logger.error(
            "No document ID available for experiment '%s'; skipping link upload.",
            experiment,
        )
        return 0

    image_folder_id = _FALLBACK_IMAGE_FOLDER
    if experiment_config.has_option("DEFAULT", "imageparentfolderid"):
        parent_folder_id = experiment_config["DEFAULT"]["imageparentfolderid"]
        date_str = datetime.now().strftime("%Y-%m-%d")
        try:
            image_folder_id = get_or_create_folder(parent_folder_id, date_str)
        except Exception as e:
            logger.warning(
                "Could not create per-day image folder: %s. Falling back to staging folder.",
                e,
            )

    script_service = establishService("script", "v1")
    linked = 0
    for file_path in display_files:
        file_id = uploadImage(file_path, image_folder_id)
        if file_id is None:
            logger.warning("Upload failed for '%s'; skipping link.", file_path)
            continue
        url = f"https://drive.google.com/file/d/{file_id}/view"
        label = f"{analyzer_id}: {Path(file_path).name}"
        result = append_link_to_scan(
            document_id, scan_number, label, url, script_service
        )
        if result == "success":
            linked += 1
            logger.info("Linked '%s' in scan log for scan %s.", label, scan_number)
        else:
            logger.warning("appendLinkToScan returned: %s", result)
    return linked


def insertImageToExperimentLog(
    scanNumber, row, column, image_path, documentID=None, experiment="Undulator"
):
    """
    Convenience: upload a local image and insert it into the 2×2 display table for a scan.

    Parameters
    ----------
    scanNumber : int
        Scan number used to locate the section/heading in the Google Doc.
    row : int
        Zero-based row index of the target table cell.
    column : int
        Zero-based column index of the target table cell.
    image_path : str
        Local path to the image file (PNG or GIF).
    documentID : str, optional
        Google Doc ID. If `None`, it is read from the experiment's INI.
    experiment : str, default 'Undulator'
        Experiment key used to pick the INI that stores the current log ID.

    Returns
    -------
    bool
        ``True`` if the image was uploaded and the Apps Script insertion
        returned a result; ``False`` on any failure.

    Notes
    -----
    - Uses a temporary Drive folder (hard-coded) as a staging area; a separate
      script is expected to purge this directory periodically.
    - INI mapping is minimal; extend `experiment_mapping` if you add experiments.
    """
    # Load per-experiment INI to resolve document ID and image folder.
    experiment_mapping = _EXPERIMENT_INI_PATHS

    experiment_config = configparser.ConfigParser()
    config_file = experiment_mapping.get(experiment, None)
    if config_file:
        config_path = config_file
        experiment_config.read(config_path)
        if not experiment_config.sections():
            logger.warning("Failed to load experiment config from: %s", config_path)
        else:
            logger.debug("Loaded experiment config: %s", config_path)

        if documentID is None:
            documentID = experiment_config["DEFAULT"]["logid"]

    # Resolve the target Drive folder for this upload.
    # If ImageParentFolderID is set in the experiment INI, create (or find) a
    # per-day subfolder there so images are stored permanently.
    # Falls back to the legacy staging folder when not configured.
    image_folder_id = _FALLBACK_IMAGE_FOLDER
    if experiment_config.has_option("DEFAULT", "imageparentfolderid"):
        parent_folder_id = experiment_config["DEFAULT"]["imageparentfolderid"]
        date_str = datetime.now().strftime("%Y-%m-%d")
        try:
            image_folder_id = get_or_create_folder(parent_folder_id, date_str)
        except Exception as e:
            logger.warning(
                "Could not create per-day image folder under %s: %s. Falling back to staging folder.",
                parent_folder_id,
                e,
            )

    image_id = uploadImage(image_path, image_folder_id)
    if image_id is None:
        logger.error("Image upload to Drive failed; skipping table insertion.")
        return False
    result = insertImageToTableCell(documentID, scanNumber, row, column, image_id, None)
    return result is not None
