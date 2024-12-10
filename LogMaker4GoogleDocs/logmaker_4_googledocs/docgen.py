"""
This module provides functions for automated document creation from 
templates in Gdocs. The second half of this code is a google apps 
script project available online. A version of the Google Project source
code will be stored in the github repository for convenience.

It is currently used for automated experiment logs at 
BELLA Center@LBNL.

Example:
    An example of how to implement the functions to generate a Google 
    doc step by step can be seen in createGdoc.py and appendScan.py

    $ python createGdoc.py parameters.ini placeholders.ini 
                                                currentvalues.ini


by Tobias Ostermayr, last updated 08/06/2020
"""

from __future__ import print_function

__version__ = '0.2'
__author__ = 'Tobias Ostermayr'


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
import sys
import httplib2

# DON'T TOUCH
SCOPES = ['https://www.googleapis.com/auth/documents','https://www.googleapis.com/auth/drive','https://www.googleapis.com/auth/spreadsheets']
# scriptconfig = configparser.ConfigParser()
# scriptconfig.read('config.ini')

# Create the ConfigParser object
scriptconfig = configparser.ConfigParser()

# Get the directory of the current script (docgen.py)
script_dir = Path(__file__).parent

# Construct the path to config.ini
config_path = script_dir / 'config.ini'

# Load the configuration file
scriptconfig.read(config_path)

# Debugging: Verify the loaded sections
if not scriptconfig.sections():
    print(f"Failed to load config file from: {config_path}")
else:
    print(f"Successfully loaded config file: {config_path}")

SCRIPT_ID = scriptconfig['DEFAULT']['script']



# DATE & TIME
today = datetime.now()
date = today.strftime("%m-%d-%y")
time = today.strftime("%H:%M")


# ========================LOCAL FUNCTIONS=============================
#Remove trailing zeros from numbers exported from MC
def format_number(num):
    """
    Remove trailing zeros from numbers exported from MC
    
    Args:
        num (float): any float number

    Returns:
        val: same number trimmed by trainling zeroes
    """
    try:
        dec = decimal.Decimal(num)
    except:
        return 'bad'
    tup = dec.as_tuple()
    delta = len(tup.digits) + tup.exponent
    digits = ''.join(str(d) for d in tup.digits)
    if delta <= 0:
        zeros = abs(tup.exponent) - len(tup.digits)
        val = '0.' + ('0'*zeros) + digits
    else:
        val = digits[:delta] + ('0'*tup.exponent) + '.' + digits[delta:]
    val = val.rstrip('0')
    if val[-1] == '.':
        val = val[:-1]
    if tup.sign:
        return '-' + val
    return val

# get path to latest file in local directory
def latestFileInDirectory(path,pattern):
    """
    Get path to latest file in local directory

    Args: 
        path (str): The directory you want to probe
        pattern (str): Search pattern

    Returns: 
        Name of the latest file or directory in path matching pattern
    """
    list_of_files = glob.glob(path+'/*'+pattern+'*') 
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    return latest_file

# Open the latest scan/ECS file and replace device parameter keys in
# placeholders.ini with current values in a new ini file (configparser 
# currentvalues).
# This step connects names in Google Templates with the current device
# value. 
# Pattern refers to a specific pattern in the filename if required, e.g., 
# "ECS Live Dump" or "Scan No "
def getValueForNameKeysECS(
        path,pattern,keylist,
        currentvalues,argcurrentvalues):
    """
    Open the latest scan/ECS file in path and replace device parameter 
    keys from keylist (placeholders.ini) with current values in a new 
    ini file (configparser currentvalues) writing them in 
    argcurrentvalues.
    This step connects names in Google Templates with the current
    device value that are stored locally. 
    Pattern refers to a specific pattern in the filename if required,
    e.g., "ECS Live Dump" or "Scan No "

    Args: 
        path (str): path to look for scan/ECS files locally
        pattern (str): search for name pattern in path
        keylist (list): keyword list of data extracted from ECS/scan
        currentvalues (configparser): configparser to store extracted 
            values in
        argcurrentvalues: path to store currentvalues config

    Returns: 
        Saves a new configuration file in the specified location,
        where all keys from a keylist (placeholders)
        are looked up in the ECS/scan file and connected to their 
        current values.
    """
    #latest ECS dump for the latest scan
    latestfile = latestFileInDirectory(path,pattern)
    latest = configparser.ConfigParser()
    latest.read(latestfile)

    #keylist is the placeholder list prepared in the ini
    for i in keylist:
        #print(i)
        tmp = i[1]
        devicename = tmp.split("&&")[0]
        #print(devicename)
        parameter = tmp.split("&&")[1]
        #print(parameter)
        for j in latest.sections(): 
            #print(j)
            try: 
                devnam = latest[j]['Device Name']
                devnam = devnam.replace('"','')
            except: devnam = None
            # print(devnam)
            if devnam == devicename:
                # print(latest[j][str(parameter)])
                try: tmp = latest[j][parameter]
                except: tmp = "No value found"
                currentvalues['DEFAULT'][i[0]] = tmp.replace('"','')
    with open(argcurrentvalues, 'w') as configfile:
        currentvalues.write(configfile)

#Append the latest scan detail info (parameter, range etc) 
# to the currentvalues.ini that contains the latest values.
def getValueForNameKeysScanFiles(path,pattern,currentvalues,argcurrentvalues):
    """
    Append the latest scan detail info (parameter, range etc) 
    to the currentvalues.ini that contains the latest values.
    Does not use keys like ECS version, but takes whole scan-
    file 'Scan Info' section.

    Args:
        path (str): path to scanfile
        pattern (str): pattern for scanfile name
        currentvalues (configparser): configparser to write values
        argcurrentvalues (str): path to save currenvalues

    Returns: 
        Config file with the standard scan info added 
    """
    latestfile = latestFileInDirectory(path,pattern)
    latest = configparser.ConfigParser()
    latest.read(latestfile)
    
    for (key, value) in latest['Scan Info'].items():
        # print(key)
        # print(value.replace('"',''))
        currentvalues['DEFAULT'][key]=value.replace('"','') 
        # print(currentvalues['DEFAULT'][key])
        if format_number(currentvalues['DEFAULT'][key]) != 'bad': 
            currentvalues['DEFAULT'][key]=format_number(
                currentvalues['DEFAULT'][key])

    with open(argcurrentvalues, 'w') as f:
        currentvalues.write(f)

#Explains itself
def cropAndScaleImage(imagepath,margin_left,margin_top,
                    margin_right,margin_bottom,scalefactor):
    """
    Crops and scales and image.

    Args:
        imagepath (str): path to image
        margin_left (int): explains itself
        margin_top (int): explains itself
        margin_right (int): explains itself
        margin_bottom (int): explains itself
        scalefactor (float): explains itself

    Returns:
        Saves a cropped and scaled image in the same path
    """
    img = Image.open(imagepath)
    #print(img.size)
    width = int(img.size[0]-margin_right)
    height = int(img.size[1]-margin_bottom)
    print('image size:', img.size[0], img.size[1])
    print('margin left, margin top, right, bottom = ', int(margin_left), int(margin_top), width, height)
    img=img.crop((int(margin_left), int(margin_top), width, height))
    img=img.resize((int(width*scalefactor), int(height*scalefactor)))
					# img=img.resize((int(width*scalefactor), int(height*scalefactor)), 
                    # Image.ANTIALIAS)
    img.save(imagepath,"PNG",quality = 94)


def scale_image(image_path, target_width_in_inches = 4.75, dpi=100):
    """
    Scale an image to a target width in inches while maintaining the aspect ratio.
    Save the scaled image with '_scaled' added to the filename.

    Args:
        image_path (str): Path to the input image.
        target_width_in_inches (float): Desired width in inches.
        dpi (int, optional): Resolution in dots per inch. Default is 96 DPI.

    Returns:
        str: Path to the scaled image.
    """
    try:
        # Calculate the target width in pixels
        target_width_pixels = int(target_width_in_inches * dpi)
        
        # Open the image
        with Image.open(image_path) as img:
            # Calculate the new height to maintain aspect ratio
            aspect_ratio = img.height / img.width
            target_height_pixels = int(target_width_pixels * aspect_ratio)
            
            # Resize the image
            scaled_img = img.resize(
                (target_width_pixels, target_height_pixels),
                Image.Resampling.LANCZOS  # Use LANCZOS for high-quality resizing
            )
            
            # Save the scaled image
            base, ext = os.path.splitext(image_path)
            scaled_image_path = f"{base}_scaled{ext}"
            scaled_img.save(scaled_image_path)
            print(f"Scaled image saved to: {scaled_image_path}")
            return scaled_image_path
    except FileNotFoundError:
        print(f"Error: File not found at {image_path}. Please provide a valid path.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
# scaled_image_path = scale_image("path/to/your/image.jpg", 5.25)

        
# ==================FUNCTIONS USING GOOGLE API========================

# Establish connection with google api service
# All following functions with a servicevar can run as standalone 
# (if None specified)
# Or use a common service established through the function 
# servicevar = establishService(apiservice,apiversion)
def establishService(apiservice, apiversion):
    """
    Handles connection with Google API and authorization.

    Args: 
        apiservice (str): Name of the Google API (e.g., 'docs', 'drive').
        apiversion (str): Version number of the API (e.g., 'v1').

    Returns:
        service (googleapiclient.discovery.Resource): Google API service object.
    """
    print('**Establish Server Connection with Google Cloud**')

    creds = None

    # Define the path to token.pickle and credentials.json
    base_path = Path(__file__).parent
    token_path = base_path / 'token.pickle'
    credentials_path = base_path / 'credentials.json'

    # Check for token.pickle to load stored credentials
    if token_path.exists():
        with token_path.open('rb') as token_file:
            creds = pickle.load(token_file)

    # If no (valid) credentials are available, log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                raise FileNotFoundError(f"Credentials file not found at: {credentials_path}")
            
            # Load credentials.json and initiate login
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with token_path.open('wb') as token_file:
            pickle.dump(creds, token_file)

    # Build the service object
    try:
        service = build(apiservice, apiversion, credentials=creds)
        print('...Service created successfully')
        return service
    except Exception as e:
        print(f"...Error in opening the service: {e}")
        raise

# Create daily log and return ID// if exits already, just return ID
def createExperimentLog(logtempID,tempfolderID,logfolderID, 
                        logfilename,argconfig,servicevar):
    """
    If no Google Docfile exists for this day yet, this function
    generates a new file from the template.

    Args:
        logtempID (str): google ID of the template google document
        tempfolderID (str): google ID of the template google folder
        logfolderID (str): target google folder for the new document
        logfilename (str): filename for the new google doc
        argconfig (str): path to the ini file containing the apps script ID
        servicevar (json?!): service passed on from previously 
                establishservice
    
    Returns:
        Returns the google ID of the new document and updates stores
        its value in the configfile with path argconfig.
    """
    print('**Create or find Log...**')
    API_SERVICE_NAME='script'
    API_VERSION='v1'
    if(servicevar == None):
        print("...establish service in createExperimentLog standalone")
        service = establishService(API_SERVICE_NAME,API_VERSION)      
    else: 
        service = servicevar
        print("...existing service used")    #documentID = ""
    #Create an execution request object.
    request = {
        "function": 'createExperimentLog', 
        "parameters": [logtempID, tempfolderID, logfolderID, logfilename],
        "devMode": True
    }

    try:
         # Make the API request.
        print('...sending request to service')
        response = service.scripts().run(body=request,
                 scriptId=SCRIPT_ID).execute()
        if 'error' in response:
            # The API executed, but the script returned an error.
    
            # Extract the first (and only) set of error details. 
            # The values of this object are the script's 
            # 'errorMessage' and 'errorType', and an list of stack 
            # trace elements.
            print("...Something went wrong in createExperimentLog")
            error = response['error']['details'][0]
            print("Script error message: {0}".format(error['errorMessage']))
        else: 
            print('...returned documend ID: ', response['response']['result'])
            documentID = response['response']['result']
            #print(str(documentID))
            config = configparser.ConfigParser()
            config.read(argconfig)
            config['DEFAULT']['LogID'] = documentID
            with open(argconfig, 'w') as configfile:
                config.write(configfile)
    except errors.HttpError as e:
        # The API encountered a problem before the script
        # started executing.
        print("...HTTP error occurred in create Experiment:")
        print(e.content)
    except Exception as e: print("...non HTTP error in opening the service")
    return(documentID)

# Append to the ExperimentLog Document from a Template
def appendToLog(templateID,documentID,search,servicevar):
    """
    Appends a google template to a google document if the search
    phrase (search) is not present in the document yet.

    Args:
        template ID (str): google ID of the template
        document ID (str): google ID of the document to write in
        search (str): search phrase to look for in google doc
        servicevar (json?!): service established with google
    
    Returns:
        If search is not found in doc, appends the template to 
        the document and returns 0. If search is found, does not
        append the template, writes stdout explanation and quits.
    """
    API_SERVICE_NAME='script'
    API_VERSION='v1'
    if(servicevar == None):
        #print("establish service in checkFile standalone")
        service = establishService(API_SERVICE_NAME,API_VERSION)   
    else: 
        service = servicevar
        #print("service from root function used")     
    #Create an execution request object.
    
    request = {
        "function": 'appendTemplate', 
        "parameters": [templateID,documentID],
        "devMode": True
    }
    # If no search term is provided, 
    # or the search term ain't in the document,
    # the function will execute and insert the template to the document
    if (search == None): search = "SomethingNobodyWouldEverWriteInADocentEver"
    tmp = None
    try: tmp = checkFileContains(documentID,search,service)
    except: print("...Failed to check file for search pattern"); return 1; 
    if (tmp == False):
        #print('Trying to append...')
        try:
            print("**Append template to document...**")
             # Make the API request.
            response = service.scripts().run(body=request,
                    scriptId=SCRIPT_ID).execute()
            
            if 'error' in response:
                # The API executed, but the script returned an error.
    
                # Extract the first (and only) set of error details. 
                # The values of this object are the script's 
                # 'errorMessage' and 'errorType', 
                # and a list of stack trace elements.
                error = response['error']['details'][0]
                print("Script error msg: {0}".format(error['errorMessage']))
            else: 
                print('...',response['response']['result'])
                #print(documentID)
                return 0
        #except errors.HttpError as e:
            # The API encountered a problem 
            # before the script started executing.
        #    print("HTTP ERROR occurred in Append to Log")
        #    print(e.content)
        except errors.HttpError as e:
        # The API encountered a problem before the script
        # started executing.
            print("...HTTP error occurred in Append To Log",e)
        except Exception as e: print("Error in Append To Log ", e)
    elif (tmp == True): print("...this Scan is already present in the Log", documentID); return 0; 
    else: print("...retry"); return 1

# Use the currentvalues.ini file created via getValueForNameKeysECS
# and getValueForScanFiles functions to replace all placeholder 
# occurances in a google document by ID
def findAndReplace(documentID,placeholdersandvalues,servicevar):
    """
    Finds placeholders in a google document and replaces
    them with values. 

    Args:
        document ID (str): google ID of the document to write in
        placeholdersandvalues: configparser containing placeholders
            as keys and the values to replace them with as values.
            Placeholders in the configparser are stripped of {{}},
            which will be used in the google docs to identify them.
        servicevar (json?!): service established with google
    
    Returns:
        Finds and replaces all instances of placeholder keys within
        {{}} in the google docs and replaces them by values stored 
        in the configparser placeholdersandvalues. This configparser
        can for instance be stored in a file (currentvalues.ini)
    """

    API_SERVICE_NAME='script'
    API_VERSION='v1'
    
    keys = list(placeholdersandvalues['DEFAULT'].keys())
    values = list(placeholdersandvalues['DEFAULT'].values())

    if(servicevar == None):
        #print("establish service in checkFile standalone")
        service = establishService(API_SERVICE_NAME,API_VERSION)      
    else: 
        service = servicevar
        #print("service from root function used")
    
    #Create an execution request object.
    
    request = {
        "function": 'findAndReplace', 
        "parameters": [documentID,keys,values],
        "devMode": False
    }

    try:
         # Make the API request.
        response = service.scripts().run(body=request,
                 scriptId=SCRIPT_ID).execute()
        if 'error' in response:
            # The API executed, but the script returned an error.
    
            # Extract the first (and only) set of error details. 
            # The values of this object are the script's 
            # 'errorMessage' and 'errorType', and
            # an list of stack trace elements.
            error = response['error']['details'][0]
            print("...Script error message: {0}".format(error['errorMessage']))
            return 1
        else: 
            #print(response['response']['result'])
            return 0
            
    except Exception as e: print("Error in findAndReplace")
    except errors.HttpError as e:
        # The API encountered a problem 
        # before the script started executing.
        print("...HTTP Error in findAndReplace", e.content)

# Find and replace placeholders in tables with images. 
# So far only for images in tables.
def findAndReplaceImage(documentID,imageID, pattern,servicevar):
    """
    Finds pattern in a gdocument and replaces
    it with an image from google drive. 

    Args:
        document ID (str): google ID of the document to write in
        imageID: ID of an image on google drive
        pattern: search pattern that should be replaced in google doc
        servicevar (json?!): service established with google
    
    Returns:
        Replaces pattern (e.g., '{{screenshot}}') in the google docs
        with an image from google drive.
        Currently works only for images in a table. 
    """
    API_SERVICE_NAME='script'
    API_VERSION='v1'

    if(servicevar == None):
        #print("establish service in checkFile standalone")
        service = establishService(API_SERVICE_NAME,API_VERSION)      
    else: 
        service = servicevar
        #print("service from root function used")
    
    #Create an execution request object.
    
    request = {
        "function": 'findAndReplaceImage', 
        "parameters": [documentID,imageID,pattern],
        "devMode": False
    }

    try:
         # Make the API request.
        response = service.scripts().run(body=request,
                 scriptId=SCRIPT_ID).execute()
        if 'error' in response:
            # The API executed, but the script returned an error.
    
            # Extract the first (and only) set of error details. 
            # The values of this object are the script's 
            # 'errorMessage' and 'errorType', and
            # an list of stack trace elements.
            error = response['error']['details'][0]
            print("Script error message: {0}".format(error['errorMessage']))
        else: 
            #print(response['response']['result'])
            return 0
            

    except errors.HttpError as e:
        # The API encountered a problem 
        # before the script started executing.
        print(e.content)


def uploadImage(localimagepath, destinationID):
    """
    Uploads a local image (PNG or GIF) to Google Drive.

    Args:
        localimagepath (str): Path to the image (PNG/GIF) to upload.
        destinationID (str): ID of the Google Drive folder to upload to.

    Returns:
        str: Google ID of the uploaded file.
    """
    API_SERVICE_NAME = 'drive'
    API_VERSION = 'v3'
    driveservice = establishService(API_SERVICE_NAME, API_VERSION)
    
    # Detect file extension and determine MIME type
    file_path = Path(localimagepath)
    mime_type, _ = mimetypes.guess_type(localimagepath)
    
    if not mime_type:
        raise ValueError(f"Unable to determine MIME type for file: {localimagepath}")
    
    if mime_type not in ['image/png', 'image/gif']:
        raise ValueError("Only PNG and GIF file types are supported.")

    # Scale the image only for PNG files
    scaled_image_path = localimagepath
    if mime_type == 'image/png':
        scaled_image_path = scale_image(localimagepath)

    # Generate metadata for the upload
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_metadata = {
        'name': f"{timestamp}_tmp.{file_path.suffix.lstrip('.')}",
        'parents': [destinationID]
    }
    
    media = MediaFileUpload(scaled_image_path, mimetype=mime_type)
    
    try:
        file = driveservice.files().create(
            body=file_metadata, media_body=media, supportsAllDrives=True
        ).execute()
        if 'error' in file:
            error = file['error']['details'][0]
            print(f"Script error message: {error['errorMessage']}")
            return None
        else:
            imageID = file['id']
            return imageID

    except errors.HttpError as e:
        print(e.content)
        return None

# Check if a google docs contains a search phrase 
def checkFileContains(fileID,search,servicevar):
    """
    Checks whether a google docs file contains a search phrase

    Args:
        fileID (str): google ID of the document to search in
        search (str): search phrase
        servicevar (json?!): service established with google
    
    Returns:
        True if google docs contains search phrase
        False if not.
    """
    API_SERVICE_NAME='script'
    API_VERSION='v1'
    
    #run standalone or within another function
    if(servicevar == None):
        #print("establish service in checkFile standalone")
        service = establishService(API_SERVICE_NAME,API_VERSION)      
    else: 
        service = servicevar
        #print("service from root function used")
    
    #Create an execution request object.
    
    request = {
        "function": 'checkFileContains', 
        "parameters": [fileID,search],
        "devMode": False
    }

    try:
         # Make the API request.
        response = service.scripts().run(body=request,
                 scriptId=SCRIPT_ID).execute()
        if 'error' in response:
            # The API executed, but the script returned an error.
    
            # Extract the first (and only) set of error details. 
            # The values of this object are the script's 
            # 'errorMessage' and 'errorType', and
            # an list of stack trace elements.
            error = response['error']['details'][0]
            print("Script error message: {0}".format(error['errorMessage']))
        else: 
            #print(response['response']['result'])
            return response['response']['result']

    except errors.HttpError as e:
        # The API encountered a problem 
        # before the script started executing.
        print(e.content)


# extracts last row of a spreadsheed file 
def lastRowOfSpreadsheet(fileID, sheetString, firstcol,lastcol, servicevar):
    """
    Extracts last row of a spreadsheet

    Args:
        fileID (str): google ID of the document to search in
        sheetString (str): Name of the Sheet in the spreadsheet document
        rangeString (str): Range to look in (e.g. A1:G1000)
        servicevar (json?!): service established with google
    
    Returns:
        Last row of the specified document within the specified range as 2 dim array 
        (use as variable[0][column])
    """
    API_SERVICE_NAME='script'
    API_VERSION='v1'
    
    #run standalone or within another function
    if(servicevar == None):
        #print("establish service in checkFile standalone")
        service = establishService(API_SERVICE_NAME,API_VERSION)      
    else: 
        service = servicevar
        #print("service from root function used")
    
    #Create an execution request object.
    
    request = {
        "function": 'lastRowFromSpreadsheet', 
        "parameters": [fileID,sheetString,firstcol,lastcol],
        "devMode": False
    }

    try:
         # Make the API request.
        response = service.scripts().run(body=request,
                 scriptId=SCRIPT_ID).execute()
        if 'error' in response:
            # The API executed, but the script returned an error.
    
            # Extract the first (and only) set of error details. 
            # The values of this object are the script's 
            # 'errorMessage' and 'errorType', and
            # an list of stack trace elements.
            error = response['error']['details'][0]
            print("Script error message: {0}".format(error['errorMessage']))
        else: 
            #print(response['response']['result'])
            return response['response']['result']

    except errors.HttpError as e:
        # The API encountered a problem 
        # before the script started executing.
        print(e.content)
        
def insertImageToTableCell(documentID, scanNumber, row, column, imageID, servicevar):
    """
    Inserts an image into a specified table cell in a Google Doc.

    Args:
        documentID (str): The Google Doc ID where the table is located.
        scanNumber (int): The scan number to locate the specific heading in the document.
        row (int): The row index of the target table cell (0-based).
        column (int): The column index of the target table cell (0-based).
        imageID (str): The Google Drive ID of the image to insert.
        servicevar (object): Pre-established Google API service object; if None, a new service will be created.

    Returns:
        dict: Result of the script execution, if successful.

    Raises:
        Exception: If there is an error during API execution or script execution.
    """
    API_SERVICE_NAME = 'script'
    API_VERSION = 'v1'

    # Check if a service object is provided; if not, establish a new service
    if servicevar is None:
        # Establish a new service if none is provided
        service = establishService(API_SERVICE_NAME, API_VERSION)
    else:
        # Use the provided service object
        service = servicevar

    # Create an execution request for the Apps Script
    request = {
        "function": 'insertImageToTableCell',
        "parameters": [documentID, scanNumber, row, column, imageID],
        "devMode": True
    }

    try:
        # Make the API request to execute the Apps Script function
        response = service.scripts().run(body=request, scriptId=SCRIPT_ID).execute()
        if 'error' in response:
            # Handle script execution errors
            error = response['error']['details'][0]
            print(f"Script error message: {error['errorMessage']}")
        else:
            # Return the result of the script execution
            return response['response']['result']

    except errors.HttpError as e:
        # Handle API-level errors
        print(e.content)


def insertImageToExperimentLog(scanNumber, row, column, image_path, documentID = None, experiment = 'Undulator'):
    """
    Uploads an image to Google Drive and inserts it into a specific table cell in a Google Doc.

    Args:
        documentID (str): The Google Doc ID where the table is located.
        scanNumber (int): The scan number to locate the specific heading in the document.
        row (int): The row index of the target table cell (0-based).
        column (int): The column index of the target table cell (0-based).
        image_path (str): The local file path to the image to be uploaded and inserted.

    Returns:
        None
    """
    # Upload the image to Google Drive and retrieve the image ID
    # note, the id is for a folder in the BELLA HTU structure,
    # but it's really just a temporary location for the image before 
    # moving into the relevant experiment log. No real need to specify
    # other directories for other experiments
    
    experiment_mapping = {'Undulator':'HTUparameters.ini'}
    
    config_file = experiment_mapping.get(experiment, None)
    if config_file:
        # Create the ConfigParser object
        experiment_config = configparser.ConfigParser()

        # Get the directory of the current script (docgen.py)
        experiment_config_dir = Path(__file__).parent

        # Construct the path to config.ini
        config_path = experiment_config_dir / config_file

        # Load the configuration file
        experiment_config.read(config_path)

        # Debugging: Verify the loaded sections
        if not experiment_config.sections():
            print(f"Failed to load config file from: {config_path}")
        else:
            print(f"Successfully loaded config file: {config_path}")
            
        documentID = experiment_config['DEFAULT']['logid']  
    
    
    image_id = uploadImage(image_path, '1O5JCAz3XF0h_spw-6kvFQOMgJHwJEvP2')

    # Insert the uploaded image into the specified table cell
    insertImageToTableCell(documentID, scanNumber, row, column, image_id, None)