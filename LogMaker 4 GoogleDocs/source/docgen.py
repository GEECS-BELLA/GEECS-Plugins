# This library is used to to make automated experiment logbooks at BELLA-HTT@LBNL on Google Drive.
# The second half of this code is a google apps script project available online. 
# A version of the Google Project source code will be stored locally for convenience.
# 
# Written by Tobias Ostermayr, last updated 04/20/2020

from __future__ import print_function
import pickle
#from wand.image import Image
from PIL import Image
import os.path
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

# DON'T TOUCH
SCOPES = "https://www.googleapis.com/auth/documents https://www.googleapis.com/auth/drive"
scriptconfig = configparser.ConfigParser()
scriptconfig.read('config.ini')
SCRIPT_ID = scriptconfig['DEFAULT']['script']


# DATE & TIME
today = datetime.now()
date = today.strftime("%m-%d-%y")
time = today.strftime("%H:%M")


# ========================LOCAL FUNCTIONS=============================
#Remove trailing zeros from numbers exported from MC
def format_number(num):
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
    list_of_files = glob.glob(path+'/*'+pattern+'*') 
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    return latest_file

# Open the latest scan/ECS file and replace device parameter keys in placeholders.ini  
# with current values in a new ini file (configparser currentvalues).
# This step connects names in Google Templates with the current device value
# Pattern refers to a specific pattern in the filename if required, e.g., "ECS Live Dump" or "Scan No "
def getValueForNameKeysECS(path,pattern,keylist,currentvalues,argcurrentvalues):
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
            try: devnam = latest[j]['Device Name'];devnam = devnam.replace('"','')
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
# to the latest_scan_values_test.ini that contains the latest device values.
def getValueForNameKeysScanFiles(path,pattern,currentvalues,argcurrentvalues):
    #latest scaninfo ini
    latestfile = latestFileInDirectory(path,pattern)
    latest = configparser.ConfigParser()
    latest.read(latestfile)
    
    for (key, value) in latest['Scan Info'].items():
        # print(key)
        # print(value.replace('"',''))
        currentvalues['DEFAULT'][key]=value.replace('"','') 
        # print(currentvalues['DEFAULT'][key])
        if format_number(currentvalues['DEFAULT'][key]) != 'bad': currentvalues['DEFAULT'][key]=format_number(currentvalues['DEFAULT'][key])

    with open(argcurrentvalues, 'w') as f:
        currentvalues.write(f)

#Explains itself
# def cropAndScaleImage(imagepath,margin_left,margin_top,margin_right,margin_bottom,scalefactor):
#     with Image(filename=imagepath) as img:
#         #print(img.size)
#         img.crop(int(margin_left), int(margin_top), width = int(img.size[0]-margin_right),  height = int(img.size[1]-margin_bottom))
#         #print(img.size)
#         #img.size
#         #print(str(int(img.size[0]*scalefactor)),str(int(img.size[1]*scalefactor)))
#         img.scale(int(img.size[0]*scalefactor),int(img.size[1]*scalefactor))
#         #print(img.size)
#         img.save(filename=imagepath)

def cropAndScaleImage(imagepath,margin_left,margin_top,margin_right,margin_bottom,scalefactor):
    img = Image.open(imagepath)
    #print(img.size)
    width = int(img.size[0]-margin_right)
    height = int(img.size[1]-margin_bottom)
    img=img.crop((int(margin_left), int(margin_top), width, height))
    img=img.resize((int(width*scalefactor), int(height*scalefactor)), Image.ANTIALIAS)
    #print(img.size)
    #img.size
    #print(str(int(img.size[0]*scalefactor)),str(int(img.size[1]*scalefactor)))
    #img.scale(int(img.size[0]*scalefactor),int(img.size[1]*scalefactor))
    #print(img.size)
    img.save(imagepath,"PNG",quality = 94)
        
# ========================FUNCTIONS USING GOOGLE API==============================

# Establish connection with google api service
# All following functions with a servicevar can run as standalone (if None is specified)
# Or use a common service established through the function 
# servicevar = establishService(apiservice,apiversion)
def establishService(apiservice,apiversion):
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    #service = build('script', 'v1', credentials=creds)

    # Call the Apps Script API
    try:
        service = build(apiservice,apiversion, credentials=creds)
        #print('Service created successfully')
        #print(service)
    #except Exception as e:
    except errors.HttpError as e:
        #print('fail')
        print(e.content)
    return(service)

# Create daily log and return ID// if exits already, just return ID
def createExperimentLog(logtempID,tempfolderID,logfolderID, logfilename,argconfig,servicevar):
    API_SERVICE_NAME='script'
    API_VERSION='v1'
    if(servicevar == None):
        #print("establish service in checkFile standalone")
        service = establishService(API_SERVICE_NAME,API_VERSION)      
    else: 
        service = servicevar
        #print("service from root function used")    documentID = ""
    #Create an execution request object.
    request = {
        "function": 'createExperimentLog', 
        "parameters": [logtempID, tempfolderID, logfolderID,logfilename],
        "devMode": True
    }

    try:
         # Make the API request.
        response = service.scripts().run(body=request,
                 scriptId=SCRIPT_ID).execute()
        
        if 'error' in response:
            # The API executed, but the script returned an error.
    
            # Extract the first (and only) set of error details. The values of
            # this object are the script's 'errorMessage' and 'errorType', and
            # an list of stack trace elements.
            error = response['error']['details'][0]
            print("Script error message: {0}".format(error['errorMessage']))
        else: 
            documentID = response['response']['result']
            #print(str(documentID))
            config = configparser.ConfigParser()
            config.read(argconfig)
            config['DEFAULT']['LogID'] = documentID
            with open(argconfig, 'w') as configfile:
                config.write(configfile)
    except errors.HttpError as e:
        # The API encountered a problem before the script started executing.
        print("here")
        print(e.content)
    return(documentID)

# Append to the ExperimentLog Document from a Template
def appendToLog(templateID,documentID,search,servicevar):
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
    # If no search term is provided, the function will execute and insert the template
    if (search == None): search = "SomethingNobodyWouldEverWriteInADocumentEver"
    if (checkFileContains(documentID,search,service) == False):
        try:
            print("Append template to document")
             # Make the API request.
            response = service.scripts().run(body=request,
                    scriptId=SCRIPT_ID).execute()
            if 'error' in response:
                # The API executed, but the script returned an error.
    
                # Extract the first (and only) set of error details. The values of
                # this object are the script's 'errorMessage' and 'errorType', and
                # an list of stack trace elements.
                error = response['error']['details'][0]
                print("Script error message: {0}".format(error['errorMessage']))
            else: return 0
                #print(response['response']['result'])
        except errors.HttpError as e:
            # The API encountered a problem before the script started executing.
            #print("here")
            print(e.content)
    else: print("This Scan is already present in the Log"); sys.exit()

# Use the latest_scan_values.ini file created via getValueForNameKeysECS and getValueForScanFiles
# functions to replace all placeholder occurances in a google document by ID
def findAndReplace(documentID,placeholdersandvalues,servicevar):


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
        "devMode": True
    }

    try:
         # Make the API request.
        response = service.scripts().run(body=request,
                 scriptId=SCRIPT_ID).execute()
        if 'error' in response:
            # The API executed, but the script returned an error.
    
            # Extract the first (and only) set of error details. The values of
            # this object are the script's 'errorMessage' and 'errorType', and
            # an list of stack trace elements.
            error = response['error']['details'][0]
            print("Script error message: {0}".format(error['errorMessage']))
        else: 
            #print(response['response']['result'])
            return 0
            

    except errors.HttpError as e:
        # The API encountered a problem before the script started executing.
        print(e.content)

# Find and replace placeholders in tables with images. So far only for images in tables.
def findAndReplaceImage(documentID,imageid, pattern,servicevar):

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
        "parameters": [documentID,imageid,pattern],
        "devMode": True
    }

    try:
         # Make the API request.
        response = service.scripts().run(body=request,
                 scriptId=SCRIPT_ID).execute()
        if 'error' in response:
            # The API executed, but the script returned an error.
    
            # Extract the first (and only) set of error details. The values of
            # this object are the script's 'errorMessage' and 'errorType', and
            # an list of stack trace elements.
            error = response['error']['details'][0]
            print("Script error message: {0}".format(error['errorMessage']))
        else: 
            #print(response['response']['result'])
            return 0
            

    except errors.HttpError as e:
        # The API encountered a problem before the script started executing.
        print(e.content)

# Upload an Image and return URL
def uploadImage(localimagepath,destinationID):
    API_SERVICE_NAME='drive'
    API_VERSION='v3'
    driveservice = establishService(API_SERVICE_NAME,API_VERSION)      
    #Create an execution request object.
    file_metadata = {'name': date + " " + time + 'tmp.png', 'parents': [destinationID]}
    media = MediaFileUpload(localimagepath,
                          mimetype='image/png'
                          )
    try:
        file = driveservice.files().create(body=file_metadata,
                                    media_body=media).execute()
        if 'error' in file:
        # The API executed, but the script returned an error.
    
        # Extract the first (and only) set of error details. The values of
        # this object are the script's 'errorMessage' and 'errorType', and
        # an list of stack trace elements.
            error = file['error']['details'][0]
            print("Script error message: {0}".format(error['errorMessage']))
        else: 
            imageID = file['id']
            #print(imageID)
            
    except errors.HttpError as e:
        # The API encountered a problem before the script started executing.
        #print("here")
        print(e.content)
    return imageID

# Check if a google docs contains a search phrase 
def checkFileContains(fileID,search,servicevar):
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
        "devMode": True
    }

    try:
         # Make the API request.
        response = service.scripts().run(body=request,
                 scriptId=SCRIPT_ID).execute()
        if 'error' in response:
            # The API executed, but the script returned an error.
    
            # Extract the first (and only) set of error details. The values of
            # this object are the script's 'errorMessage' and 'errorType', and
            # an list of stack trace elements.
            error = response['error']['details'][0]
            print("Script error message: {0}".format(error['errorMessage']))
        else: 
            #print(response['response']['result'])
            return response['response']['result']

    except errors.HttpError as e:
        # The API encountered a problem before the script started executing.
        print(e.content)

