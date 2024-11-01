#!/usr/bin/env python
# coding: utf-8

# In[5]:


import configparser
import re
import requests
import scram
import requests
import hszinc
from hszinc import parse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import csv
import sys


# In[6]:


CONFIG_FILE = "./spyspark.cfg"
MAX_ATTEMPTS = 3

# Define global module variables, in particular config object
config = configparser.ConfigParser()
result_list = config.read(CONFIG_FILE)
if not result_list:
    raise Exception("Missing config file spyspark.cfg")
host_addr = config['Host']['Address']
skysparkquery = config['Query']['query']


# In[9]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions to interact with SkySpark database using Axon queries

Module includes the following functions:
request        Send Axon request to SkySpark, return resulting text
__name__       Simple console to send REST request to SkySpark

Created on Sun Nov 19 15:29:51 2017
Last updated on 2024-20-28 by Chetanya for GEECS System
"""



# Define constants
CONFIG_FILE = "./spyspark.cfg"
MAX_ATTEMPTS = 3

# Define global module variables, in particular config object
config = configparser.ConfigParser()
result_list = config.read(CONFIG_FILE)
if not result_list:
    raise Exception("Missing config file spyspark.cfg")
host_addr = config['Host']['Address']


# Exception raised if empty result is received from SkySpark
class AxonException(Exception):
    pass


def about() -> str:
    uri = host_addr + "about"
    return request(uri)


def axon_request(query: str, result_type: str = "text/zinc") -> str:
    uri = host_addr + "eval"
    
    # Encode special characters
    #query = query.replace("\\$","\\\\\$").replace('"', '\\\"')
    # Encode request as Zinc
    data = f"""ver:"3.0"\nexpr\n"{query}"\n""".encode('utf-8')

    return request(uri, data, result_type, request_type="text/zinc")


def his_write(data) -> str:
    uri = host_addr + "hisWrite"
    return request(uri, data)


def commit(data) -> str:
    uri = host_addr + "commit"
    return request(uri, data)


def request(request_uri: str, data: str = None,
            result_type: str = "text/zinc",
            request_type: str = "text/zinc") -> str:
    """Process REST operation, return resulting text
    
    Use SkySpark REST API with uri provided as first argument.
    Use authorization token stored in spyspark.cfg. If
    an authorization issue is detected, attempt to re-authorize. If other
    HTTP issues are detected, raise Exception. Return result as string.
    
    If the Axon query returns 'empty\n', a custom AxonException is raised.
    This can occur if there are no results or if the query is bad.
    
    Keyword arguments:
    request_uri  -- REST Uri to use with REST operation
    data         -- Data to use with POST request, or None for GET
    result_type  -- Requested MIME type in which to receive results
                    (default: "text/zinc" for Zinc format)
    request_type -- MIME type in which the request data is provided
    """
    # Attempt to send request; if an authorization issue is detected,
    # retry after updating the authorization token
    for i in range(0, MAX_ATTEMPTS):
        auth_token = scram.current_token()
        headers= {"authorization": "BEARER authToken="+auth_token,
                  "accept": result_type,
                  "content-type": request_type}
        if data is None:
            r = request.get(request_uri, headers=headers)
        else:
            r = requests.post(request_uri, data=data, headers=headers)
        if r.status_code == 200:
            if r.text != "empty\n":
                if result_type == "text/csv":
                    text = re.sub('â\x9c\x93','True',r.text)    # Checkmarks
                    text = re.sub('Â','',text)
                    return text
                else:
                    return r.text
            else:
                raise AxonException("Empty result, check query")
        if r.status_code == 400:    # Missing required header
            raise Exception("HTTP request is missing a required header")
        if r.status_code == 404:    # Invalid URI
            raise Exception("URI does not map to a valid operation URI")    
        if r.status_code == 406:    # Invalid "accept" header
            raise Exception("Unsupported MIME type requested")
        if r.status_code == 403:    # Authorization issue, try to reauthorize
            scram.update_token()
        else:
            raise Exception("HTTP error: %d" % r.status_code)


def parse_zinc_output(zinc_output):
    try:
        grid = hszinc.parse(zinc_output)
        return grid
    except Exception as e:
        print(f"Failed to parse Zinc output: {e}")
        return None    

def process_data(grid):
    rows = []
    # Iterate over the rows in the grid
    for row in grid:
        # Convert the row (which is a dictionary) to a list of values

        row_values = [row[col] for col in grid.column.keys()]
        rows.append(row_values)

    # Convert the list of lists to a NumPy array
    array = np.array(rows)
    df = pd.DataFrame(array)
    df2 = df.loc[:,[0,35,25,16, 17]]
    df2.columns = ["Name", "Description","Update_Time","Status","Value"]
    df2["System_Time"] = dt.datetime.now()
    df2["System_Time"] = df2["System_Time"].dt.tz_localize('America/Los_Angeles')
    df2["Update_Time"] = df2['Update_Time'].dt.tz_convert('America/Los_Angeles')
    df2['Data_Age'] = df2['System_Time']  - df2['Update_Time']
    df3 = df2
    df3
    def extract_number_and_unit(temp_string):
        temp_string = str(temp_string)
        pattern =  r"([-+]?\d*\.\d+|\d+)\s*([^\d\s]+)"
        match = re.search(pattern, temp_string)
        if match:
            number = match.group(1)
            unit = match.group(2).replace('Â°F', 'F')  # Remove the unwanted character
            return number, unit
        return None, None

    df3[['Value', 'Unit']]=df3['Value'].apply(lambda x:pd.Series(extract_number_and_unit(x)))

    df3 = df3.iloc[:,[0,1,3,4,7,2,5,6]]
    
    df3 = df3.to_numpy()   
    return(df3)
        
        
if __name__ == '__main__':
    """Simple console to send REST request to SkySpark and display results"""
    ref = "https://skyfoundry.com/doc/docSkySpark/Axon"
    sample = "read(point and siteRef->dis==\"Building 77\" and " + \
             "equipRef->dis==\"AHU-33\" and discharge and air " + \
             "and temp and sensor).hisRead(yesterday))\n" +\
             "Enter 'q' or 'quit' to exit"
    query = skysparkquery
    data = axon_request(query)
    with open('datfile.txt','w') as file:
        file.write(data)
    zinc_output = data
    grid = parse_zinc_output(zinc_output)
    output = process_data(grid)
    print(output)
    with open('datfile2.txt','w') as file:
        file.write(output)
    
