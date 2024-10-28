#!/usr/bin/env python
# coding: utf-8

# In[124]:


import configparser
import re
import requests
import scram
import requests
from hszinc import parse
import hszinc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt


# In[125]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions to interact with SkySpark database using Axon queries

Module includes the following functions:
request        Send Axon request to SkySpark, return resulting text
__name__       Simple console to send REST request to SkySpark

Created on Sun Nov 19 15:29:51 2017
Last updated on 2019-08-23

@author: rvitti
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
    query = query.replace("\\$","\\\\\$").replace('"', '\\\"')
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



if __name__ == '__main__':
    """Simple console to send REST request to SkySpark and display results"""
    ref = "https://skyfoundry.com/doc/docSkySpark/Axon"
    sample = "read(point and siteRef->dis==\"Building 77\" and " + \
             "equipRef->dis==\"AHU-33\" and discharge and air " + \
             "and temp and sensor).hisRead(yesterday))\n" +\
             "Enter 'q' or 'quit' to exit"
    query = ""
    while query.lower() != "quit" and query.lower() != "q":
        query = input("Enter Axon query:\n>")
        response = axon_request(query)
        if query.lower() == "help":
            print("""\nReference: %s\nExample: %s""" % (ref, sample))
        elif query.lower() != "quit" and query.lower() != "q":
            try:
                #print(axon_request(query))
                print('hello')
                data = axon_request(query)
                with open('datfile.txt','w') as file:
                    file.write(data)
            except AxonException as e:
                print(e.args[0]+'\n')


# In[126]:


def read_file(file_name):
    try:
        with open(file_name, 'r') as file:
            content = file.read()
        return content
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def parse_zinc_output(zinc_output):
    try:
        grid = hszinc.parse(zinc_output)
        return grid
    except Exception as e:
        print(f"Failed to parse Zinc output: {e}")
        return None        
def main():
    file_name = "datfile.txt"  # Update with your file name
    
    # Read the text file
    zinc_output = read_file(file_name)
    if zinc_output is None:
        return
    
    print("Zinc Output:")
    print(zinc_output)

    # Parse the Zinc output
    grid = parse_zinc_output(zinc_output)
    if grid is None:
        return
    
    print("\nParsed Grid:")
    for row in grid:
        print(row)

    export_to_csv(grid, 'parsedfile.csv')

    plot_data(grid)

import csv

def export_to_csv(grid, csv_file_name):
    try:
        with open(csv_file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write the header
            header = grid.column.keys()
            writer.writerow(header)
            
            # Write the rows
            for row in grid:
                writer.writerow([row.get(col, '') for col in header])
                
        print(f"Data successfully exported to {csv_file_name}")
    except Exception as e:
        print(f"An error occurred while writing to the CSV file: {e}")



def plot_data(grid):
    try:
        # Assuming the grid has columns 'time' and 'value'
        times = [row['ts'] for row in grid]
        values = [row['v0'] for row in grid]

        plt.figure(figsize=(10, 5))
        plt.plot(times, values, marker='o', linestyle='-', color='b')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Line Graph of Parsed Data')
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"An error occurred while plotting the data: {e}")

if __name__ == "__main__":
    main()


# In[127]:


file_name = "datfile.txt"  # Update with your file name
zinc_output = read_file(file_name)
grid = parse_zinc_output(zinc_output)

# Create a list of lists to store the row data
rows = []

# Iterate over the rows in the grid
for row in grid:
    # Convert the row (which is a dictionary) to a list of values

    row_values = [row[col] for col in grid.column.keys()]
    rows.append(row_values)

# Convert the list of lists to a NumPy array
array = np.array(rows)
df = pd.DataFrame(array)


# In[128]:


# Cleanup of Extracted Data
df2 = df.loc[:,[35,5,25,16, 17]]
df2.columns = ["Name", "Description","Update_Time","Status","Value"]
df2["System_Time"] = dt.datetime.now()
df2["System_Time"] = df2["System_Time"].dt.tz_localize('America/Los_Angeles')
df2["Update_Time"] = df2['Update_Time'].dt.tz_convert('America/Los_Angeles')
df2['Data_Age'] = df2['System_Time']  - df2['Update_Time']
df2


# In[129]:


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
df3


# In[ ]:





# In[ ]:





# In[ ]:




