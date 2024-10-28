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
import configparser
import re
import requests

import scram


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
