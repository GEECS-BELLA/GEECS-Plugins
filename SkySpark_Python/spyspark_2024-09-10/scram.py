#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions to authenticate to a server using SCRAM

Module includes the following functions:
current_token   Retrieve most recent authorization token
update_token    Use stored information to update authorization token
hello           Send HELLO message for handshake with server
first_message   Send HELLO and first client message to request authentication
final_message   Send all client messages to request authentication from server
parse_auth      Parse WWW-Authenticate string from server response header
parse_msg       Parse SCRAM message string and return a dictionary
    
References:
RFC5802
https://project-haystack.org/doc/Auth
http://www.alienfactory.co.uk/articles/skyspark-scram-over-sasl

Created on Sun Nov 19 18:22:58 2017
Last updated on 2017-11-20

@author: rvitti
"""
import configparser
import getpass
import hashlib
import hmac
import re
import requests
import secrets

import scram_utils as scram_u


# Define constants
CONFIG_FILE = "./spyspark.cfg"
MAX_ATTEMPTS = 3
NONCE_LEN = 32

# Define global module variables, in particular config object
config = configparser.ConfigParser()
result_list = config.read(CONFIG_FILE)
if not result_list:
    raise Exception("Missing config file spyspark.cfg")
host_ops_addr = config['Host']['Address'] + "about"


# Exception raised if requested authentication method from server is not SCRAM
class AuthException(Exception):
    pass

# Exception raised if username / password pair is not regognized by server
class LoginException(Exception):
    pass


def current_token() -> str:
    """Return most recent authorization token for HTTP requests"""
    try:
        return(config['Authorization']['Token'])
    except KeyError:
        return("")


def update_token() -> bool:
    """Update authorization token for HTTP requests using stored information
    
    Use the username and salted password stored in the configuration file
    in order to attempt and update the authorization token.
    If the salt has changed or the update otherwise fails, attempt to
    re-authorize using user input.
    
    Return True if the update succeeded.
    """
    try:
        username = config['Authorization']['User']
        salted_password = config['Authorization']['SaltedPassword']
    except KeyError:
        username = None
        salted_password = None
    
    for i in range(0, MAX_ATTEMPTS):
        try:
            auth_dict = final_message(username, salted_password)
            config['Authorization']['User'] = auth_dict['user']
            config['Authorization']['SaltedPassword'] = auth_dict['salted_pwd']
            config['Authorization']['Salt'] = auth_dict['salt']
            config['Authorization']['Iterations'] = auth_dict['iterations']
            config['Authorization']['Token'] = auth_dict['authToken']
            with open(CONFIG_FILE, 'w') as configfile:
                config.write(configfile)
            return True
        except LoginException as e:
            print("Stored or entered username or password invalid")
            username = None
            salted_password = None


def hello(username: str) -> dict:
    """Send HELLO message for handshake with server
    
    Return a dictionary containing the authentication method, and the
    following additional expected keys: hash and handshakeToken.
    All dictionary values are text strings.
    
    Keyword arguments:
    username     -- username, as text string
    """
    b64_username = scram_u.b64_encode(username.encode("utf-8"), padding=False)
    headers = {"authorization": b'HELLO username='+b64_username}
    r = requests.get(host_ops_addr, headers=headers)
    if r.status_code != 401:
        raise Exception("HTTP error: %d" % r.status_code)
    try:
        auth = r.headers["WWW-Authenticate"]
    except:
        raise Exception("WWW-Authenticate field expected but not found"+\
                        "in response header")
    response = parse_auth(auth)
    response['user'] = username
    return(response)


def first_message(username: str) -> dict:
    """Send HELLO and first client message, request authentication from server
    
    Send HELLO and first client message and process response from server.
    Return  a dictionary containing the authentication method, and the
    following additional expected keys: data, hash and handshakeToken. Also
    include client nonce and first message bare.
    All dictionary values are text strings.
    
    Keyword arguments:
    username     -- username, as text string
    """
    # Send HELLO message and test response
    auth_dict = hello(username)
    if auth_dict['method'] != "scram":
        raise AuthException("Server requested an authentication method" +
                            "other than SCRAM")
    
    # Create bare first message.  Channel binding not used in this module.
    gs2_cbind_flag = "n"
    gs2_header = gs2_cbind_flag + "," + ","
    auth_dict['gs2_header'] = gs2_header
    auth_dict['nonce'] = secrets.token_urlsafe(NONCE_LEN)
    c_1st_msg_bare = "n=" + username + ",r=" + auth_dict['nonce']
    auth_dict['c_1st_msg_bare'] = c_1st_msg_bare
    
    # Complete and encode first message.
    c_1st_msg = gs2_header + c_1st_msg_bare
    b64_c_1st_msg = scram_u.b64_encode(c_1st_msg.encode("utf-8"), 
                                       padding=False)
    
    # Complete header and make authentication request
    auth_bytes = \
         auth_dict['method'].encode("utf-8") + b' ' + \
         b'handshakeToken=' + auth_dict['handshakeToken'].encode("utf-8") + \
         b',' + b'data=' + b64_c_1st_msg
    headers = {"authorization" : auth_bytes}
    r = requests.get(host_ops_addr, headers=headers)
    
    # Check status code and response header contents
    if r.status_code != 401:
        raise Exception("HTTP error: %d" % r.status_code)
    try:
        auth = r.headers["WWW-Authenticate"]
    except:
        raise Exception("WWW-Authenticate field expected but not found"+\
                        "in response header")
        
    response = parse_auth(auth)
    auth_dict.update(response)
    return(auth_dict)


def final_message(username: str = None, salted_password: str = None) -> dict:
    """Send all client messages to request authentication from server
    
    Send all client messages and process responses from server. Return 
    a dictionary containing all authentication information, including the
    authorization token.
    All dictionary values are text strings.
    
    Keyword arguments:
    username         -- username, as text string (default: None)
    salted_password  -- salted password, as text string (default: None)
    """
    # Send HELLO and first client message and test response
    if salted_password is None:
        username = input("username: ")
    auth_dict = first_message(username)
    if auth_dict['method'] != "scram":
        raise AuthException("Server requested an authentication method" +
                            "other than SCRAM")
    
    # Decode data and test nonce
    s_1st_msg = scram_u.b64_decode(auth_dict['data']).decode('utf-8')
    s_msg_values = parse_msg(s_1st_msg)
    if s_msg_values['r'][0:len(auth_dict['nonce'])] != auth_dict['nonce']:
        raise Exception('Nonce mismatch')
        
    # If salted_password is not available, request user input
    if salted_password is None:
        salted_password = scram_u.h_i(
                getpass.getpass("password: ").encode('utf-8'),
                scram_u.b64_decode(s_msg_values['s'].encode('utf-8')),
                int(s_msg_values['i']),
                auth_dict['hash'])
    else:
        salted_password = scram_u.b64_decode(salted_password.encode('utf-8'))
        
    # Add most recent password info to auth_dict
    auth_dict['salted_pwd'] = scram_u.b64_encode(salted_password)\
                              .decode('utf-8')
    auth_dict['salt'] = s_msg_values['s']
    auth_dict['iterations'] = s_msg_values['i']
    
    # Compute keys
    client_key = hmac.new(salted_password, "Client Key".encode('utf-8'), 
                          auth_dict['hash']).digest()
    server_key = hmac.new(salted_password, "Server Key".encode('utf-8'), 
                          auth_dict['hash']).digest()
    stored_key = hashlib.new(auth_dict['hash'], client_key).digest()
    
    # Create client final message without proof
    chan_bind_b64 = scram_u.b64_encode(auth_dict['gs2_header'].encode('utf-8'))
    chan_binding = "c=" + chan_bind_b64.decode('utf-8')
    c_final_msg_wo_proof = chan_binding + ",r=" + s_msg_values['r']
    
    auth_message = auth_dict['c_1st_msg_bare'] + "," + s_1st_msg + "," + \
                   c_final_msg_wo_proof
    
    # Compute signatures
    c_signature = hmac.new(stored_key, auth_message.encode('utf-8'), 
                           auth_dict['hash']).digest()
    s_signature = hmac.new(server_key, auth_message.encode('utf-8'), 
                           auth_dict['hash']).digest()

    # Complete client final message with proof and encode
    c_proof = scram_u.b_xor(client_key, c_signature)
    proof = "p=" + scram_u.b64_encode(c_proof).decode('utf-8')
    c_final_msg = c_final_msg_wo_proof + "," + proof
    b64_c_final_message = scram_u.b64_encode(c_final_msg.encode("utf-8"),
                                          padding=False)

    # Complete header and make authentication request
    auth_bytes = \
         auth_dict['method'].encode("utf-8") + b' ' + \
         b'handshakeToken=' + auth_dict['handshakeToken'].encode("utf-8") + \
         b',' + b'data=' + b64_c_final_message
    headers = {"authorization" : auth_bytes}
    r = requests.get(host_ops_addr, headers=headers)
    
    # Check status code and response header contents
    if r.status_code == 403:
        raise LoginException("Username or password is not valid")
    elif r.status_code != 200:
        raise Exception("HTTP error: %d" % r.status_code)
    try:
        auth = r.headers["Authentication-Info"]
    except:
        raise Exception("Authentication-Info field expected but not found"+ \
                        "in response header")
    
    response = parse_auth(auth, method=False)
    auth_dict.update(response)
    
    # Validate server key
    s_final_message = scram_u.b64_decode(auth_dict['data'].encode('utf-8')) \
                      .decode('utf-8')
    s_msg_values = parse_msg(s_final_message)
    
    if not hmac.compare_digest(scram_u.b64_decode(s_msg_values['v']\
                                                  .encode('utf-8')), 
                               s_signature):
        raise Exception('Server signature is incorrect')    

    return(auth_dict)


def parse_auth(auth: str, method: bool = True) -> dict:
    """Parse WWW-Authenticate string from server response header
    
    Parse WWW-Authenticate string from server response header and return
    a dictionary with authentication method and other provided keys.
    All dictionary values are text strings.
    
    Keyword arguments:
    auth   -- WWW-Authenticate text string
    """
    parsed = {}
    if method:
        parsed['method'] = auth.split(" ")[0]
        auth = auth.split(parsed['method']+" ")[1]
    
    # Rest of text field contains keys separated with ", "
    auth_pairs = auth.split(", ")
    for pair in auth_pairs:
        key = pair.split("=")[0]
        value = pair.split("=")[1]
        parsed[key] = value
    if 'hash' in parsed:
        parsed['hash'] = re.sub('-','',parsed['hash'])
    return(parsed)


def parse_msg(msg: str) -> dict:
    """Parse SCRAM message string and return dictionary
    
    All dictionary values are text strings.
    
    Keyword arguments:
    msg    -- SCRAM message
    """
    parsed = {}
    msg_pairs = msg.split(",")
    for pair in msg_pairs:
        key = pair.split("=")[0]
        value = pair.split("=")[1]
        parsed[key] = value
    return(parsed)