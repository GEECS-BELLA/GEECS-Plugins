#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions for SCRAM authentication client

Module includes the following functions:
b_xor      Return bit-wise xor
h_i        SCRAM Hi() function
b64encode  Encode bytes-like object using Base64, with or without padding
b64decode  Decode Base64 encoded bytes-like object, add padding as needed

Created on Fri 2017-11-17 18:30:13
Last updated on 2017-11-18

@author: rvitti
"""
import hmac
import base64


def b_xor(x: bytes, y: bytes) -> bytes:
    """Perform bitwise xor of two bytes-like objects, return resulting bytes"""
    return bytes([b1 ^ b2 for b1, b2 in zip(x, y)])


def h_i(key: bytes, salt: bytes, i: int, hash_name: str) -> bytes:
    """SCRAM Hi() function, essentially PBKDF2 [RFC2898] with hmac as PRF
    
    Produce a derived key from a base key, using a salt and other
    parameters.  Typically applied to passwords to derive salted
    passwords that can be more safely stored.
    
    Return derived key as bytes.
    
    Reference: RFC5802, Section 2, Paragraph 2.2 Notation
    
    Keyword arguments:
    key       -- bytes-like object to encode; "str" in RFC5802 (octet string)
    salt      -- bytes-like object to use as salt
    i         -- iteration count, integer
    hash_name -- name of the hash algorithm for hmac objects to use
    """
    adder = 1
    salt_plus = salt + adder.to_bytes(4, 'big')
    
    # Calculate first hmac digest and initialize the derived key hi
    u_prev = hmac.new(key, salt_plus, hash_name).digest()
    hi = u_prev
    
    # Calculate subsequent i-1 hmac digests and update hi with each digest
    for n in range(1,i):
        u = hmac.new(key, u_prev, hash_name).digest()
        hi = b_xor(hi, u)
        u_prev = u 
    return(hi)


def b64_encode(s: bytes, padding: bool = True) -> bytes:
    """Encode bytes-like object using Base64, with or without padding
    
    HTTP Authorization headers for SkySpark authentication use Base64
    URL encoding without padding.  See RFC7515 for example.
    
    Return the encoded bytes.

    Keyword arguments:
    s       -- bytes-like object to encode
    padding -- boolean to indicate if padding shall be used (default: True) 
    """
    b64_s = base64.b64encode(s)
    
    # base64.b64encode adds padding if the length of the input bytes
    # sequence is not a multiple of 3 (Base64 has 4:3 ratio).  Remove
    # padding as necessary if requested.
    if not padding and len(s) % 3 > 0:
        padding_len = 3 - len(s) % 3
        return(b64_s[0:-padding_len])
    else:
        return(b64_s)
        

def b64_decode(s: bytes) -> bytes:
    """Decode the Base64 encoded bytes-like object

    HTTP Authorization headers for SkySpark authentication use Base64
    URL encoding without padding.
    
    Add padding before decoding as needed.  Return the decoded bytes.

    Keyword arguments:
    s       -- bytes-like object to decode
    """
    if len(s) % 4 > 0:
        padding_len = 4 - len(s) % 4
        s += b'='*padding_len
    return(base64.b64decode(s))