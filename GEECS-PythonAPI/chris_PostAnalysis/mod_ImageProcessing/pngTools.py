# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 22:24:29 2019

@author: ATAP
"""
import sys
sys.path.append('..')
import png #this package does low level i/o on pngs
import imageio as im

def nBitPNG(fname):
    """
    This reads in a png and scales pixel values to compensate for variable bit 
    depth. This is an issue for 12 bit pngs created by NI Vision, which get 
    saved with varying numbers of significant bits. This scaling is required 
    if numerical pixel values are to be read out correctly, and no image 
    analysis package seems to do it natively.\n
    Based on Kei's "f12bitPngOpnV04" MATLAB function.\n
    INPUTS:\n
    fname: <str> path to png\n
    OUTPUTS:\n
    scaled_image: <2-D float numpy array> image scaled according to significant bits
    """
    
    # get significant bits (try exception handling here)
    try:
        sBIT = chunkBytes(fname,b'sBIT')
        sig_bits = int.from_bytes(sBIT,'little')
#    except ChunkErrorsBit:
#        print('No sBIT chunk in ' + fname)
#        sig_bits = 16
    except: 
        print('Problem reading chunks in ' + fname)
        sig_bits = 16
    
    # read png and scale
    raw_png = im.imread(fname,as_gray=True)
    scaled_image = raw_png/(2**(16-sig_bits))
    
    return scaled_image

def chunkBytes(fname,chunk_name):
    """
    Looks for a named chunk in a png and returns its value if found. There is 
    no exception handling if the named chunk isn't found, in which case an 
    error is thrown.\n
    INPUTS\n
    fname: <str> path to png\n
    chunk_name: <bytes>(e.g. b'sBIT') chunk name\n
    OUTPUTS\n
    chunk_val: <bytes>(e.g. b'sBIT') value of named chunk, if found\n
    """
    chunkread = png.Reader(filename=fname)
    
    curr_name = b'turd'
    while curr_name != chunk_name:
    #while chunkname != b'IEND':
        
        if curr_name == b'IEND':
            chunk_val=b'\x00'
            break
        else:
            curr_name,chunk_val = chunkread.chunk()
    
    return chunk_val
