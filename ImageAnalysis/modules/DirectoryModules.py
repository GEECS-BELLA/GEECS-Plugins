"""
Created on Mon Jul 24

Module for compiling filepaths

@author: chris
"""

const_superpath = "Z:/data/Undulator/"

def CompileDailyPath(day, month, year):
    yearpath = 'Y'+'{:04d}'.format(year)
    monthpath = '{:02d}'.format(month) + '-' + MonthLookup(month)
    daypath = '{:02d}'.format(year%100) + '_' + '{:02d}'.format(month) + '{:02d}'.format(day)
    return const_superpath + yearpath + '/' + monthpath + '/' + daypath + '/scans'

def CompilePlotInfo(day, month, year, scan, shot, cameraStr):
    dateStr = '{:02d}'.format(month) + '/' + '{:02d}'.format(day) + '/' + '{:02d}'.format(year%100)
    scanStr = 'Scan '+'{:03d}'.format(scan)
    shotStr = 'Shot '+'{:03d}'.format(shot)
    return cameraStr +': ' + dateStr + ' ' + scanStr + ' ' + shotStr

def CompileFileLocation(superpath, scannumber, shotnumber, imagename, suffix=".png"):
    """
    Given the scan and shot number, compile the full path to the image

    Parameters
    ----------
    superpath : String
        Path to the folder where the datasets are stored.
        (NOTE:  Will update more once I understand the file structure of where things are saved)
    scannumber : Int
        The set number.
    shotnumber : Int
        The shot number.
    imagename : String
        Name of the cameara's image.
    suffix : String
        What filetype it is.  Defaults to .png

    Returns
    -------
    The filepath for the image.

    """
    scanpath = "Scan" + "{:03d}".format(scannumber)
    shotpath = scanpath + "_" + imagename + "_" + "{:03d}".format(shotnumber) + suffix
    fullpath = superpath + "/" + scanpath + "/" + imagename + "/" + shotpath

    return fullpath

def CompileTDMSFilepath(superpath, scannumber):
    """
    Given the scan number, compile the path to the tdms file

    Parameters
    ----------
    superpath : String
        Path to the folder where the datasets are stored.
        (NOTE:  Will update more once I understand the file structure of where things are saved).
    scannumber : Int
        The set number.

    Returns
    -------
    tdms_filepath : String
        The filepath for the tdms file.

    """
    scanpath = "Scan" + "{:03d}".format(scannumber)
    tdms_filepath = superpath + "/" + scanpath + "/" + scanpath + ".tdms"
    return tdms_filepath

def MonthLookup(month):
    monthabbr = 'ERR'
    if month == 1:
        monthabbr = 'Jan'
    elif month == 2:
        monthabbr = 'Feb'
    elif month == 3:
        monthabbr = 'Mar'
    elif month == 4:
        monthabbr = 'Apr'
    elif month == 5:
        monthabbr = 'May'
    elif month == 6:
        monthabbr = 'Jun'
    elif month == 7:
        monthabbr = 'Jul'
    elif month == 8:
        monthabbr = 'Aug'
    elif month == 9:
        monthabbr = 'Sep'
    elif month == 10:
        monthabbr = 'Oct'
    elif month == 11:
        monthabbr = 'Nov'
    elif month == 12:
        monthabbr = 'Dec'
    else:
        print("Not a valid month!!")
    return monthabbr
