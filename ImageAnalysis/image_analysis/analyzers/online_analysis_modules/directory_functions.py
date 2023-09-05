"""
Created on Mon Jul 24

Module for compiling filepaths

@author: chris
"""

import os

SUPER_PATH = "Z:/data/Undulator/"


def compile_daily_path(day, month, year):
    year_path = 'Y'+'{:04d}'.format(year)
    month_path = '{:02d}'.format(month) + '-' + month_lookup(month)
    day_path = '{:02d}'.format(year%100) + '_' + '{:02d}'.format(month) + '{:02d}'.format(day)
    return SUPER_PATH + year_path + '/' + month_path + '/' + day_path + '/scans'


def compile_plot_info(day, month, year, scan, shot=None, camera_name =''):
    date_string = '{:02d}'.format(month) + '/' + '{:02d}'.format(day) + '/' + '{:02d}'.format(year%100)
    scan_string = 'Scan '+'{:03d}'.format(scan)
    if shot is not None:
        shot_string = 'Shot '+'{:03d}'.format(shot)
    else:
        shot_string = ''
    return camera_name + ': ' + date_string + ' ' + scan_string + ' ' + shot_string


def get_number_of_shots(super_path, scan_number, image_name):
    scan_path = "Scan"+"{:03d}".format(scan_number)
    num_shots = 0
    for file in os.listdir(super_path + "/" + scan_path + "/" + image_name + "/"):
        if file.endswith(".png"):
            num_shots = num_shots + 1
    print(num_shots)
    return num_shots


def compile_file_location(super_path, scan_number, shot_number, image_name, suffix=".png"):
    """
    Given the scan and shot number, compile the full path to the image

    Parameters
    ----------
    super_path : String
        Path to the folder where the datasets are stored.
        (NOTE:  Will update more once I understand the file structure of where things are saved)
    scan_number : Int
        The set number.
    shot_number : Int
        The shot number.
    image_name : String
        Name of the camera's image.
    suffix : String
        What filetype it is.  Defaults to .png

    Returns
    -------
    The filepath for the image.

    """
    scan_path = "Scan" + "{:03d}".format(scan_number)
    shot_path = scan_path + "_" + image_name + "_" + "{:03d}".format(shot_number) + suffix
    full_path = super_path + "/" + scan_path + "/" + image_name + "/" + shot_path

    return full_path


def month_lookup(month):
    month_abbreviation = 'ERR'
    if month == 1:
        month_abbreviation = 'Jan'
    elif month == 2:
        month_abbreviation = 'Feb'
    elif month == 3:
        month_abbreviation = 'Mar'
    elif month == 4:
        month_abbreviation = 'Apr'
    elif month == 5:
        month_abbreviation = 'May'
    elif month == 6:
        month_abbreviation = 'Jun'
    elif month == 7:
        month_abbreviation = 'Jul'
    elif month == 8:
        month_abbreviation = 'Aug'
    elif month == 9:
        month_abbreviation = 'Sep'
    elif month == 10:
        month_abbreviation = 'Oct'
    elif month == 11:
        month_abbreviation = 'Nov'
    elif month == 12:
        month_abbreviation = 'Dec'
    else:
        print("Not a valid month!!")
    return month_abbreviation
