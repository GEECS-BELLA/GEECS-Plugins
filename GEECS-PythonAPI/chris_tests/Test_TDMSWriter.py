"""
Fri 7-18-2023

Made a module for a quick test of a TDMS file writer/reader.  Mostly to perform quick tests before I get the s file
integration for GEECS up and running.

@Chris
"""

import sys
import numpy as np

sys.path.insert(0, "../")
import chris_PostAnalysis.mod_ImageProcessing.QuickTDMSWriterReader as TDMSFuncs


# Generate some dummy data and a place to store

camera_name = 'U_HiResMagSpec'

magSpecDict_shot1 = {
        "Shot-Number": 1,
        "Clipped-Percentage": 0,
        "Saturation-Counts": 50,
        "Charge-On-Camera": 10,
        "Picoscope-Charge": 10,
        "Peak-Charge": 1,
        "Peak-Charge-Energy": 100,
        "Average-Energy": 100,
        "Energy-Spread": 1,
        "Average-Beam-Size": 20,
        "Projected-Beam-Size": 10,
        "Beam-Tilt": 0,
        "Beam-Intercept": 0
    }

magSpecDict_shot2 = {
    "Shot-Number": 2,
    "Clipped-Percentage": 0,
    "Saturation-Counts": 100,
    "Charge-On-Camera": 20,
    "Picoscope-Charge": 20,
    "Peak-Charge": 2,
    "Peak-Charge-Energy": 200,
    "Average-Energy": 200,
    "Energy-Spread": 2,
    "Average-Beam-Size": 80,
    "Projected-Beam-Size": 20,
    "Beam-Tilt": 0,
    "Beam-Intercept": 0
}

data_day = 1
data_month = 1
data_year = 2000
scan_number = 1

# Check if data already exists

tdmsFilepath = TDMSFuncs.CompileFilename(data_day, data_month, data_year, scan_number)
print(tdmsFilepath)

doOverwrite = False
if not TDMSFuncs.CheckExist(tdmsFilepath) or doOverwrite:
    print("-> Doesn't exist or overwriting.")
    # Write Data

    with TDMSFuncs.GetTDMSWriter(tdmsFilepath) as tdms_writer:
        TDMSFuncs.WriteAnalyzeDictionary(tdms_writer, camera_name, magSpecDict_shot1)
        TDMSFuncs.WriteAnalyzeDictionary(tdms_writer, camera_name, magSpecDict_shot2)

# Read Data

tdms_data = TDMSFuncs.ReadFullTDMSScan(tdmsFilepath)

array_name = 'Average-Beam-Size'
array = TDMSFuncs.ReturnChannelArray(tdms_data, camera_name, array_name)
print(array)
