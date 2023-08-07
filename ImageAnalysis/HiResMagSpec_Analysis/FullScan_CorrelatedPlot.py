# Imports
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "../")
import modules.QuickTDMSWriterReader as TDMSFuncs
import modules.DirectoryModules as DirectoryFunc
import modules.CedossMathTools as MathTools

# Define constants and filepaths
data_day = 29
data_month = 6
data_year = 2023
scan_number = 23
superpath = DirectoryFunc.CompileDailyPath(data_day, data_month, data_year)
image_name = "U_HiResMagCam"

tdms_output_filepath = TDMSFuncs.CompileFilename(data_day, data_month, data_year, scan_number)

# Check if tdms output file exists
if not TDMSFuncs.CheckExist(tdms_output_filepath):
    print("Didn't find your file!")

# If so, load in the necessary arrays from the tdms file
else:
    tdms_data = TDMSFuncs.ReadFullTDMSScan(tdms_output_filepath)

    shot_number =           TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Shot-Number')
    clipped_percent =       TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Clipped-Percentage')
    saturation_counts =     TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Saturation-Counts')
    camera_charge =         TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Charge-On-Camera')
    picoscope_charge =      TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Picoscope-Charge')
    peak_energy =           TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Peak-Charge-Energy')
    peak_charge =           TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Peak-Charge')
    average_energy =        TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Average-Energy')
    energy_spread =         TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Energy-Spread-Percent')
    average_size =          TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Average-Beam-Size')
    projected_size =        TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Projected-Beam-Size')
    beam_tilt =             TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Beam-Tilt')
    beam_intercept =        TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Beam-Intercept-100MeV')

    energy_spread = energy_spread*100

    # Filter out the "bad" shots
    inputList = [[clipped_percent, '<', 0.05],
                 [saturation_counts, '<', 200],
                 [camera_charge, '>=', 2]]
    filter = MathTools.GetInequalityIndices(inputList)

    xaxis = peak_charge;     xaxis_label = 'Peak Charge (pC/MeV)'
    yaxis = average_size;      yaxis_label = 'Average Size ('+r'$\mu$'+'m)'
    caxis = picoscope_charge;   caxis_label = 'Picoscope Charge (pC)'

    plotInfo = DirectoryFunc.CompilePlotInfo(data_day, data_month, data_year, scan_number, shot=None, cameraStr=image_name)
    plt.set_cmap('plasma')
    plt.scatter(xaxis, yaxis, marker="s", c=caxis, s=4, label="All Shots")
    plt.scatter(xaxis[filter], yaxis[filter], marker="o", c=caxis[filter], label="Good Shots")
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.colorbar(label=caxis_label)
    plt.scatter(np.average(xaxis[filter]), np.average(yaxis[filter]), marker="+", s=80, c="k",
                label="Average")
    plt.xlim([min(xaxis[filter])*0.95, max(xaxis[filter])*1.05])
    plt.legend()
    plt.title(plotInfo)
    plt.show()
