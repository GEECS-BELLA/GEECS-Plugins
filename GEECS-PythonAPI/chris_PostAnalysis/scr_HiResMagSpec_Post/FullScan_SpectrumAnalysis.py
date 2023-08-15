# Imports
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "../../")
import chris_PostAnalysis.mod_ImageProcessing.QuickTDMSWriterReader as TDMSFuncs
import online_analysis.HTU.OnlineAnalysisModules.DirectoryModules as DirectoryFunc
import online_analysis.HTU.OnlineAnalysisModules.CedossMathTools as MathTools

# Define constants and filepaths
"""
data_day = 29
data_month = 6
data_year = 2023
scan_number = 23
superpath = DirectoryFunc.CompileDailyPath(data_day, data_month, data_year)
image_name = "U_HiResMagCam"
"""

data_day = 9
data_month = 8
data_year = 2023
scan_number = 9
superpath = DirectoryFunc.CompileDailyPath(data_day, data_month, data_year)
image_name = "UC_TestCam"

tdms_output_filepath = TDMSFuncs.CompileFilename(data_day, data_month, data_year, scan_number)

# Check if tdms output file exists
if not TDMSFuncs.CheckExist(tdms_output_filepath):
    print("Didn't find your file!")

# If so, load in the necessary arrays from the tdms file
else:
    tdms_data = TDMSFuncs.ReadFullTDMSScan(tdms_output_filepath)

    #shot_number =           TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Shot-Number')
    clipped_percent =       TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Clipped-Percentage')
    saturation_counts =     TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Saturation-Counts')
    camera_charge =         TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Charge-On-Camera')
    #picoscope_charge =      TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Picoscope-Charge')
    peak_energy =           TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Peak-Charge-Energy')
    peak_charge =           TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Peak-Charge')
    average_energy =        TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Average-Energy')
    energy_spread =         TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Energy-Spread')
    average_size =          TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Average-Beam-Size')
    projected_size =        TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Projected-Beam-Size')
    beam_tilt =             TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Beam-Tilt')
    beam_intercept =        TDMSFuncs.ReturnChannelArray(tdms_data, image_name, 'Beam-Intercept')

    #Try to calculate correlation coefficients between them all

    # Filter out the "bad" shots
    inputList = [[clipped_percent, '<', 0.05],
                 [saturation_counts, '<', 200],
                 [camera_charge, '>=', 5]]
    filter = MathTools.GetInequalityIndices(inputList)

    # Get to work plotting!
    plt.set_cmap('plasma')
    plt.scatter(peak_energy, peak_charge, marker="s", c=energy_spread, s=4, label="All Shots")
    plt.scatter(peak_energy[filter], peak_charge[filter], marker="o", c=energy_spread[filter], label="Good Shots")
    plt.xlabel("Energy at Peak Charge (MeV)")
    plt.ylabel("Peak Charge (pC/um*MeV)")
    plt.colorbar(label="Energy Spread (MeV)")
    plt.scatter(np.average(peak_energy[filter]), np.average(peak_charge[filter]), marker="+", s=80, c="k", label="Average")
    plt.xlim([min(peak_energy[filter])*0.95, max(peak_energy[filter])*1.05])
    #plt.ylim([min(peak_charge[filter])*0.9, max(peak_charge[filter])*1.1])
    plt.legend()
    plt.show()

    plt.set_cmap('plasma')
    plt.scatter(average_energy, peak_charge, marker="s", c=energy_spread, s=4, label="All Shots")
    plt.scatter(average_energy[filter], peak_charge[filter], marker="o", c=energy_spread[filter], label="Good Shots")
    plt.xlabel("Average Energy (MeV)")
    plt.ylabel("Peak Charge (pC/um*MeV)")
    plt.colorbar(label="Energy Spread (MeV)")
    plt.scatter(np.average(average_energy[filter]), np.average(peak_charge[filter]), marker="+", s=80, c="k",
                label="Average")
    plt.xlim([min(average_energy[filter])*0.95, max(average_energy[filter])*1.05])
    plt.legend()
    plt.show()
