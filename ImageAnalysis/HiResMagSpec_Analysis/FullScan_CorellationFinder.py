"""

"""

# Imports
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "../")
import modules.QuickTDMSWriterReader as TDMSFuncs
import modules.DirectoryModules as DirectoryFunc
import modules.CedossMathTools as MathTools

# Define constants and filepaths
data_day = 25
data_month = 7
data_year = 2023
scan_number = 25
#superpath = DirectoryFunc.CompileDailyPath(data_day, data_month, data_year)
image_name = "U_HiResMagCam"

tdms_output_filepath = TDMSFuncs.CompileFilename(data_day, data_month, data_year, scan_number)

# Check if tdms output file exists
if not TDMSFuncs.CheckExist(tdms_output_filepath):
    print("Didn't find your file!")

# If so, load in the necessary arrays from the tdms file
else:
    tdms_data = TDMSFuncs.ReadFullTDMSScan(tdms_output_filepath)

    # Now this is where we do some coding.  Make a list of all of the things we want to include in the matrix, then
    #  load a massive matrix of all the data.  Additionally load an array with the names so we know what is what

    channel_array = np.array([
        "Shot-Number", "Clipped-Percentage", "Saturation-Counts", "Charge-On-Camera", "Picoscope-Charge", "Peak-Charge",
        "Peak-Charge-Energy", "Average-Energy", "Energy-Spread", "Average-Beam-Size", "Projected-Beam-Size",
        "Beam-Tilt", "Beam-Intercept"])

    num_data = len(channel_array)
    num_shot = len(TDMSFuncs.ReturnChannelArray(tdms_data, image_name, channel_array[0]))
    data_matrix = np.zeros((num_data, num_shot))
    for i in range(num_data):
        data_matrix[i] = TDMSFuncs.ReturnChannelArray(tdms_data, image_name, channel_array[i])

    # Filter out the bad shots
    clipped_percent = data_matrix[np.where(channel_array == "Clipped-Percentage")[0][0]]
    saturation_counts = data_matrix[np.where(channel_array == "Saturation-Counts")[0][0]]
    camera_charge = data_matrix[np.where(channel_array == "Charge-On-Camera")[0][0]]

    inputList = [[clipped_percent, '<', 0.05],
                 [saturation_counts, '<', 200],
                 [camera_charge, '>=', 5]]
    filter = MathTools.GetInequalityIndices(inputList)

    # We want to initialize a 2D array and have a tournament group stage type 2x for loop
    results_matrix = np.zeros((num_data, num_data))

    for m in range(num_data):
        for n in range(m):
            #print(m,n)
            data_x = data_matrix[m, filter]
            data_y = data_matrix[n, filter]
            #print(np.corrcoef(data_x, data_y))
            results_matrix[m, n] = np.corrcoef(data_x, data_y)[0,1]

    #print(results_matrix)
    plotInfo = DirectoryFunc.CompilePlotInfo(data_day, data_month, data_year, scan_number, shot=None, cameraStr=image_name)

    plt.set_cmap("PiYG")
    plt.imshow(results_matrix)
    plt.clim((-1,1))
    plt.colorbar(label="Correlation Coefficient")
    plt.grid()
    for i in range(len(channel_array)):
        plt.text(len(channel_array)-6.4, i*0.6, str(i) + ': ' + channel_array[i])
    plt.title(plotInfo)
    plt.xlabel("Data Type")
    plt.ylabel("Data Type")
    plt.show()