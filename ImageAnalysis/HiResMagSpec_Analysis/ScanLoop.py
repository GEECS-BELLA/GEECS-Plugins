# Imports
import sys
import numpy as np

sys.path.insert(0, "../")
import modules.QuickTDMSWriterReader as TDMSFuncs
import modules.MagSpecAnalysis as MagSpecAnalysis
import modules.DirectoryModules as DirectoryFunc

# Define necessary constants and filepaths
doOverwrite = False

data_day = 25
data_month = 7
data_year = 2023
scan_number = 25
superpath = DirectoryFunc.CompileDailyPath(data_day, data_month, data_year)
image_name = "U_HiResMagCam"

num_shots = DirectoryFunc.GetNumberOfShots(superpath, scan_number, image_name)
shot_arr = np.array(range(num_shots)) + 1

# Check if the file already exists, otherwise only loop if you want to overwrite
tdms_output_filepath = TDMSFuncs.CompileFilename(data_day, data_month, data_year, scan_number)
if not TDMSFuncs.CheckExist(tdms_output_filepath) or doOverwrite:
    print("Generating a new TDMS File...")

    # Otherwise, open the file for writing:
    with TDMSFuncs.GetTDMSWriter(tdms_output_filepath) as tdms_writer:

        # For each shot, perform the analysis and return the analyze dict
        for i in range(len(shot_arr)):
            if i % 10 == 0:
                print(i / len(shot_arr) * 100, "%")
            shot_number = shot_arr[i]
            tdms_filepath = DirectoryFunc.CompileTDMSFilepath(superpath, scan_number)
            interpSpec_filepath = DirectoryFunc.CompileFileLocation(superpath, scan_number, shot_number,
                                                                    imagename='U_HiResMagCam-interpSpec', suffix=".txt")
            raw_image = MagSpecAnalysis.LoadImage(superpath, scan_number, shot_number, image_name, doThreshold=False,
                                                  doNormalize=False)
            analyze_dict = MagSpecAnalysis.AnalyzeImage(raw_image, tdms_filepath, interpSpec_filepath, shot_number)

            # Afterwards, write the analyze dict into the tdms file
            TDMSFuncs.WriteAnalyzeDictionary(tdms_writer, image_name, analyze_dict)

    print("Finished!", tdms_output_filepath)

else:
    print("File either already exists or overwriting not permitted")
