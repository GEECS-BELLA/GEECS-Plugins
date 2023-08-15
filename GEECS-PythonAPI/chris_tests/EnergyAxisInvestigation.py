# Imports
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "../")
import online_analysis.HTU.OnlineAnalysisModules.DirectoryModules as DirectoryFunc
import chris_PostAnalysis.mod_ImageProcessing as pngTools
import online_analysis.HTU.OnlineAnalysisModules.EnergyAxisLookup_HiRes as EnergyAxis
import online_analysis.HTU.OnlineAnalysisModules.HiResMagSpecAnalysis as MagSpecAnalysis

print("Not yet fully fixed!  Exiting!")
sys.exit()

# Load both the interpolated and raw images

data_day = 29#13#29
data_month =6#7#6
data_year = 2023

scan_number = 23#20#23
shot_number = 88

superpath = DirectoryFunc.CompileDailyPath(data_day, data_month, data_year)
int_name = "U_HiResMagCam-interp"
raw_name = "U_HiResMagCam"

int_image = MagSpecAnalysis.LoadImage(superpath, scan_number, shot_number, int_name, doThreshold = False, doNormalize = False)
raw_image = MagSpecAnalysis.LoadImage(superpath, scan_number, shot_number, raw_name, doThreshold = False, doNormalize = False)

# Rotate the raw image 180 degrees

raw_image = np.copy(raw_image[::-1, ::-1])

# Load the energy axis from the Spec file

interpSpec_filepath = DirectoryFunc.CompileFileLocation(superpath, scan_number, shot_number,
                                                        imagename = 'U_HiResMagCam-interpSpec', suffix=".txt")
energy_arr, spec_charge_arr = MagSpecAnalysis.ParseInterpSpec(interpSpec_filepath)

# Calculate the energy projection for both images.

int_charge_arr = MagSpecAnalysis.CalculateChargeDensityDistribution(int_image)
raw_charge_arr = MagSpecAnalysis.CalculateChargeDensityDistribution(raw_image)

# Normalize the interpolated projection to the maximum of the raw projection

int_charge_arr = int_charge_arr * np.max(raw_charge_arr)/np.max(int_charge_arr)

# Make up an energy axis for the raw image

linear_axis = np.linspace(energy_arr[0], energy_arr[-1], len(raw_charge_arr))

mm_axis = np.linspace(0,len(raw_charge_arr),len(raw_charge_arr))*0.043
xbeam = mm_axis[int(len(mm_axis)/2)]#36 #10.5 #24
print(xbeam)
#Ebend = energy_arr[int(len(energy_arr)/2)]# 106 #94 #100

E0 = energy_arr[0]
Em = energy_arr[-1]
px = mm_axis[-1]
Ebend = (E0 * Em * px)/(E0*xbeam - xbeam*Em + Em*px)
print(Ebend)

dnom = xbeam * energy_arr[0] / (energy_arr[0] - Ebend)
print(dnom)

estimated_axis = dnom * Ebend / (mm_axis + (dnom - xbeam))

#estimated_axis = estimated_axis + 1.15
#positive 0.001 and negative 0.01
estimated_axis = estimated_axis + (0.03)*np.power(estimated_axis-Ebend,1) + (-0.007)*np.power(estimated_axis-Ebend,2) + 1.17
#estimated_axis = estimated_axis - 0.00002*np.power(estimated_axis-Ebend,4)
#print(1.15/(E0-Ebend)**2)
# Rotate here if not earlier

#raw_charge_arr = np.flip(raw_charge_arr)

pixel_axis = np.arange(0, len(raw_charge_arr))
lookup_axis = EnergyAxis.return_default_energy_axis(pixel_axis)


# Plot and compare

plotInfo = DirectoryFunc.CompilePlotInfo(data_day, data_month, data_year, scan_number, shot_number, "Projection Comparison")

plt.plot(energy_arr, int_charge_arr, c = 'g', label="Interpolated Projection")
plt.plot(linear_axis, raw_charge_arr, c = 'b', ls = 'dashed', label="Raw Projection")
plt.plot(estimated_axis, raw_charge_arr, c = 'r', ls = 'dotted', label="Estimated Projection")
plt.plot(lookup_axis, raw_charge_arr, c='orange', ls='dotted', label="Lookup")
plt.legend(title="Want Green and Red to Match")
plt.title(plotInfo)
plt.xlabel("Energy (MeV)")
plt.ylabel("Un-normalized Projection")
plt.show()

# Find a way to get the axes to match.

#