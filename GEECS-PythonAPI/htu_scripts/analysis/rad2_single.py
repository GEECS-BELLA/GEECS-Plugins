"""
Thursday 9/7/2023

Calls a single instance of the UndulatorExitCam's analysis and plots the returned values.

@Chris
"""

from pathlib import Path
import matplotlib.pyplot as plt
from image_analysis.utils import read_imaq_png_image

from geecs_python_api.controls.data_acquisition.data_acquisition import DataInterface
from geecs_python_api.controls.data_acquisition.scan_analysis import VisaEBeamAnalysis, CameraImageAnalysis

data_day = '19'  # 28
data_month = 'Nov'#11  # 3
data_year = '2024'
scan_number = 17  # 122
image_name = "UC_UndulatorRad2"

data_interface = DataInterface()
data_interface.year = data_year
data_interface.month = data_month
data_interface.day = data_day
(raw_data_path, analysis_data_path) = data_interface.create_data_path(scan_number)

scan_directory = raw_data_path / f"Scan{scan_number:03d}"
analysis_class = CameraImageAnalysis(scan_directory, image_name)

shot_number = 30
print("Beam Charge [pc]: ", analysis_class.auxiliary_data["U_BCaveICT Python Results.ChA Alias:U_BCaveICT Charge pC"][shot_number-1])

fullpath = scan_directory / image_name / f"Scan{scan_number:03d}_{image_name}_{shot_number:03d}.png"
raw_image = read_imaq_png_image(Path(fullpath))*1.0

"""
start = time.perf_counter()

current_directory = Path(__file__)
config_file = 'undulatorrad2cam_may2_settings.ini'
config_filepath = current_directory.parents[1] / "ini_analyzerconfigs" / config_file
analyzer = LightSpec().apply_config(config_filepath)

#analyzer = labview_adapters.analyzer_from_device_type("UC_UndulatorRad2")


analyzer_returns = analyzer.analyze_image(raw_image)

analyzer_dict = analyzer_returns['analyzer_return_dictionary']
print("Elapsed Time: ", time.perf_counter() - start, "s")
print(analyzer_dict)

analyzer_lineouts = analyzer_returns['analyzer_return_lineouts']
wavelength = analyzer_lineouts[0]
processed_image = analyzer_returns["processed_image_uint16"]
"""

plt.figure(figsize=(20,10))
plt.title("")
plt.xlabel("Calibrated Wavelength (nm)")
plt.ylabel("Height (arb. units)")
plt.imshow(raw_image,
           # extent=[wavelength[0],wavelength[-1],-1,1],
           aspect='auto', vmin=0, vmax=250
           )
plt.show()

"""
plt.plot(analyzer_lineouts[0], analyzer_lineouts[1])
plt.xlabel("Calibrated Wavelength (nm)")
plt.ylabel("Photon Counts (arb. units)")
plt.show()
"""
