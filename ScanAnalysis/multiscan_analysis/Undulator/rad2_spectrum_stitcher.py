from geecs_data_utils import ScanData
from scan_analysis.analyzers.Undulator.rad2_spec_analysis import Rad2SpecAnalysis

import numpy as np
import matplotlib.pyplot as plt

# Get the tag for the given date+scan number
tag = ScanData.get_scan_tag(year=2025, month=3, day=6, number=90, experiment='Undulator')

# Load the analyzer to determine if it is visa9 (it is not running any new analysis)
analyzer = Rad2SpecAnalysis(scan_tag=tag, skip_plt_show=False, debug_mode=False, background_mode=False)
if analyzer.get_visa_station() == 9:

    # Get analysis folder and spectrum file
    analysis_folder = ScanData.get_daily_scan_folder(tag=tag).parent / 'analysis' / f"Scan{tag.number:03d}"
    spectrum_file = analysis_folder / 'UC_UndulatorRad2' / 'CameraImageAnalysis' / 'scan_spectrum.npy'

    # Load full file, extract energy from the first row and data from the rest
    lineout_data = np.load(spectrum_file)
    energy_axis = lineout_data[0]
    scan_spectrums = lineout_data[1:]

    number_of_shots = np.shape(scan_spectrums)[0]

    plt.imshow(scan_spectrums, extent=([energy_axis[0], energy_axis[-1], 0, number_of_shots]))
    plt.show()
