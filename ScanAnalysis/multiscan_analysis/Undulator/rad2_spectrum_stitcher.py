"""
Plot Rad2 per-shot spectrum for a given scan.

This script loads and visualizes the spectrum data saved by
`Rad2SpecAnalysis` for a specific scan. It uses the scan tag
(year, month, day, number, experiment) to locate the corresponding
analysis folder, loads the saved spectrum (`scan_spectrum.npy`),
and plots the shot-by-shot lineouts as an image.

Workflow
--------
1. Construct a `ScanTag` for the target scan.
2. Initialize `Rad2SpecAnalysis` and check if the analyzer
   corresponds to visa station 9.
3. If so, locate the analysis folder and spectrum file.
4. Load the spectrum file:
   - The first row contains the energy axis.
   - Subsequent rows contain the per-shot spectra.
5. Plot the spectra as a 2D image with energy on the x-axis
   and shot index on the y-axis.

Dependencies
------------
- geecs_data_utils
- scan_analysis
- numpy
- matplotlib

Notes
-----
If the analyzer does not correspond to visa station 9, no plot
is generated. The script assumes the spectrum file exists;
missing files will raise a `FileNotFoundError`.

"""

from geecs_data_utils import ScanPaths, ScanTag
from scan_analysis.analyzers.Undulator.rad2_spec_analysis import Rad2SpecAnalysis

import numpy as np
import matplotlib.pyplot as plt

# Get the tag for the given date+scan number
tag = ScanTag(year=2025, month=3, day=6, number=90, experiment="Undulator")

# Load the analyzer to determine if it is visa9 (it is not running any new analysis)
analyzer = Rad2SpecAnalysis(
    skip_plt_show=False, debug_mode=False, background_mode=False
)
if analyzer.get_visa_station() == 9:
    # Get analysis folder and spectrum file
    analysis_folder = (
        ScanPaths.get_daily_scan_folder(tag=tag).parent
        / "analysis"
        / f"Scan{tag.number:03d}"
    )
    spectrum_file = (
        analysis_folder
        / "UC_UndulatorRad2"
        / "CameraImageAnalyzer"
        / "scan_spectrum.npy"
    )

    # Load full file, extract energy from the first row and data from the rest
    lineout_data = np.load(spectrum_file)
    energy_axis = lineout_data[0]
    scan_spectrums = lineout_data[1:]

    number_of_shots = np.shape(scan_spectrums)[0]

    plt.imshow(
        scan_spectrums, extent=([energy_axis[0], energy_axis[-1], 0, number_of_shots])
    )
    plt.show()
