""" Example scanner tools integration test with pytest

pytest is a testing framework that is very popular due to its simplicity.

"""

import pytest
import numpy as np

from pathlib import Path

from geecs_python_api.analysis.scans.scan_data import ScanData, ScanTag
from image_analysis.analyzers.density_from_phase_analysis import PhaseAnalysisConfig, PhasePreprocessor, PhaseDataLoader, DensityAnalysis

def get_path_to_phase_file():
    st = ScanTag(2025, 2, 26, 13, experiment='Undulator')
    s_data = ScanData(tag=st)
    path_to_file = Path(s_data.get_analysis_folder() / "U_HasoLift" / "HasoAnalysis" / 'Scan013_U_HasoLift_002_postprocessed.tsv')
    return path_to_file

def test_get_path_to_haso_file():
    path_to_file = get_path_to_phase_file()
    return path_to_file.is_file()

def test_phase_processing():
    phase_file_path: Path = get_path_to_phase_file()

    config: PhaseAnalysisConfig = PhaseAnalysisConfig(
        pixel_scale=14.64,            # um per pixel (vertical)
        wavelength_nm=800,           # Probe laser wavelength in nm
        threshold_fraction=0.2,      # Threshold fraction for pre-processing
        roi=(1, -1, 1, -1),      # Example ROI: (x_min, x_max, y_min, y_max)
        background=None  # Background is now a Path

    )

    processor  = PhasePreprocessor(phase_file_path, config)
    processor.process_phase()



if __name__ == "__main__":
    pytest.main()