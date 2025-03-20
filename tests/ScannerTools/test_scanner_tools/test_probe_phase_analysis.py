""" Example scanner tools integration test with pytest

pytest is a testing framework that is very popular due to its simplicity.

"""

import pytest

from pathlib import Path

from geecs_python_api.analysis.scans.scan_data import ScanData, ScanTag
from image_analysis.offline_analyzers.density_from_phase_analysis import PhaseAnalysisConfig, PhasePreprocessor, PhaseDownrampProcessor

def get_path_to_phase_file():
    st = ScanTag(2025, 3, 6, 110, experiment='Undulator')
    s_data = ScanData(tag=st)
    path_to_file = Path(s_data.get_folder() / "U_HasoLift" / 'Scan110_U_HasoLift_061_postprocessed.tsv')
    # path_to_file = Path(s_data.get_analysis_folder() / "U_HasoLift" / "HasoAnalysis" / 'Scan016_U_HasoLift_002_postprocessed.tsv')
    print(path_to_file)
    return path_to_file

def get_path_to_bkg_file():
    st = ScanTag(2025, 3, 6, 15, experiment='Undulator')
    s_data = ScanData(tag=st)
    # path_to_file = Path(s_data.get_analysis_folder() / "U_HasoLift" / "HasoAnalysis" / 'average_phase2.tsv')
    path_to_file = Path(s_data.get_folder() / "U_HasoLift" / 'average_phase.tsv')
    return path_to_file

def test_get_path_to_haso_file():
    path_to_file = get_path_to_phase_file()
    return path_to_file.is_file()

def test_phase_processing():
    phase_file_path: Path = get_path_to_phase_file()
    print(f'file to process: {phase_file_path}')
    bkg_file_path = get_path_to_bkg_file()
    print(f'bkg to process: {bkg_file_path}')

    config: PhaseAnalysisConfig = PhaseAnalysisConfig(
        pixel_scale=10.1,            # um per pixel (vertical)
        wavelength_nm=800,           # Probe laser wavelength in nm
        threshold_fraction=0.05,      # Threshold fraction for pre-processing
        roi=(10, -10, 75, -250),      # Example ROI: (x_min, x_max, y_min, y_max)
        background=bkg_file_path  # Background is now a Path
    )

    processor  = PhaseDownrampProcessor(config)
    processor.use_interactive = True
    processor.analyze_image( file_path = phase_file_path)


if __name__ == "__main__":
    pytest.main()