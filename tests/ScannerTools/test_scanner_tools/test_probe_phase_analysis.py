"""
Example scanner tools integration test with pytest.

This module demonstrates how to integrate GEECS scanner tools with
the pytest testing framework. It includes utility functions to build
paths to specific phase and background files, and tests that verify
file access and run a phase-processing pipeline.

Notes
-----
- pytest is a popular Python testing framework due to its simplicity
  and expressive test syntax.
- These tests assume that scan data is available in the expected
  directory structure (via `ScanPaths` and `ScanData`).
"""

import pytest
from pathlib import Path

from geecs_data_utils import ScanTag, ScanPaths
from image_analysis.offline_analyzers.density_from_phase_analysis import (
    PhaseAnalysisConfig,
    PhaseDownrampProcessor,
)


def get_path_to_phase_file() -> Path:
    """
    Construct the path to a phase analysis TSV file.

    Returns
    -------
    Path
        Path object pointing to the postprocessed phase TSV file
        for a specific scan.

    Notes
    -----
    The returned file corresponds to the `U_HasoLift` device
    and is tied to the hardcoded `ScanTag` (2025-03-06, number 110).
    """
    st = ScanTag(year=2025, month=3, day=6, number=110, experiment="Undulator")
    s_paths = ScanPaths(tag=st)
    path_to_file = Path(
        s_paths.get_folder() / "U_HasoLift" / "Scan110_U_HasoLift_061_postprocessed.tsv"
    )
    print(path_to_file)
    return path_to_file


def get_path_to_bkg_file() -> Path:
    """
    Construct the path to a background TSV file.

    Returns
    -------
    Path
        Path object pointing to the average background phase TSV file.

    Notes
    -----
    The returned file corresponds to the `U_HasoLift` device
    for scan 15 (2025-03-06).
    """
    st = ScanTag(year=2025, month=3, day=6, number=15, experiment="Undulator")
    s_paths = ScanPaths(tag=st)
    path_to_file = Path(s_paths.get_folder() / "U_HasoLift" / "average_phase.tsv")
    return path_to_file


def test_get_path_to_haso_file():
    """
    Verify that the expected Haso phase file exists.

    Returns
    -------
    bool
        True if the file exists, False otherwise.

    Notes
    -----
    This test uses `get_path_to_phase_file` to locate the file.
    """
    path_to_file = get_path_to_phase_file()
    return path_to_file.is_file()


def test_phase_processing():
    """
    Run phase analysis using `PhaseDownrampProcessor`.

    This test configures a phase processor with example parameters
    and runs analysis on a specified image file, using a background
    file for correction.

    Notes
    -----
    - `PhaseAnalysisConfig` is instantiated with hardcoded
      pixel scale, wavelength, threshold, ROI, and background file.
    - Processor is run in interactive mode for debugging/visualization.
    """
    phase_file_path: Path = get_path_to_phase_file()
    print(f"file to process: {phase_file_path}")
    bkg_file_path = get_path_to_bkg_file()
    print(f"bkg to process: {bkg_file_path}")

    config: PhaseAnalysisConfig = PhaseAnalysisConfig(
        pixel_scale=10.1,  # um per pixel (vertical)
        wavelength_nm=800,  # Probe laser wavelength in nm
        threshold_fraction=0.05,  # Threshold fraction for pre-processing
        roi=(10, -10, 75, -250),  # Example ROI: (x_min, x_max, y_min, y_max)
        background=bkg_file_path,  # Background file path
    )

    processor = PhaseDownrampProcessor(config)
    processor.use_interactive = True
    processor.analyze_image_file(image_filepath=phase_file_path)


if __name__ == "__main__":
    pytest.main()
