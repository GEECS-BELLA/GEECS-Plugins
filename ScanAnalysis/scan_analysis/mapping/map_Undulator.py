"""Analyzer mapping for the Undulator experiment.

This module declares a list of :class:`scan_analysis.base.ScanAnalyzerInfo` objects
(`undulator_analyzers`) used by the scan runner to decide which analyzers to
instantiate for a given scan. Each entry specifies:

- `scan_analyzer_class`: the concrete analyzer class.
- `requirements`: devices/data that must be present (supports sets and nested AND/OR dicts).
- `device_name`: optional device association used by the analyzer.
- `scan_analyzer_kwargs`: constructor kwargs forwarded to the analyzer.

Examples
--------
>>> from geecs_data_utils import ScanData
>>> from scan_analysis.mapping.map_Undulator import undulator_analyzers
>>> tag = ScanData.get_scan_tag(2025, 4, 3, number=2, experiment='Undulator')
>>> # Pass `undulator_analyzers` into your orchestrator (see execute_analysis.analyze_scan).
"""

from geecs_data_utils import ScanData, ScanTag

from scan_analysis.base import ScanAnalyzerInfo as Info
from scan_analysis.analyzers.common.array2D_scan_analysis import Array2DScanAnalyzer

from scan_analysis.analyzers.Undulator.hi_res_mag_cam_analysis import (
    HiResMagCamAnalysis,
)
from scan_analysis.analyzers.Undulator.mag_spec_stitcher_analysis import (
    MagSpecStitcherAnalyzer,
)
from scan_analysis.analyzers.Undulator.rad2_spec_analysis import Rad2SpecAnalysis
from scan_analysis.analyzers.Undulator.HIMG_with_average_saving import HIMGWithAveraging
from scan_analysis.analyzers.Undulator.hamaspectro_analysis import (
    FiberSpectrometerAnalyzer,
)
from scan_analysis.analyzers.Undulator.frog_analysis import FrogAnalyzer

from image_analysis.offline_analyzers.Undulator.EBeamProfile import EBeamProfileAnalyzer
from image_analysis.offline_analyzers.Undulator.ACaveMagCam3 import (
    ACaveMagCam3ImageAnalyzer,
)
from image_analysis.offline_analyzers.HASO_himg_has_processor import (
    HASOHimgHasProcessor,
)
from image_analysis.offline_analyzers.density_from_phase_analysis import (
    PhaseAnalysisConfig,
    PhaseDownrampProcessor,
)

from dataclasses import asdict


def get_path_to_bkg_file():
    """Return a default background TSV for HASO phase processing (Undulator example)."""
    st = ScanTag(2025, 3, 6, 15, experiment="Undulator")
    s_data = ScanData(tag=st)
    path_to_file = s_data.get_folder() / "U_HasoLift" / "average_phase.tsv"
    return path_to_file


bkg_file_path = get_path_to_bkg_file()
phase_analysis_config: PhaseAnalysisConfig = PhaseAnalysisConfig(
    pixel_scale=10.1,  # um per pixel (vertical)
    wavelength_nm=800,  # Probe laser wavelength in nm
    threshold_fraction=0.05,  # Threshold fraction for pre-processing
    roi=(10, -10, 75, -250),  # Example ROI: (x_min, x_max, y_min, y_max)
    background_path=bkg_file_path,  # Background is now a Path
)

phase_analysis_config_dict = asdict(phase_analysis_config)

e_beam_profile_camera_devices = [
    "UC_ALineEbeam1",
    "UC_ALineEBeam2",
    "UC_ALineEBeam3",
    "UC_VisaEBeam1",
    "UC_VisaEBeam2",
    "UC_VisaEBeam3",
    "UC_VisaEBeam4",
    "UC_VisaEBeam5",
    "UC_VisaEBeam6",
    "UC_VisaEBeam7",
    "UC_VisaEBeam8",
]

e_beam_profile_camera_analyzers = [
    Info(
        scan_analyzer_class=Array2DScanAnalyzer,
        requirements={device},
        device_name=device,
        scan_analyzer_kwargs={
            "image_analyzer": EBeamProfileAnalyzer(camera_name=device)
        },
    )
    for device in e_beam_profile_camera_devices
]

undulator_analyzers = [
    *e_beam_profile_camera_analyzers,
    Info(
        scan_analyzer_class=MagSpecStitcherAnalyzer,
        requirements={"U_BCaveMagSpec"},
        device_name="U_BCaveMagSpec",
    ),
    Info(
        scan_analyzer_class=Rad2SpecAnalysis,
        requirements={
            "AND": ["UC_UndulatorRad2", {"OR": ["U_BCaveICT", "U_UndulatorExitICT"]}]
        },
    ),
    Info(scan_analyzer_class=HiResMagCamAnalysis, requirements={"UC_HiResMagCam"}),
    Info(
        scan_analyzer_class=FiberSpectrometerAnalyzer,
        requirements={"U_HamaSpectro"},
        device_name="U_HamaSpectro",
    ),
    Info(
        scan_analyzer_class=FrogAnalyzer,
        requirements={"U_FROG_Grenouille-Temporal"},
        device_name="U_FROG_Grenouille-Temporal",
    ),
    Info(
        scan_analyzer_class=HIMGWithAveraging,
        requirements={"U_HasoLift"},
        device_name="U_HasoLift",
        scan_analyzer_kwargs={
            "image_analyzer": HASOHimgHasProcessor(),
            "file_tail": ".himg",
        },
    ),
    Info(
        scan_analyzer_class=Array2DScanAnalyzer,
        requirements={"UC_ACaveMagCam3"},
        device_name="UC_ACaveMagCam3",
        scan_analyzer_kwargs={"image_analyzer": ACaveMagCam3ImageAnalyzer()},
    ),
    Info(
        scan_analyzer_class=Array2DScanAnalyzer,
        requirements={"U_HasoLift"},
        device_name="U_HasoLift",
        scan_analyzer_kwargs={
            "image_analyzer": PhaseDownrampProcessor(**phase_analysis_config_dict),
            "file_tail": "_postprocessed.tsv",
        },
    ),
]
