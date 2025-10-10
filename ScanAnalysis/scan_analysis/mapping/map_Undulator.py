"""Analyzer mapping for the Undulator experiment.

This module declares a list of :class:`scan_analysis.base.ScanAnalyzerInfo` objects
(`undulator_analyzers`) used by the scan runner to decide which analyzers to
instantiate for a given scan. Each entry specifies:

- `scan_analyzer_class`: the concrete analyzer class.
- `requirements`: devices/data that must be present (supports sets and nested AND/OR dicts).
- `device_name`: optional device association used by the analyzer.
- `scan_analyzer_kwargs`: constructor kwargs forwarded to the analyzer.
"""

from geecs_data_utils import ScanData, ScanPaths

from scan_analysis.base import ScanAnalyzerInfo as Info
from scan_analysis.analyzers.common.array2D_scan_analysis import Array2DScanAnalyzer
from scan_analysis.analyzers.Undulator.camera_image_analysis import CameraImageAnalyzer

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
from scan_analysis.analyzers.Undulator.ict_plot_analysis import ICTPlotAnalysis
from scan_analysis.analyzers.Undulator.aline3_dispersion_analysis import (
    ALine3DispersionAnalysis,
)


from image_analysis.offline_analyzers.beam_analyzer import BeamAnalyzer
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

from image_analysis.config_loader import set_config_base_dir
from dataclasses import asdict

set_config_base_dir(ScanPaths.paths_config.image_analysis_configs_path)



def get_path_to_bkg_file():
    """Return a default background TSV for HASO phase processing (Undulator example)."""
    s_data = ScanData.from_date(year=2025, month=3, day=6, number=15, experiment="Undulator")
    path_to_file = s_data.paths.get_folder() / "U_HasoLift" / "average_phase.tsv"
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
            "image_analyzer": BeamAnalyzer(camera_config_name=device)
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
        scan_analyzer_class=ICTPlotAnalysis,
        requirements={"U_BCaveICT"},
        device_name="U_BCaveICT",
    ),
    Info(
        scan_analyzer_class=Rad2SpecAnalysis,
        requirements={
            "AND": ["UC_UndulatorRad2", {"OR": ["U_BCaveICT", "U_UndulatorExitICT"]}]
        },
    ),
    Info(
        scan_analyzer_class=ALine3DispersionAnalysis,
        requirements={"UC_ALineEBeam3"},
        device_name="UC_ALineEBeam3",
    ),
    Info(
        scan_analyzer_class=CameraImageAnalyzer,
        requirements={"UC_TC_Phosphor"},
        device_name="UC_TC_Phosphor",
    ),
    Info(
        scan_analyzer_class=CameraImageAnalyzer,
        requirements={"UC_DiagnosticsPhosphor"},
        device_name="UC_DiagnosticsPhosphor",
    ),
    Info(
        scan_analyzer_class=CameraImageAnalyzer,
        requirements={"UC_Phosphor1"},
        device_name="UC_Phosphor1",
    ),
    Info(
        scan_analyzer_class=CameraImageAnalyzer,
        requirements={"UC_ModeImager"},
        device_name="UC_ModeImager",
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
