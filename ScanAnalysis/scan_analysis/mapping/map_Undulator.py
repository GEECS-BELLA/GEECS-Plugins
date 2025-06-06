from geecs_data_utils import ScanData, ScanTag

from scan_analysis.base import AnalyzerInfo as Info
from scan_analysis.analyzers.common.array2D_scan_analysis import Array2DScanAnalyzer

from scan_analysis.analyzers.Undulator.hi_res_mag_cam_analysis import HiResMagCamAnalysis
from scan_analysis.analyzers.Undulator.mag_spec_stitcher_analysis import MagSpecStitcherAnalyzer
from scan_analysis.analyzers.Undulator.rad2_spec_analysis import Rad2SpecAnalysis
from scan_analysis.analyzers.Undulator.visa_ebeam_analysis import VisaEBeamAnalysis
from scan_analysis.analyzers.Undulator.camera_image_analysis import CameraImageAnalyzer
from scan_analysis.analyzers.Undulator.HIMG_with_average_saving import HIMGWithAveraging
from scan_analysis.analyzers.Undulator.hamaspectro_analysis import FiberSpectrometerAnalyzer
from scan_analysis.analyzers.Undulator.frog_analysis import FrogAnalyzer

from image_analysis.offline_analyzers.Undulator.ACaveMagCam3 import ACaveMagCam3ImageAnalyzer
from image_analysis.offline_analyzers.HASO_himg_has_processor import HASOHimgHasProcessor
from image_analysis.offline_analyzers.density_from_phase_analysis import PhaseAnalysisConfig, PhaseDownrampProcessor

from dataclasses import asdict

def get_path_to_bkg_file():
    st = ScanTag(2025, 3, 6, 15, experiment='Undulator')
    s_data = ScanData(tag=st)
    path_to_file = s_data.get_folder() / 'U_HasoLift' / 'average_phase.tsv'
    return path_to_file


bkg_file_path = get_path_to_bkg_file()
phase_analysis_config: PhaseAnalysisConfig = PhaseAnalysisConfig(
    pixel_scale=10.1,  # um per pixel (vertical)
    wavelength_nm=800,  # Probe laser wavelength in nm
    threshold_fraction=0.05,  # Threshold fraction for pre-processing
    roi=(10, -10, 75, -250),  # Example ROI: (x_min, x_max, y_min, y_max)
    background_path=bkg_file_path  # Background is now a Path
)

phase_analysis_config_dict = asdict(phase_analysis_config)


undulator_analyzers = [
    Info(analyzer_class=MagSpecStitcherAnalyzer,
         requirements={'U_BCaveMagSpec'},
         device_name='U_BCaveMagSpec'),
    Info(analyzer_class=VisaEBeamAnalysis,
         requirements={'OR': ['UC_VisaEBeam1', 'UC_VisaEBeam2', 'UC_VisaEBeam3', 'UC_VisaEBeam4',
                              'UC_VisaEBeam5', 'UC_VisaEBeam6', 'UC_VisaEBeam7', 'UC_VisaEBeam8']}),
    Info(analyzer_class=Rad2SpecAnalysis,
         requirements={'AND': ['UC_UndulatorRad2', {'OR': ['U_BCaveICT', 'U_UndulatorExitICT']}]}),
    Info(analyzer_class=CameraImageAnalyzer,
         requirements={'UC_ALineEbeam1'},
         device_name='UC_ALineEbeam1'),
    Info(analyzer_class=CameraImageAnalyzer,
         requirements={'UC_ALineEBeam2'},
         device_name='UC_ALineEBeam2'),
    Info(analyzer_class=CameraImageAnalyzer,
         requirements={'UC_ALineEBeam3'},
         device_name='UC_ALineEBeam3'),
    Info(analyzer_class=CameraImageAnalyzer,
         requirements={'UC_TC_Phosphor'},
         device_name='UC_TC_Phosphor'),
    Info(analyzer_class=CameraImageAnalyzer,
         requirements={'UC_DiagnosticsPhosphor'},
         device_name='UC_DiagnosticsPhosphor'),
    Info(analyzer_class=CameraImageAnalyzer,
         requirements={'UC_Phosphor1'},
         device_name='UC_Phosphor1'),
    Info(analyzer_class=CameraImageAnalyzer,
         requirements={'UC_ModeImager'},
         device_name='UC_ModeImager'),
    Info(analyzer_class=HiResMagCamAnalysis,
         requirements={'UC_HiResMagCam'}),
    Info(analyzer_class=FiberSpectrometerAnalyzer,
         requirements={'U_HamaSpectro'},
         device_name='U_HamaSpectro'),
    Info(analyzer_class=FrogAnalyzer,
         requirements={'U_FROG_Grenouille-Temporal'},
         device_name='U_FROG_Grenouille-Temporal'),
    Info(analyzer_class=Array2DScanAnalyzer,
         requirements={'UC_ACaveMagCam3'},
         device_name='UC_ACaveMagCam3',
         extra_kwargs={'image_analyzer':ACaveMagCam3ImageAnalyzer()}),
    Info(analyzer_class=HIMGWithAveraging,
         requirements={'U_HasoLift'},
         device_name='U_HasoLift',
         extra_kwargs={'image_analyzer': HASOHimgHasProcessor(), 'file_tail':".himg"}),
    Info(analyzer_class=Array2DScanAnalyzer,
         requirements={'U_HasoLift'},
         device_name='U_HasoLift',
         extra_kwargs={'image_analyzer': PhaseDownrampProcessor(**phase_analysis_config_dict),
                       'file_tail':"_postprocessed.tsv"})
]
