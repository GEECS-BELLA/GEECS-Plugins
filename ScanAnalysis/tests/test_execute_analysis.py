import unittest

from geecs_data_utils import ScanData, ScanTag
from scan_analysis.execute_scan_analysis import analyze_scan
from pathlib import Path
from dataclasses import asdict

from scan_analysis.base import ScanAnalyzerInfo as Info
from scan_analysis.analyzers.common.array2D_scan_analysis import Array2DScanAnalyzer

class TestExecuteAnalysis(unittest.TestCase):
    def test_init(self):
        from scan_analysis.mapping.map_Undulator import undulator_analyzers

        perform_analysis = False

        # Given scan tag and string for analysis:
        test_tag = ScanTag(year=2024, month=11, day=5, number=5, experiment='Undulator')
        test_analyzer = undulator_analyzers[0]  # MagSpec

        # Convert string to analysis class and call analysis with scan tag
        analyze_scan(test_tag, [test_analyzer], debug_mode=not perform_analysis)
        print("Done with MagSpec")

        ######################

        test_tag = ScanTag(year=2024, month=11, day=26, number=19, experiment='Undulator')
        test_analyzer = undulator_analyzers[1]  # Visa Undulator

        # Convert string to analysis class and call analysis with scan tag
        analyze_scan(test_tag, [test_analyzer], debug_mode=not perform_analysis)
        print("Done with Visa")


    def test_ACaveMagCam3(self):
        from image_analysis.offline_analyzers.Undulator.ACaveMagCam3 import ACaveMagCam3ImageAnalyzer
        perform_analysis = True
        analyzer_info = Info(analyzer_class=Array2DScanAnalyzer,
                             requirements={'UC_ACaveMagCam3'},
                             device_name='UC_ACaveMagCam3',
                             extra_kwargs={'image_analyzer':ACaveMagCam3ImageAnalyzer()})
        test_tag = ScanTag(year=2025, month=3, day=6, number=39, experiment='Undulator')
        test_analyzer = analyzer_info
        analyze_scan(test_tag, [analyzer_info], debug_mode=not perform_analysis)

    def test_HasoLift(self):
        from image_analysis.offline_analyzers.HASO_himg_has_processor import HASOHimgHasProcessor
        from image_analysis.offline_analyzers.HASO_himg_has_processor import SlopesMask, HasoHimgHasConfig
        mask = SlopesMask(top=75, bottom=246, left=10, right=670)
        analysis_config = HasoHimgHasConfig()
        analysis_config.mask = mask
        analysis_config.wakekit_config_file_path = Path(
            'C:/Users/Loasis.loasis/Documents/GitHub/GEECS-Plugins/ImageAnalysis/image_analysis/third_party_sdks/wavekit_43/WFS_HASO4_LIFT_680_8244_gain_enabled.dat')
        analysis_config.wakekit_config_file_path: Path = Path(
            'Z:/software/control-all-loasis/HTU/Active Version/GEECS-Plugins/ImageAnalysis/image_analysis/third_party_sdks/wavekit_43/WFS_HASO4_LIFT_680_8244_gain_enabled.dat')

        analysis_config_dict = asdict(analysis_config)

        analyzer_info = Info(analyzer_class=Array2DScanAnalyzer,
                             requirements={'U_HasoLift'},
                             device_name='U_HasoLift',
                             extra_kwargs={'image_analyzer':HASOHimgHasProcessor(**analysis_config_dict),
                          'file_tail':".himg"}
                             )

        test_tag = ScanTag(year=2025, month=2, day=19, number=2, experiment='Undulator')
        analyze_scan(test_tag, [analyzer_info])

    def test_DensityDownRampPhase(self):
        from image_analysis.offline_analyzers.density_from_phase_analysis import PhaseAnalysisConfig, \
            PhaseDownrampProcessor

        def get_path_to_bkg_file():
            st = ScanTag(2025, 3, 6, 15, experiment='Undulator')
            s_data = ScanData(tag=st)
            path_to_file = s_data.get_folder() / 'U_HasoLift' / 'average_phase.tsv'

            return path_to_file

        bkg_file_path = get_path_to_bkg_file()
        config: PhaseAnalysisConfig = PhaseAnalysisConfig(
            pixel_scale=10.1,  # um per pixel (vertical)
            wavelength_nm=800,  # Probe laser wavelength in nm
            threshold_fraction=0.05,  # Threshold fraction for pre-processing
            roi=(10, -10, 75, -250),  # Example ROI: (x_min, x_max, y_min, y_max)
            background_path=bkg_file_path  # Background is now a Path
        )
        config_dict = asdict(config)
        analyzer_info = Info(analyzer_class=Array2DScanAnalyzer,
                             requirements={'U_HasoLift'},
                             device_name='U_HasoLift',
                             extra_kwargs={'image_analyzer': PhaseDownrampProcessor(**config_dict),
                            'file_tail': "_postprocessed.tsv"})

        test_tag = ScanTag(year=2025, month=3, day=6, number=16, experiment='Undulator')
        analyze_scan(test_tag, [analyzer_info])

    def test_VisaEBeamAnalyzer(self):
        from image_analysis.offline_analyzers.Undulator.VisaEBeam import VisaEBeam

        config_dict = {'camera_name':'UC_VisaEBeam1'}
        analyzer_info = Info(analyzer_class=Array2DScanAnalyzer,
                             requirements={'UC_VisaEBeam1'},
                             device_name='UC_VisaEBeam1',
                             extra_kwargs={'image_analyzer': VisaEBeam(**config_dict)})

        test_tag = ScanTag(year=2024, month=12, day=5, number=11, experiment='Undulator')
        analyze_scan(test_tag, [analyzer_info])


if __name__ == "__main__":
    unittest.main()
