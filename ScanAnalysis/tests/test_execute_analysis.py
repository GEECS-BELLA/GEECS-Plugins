import unittest

from geecs_python_api.controls.api_defs import ScanTag
from geecs_python_api.analysis.scans.scan_data import ScanData
# from scan_analysis.mapping.map_Undulator import undulator_analyzers
from scan_analysis.execute_scan_analysis import analyze_scan

from scan_analysis.base import AnalyzerInfo as Info
from scan_analysis.analyzers.Undulator.array2D_scan_analysis import Array2DScanAnalysis
from scan_analysis.analyzers.Undulator.HIMG_with_average_saving import HIMGWithAveraging

from image_analysis.offline_analyzers.basic_image_analysis import BasicImageAnalyzer
from image_analysis.offline_analyzers.HASO_himg_has_processor import HASOHimgHasProcessor
from image_analysis.offline_analyzers.density_from_phase_analysis import PhaseAnalysisConfig, PhaseDownrampProcessor

from pathlib import Path


class TestExecuteAnalysis(unittest.TestCase):
    def test_init(self):
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
        perform_analysis = False
        analyzer_info = Info(analyzer_class=Array2DScanAnalysis,
             requirements={'UC_ACaveMagCam3'},
             device_name='UC_ACaveMagCam3',
             image_analyzer_class=BasicImageAnalyzer)
        test_tag = ScanTag(year=2025, month=3, day=6, number=39, experiment='Undulator')
        test_analyzer = analyzer_info
        analyze_scan(test_tag, [analyzer_info], debug_mode=not perform_analysis)

    def test_HasoLift(self):
        analyzer_info = Info(analyzer_class=HIMGWithAveraging,
            requirements={'U_HasoLift'},
            device_name='U_HasoLift',
            image_analyzer_class=HASOHimgHasProcessor,
            file_pattern = "*_{shot_num:03d}.himg")
        test_tag = ScanTag(year=2025, month=3, day=6, number=15, experiment='Undulator')
        analyze_scan(test_tag, [analyzer_info])

    def test_DensityDownRampPhase(self):

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
            background=bkg_file_path  # Background is now a Path
        )

        analyzer_info = Info(analyzer_class=Array2DScanAnalysis,
                requirements={'U_HasoLift'},
                device_name='U_HasoLift',
                image_analyzer_class=PhaseDownrampProcessor,
                file_pattern = "*_{shot_num:03d}_postprocessed.tsv",
                image_analysis_config = config)

        test_tag = ScanTag(year=2025, month=3, day=6, number=16, experiment='Undulator')
        analyze_scan(test_tag, [analyzer_info])


if __name__ == "__main__":
    unittest.main()
