import unittest 

from pathlib import Path

# for this import to work offline, a parent of this folder needs to have a folder 
# called "user data". The file dialog for Confirmation.INI can be ignored.
from geecs_python_api.analysis.scans.quad_scan_analysis import QuadAnalysis
from geecs_python_api.controls.experiment.htu import HtuExp
from geecs_python_api.analysis.scans.scan_data import ScanTag
from geecs_python_api.tools.images.filtering import FiltersParameters

class QuadAnalysisTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.htu = HtuExp(get_info=False)
        # the HtuExp singleton's base_path and is_offline attributes needs to be 
        # set for the internal call to ScanData
        self.htu.base_path = Path(__file__).parents[3] / 'data' 
        self.htu.is_offline = True

        self.camera = 'A3'
        self.emq_number = 3
        self.quad_to_screen_distance = 2.126  # [m]

        return super().setUp()

    def test_quad_analysis(self):

        quad_analysis = QuadAnalysis(ScanTag(2023, 8, 8, 22), self.emq_number, self.camera, fwhms_metric='median', quad_2_screen=self.quad_to_screen_distance)

        filters = FiltersParameters(contrast=1.333, hp_median=2, hp_threshold=3., denoise_cycles=0, gauss_filter=5.,
                                    com_threshold=0.8, bkg_image=None, box=True, ellipse=False)

        quad_analysis.analyze(None, initial_filtering=filters, ask_rerun=False, blind_loads=True,
                              store_images=False, store_scalars=False, save_plots=False, save=False)

if __name__ == "__main__":
    unittest.main()
