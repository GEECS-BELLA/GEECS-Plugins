import unittest

from pathlib import Path

import os
print(Path(os.curdir).resolve())

from geecs_python_api.analysis.scans.scan_analysis import ScanAnalysis
from geecs_python_api.analysis.scans.scan_data import ScanData, ScanTag
from geecs_python_api.analysis.scans.scan_images import ScanImages
from geecs_python_api.analysis.devices.UC_BeamSpot import UC_BeamSpotScanImages  # TODO uncertain where this is located

class ScanImagesTestCase(unittest.TestCase):
    
    def test_UC_BeamSpotScanImages(self):
        scan_folder = ScanData.build_folder_path(ScanTag(2023, 8, 3, 37), base_directory = Path(r'Z:\data'), experiment = 'Undulator')
        scan_data = ScanData(scan_folder, ignore_experiment_name=True)
        scan_images = UC_BeamSpotScanImages(scan_data, 'UC_ALineEBeam3')
        scan_images.camera_roi = [265, 858, 397, 740]
        scan_images.analyze_image(Path(__file__).parents[3]/'data'/'Scan037_UC_ALineEBeam3_022.png')

if __name__ == "__main__":
    unittest.main()
