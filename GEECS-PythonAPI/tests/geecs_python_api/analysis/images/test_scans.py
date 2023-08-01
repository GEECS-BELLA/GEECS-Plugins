from unittest import TestCase

from geecs_python_api.analysis.images.scans.scan_analysis import ScanAnalysis
from geecs_python_api.analysis.images.scans.scan_data import ScanData
from geecs_python_api.analysis.images.scans.scan_images import ScanImages

class ScanDataTestCase(TestCase):
    pass

class ScanImagesTestCase(TestCase):
    pass
        
class ScanAnalysisTestCase(TestCase):
    def test_init(self):
        scan = ScanData(folder="Z:/data/Undulator/Y2023/05-May/23_0501/scans/Scan002")
        scan_images = ScanImages(scan, "UC_VisaEBeam3")
        scan_analysis = ScanAnalysis(scan, scan_images)

