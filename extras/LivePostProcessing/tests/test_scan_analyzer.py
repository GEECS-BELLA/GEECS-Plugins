import unittest
from livepostprocessing.scan_analyzer import ScanAnalyzer

class TestScanWatch(unittest.TestCase):
    def test_init(self):
        scan_analyzer = ScanAnalyzer()

    def test_analyze_scan(self):
        sa = ScanAnalyzer(experiment_data_folder=r"C:\GEECS\Developers Version\source\GEECS-Plugins\LivePostProcessing\tests\data\experiment")
        sa.analyze_scan("23_0518", 27)

if __name__ == "__main__":
    unittest.main()
