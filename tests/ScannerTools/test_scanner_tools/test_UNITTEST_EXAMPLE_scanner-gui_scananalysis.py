""" Example scanner tools integration test with unittest 

Unittest is Python's built-in testing framework. It has more features and ways 
to organize tests than pytest, but it is more verbose. 

"""

import unittest

from pathlib import Path

from geecs_scanner.data_acquisition.scan_manager import ScanManager
from scan_analysis.execute_scan_analysis import analyze_scan

class TestScannerGuiScanAnalysis(unittest.TestCase):

    def setUp(self):
        # Setup code to run before each test
        pass

    def tearDown(self):
        # Cleanup code to run after each test
        pass

    def test_example(self):
        # Example test case
        self.assertEqual(1 + 1, 2)

    def test_pass_scan_manager_config_to_scan_analysis(self):
        # Test case to check if scan manager config is passed to scan analysis
        scan_manager = ScanManager()
        analyze_scan(scan_manager.scan_tag, scan_manager.analyzer_list, debug_mode=False)        

        # use unittest's built-in assertions to check that the output is as expected
        self.assertEquals(1+1, 2)
        self.assertTrue(Path("/path/to/analysis/output").exists())

if __name__ == '__main__':
    unittest.main()