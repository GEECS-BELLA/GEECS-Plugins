import unittest

from geecs_python_api.controls.api_defs import ScanTag
from scan_analysis.mapping.map_Undulator import undulator_analyzers
from scan_analysis.execute_scan_analysis import analyze_scan


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


if __name__ == "__main__":
    unittest.main()
