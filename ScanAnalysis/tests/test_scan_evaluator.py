import unittest
from scan_analysis.mapping.scan_evaluator import check_for_analysis_match
from scan_analysis.mapping.map_Undulator import undulator_analyzers


class TestScanEvaluator(unittest.TestCase):
    def test_init(self):
        exp = 'Undulator'
        do_print = False

        if do_print:
            print("Scan 21 should just be the VISA, Rad2")
        folder = "Z:\\data\\Undulator\\Y2024\\11-Nov\\24_1105\\scans\\Scan021"
        results = check_for_analysis_match(folder, undulator_analyzers)
        self.assertEqual(len(results), 2)
        for a in results:
            self.assertIsNone(a.device_name)
            if do_print:
                print(a.analyzer_class.__name__, a.device_name)

        if do_print:
            print("Scan 7 should just be the Mag Spec and HiRes")
        folder = "Z:\\data\\Undulator\\Y2024\\11-Nov\\24_1105\\scans\\Scan007"
        results = check_for_analysis_match(folder, undulator_analyzers)
        self.assertEqual(len(results), 2)
        for a in results:
            if do_print:
                print(a.analyzer_class.__name__, a.device_name)

        if do_print:
            print("Scan 12 should just be the ALine3")
        folder = "Z:\\data\\Undulator\\Y2024\\11-Nov\\24_1105\\scans\\Scan012"
        results = check_for_analysis_match(folder, undulator_analyzers)
        self.assertEqual(len(results), 1)
        for a in results:
            self.assertEqual(a.analyzer_class.__name__, 'CameraImageAnalysis')
            self.assertEqual(a.device_name, 'UC_ALineEBeam3')
            if do_print:
                print(a.analyzer_class.__name__, a.device_name)

        if do_print:
            print("Previously using Master Control, nearly all would be flagged every time")
        folder = "Z:\\data\\Undulator\\Y2024\\06-Jun\\24_0606\\scans\\Scan003"
        results = check_for_analysis_match(folder, undulator_analyzers)
        for a in results:
            if do_print:
                print(a.analyzer_class.__name__, a.device_name)
        self.assertEqual(len(results), len(undulator_analyzers)-2)  # 2 less because no aline2 and tc_phosphor


if __name__ == "__main__":
    unittest.main()
