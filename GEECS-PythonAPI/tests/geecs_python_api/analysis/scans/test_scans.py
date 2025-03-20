import unittest 

from geecs_python_api.analysis.scans.scan_analysis import ScanAnalysis
from geecs_python_api.analysis.scans.scan_data import ScanData
from geecs_python_api.analysis.scans.scan_images import ScanImages


class ScanDataTestCase(unittest.TestCase):
    pass


class ScanImagesTestCase(unittest.TestCase):
    pass


class ScanAnalysisTestCase(unittest.TestCase):
    def test_init(self):
        experiment_name = 'Undulator'
        scan_tag = ScanData.get_scan_tag(23, '8', 9, 4)

        scan_folder = ScanData.get_scan_folder_path(scan_tag)
        scan_data = ScanData(scan_folder)

        files = scan_data.get_folders_and_files()

        print(files['devices'])
        print(files['files'])
        print(scan_data.get_folder())
        print(scan_data.get_analysis_folder())
        print(ScanData.get_device_shot_path(scan_tag, 'UC_Device', 5))

        scan_tag = ScanData.get_scan_tag('2024', 'nov', 19, 18)
        scan_data = ScanData(tag=scan_tag)
        files = scan_data.get_folders_and_files()

        print(files['devices'])
        print(files['files'])
        print(scan_data.get_folder())
        print(scan_data.get_analysis_folder())
        print(ScanData.get_device_shot_path(scan_tag, files['devices'][0], 5))

        next_folder = ScanData.get_next_scan_folder(experiment=experiment_name)
        print(next_folder)
        """ Code below will generate a new folder """
        # scan_data = ScanData.build_next_scan_data(experiment=experiment_name)
        # print("Next Folder: (w/ creation)")
        # print(scan_data.get_folder())
        # print(scan_data.get_analysis_folder())
        """  ### ### ### ### ### ### ### ### ###  """

        scan_data = ScanData.get_latest_scan_data(experiment=experiment_name, year=2024, month=11, day=21)
        print(scan_data.get_folder())

        scan = ScanData(folder="Z:/data/Undulator/Y2023/05-May/23_0501/scans/Scan002")
        scan_images = ScanImages(scan, "UC_VisaEBeam3")
        scan_analysis = ScanAnalysis(scan, scan_images, key_device="U_S1H")

    def test_background_check(self):
        experiment_name = 'Undulator'
        background_scan_tag = ScanData.get_scan_tag(year=25, month=3, day=6, number=24, experiment='Undulator')
        normal_scan_tag = ScanData.get_scan_tag(year=25, month=3, day=6, number=25, experiment="Undulator")
        old_scan_tag = ScanData.get_scan_tag(year=2023, month=8, day=9, number=4, experiment="Undulator")

        assert ScanData.is_background_scan(tag=background_scan_tag) == True
        assert ScanData.is_background_scan(tag=normal_scan_tag) == False
        assert ScanData.is_background_scan(tag=old_scan_tag) == False

    def test_fresh_sfile(self):
        scan_tag = ScanData.get_scan_tag(year=25, month=3, day=6, number=24, experiment='Undulator')
        scan_data = ScanData(tag=scan_tag)

        # Copy it twice to ensure repeatability
        # scan_data.copy_fresh_sfile_to_analysis()
        # scan_data.copy_fresh_sfile_to_analysis()


if __name__ == "__main__":
    unittest.main()
