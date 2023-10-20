import unittest
from pathlib import Path
from configparser import ConfigParser
from shutil import rmtree
from typing import NamedTuple
import logging

import pandas as pd
import numpy as np

from imageio.v3 import imwrite

from livepostprocessing.scan_analyzer import ScanAnalyzer
from livepostprocessing.orm import Scan

from image_analysis import ureg, Q_, Quantity
from image_analysis.algorithms.grenouille import GrenouilleRetrieval
from image_analysis.algorithms.qwlsi import QWLSIImageAnalyzer

experiment_data_folder = Path(__file__).parent / 'data' / 'experiment'
run_id = "00_0101"
scan_number = 1

class TestScanAnalyzer(unittest.TestCase):
    def _create_test_data(self):
    
        run_id = "00_0101"
        scan_number = 1
        # create scan directory 
        scan = Scan(run_id, scan_number, experiment_data_folder=experiment_data_folder)
        scan.path.mkdir(parents=True, exist_ok=True)
        scan.analysis_path.mkdir(parents=True, exist_ok=True)

        def _write_one_shot_scandata_file():
            scan_data = pd.DataFrame([
                pd.Series({'Shotnumber': 1, 'Bin #': 1, 'scan': scan_number}, name=(run_id, scan_number, 1))
            ]).rename_axis(index=['run_id', 'scan', 'shot'])
            scan_data.to_csv(scan.path / f"ScanDataScan{scan_number:03d}.txt", sep='\t', index=False)
            scan_data.to_csv(scan.analysis_path.parent / f"s{scan_number:d}.txt", sep='\t', index=False)

        def _write_scaninfo_file():
            cp = ConfigParser()
            cp.optionxform = str
            cp.add_section("Scan Info")
            cp["Scan Info"].update(
                {"Scan No": f"\"{scan_number}\"",
                "ScanStartInfo": "\"\"",
                "Scan Parameter": "\"Shotnumber\"",
                "ScanEndInfo": "\"\"",
                }
            )

            with (scan.path / f"ScanInfoScan{scan_number:03d}.ini").open('w') as f:
                cp.write(f)

        def _create_U_PhasicsFileCopy_image():
            qia = QWLSIImageAnalyzer(reconstruction_method='velghe')
            
            qia.CAMERA_RESOLUTION = Q_(4.0, 'um')
            qia.GRATING_CAMERA_DISTANCE = Q_(1.0, 'mm')

            x = np.arange(128) * qia.CAMERA_RESOLUTION
            y = np.arange(96) * qia.CAMERA_RESOLUTION
            x0 = 50 * qia.CAMERA_RESOLUTION
            y0 = 30 * qia.CAMERA_RESOLUTION
            x_sig = 20 * qia.CAMERA_RESOLUTION
            y_sig = 10 * qia.CAMERA_RESOLUTION
            wavefront_ampl = Q_(3000, 'nm')

            X, Y = np.meshgrid(x, y)
            wavefront = wavefront_ampl * np.exp(-np.square(X - x0) / (2 * x_sig**2) - np.square(Y - y0) / (2 * y_sig**2))

            grad_wavefront = NamedTuple('gradient_2d', x=np.ndarray, y=np.ndarray)(
                x = -2*(X - x0) / (2 * x_sig**2) * wavefront, 
                y = -2*(Y - y0) / (2 * y_sig**2) * wavefront, 
            )

            interferogram = sum([
                np.cos(2*np.pi * (dsc.nu_x * X + dsc.nu_y * Y)
                        - 2*np.pi * qia.GRATING_CAMERA_DISTANCE * (dsc.nu_x * grad_wavefront.x + dsc.nu_y * grad_wavefront.y)
                    )
                for dsc in qia.diffraction_spot_centers
            ]).m

            U_PhasicsFileCopy_folder = scan.path / "U_PhasicsFileCopy"
            U_PhasicsFileCopy_folder.mkdir()
            imwrite(U_PhasicsFileCopy_folder / f"Scan{scan_number:03d}_U_PhasicsFileCopy_001.tif",
                    interferogram
            ) 


        def _create_U_FROG_Grenouille_image():
            gr = GrenouilleRetrieval(calculate_next_E_method='integration')

            # ## Calculate sample grenouille trace from example E field pulse
            pulse_time_step = Q_(5, 'fs')
            pulse_t = (np.arange(127) - 64) * pulse_time_step
            pulse_E = np.exp(-1/2 * np.square(pulse_t) / Q_(30, 'fs')**2) * np.exp(1j * pulse_t * 2*np.pi / Q_(200, 'fs'))
            pulse_E = pulse_E.m_as('')

            grenouille_trace_center_wavelength = Q_(400, 'nm')
            grenouille_trace_wavelength_step = Q_(0.7, 'nm')
            grenouille_trace_time_delay_step = Q_(2.1, 'fs')
            grenouille_trace_shape = (80, 81)

            grenouille_trace = gr.simulate_grenouille_trace(pulse_E, pulse_time_step, grenouille_trace_shape, 
                                grenouille_trace_center_wavelength=grenouille_trace_center_wavelength, 
                                grenouille_trace_wavelength_step=grenouille_trace_wavelength_step, 
                                grenouille_trace_time_delay_step=grenouille_trace_time_delay_step,
                            )

            U_FROG_Grenouille_folder = scan.path / "U_FROG_Grenouille"
            U_FROG_Grenouille_folder.mkdir()
            
            # scale to 0..1
            grenouille_trace = (grenouille_trace - grenouille_trace.min()) / (grenouille_trace.max() - grenouille_trace.min())
            # scale to 0..255
            grenouille_trace = (255 * grenouille_trace).astype(int)

            imwrite(U_FROG_Grenouille_folder / f"Scan{scan_number:03d}_U_FROG_Grenouille_001.png",
                    np.transpose(grenouille_trace)
            )
        
        _write_one_shot_scandata_file()
        _write_scaninfo_file()
        _create_U_PhasicsFileCopy_image()
        _create_U_FROG_Grenouille_image()

    def setUp(self):
        self._create_test_data()

    def _analyze_scan_and_assert_files_created(self):
        with self.assertLogs('scan_analyzer', logging.INFO):
            self.sa.analyze_scan(run_id, scan_number)
        self.assertTrue((self.sa.scan.analysis_path / "U_PhasicsFileCopy-wavefront_nm" / f"Scan{scan_number:03d}_U_PhasicsFileCopy-wavefront_nm_001.tif").exists())
        self.assertTrue((self.sa.scan.analysis_path / "U_FROG_Grenouille-pulse_E_field_AU"/ f"Scan{scan_number:03d}_U_FROG_Grenouille-pulse_E_field_AU_001.dat").exists())
        
        updated_s_file_data = pd.read_csv(self.sa.scan.analysis_path.parent / f"s{scan_number:d}.txt", sep='\t')
        self.assertIn("U_PhasicsFileCopy peak_density_cm-3", updated_s_file_data.columns)
        self.assertIn("U_FROG_Grenouille fwhm_fs", updated_s_file_data.columns)

    def test_analyze_scan(self):
        self.sa = ScanAnalyzer(experiment_data_folder=experiment_data_folder)

        self._analyze_scan_and_assert_files_created()

    def test_analyze_scan_with_lambda(self):
        self.sa = ScanAnalyzer(experiment_data_folder=experiment_data_folder)
        self.sa.image_analyzers['U_FROG_Grenouille'] = U_FROG_GrenouilleAWSLambdaImageAnalyzer()

        self._analyze_scan_and_assert_files_created()

    def tearDown(self) -> None:
        rmtree(experiment_data_folder)

if __name__ == "__main__":
    unittest.main()
