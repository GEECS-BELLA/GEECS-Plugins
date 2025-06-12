from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from geecs_scanner.optimization.base_evaluator import BaseEvaluator
from geecs_scanner.data_acquisition.scan_data_manager import ScanDataManager
from geecs_scanner.data_acquisition.data_logger import DataLogger

from scan_analysis.base import ScanAnalyzerInfo
from scan_analysis.execute_scan_analysis import analyze_scan, instantiate_scan_analyzer
from scan_analysis.analyzers.common.array2D_scan_analysis import Array2DScanAnalyzer
from image_analysis.utils import read_imaq_image
from image_analysis.offline_analyzers.Undulator.EBeamProfile import EBeamProfileAnalyzer
from geecs_data_utils import ScanTag, ScanData

class ALine3SizeEval(BaseEvaluator):
    def __init__(self, device_requirements=None,
                 scan_data_manager: Optional[ScanDataManager] = None,
                 data_logger: Optional[DataLogger] = None):

        required_keys={}

        super().__init__(
            device_requirements = device_requirements,
            required_keys = required_keys,
            scan_data_manager = scan_data_manager,
            data_logger = data_logger
        )

        self.dev_name = 'UC_ALineEBeam3'
        config_dict = {'camera_name': self.dev_name}
        self.scan_analyzer = Array2DScanAnalyzer(
                                            device_name=self.dev_name,
                                            image_analyzer=EBeamProfileAnalyzer(**config_dict)
                                            )

        # use live_analysis option for the scan_analyzer so that it knows not to try to load
        # data from an sFile already written to disk (which doesn't happen until the end of scan)
        self.scan_analyzer.live_analysis = True
        self.scan_analyzer.use_colon_scan_param = False  # default is file-style

        self.output_key = 'f' # string name of optimization function defined in config, don't change
        self.objective_tag: str = 'PlaceHolder' # string to append to logged objective value

    def evaluate_all_shots(self, shot_entries: list[dict]) -> list[float]:
        """
        Evaluate the objective function for a list of shots.

        Args:
            shot_entries (list[dict]): List of shot_entry dicts as produced by _gather_shot_entries.

        Returns:
            list[float]: List of scalar objective values, one per shot.
        """

        # set the 'aux' data manually to isolate the current bin to get analyzed by the ScanAnalyzer
        self.scan_analyzer.auxiliary_data = self.current_data_bin
        self.scan_analyzer.run_analysis(scan_tag=self.scan_tag)

        # grab the path to the saved average image from the scan analyzer and load
        avg_image_path =  self.scan_analyzer.saved_avg_image_paths[self.bin_number]
        avg_image = read_imaq_image(avg_image_path)

        # run standalone analyis using the image_analyzer, passing the argument that preprocessing
        # has already been done, e.g. ROI, background etc.
        result = self.scan_analyzer.image_analyzer.analyze_image(avg_image,
                                                        auxiliary_data={"preprocessed":True})

        # extract the scalar results returned by the image analyzer
        scalar_results = result['analyzer_return_dictionary']

        # define keys to extract values to use for the objective function
        x_key = f'{self.dev_name}_x_fwhm'
        y_key = f'{self.dev_name}_y_fwhm'

        objective_value = self.objective_fn( x = scalar_results[x_key],
                                        y = scalar_results[y_key])

        for shot_entry in shot_entries:
            self.log_objective_result(shot_num=shot_entry['shot_number'],scalar_value=objective_value)

        return objective_value

    @staticmethod
    def objective_fn(x, y):
        calibration = 24.4e-3 # spatial calibration in um/pixel
        return (x*calibration)**2 + (y*calibration)**2

    def _get_value(self, input_data: Dict) -> Dict:

        shot_entries = self._gather_shot_entries(   shot_numbers=self.current_shot_numbers,
                                                    scalar_variables=self.required_keys,
                                                    non_scalar_variables=['UC_ALineEBeam3'])

        result = self.evaluate_all_shots(shot_entries)

        return {self.output_key: result}