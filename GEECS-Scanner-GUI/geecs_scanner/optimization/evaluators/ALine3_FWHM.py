from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from geecs_scanner.optimization.base_evaluator import BaseEvaluator
from geecs_scanner.data_acquisition.scan_data_manager import ScanDataManager
from geecs_scanner.data_acquisition.data_logger import DataLogger
from image_analysis.utils import read_imaq_image
from image_analysis.offline_analyzers.Undulator.EBeamProfile import EBeamProfileAnalyzer

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

        dev_name = 'UC_ALineEBeam3'
        config_dict = {'camera_name': dev_name}
        self.image_analyzer = EBeamProfileAnalyzer(**config_dict)

        self.output_key = 'f' # string name of optimization function defined in config, don't change

        self.objective_tag: str = 'PlaceHolder' # string to append to logged objective value

    def analyze_image(self, shot_entry: dict) -> np.array:
        """
        Compute a scalar result for a single shot by summing all scalar values.
        Images are loaded to verify their availability but not used in the computation.

        Args:
            shot_entry (dict): A dictionary with keys:
                - 'shot_number' (int)
                - 'scalars' (dict[str, float])
                - 'image_paths' (dict[str, Path])

        Returns:
            float: The sum of scalar values for the shot,
                   or NaN if an error occurs.
        """
        try:
            path = shot_entry['image_paths']['UC_ALineEBeam3']
            result = self.image_analyzer.analyze_image_file(image_filepath=path)

            image = result['processed_image']
            # self.log_objective_result(shot_num=shot_entry['shot_number'],scalar_value=objective)

            return image

        except Exception as e:
            logging.error(f"Error evaluating objective for shot {shot_entry.get('shot_number')}: {e}")
            return

    def evaluate_all_shots(self, shot_entries: list[dict]) -> list[float]:
        """
        Evaluate the objective function for a list of shots.

        Args:
            shot_entries (list[dict]): List of shot_entry dicts as produced by _gather_shot_entries.

        Returns:
            list[float]: List of scalar objective values, one per shot.
        """
        images = []
        for shot_entry in shot_entries:
            image = self.evaluate_objective_fn_per_shot(shot_entry)
            if image:
                images.append(image)

        average_image = np.mean(images, axis=0)

        avg_result = self.image_analyzer.analyze_image(image=average_image, auxiliary_data={'preprocessed':True})

        scalar_results = avg_result['analyzer_return_dictionary']
        x_key = f'{self.image_analyzer.camera_name}_x_fwhm'
        y_key = f'{self.image_analyzer.camera_name}_y_fwhm'

        objective_value = self.objective_fn( x = scalar_results[x_key],
                                        y = scalar_results[y_key])

        for shot_entry in shot_entries:
            self.log_objective_result(shot_num=shot_entry['shot_number'],scalar_value=objective_value)

        return objective_value

    @staticmethod
    def objective_fn(x, y):
        return np.abs(4-x) + np.abs(4-y)

    def _get_value(self, input_data: Dict) -> Dict:

        shot_entries = self._gather_shot_entries(   shot_numbers=self.current_shot_numbers,
                                                    scalar_variables=self.required_keys,
                                                    non_scalar_variables=['UC_ALineEBeam3'])

        result = self.evaluate_all_shots(shot_entries)

        return {self.output_key: result}