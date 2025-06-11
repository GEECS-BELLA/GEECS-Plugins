from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional
import numpy as np

from geecs_scanner.optimization.base_evaluator import BaseEvaluator
from geecs_scanner.data_acquisition.scan_data_manager import ScanDataManager
from geecs_scanner.data_acquisition.data_logger import DataLogger
from image_analysis.utils import read_imaq_image
from image_analysis.offline_analyzers.Undulator.MagSpecStitcher import VisaEBeam



class TestEvaluator(BaseEvaluator):
    def __init__(self, device_requirements=None,
                 scan_data_manager: Optional[ScanDataManager] = None,
                 data_logger: Optional[DataLogger] = None):
        # required_keys = {
        #     'var1': 'UC_TC_Phosphor:acq_timestamp',
        #     'var2': 'UC_ALineEBeam3:acq_timestamp',
        # }

        required_keys={}

        super().__init__(
            device_requirements = device_requirements,
            required_keys = required_keys,
            scan_data_manager = scan_data_manager,
            data_logger = data_logger
        )

        dev_name = 'U_BCaveMagSpec'
        test_dict = {'camera_name': dev_name}
        self.image_analyzer = VisaEBeam(**test_dict)

        self.output_key = 'f'
        self.objective_tag: str = 'TestVal'

    def evaluate_objective_fn_per_shot(self, shot_entry: dict) -> float:
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
            path = shot_entry['image_paths']['U_BCaveMagSpec']
            result = self.image_analyzer.analyze_image_file(image_filepath=path)

            objective = result['analyzer_return_dictionary']['optimization_target']
            self.log_objective_result(shot_num=shot_entry['shot_number'],scalar_value=objective)

            return objective

        except Exception as e:
            logging.error(f"Error evaluating objective for shot {shot_entry.get('shot_number')}: {e}")
            return float("nan")

    def evaluate_all_shots(self, shot_entries: list[dict]) -> list[float]:
        """
        Evaluate the objective function for a list of shots.

        Args:
            shot_entries (list[dict]): List of shot_entry dicts as produced by _gather_shot_entries.

        Returns:
            list[float]: List of scalar objective values, one per shot.
        """
        results = []
        for shot_entry in shot_entries:
            result = self.evaluate_objective_fn_per_shot(shot_entry)
            results.append(result)

        return results

    def _get_value(self, input_data: Dict) -> Dict:

        shot_entries = self._gather_shot_entries(   shot_numbers=self.current_shot_numbers,
                                                    scalar_variables=self.required_keys,
                                                    non_scalar_variables=['U_BCaveMagSpec'])

        results = self.evaluate_all_shots(shot_entries)

        return {self.output_key: np.mean(results)}