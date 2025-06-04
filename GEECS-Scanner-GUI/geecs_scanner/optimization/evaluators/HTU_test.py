from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional
import numpy as np

from geecs_scanner.optimization.base_evaluator import BaseEvaluator
from geecs_scanner.data_acquisition.scan_data_manager import ScanDataManager
from geecs_scanner.data_acquisition.data_logger import DataLogger
from image_analysis.utils import read_imaq_image



class TestEvaluator(BaseEvaluator):
    def __init__(self, device_requirements=None,
                 scan_data_manager: Optional[ScanDataManager] = None,
                 data_logger: Optional[DataLogger] = None):
        required_keys = {
            'var1': 'UC_TC_Phosphor:acq_timestamp',
            'var2': 'UC_TC_Phosphor:MeanCounts',
            'var3': 'UC_ALineEBeam3:acq_timestamp',
        }

        super().__init__(
            device_requirements = device_requirements,
            required_keys = required_keys,
            scan_data_manager = scan_data_manager,
            data_logger = data_logger
        )

        self.output_key = 'f'

    @staticmethod
    def evaluate_objective_fn_per_shot(shot_entry: dict) -> float:
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
            # Load all images (just to verify that they are accessible)
            images = {}
            for device, path in shot_entry.get('image_paths', {}).items():
                try:
                    images[device] = read_imaq_image(path)
                except Exception as e:
                    logging.warning(f"Failed to load image for {device}, shot {shot_entry['shot_number']}: {e}")

            shot_entry['images'] = images  # Store for downstream logic if needed

            # Sum all scalar values
            scalar_sum = sum(shot_entry.get('scalars', {}).values())
            return scalar_sum

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
                                                    non_scalar_variables=['UC_ALineEBeam3'])

        results = self.evaluate_all_shots(shot_entries)

        return {self.output_key: np.mean(results)}