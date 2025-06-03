from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Any, Optional, Union, List
import yaml
import pandas as pd

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
        self.log_entries: Optional[Dict[float, Dict[str, Any]]] = None

    def _get_value(self, input_data: Union[
            pd.DataFrame,
            List[Dict[str, float]],
            Dict[str, List[float]],
            Dict[str, float],
        ],
        ) -> pd.DataFrame:

        logging.info(f'input data passed to evaluator method: {input_data}')

        objective = self.current_data_bin[self.required_keys['var1']]

        logging.info(f'objective value for bin {self.bin_number} is {objective}')
        logging.info(f'shot numbers for bin {self.bin_number} are {self.current_shot_numbers}')

        for i in self.current_shot_numbers:
            file_path = self.get_device_shot_path(device_name = "UC_ALineEBeam3", shot_number = i, file_extension = '.png')
            try:
                read_imaq_image(file_path)
                logging.info(f'loaded image: {file_path}')
            except:
                logging.warning(f'no image found for {file_path}')

        return {self.output_key: objective}