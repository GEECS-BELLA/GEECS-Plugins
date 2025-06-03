from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Any, Optional, Union, List
import yaml
import pandas as pd
from calibrations.Undulator.calibration_scripts.magspec_charge_calibration.single_scan_charge_calibration_acavemagcam3 import \
    shot_number

from geecs_scanner.optimization.base_evaluator import BaseEvaluator
from geecs_scanner.data_acquisition.scan_data_manager import ScanDataManager

class TestEvaluator(BaseEvaluator):
    def __init__(self, device_requirements=None, scan_data_manager=None):
        required_keys = {
            'var1': 'UC_TC_Phosphor:acq_timestamp',
            'var2': 'UC_TC_Phosphor:MeanCounts',
            'var3': 'UC_ALineEBeam3:acq_timestamp',
        }
        super().__init__(
            device_requirements=device_requirements,
            required_keys=required_keys,
            scan_data_manager=scan_data_manager
        )

        self.output_key = 'f'
        self.log_entries: Optional[Dict[float, Dict[str, Any]]] = None

    def get_value(self, input_data: Union[
            pd.DataFrame,
            List[Dict[str, float]],
            Dict[str, List[float]],
            Dict[str, float],
        ],
        ) -> pd.DataFrame:

        logging.info(f'input data passed to evaluator method: {input_data}')
        if self.log_entries is None:
            raise RuntimeError("log_entries not set on evaluator")

        # Assuming scan_manager sets current_bin externally
        bin_num = max(entry['Bin #'] for entry in self.log_entries.values())  # or passed in explicitly

        entries = self.filter_log_entries_by_bin(self.log_entries, bin_num)
        acq_times = [entry[self.required_keys['var1']] for entry in entries if self.required_keys['var1'] in entry]

        self.get_device_shot_path(device_name = "test", shot_number = 1, file_extension = '.png')

        if not acq_times:
            raise ValueError("No acq_times data found in current bin")

        mean_acq_times = sum(acq_times) / len(acq_times)
        return {self.output_key: mean_acq_times}