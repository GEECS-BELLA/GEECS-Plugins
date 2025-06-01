from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Any, Optional
import yaml

from geecs_scanner.optimization.base_evaluator import BaseEvaluator

class TestEvaluator(BaseEvaluator):
    """
    Evaluator that extracts the beam size value from logged scalar data.
    Assumes 'beam_size' is recorded in the DataLogger and represents the objective to minimize.
    """

    def __init__(self, device_requirements=None):
        required_keys = {
            'var1': 'UC_TC_Phosphor:acq_timestamp',
            'var2': 'UC_TC_Phosphor:MeanCounts',
            'var3': 'UC_ALineEBeam3:acq_timestamp',
        }
        super().__init__(device_requirements=device_requirements, required_keys=required_keys)
        self.output_key = 'f'

    def get_value(self, log_entries: Dict[float, Dict[str, Any]]) -> Dict[str, Any]:
        # Assuming scan_manager sets current_bin externally
        bin_num = max(entry['Bin #'] for entry in log_entries.values())  # or passed in explicitly

        entries = self.filter_log_entries_by_bin(log_entries, bin_num)
        acq_times = [entry[required_keys['var1']] for entry in entries if required_keys['var1'] in entry]

        if not acq_times:
            raise ValueError("No acq_times data found in current bin")

        mean_acq_times = sum(acq_times) / len(acq_times)
        return {'f': mean_acq_times}