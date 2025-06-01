from optimization.base_evaluator import BaseEvaluator

# Devices that must be recorded for this evaluator to function
DEVICE_REQUIREMENTS = ['UC_TC_Phosphor:acq_timestamp']  # Replace with your actual device/variable names

class TestEvaluator(BaseEvaluator):
    """
    Evaluator that extracts the beam size value from logged scalar data.
    Assumes 'beam_size' is recorded in the DataLogger and represents the objective to minimize.
    """

    def __init__(self):
        super().__init__()
        self.device_requirements = DEVICE_REQUIREMENTS
        self.output_key = 'f'  # Consistent with VOCS objective key

    def get_value(self, log_entries: Dict[float, Dict[str, Any]]) -> Dict[str, Any]:
        # Assuming scan_manager sets current_bin externally
        bin_num = max(entry['Bin #'] for entry in log_entries.values())  # or passed in explicitly

        entries = self.filter_log_entries_by_bin(log_entries, bin_num)
        beam_sizes = [entry['UC_TC_Phosphor:acq_timestamp'] for entry in entries if 'UC_TC_Phosphor:acq_timestamp' in entry]

        if not beam_sizes:
            raise ValueError("No beam size data found in current bin")

        mean_beam_size = sum(beam_sizes) / len(beam_sizes)
        return {'f': mean_beam_size}