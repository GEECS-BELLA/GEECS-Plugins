# optimization/base_evaluator.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseEvaluator(ABC):
    """
    Base class for evaluator logic. Each evaluator must implement `get_value(inputs)`
    and provide a list of required devices.
    """

    def __init__(self):
        self.device_requirements: List[str] = []

    @staticmethod
    def filter_log_entries_by_bin(
            log_entries: Dict[float, Dict[str, Any]],
            bin_num: int
    ) -> List[Dict[str, Any]]:
        return [entry for entry in log_entries.values() if entry.get('Bin #') == bin_num]

    @abstractmethod
    def get_value(self, log_entries: Dict[float, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the objective function.

        Args:
            log_entries: Dict data recorded by DataLogger

        Returns:
            Dict of measured objectives and constraints.
        """
        pass

    def __call__(self, log_entries:  Dict[float, Dict[str, Any]]) -> Dict[str, Any]:
        return self.get_value(log_entries)