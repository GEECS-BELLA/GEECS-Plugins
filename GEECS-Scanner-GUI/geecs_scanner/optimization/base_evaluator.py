# optimization/base_evaluator.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import logging
import pandas as pd
from pathlib import Path

from geecs_scanner.data_acquisition.scan_data_manager import ScanDataManager

class BaseEvaluator(ABC):
    """
    Base class for evaluator logic. Each evaluator must implement `get_value(inputs)`
    and provide device requirements in dictionary format.
    """

    def __init__(
        self,
        device_requirements: Optional[Dict[str, Any]] = None,
        required_keys: Optional[Dict[str, str]] = None,
        scan_data_manager: Optional[ScanDataManager] = None

    ):
        self.device_requirements = device_requirements or {}
        self.required_keys = required_keys or {}
        self.scan_data_manager = scan_data_manager
        self.log_entries: Optional[Dict[float, Dict[str, Any]]] = None

        self.scan_tag = self.scan_data_manager.scan_data.get_tag()

        # Validate required keys if provided
        if self.required_keys:
            try:
                self.validate_variable_keys_against_requirements(self.required_keys)
            except ValueError as e:
                logging.warning(f'Key validation failed: {e}')

    def get_device_shot_path(self, device_name: str, shot_number: int, file_extension: str = '.png') -> Path:
        """
        Helper to get full path to a device shot file using the evaluator's scan_data_manager.

        Args:
            device_name (str): Device name whose file you want.
            shot_number (int): Shot number for the file.
            file_extension (str): File extension or suffix.

        Returns:
            Path: Full path to the image or data file.
        """

        return self.scan_data_manager.scan_data.get_device_shot_path(
            device_name=device_name,
            shot_number=shot_number,
            tag=self.scan_tag,
            file_extension=file_extension
        )

    def validate_variable_keys_against_requirements(self, variable_map: dict[str, str]) -> None:
        """
        Validates that each 'device:variable' in the map exists in device_requirements.

        Args:
            variable_map: Dict mapping aliases to 'device:variable' strings.

        Raises:
            ValueError if any variable is not in device_requirements.
        """
        device_reqs = self.device_requirements.get('Devices', {})

        logging.info(f'device reqs pass to evaluator: {device_reqs}')

        declared_vars = {
            f"{device}:{var}"
            for device, req in device_reqs.items()
            for var in req.get('variable_list', [])
        }
        logging.info(f'declared variables in evaluator: {declared_vars}')
        logging.info(f'looking for matches in variable_map.values() {variable_map.values()}')
        missing = [key for key in variable_map.values() if key not in declared_vars]
        if missing:
            raise ValueError(f"The following keys are not in device_requirements: {missing}")

    @staticmethod
    def filter_log_entries_by_bin(
            log_entries: Dict[float, Dict[str, Any]],
            bin_num: int
    ) -> List[Dict[str, Any]]:
        return [entry for entry in log_entries.values() if entry.get('Bin #') == bin_num]

    @abstractmethod
    def get_value(self, input_data: Union[
            pd.DataFrame,
            List[Dict[str, float]],
            Dict[str, List[float]],
            Dict[str, float],
        ],
        ) -> pd.DataFrame:

        """
        Evaluate the objective function.

        Args:
            input_data: data representing the set values for the control variables

        Returns:
            Dict of measured objectives and constraints.
        """

        pass

    def __call__(self, input_data: Union[
            pd.DataFrame,
            List[Dict[str, float]],
            Dict[str, List[float]],
            Dict[str, float],
        ],
        ) -> pd.DataFrame:

        return self.get_value(input_data)