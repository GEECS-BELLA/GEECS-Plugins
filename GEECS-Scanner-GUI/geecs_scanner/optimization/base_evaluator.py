# optimization/base_evaluator.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import logging
import pandas as pd
from pathlib import Path

from geecs_scanner.data_acquisition.scan_data_manager import ScanDataManager
from geecs_scanner.data_acquisition.data_logger import DataLogger

class BaseEvaluator(ABC):
    """
    Base class for evaluator logic. Each evaluator must implement `get_value(inputs)`
    and provide device requirements in dictionary format.
    """

    def __init__(
        self,
        device_requirements: Optional[Dict[str, Any]] = None,
        required_keys: Optional[Dict[str, str]] = None,
        scan_data_manager: Optional[ScanDataManager] = None,
        data_logger: Optional[DataLogger] = None

    ):
        self.device_requirements = device_requirements or {}
        self.required_keys = required_keys or {}
        self.scan_data_manager = scan_data_manager
        self.data_logger = data_logger
        self.bin_number: int = 0
        self.log_entries: Optional[Dict[float, Dict[str, Any]]] = None
        self.log_df: Optional[pd.DataFrame] = None  # initialize a dataframe version of the log_entries
        self.current_data_bin: Optional[pd.DataFrame] = None
        self.current_shot_numbers: Optional[List] = None
        self.objective_tag: str = 'default'

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

    def convert_log_entries_to_df(self):
        """
        Helper to get convert the log_entries dict into df for easier manipulation
        Note: a similar conversion happens to log_entries at teh end of a scan. this
        is a bit of a redundant step. log_entries should probably eventually be converted
        to a dataframe from the beginning in data_logger

        """
        self.log_entries = self.data_logger.log_entries
        self.log_df = pd.DataFrame.from_dict(self.log_entries, orient='index')
        self.log_df = self.log_df.sort_values(by='Elapsed Time').reset_index(drop=True)
        self.log_df['Shotnumber'] = self.log_df.index + 1

    def get_shotnumbers_for_bin(self, bin_number: int) -> None:
        self.current_shot_numbers = self.log_df[self.log_df["Bin #"] == bin_number]["Shotnumber"].values

    def get_current_data(self)->None:
        """
         simple method to update the current_data_bin dataframe to isolate just the data from
         the current data. Also, extract the corresponding shot_numbers and return those
         """
        self.convert_log_entries_to_df()
        self.bin_number = self.data_logger.bin_num
        self.get_shotnumbers_for_bin(self.bin_number)
        self.current_data_bin = self.log_df[self.log_df["Bin #"] == self.bin_number].copy()


    def _gather_shot_entries(
            self,
            shot_numbers: list[int],
            scalar_variables: dict[str, str],
            non_scalar_variables: list[str]
    ) -> list[dict]:
        """
        Parse each shot into a dictionary of scalars and image file paths.

        Args:
            shot_numbers (list[int]): List of shot numbers to process.
            scalar_variables (dict[str, str]): Mapping from short names to column names in current_data_bin.
            non_scalar_variables (list[str]): Devices to associate with image paths.

        Returns:
            List[dict]: Each dict has keys:
                'shot_number', 'scalars', 'image_paths'
        """
        entries = []

        for shot in shot_numbers:
            entry = {
                'shot_number': shot,
                'scalars': {},
                'image_paths': {}
            }

            # Extract scalars based on exact match of 'Shotnumber'
            shot_row = self.current_data_bin[self.current_data_bin['Shotnumber'] == shot]
            if shot_row.empty:
                logging.warning(f"No row found for shot number {shot}")
            else:
                for short_name, full_key in scalar_variables.items():
                    if full_key in shot_row.columns:
                        value = shot_row[full_key].values[0]
                        entry['scalars'][short_name] = value
                        logging.info(f"Scalar for '{short_name}' from '{full_key}' on shot {shot}: {value}")
                    else:
                        logging.warning(f"Key '{full_key}' not found in current_data_bin columns")

            # Image paths
            for device in non_scalar_variables:
                path = self.get_device_shot_path(device_name=device, shot_number=shot, file_extension=".png")
                entry['image_paths'][device] = path

            entries.append(entry)

        return entries

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

    def get_value(self, input_data: Dict) -> Dict:

        """
        Evaluate the objective function.

        This method first retrieves the most recent data from the DataLogger
        and then delegates the actual evaluation logic to the subclass-defined
        `_get_value()` method.

        Args:
            input_data: A specification of control variable settings. Can be one of:
                - A pandas DataFrame of input rows,
                - A list of dictionaries mapping variable names to values,
                - A dictionary of variable names to lists of values,
                - A dictionary of variable names to single float values.

        Returns:
            pd.DataFrame: A DataFrame containing evaluated objective(s) and constraint(s).
        """

        self.get_current_data()
        return self._get_value(input_data = input_data)

    def log_objective_result(self, shot_num: int, scalar_value: float):
        """
        Add the computed scalar value to the log_entries under the appropriate elapsed time.

        Args:
            shot_num (int): The shot number being processed.
            scalar_value (float): The scalar result to log.
        """
        try:
            elapsed_time = self.current_data_bin.loc[
                self.current_data_bin['Shotnumber'] == shot_num, 'Elapsed Time'
            ].values[0]
        except Exception as e:
            logging.warning(f"Could not extract Elapsed Time for shot {shot_num}: {e}")
            return

        if elapsed_time:
            key = f"Objective:{self.objective_tag}"
            self.data_logger.log_entries[elapsed_time][key] = scalar_value
            logging.info(f"Logged {key} = {scalar_value} for shot {shot_num}")
        else:
            logging.warning(f"Cannot log result: no valid data_logger or elapsed_time={elapsed_time}")

    @abstractmethod
    def _get_value(self, input_data: Dict) -> Dict:
        """
        Abstract method for computing objectives and constraints.

        This method must be implemented by subclasses and contains the core logic
        for evaluating the objective function, assuming current data has already
        been updated via `get_current_data()`.

        Args:
            input_data (Dict): A specification of control variable settings. Format same as in `get_value`.

        Returns:
            Dict: containing results of objective function calculation.
        """

        pass

    def __call__(self, input_data: Dict) -> Dict:
        """
        Makes the evaluator instance callable.

        This is equivalent to calling `get_value(input_data)`.

        Args:
            input_data (Dict): A specification of control variable settings. Format same as in `get_value`.

        Returns:
            Dict: containing results of objective function calculation.
        """

        return self.get_value(input_data)