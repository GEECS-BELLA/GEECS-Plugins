"""
Base evaluator module for optimization algorithms.

This module provides the abstract base class for all optimization evaluators in the GEECS
scanner system. Evaluators are responsible for computing objective functions and constraints
based on experimental data, enabling automated optimization of laser and accelerator parameters.

The module supports both scalar and non-scalar data evaluation, with built-in data management
capabilities for handling shot-by-shot measurements and image analysis.

Classes
-------
BaseEvaluator
    Abstract base class for all optimization evaluators.

Notes
-----
All evaluators must implement the abstract `_get_value` method and should define
appropriate device requirements for data acquisition.
"""

# optimization/base_evaluator.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Dict, Any, Optional

if TYPE_CHECKING:
    from geecs_scanner.data_acquisition.scan_data_manager import ScanDataManager
    from geecs_scanner.data_acquisition.data_logger import DataLogger

import logging
import pandas as pd
from pathlib import Path

# Module-level logger
logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """
    Abstract base class for optimization evaluators.

    This class provides the foundation for all optimization evaluators in the GEECS
    scanner system. Evaluators compute objective functions and constraints based on
    experimental data, enabling automated optimization of system parameters.

    The base evaluator handles data management, shot organization, and provides
    utilities for accessing both scalar measurements and image data from devices.

    Parameters
    ----------
    device_requirements : dict, optional
        Dictionary specifying required devices and their variables for evaluation.
        Format: {"Devices": {"device_name": {"variable_list": ["var1", "var2"]}}}
    required_keys : dict, optional
        Mapping of short names to full device:variable keys for validation.
    scan_data_manager : ScanDataManager, optional
        Manager for accessing scan data and file paths.
    data_logger : DataLogger, optional
        Logger instance for accessing shot data and bin information.

    Attributes
    ----------
    bin_number : int
        Current data bin number being processed.
    log_entries : dict, optional
        Dictionary of logged data entries indexed by elapsed time.
    log_df : pd.DataFrame, optional
        DataFrame version of log_entries for easier manipulation.
    current_data_bin : pd.DataFrame, optional
        Data for the current bin being evaluated.
    current_shot_numbers : list, optional
        Shot numbers in the current data bin.
    objective_tag : str
        Tag for identifying objective values in logs (default: 'default').
    scan_tag : str
        Tag identifying the current scan.

    Methods
    -------
    get_value(input_data)
        Main evaluation method that updates data and calls _get_value.
    _get_value(input_data)
        Abstract method for subclass-specific evaluation logic.
    get_device_shot_path(device_name, shot_number, file_extension)
        Get full path to device data files.
    validate_variable_keys_against_requirements(variable_map)
        Validate that required variables are available.
    log_objective_result(shot_num, scalar_value)
        Log computed objective values.

    Notes
    -----
    - Subclasses must implement the abstract `_get_value` method
    - Device requirements should be specified to ensure proper data availability
    - The evaluator automatically handles data bin management and shot organization
    """

    def __init__(
        self,
        device_requirements: Optional[Dict[str, Any]] = None,
        required_keys: Optional[Dict[str, str]] = None,
        scan_data_manager: Optional[ScanDataManager] = None,
        data_logger: Optional[DataLogger] = None,
    ):
        self.device_requirements = device_requirements or {}
        self.required_keys = required_keys or {}
        self.scan_data_manager = scan_data_manager
        self.data_logger = data_logger
        self.bin_number: int = 0
        self.log_entries: Optional[Dict[float, Dict[str, Any]]] = None
        self.log_df: Optional[pd.DataFrame] = (
            None  # initialize a dataframe version of the log_entries
        )
        self.current_data_bin: Optional[pd.DataFrame] = None
        self.current_shot_numbers: Optional[List] = None
        self.objective_tag: str = "default"

        self.scan_tag = self.scan_data_manager.scan_paths.get_tag()

        # Validate required keys if provided
        if self.required_keys:
            try:
                self.validate_variable_keys_against_requirements(self.required_keys)
            except ValueError as e:
                logger.warning("Key validation failed: %s", e)

    def get_device_shot_path(
        self, device_name: str, shot_number: int, file_extension: str = ".png"
    ) -> Path:
        """
        Get full path to a device shot file.

        Constructs the complete file path for a specific device's data file
        from a particular shot using the scan data manager.

        Parameters
        ----------
        device_name : str
            Name of the device whose file path is requested.
        shot_number : int
            Shot number for the specific data file.
        file_extension : str, default='.png'
            File extension or suffix for the data file.

        Returns
        -------
        Path
            Complete path to the device's data file for the specified shot.

        """
        return self.scan_data_manager.scan_paths.get_device_shot_path(
            device_name=device_name,
            shot_number=shot_number,
            tag=self.scan_tag,
            file_extension=file_extension,
        )

    def convert_log_entries_to_df(self):
        """
        Convert log entries dictionary to DataFrame format.

        Transforms the log_entries dictionary from the data logger into a pandas
        DataFrame for easier data manipulation and analysis. Adds shot numbers
        and sorts by elapsed time.

        Notes
        -----
        This conversion is somewhat redundant as a similar conversion happens
        at the end of a scan. Future versions should consider converting
        log_entries to DataFrame format from the beginning in data_logger.

        See Also
        --------
        get_current_data : Method that calls this conversion as part of data update.
        """
        self.log_entries = self.data_logger.log_entries
        self.log_df = pd.DataFrame.from_dict(self.log_entries, orient="index")
        self.log_df = self.log_df.sort_values(by="Elapsed Time").reset_index(drop=True)
        self.log_df["Shotnumber"] = self.log_df.index + 1

    def get_shotnumbers_for_bin(self, bin_number: int) -> None:
        """
        Extract shot numbers for a specific data bin.

        Filters the log DataFrame to find all shot numbers that belong
        to the specified bin number and stores them in current_shot_numbers.

        Parameters
        ----------
        bin_number : int
            The bin number to filter shot numbers for.

        Notes
        -----
        This method modifies the current_shot_numbers attribute in place.
        """
        self.current_shot_numbers = self.log_df[self.log_df["Bin #"] == bin_number][
            "Shotnumber"
        ].values

    def get_current_data(self) -> None:
        """
        Update current data bin with latest information.

        Refreshes the current_data_bin DataFrame to contain only data from
        the current bin being processed. Also extracts corresponding shot
        numbers for the current bin.

        Notes
        -----
        This method performs several operations:
        1. Converts log entries to DataFrame format
        2. Updates bin number from data logger
        3. Extracts shot numbers for current bin
        4. Filters data to current bin only

        See Also
        --------
        convert_log_entries_to_df : Converts log entries to DataFrame.
        get_shotnumbers_for_bin : Extracts shot numbers for a bin.
        """
        self.convert_log_entries_to_df()
        self.bin_number = self.data_logger.bin_num
        self.get_shotnumbers_for_bin(self.bin_number)
        self.current_data_bin = self.log_df[
            self.log_df["Bin #"] == self.bin_number
        ].copy()

    def _gather_shot_entries(
        self,
        shot_numbers: list[int],
        scalar_variables: dict[str, str],
        non_scalar_variables: list[str],
    ) -> list[dict]:
        """
        Parse shot data into structured dictionaries.

        Processes each shot to extract scalar measurements and construct
        file paths for non-scalar data (e.g., images), organizing the
        information into a standardized format.

        Parameters
        ----------
        shot_numbers : list of int
            List of shot numbers to process.
        scalar_variables : dict of str to str
            Mapping from short variable names to full column names
            in current_data_bin DataFrame.
        non_scalar_variables : list of str
            List of device names for which to construct image file paths.

        Returns
        -------
        list of dict
            Each dictionary contains:
            - 'shot_number' : int - The shot number
            - 'scalars' : dict - Scalar measurements for this shot
            - 'image_paths' : dict - File paths to device images
        """
        entries = []

        for shot in shot_numbers:
            entry = {"shot_number": shot, "scalars": {}, "image_paths": {}}

            # Extract scalars based on exact match of 'Shotnumber'
            shot_row = self.current_data_bin[
                self.current_data_bin["Shotnumber"] == shot
            ]
            if shot_row.empty:
                logger.warning("No entry in data table found for shot number %s", shot)
            else:
                for short_name, full_key in scalar_variables.items():
                    if full_key in shot_row.columns:
                        value = shot_row[full_key].values[0]
                        entry["scalars"][short_name] = value
                        logger.info(
                            "Scalar for '%s' from '%s' on shot %s: %s",
                            short_name,
                            full_key,
                            shot,
                            value,
                        )
                    else:
                        logger.warning(
                            "Key '%s' not found in current_data_bin columns", full_key
                        )

            # Image paths
            for device in non_scalar_variables:
                path = self.get_device_shot_path(
                    device_name=device, shot_number=shot, file_extension=".png"
                )
                entry["image_paths"][device] = path

            entries.append(entry)

        return entries

    def validate_variable_keys_against_requirements(
        self, variable_map: dict[str, str]
    ) -> None:
        """
        Validate that required variables exist in device requirements.

        Checks that each 'device:variable' key in the variable map corresponds
        to a variable declared in the device_requirements configuration.

        Parameters
        ----------
        variable_map : dict of str to str
            Dictionary mapping short alias names to full 'device:variable' strings.

        Raises
        ------
        ValueError
            If any variable in the map is not found in device_requirements.
        """
        device_reqs = self.device_requirements.get("Devices", {})

        logger.info("Device requirements passed to evaluator: %s", device_reqs)

        declared_vars = {
            f"{device}:{var}"
            for device, req in device_reqs.items()
            for var in req.get("variable_list", [])
        }
        logger.info("Declared variables in evaluator: %s", declared_vars)
        logger.info(
            "Looking for matches in variable_map.values(): %s", variable_map.values()
        )
        missing = [key for key in variable_map.values() if key not in declared_vars]
        if missing:
            raise ValueError(
                f"The following keys are not in device_requirements: {missing}"
            )

    @staticmethod
    def filter_log_entries_by_bin(
        log_entries: Dict[float, Dict[str, Any]], bin_num: int
    ) -> List[Dict[str, Any]]:
        """
        Filter log entries by bin number.

        Extracts all log entries that belong to a specific data bin number
        from the complete log entries dictionary.

        Parameters
        ----------
        log_entries : dict of float to dict
            Complete log entries dictionary indexed by elapsed time.
        bin_num : int
            Bin number to filter entries for.

        Returns
        -------
        list of dict
            List of log entry dictionaries that belong to the specified bin.
        """
        return [
            entry for entry in log_entries.values() if entry.get("Bin #") == bin_num
        ]

    def get_value(self, input_data: Dict) -> Dict:
        """
        Evaluate the objective function with current data.

        This is the main evaluation method that first retrieves the most recent
        data from the DataLogger and then delegates the actual evaluation logic
        to the subclass-defined `_get_value()` method.

        Parameters
        ----------
        input_data : dict
            Specification of control variable settings. Can be one of:
            - A pandas DataFrame of input rows
            - A list of dictionaries mapping variable names to values
            - A dictionary of variable names to lists of values
            - A dictionary of variable names to single float values

        Returns
        -------
        dict
            Dictionary containing evaluated objective(s) and constraint(s).
            The exact structure depends on the specific evaluator implementation.

        """
        self.get_current_data()
        return self._get_value(input_data=input_data)

    def log_objective_result(self, shot_num: int, scalar_value: float):
        """
        Log computed objective value for a specific shot.

        Adds the computed scalar objective value to the log_entries dictionary
        under the appropriate elapsed time key for the specified shot number.

        Parameters
        ----------
        shot_num : int
            The shot number being processed.
        scalar_value : float
            The computed objective value to log.

        Notes
        -----
        The objective value is stored with a key format "Objective:{objective_tag}"
        where objective_tag is the evaluator's objective identifier.

        """
        try:
            elapsed_time = self.current_data_bin.loc[
                self.current_data_bin["Shotnumber"] == shot_num, "Elapsed Time"
            ].values[0]
        except Exception as e:
            logger.warning(
                "Could not extract Elapsed Time for shot %s: %s", shot_num, e
            )
            return

        if elapsed_time:
            key = f"Objective:{self.objective_tag}"
            self.data_logger.log_entries[elapsed_time][key] = scalar_value
            logger.info("Logged %s = %s for shot %s", key, scalar_value, shot_num)
        else:
            logger.warning(
                "Cannot log result: no valid data_logger or elapsed_time=%s",
                elapsed_time,
            )

    @abstractmethod
    def _get_value(self, input_data: Dict) -> Dict:
        """
        Abstract method for computing objectives and constraints.

        This method must be implemented by subclasses and contains the core logic
        for evaluating the objective function. It is called by `get_value()` after
        current data has been updated via `get_current_data()`.

        Parameters
        ----------
        input_data : dict
            Specification of control variable settings. Format is the same as
            in `get_value()` method.

        Returns
        -------
        dict
            Dictionary containing results of objective function calculation.
            The structure depends on the specific evaluator implementation.

        Notes
        -----
        Subclasses must implement this method to define their specific
        evaluation logic. The method should assume that current_data_bin
        and related attributes have been updated with the latest data.
        """
        pass

    def __call__(self, input_data: Dict) -> Dict:
        """
        Make the evaluator instance callable.

        This method allows the evaluator to be used as a callable function,
        which is equivalent to calling `get_value(input_data)`.

        Parameters
        ----------
        input_data : dict
            Specification of control variable settings. Format is the same as
            in `get_value()` method.

        Returns
        -------
        dict
            Dictionary containing results of objective function calculation.

        """
        return self.get_value(input_data)
