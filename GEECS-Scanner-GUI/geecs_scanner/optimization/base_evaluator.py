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

Examples
--------
Creating a custom evaluator:

>>> class MyEvaluator(BaseEvaluator):
...     def _get_value(self, input_data):
...         # Custom evaluation logic
...         return {"objective": computed_value}

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

    Examples
    --------
    Implementing a custom evaluator:

    >>> class BeamSizeEvaluator(BaseEvaluator):
    ...     def _get_value(self, input_data):
    ...         # Get current shot data
    ...         self.get_current_data()
    ...
    ...         # Compute beam size from image analysis
    ...         beam_size = self.analyze_beam_images()
    ...
    ...         return {"beam_size": beam_size}

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
                logging.warning(f"Key validation failed: {e}")

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

        Examples
        --------
        >>> evaluator = MyEvaluator(scan_data_manager=sdm)
        >>> path = evaluator.get_device_shot_path("camera1", 42, ".tiff")
        >>> print(path)
        /path/to/scan/data/camera1/shot_042.tiff
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

        Examples
        --------
        >>> scalar_vars = {"energy": "LaserEnergy:value", "power": "LaserPower:value"}
        >>> non_scalar_vars = ["camera1", "camera2"]
        >>> entries = evaluator._gather_shot_entries([1, 2], scalar_vars, non_scalar_vars)
        >>> print(entries[0])
        {
            'shot_number': 1,
            'scalars': {'energy': 1.5, 'power': 100.0},
            'image_paths': {'camera1': '/path/to/camera1_001.png', ...}
        }
        """
        entries = []

        for shot in shot_numbers:
            entry = {"shot_number": shot, "scalars": {}, "image_paths": {}}

            # Extract scalars based on exact match of 'Shotnumber'
            shot_row = self.current_data_bin[
                self.current_data_bin["Shotnumber"] == shot
            ]
            if shot_row.empty:
                logging.warning(f"No row found for shot number {shot}")
            else:
                for short_name, full_key in scalar_variables.items():
                    if full_key in shot_row.columns:
                        value = shot_row[full_key].values[0]
                        entry["scalars"][short_name] = value
                        logging.info(
                            f"Scalar for '{short_name}' from '{full_key}' on shot {shot}: {value}"
                        )
                    else:
                        logging.warning(
                            f"Key '{full_key}' not found in current_data_bin columns"
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

        Examples
        --------
        >>> device_reqs = {"Devices": {"laser": {"variable_list": ["power", "energy"]}}}
        >>> evaluator = MyEvaluator(device_requirements=device_reqs)
        >>> var_map = {"pwr": "laser:power", "eng": "laser:energy"}
        >>> evaluator.validate_variable_keys_against_requirements(var_map)  # No error
        >>> bad_map = {"temp": "laser:temperature"}  # temperature not in variable_list
        >>> evaluator.validate_variable_keys_against_requirements(bad_map)  # Raises ValueError
        """
        device_reqs = self.device_requirements.get("Devices", {})

        logging.info(f"device reqs pass to evaluator: {device_reqs}")

        declared_vars = {
            f"{device}:{var}"
            for device, req in device_reqs.items()
            for var in req.get("variable_list", [])
        }
        logging.info(f"declared variables in evaluator: {declared_vars}")
        logging.info(
            f"looking for matches in variable_map.values() {variable_map.values()}"
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

        Examples
        --------
        >>> log_data = {
        ...     1.0: {"Bin #": 1, "value": 10},
        ...     2.0: {"Bin #": 1, "value": 15},
        ...     3.0: {"Bin #": 2, "value": 20}
        ... }
        >>> bin1_entries = BaseEvaluator.filter_log_entries_by_bin(log_data, 1)
        >>> len(bin1_entries)
        2
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

        Examples
        --------
        >>> evaluator = MyEvaluator(...)
        >>> input_vars = {"laser_power": 100.0, "focus_position": 5.2}
        >>> result = evaluator.get_value(input_vars)
        >>> print(result)
        {"objective": 0.85, "constraint": 0.1}

        See Also
        --------
        _get_value : Abstract method implemented by subclasses.
        get_current_data : Method that updates current data bin.
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

        Examples
        --------
        >>> evaluator.log_objective_result(42, 0.85)
        # Logs objective value 0.85 for shot 42
        """
        try:
            elapsed_time = self.current_data_bin.loc[
                self.current_data_bin["Shotnumber"] == shot_num, "Elapsed Time"
            ].values[0]
        except Exception as e:
            logging.warning(f"Could not extract Elapsed Time for shot {shot_num}: {e}")
            return

        if elapsed_time:
            key = f"Objective:{self.objective_tag}"
            self.data_logger.log_entries[elapsed_time][key] = scalar_value
            logging.info(f"Logged {key} = {scalar_value} for shot {shot_num}")
        else:
            logging.warning(
                f"Cannot log result: no valid data_logger or elapsed_time={elapsed_time}"
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

        Examples
        --------
        Example implementation in a subclass:

        >>> def _get_value(self, input_data):
        ...     # Process current shot data
        ...     beam_sizes = []
        ...     for shot in self.current_shot_numbers:
        ...         image_path = self.get_device_shot_path("camera", shot)
        ...         beam_size = analyze_beam_image(image_path)
        ...         beam_sizes.append(beam_size)
        ...
        ...     # Compute objective
        ...     avg_beam_size = np.mean(beam_sizes)
        ...     return {"beam_size": avg_beam_size}
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

        Examples
        --------
        >>> evaluator = MyEvaluator(...)
        >>> input_vars = {"laser_power": 100.0}
        >>>
        >>> # These two calls are equivalent:
        >>> result1 = evaluator.get_value(input_vars)
        >>> result2 = evaluator(input_vars)
        >>> assert result1 == result2

        See Also
        --------
        get_value : The main evaluation method.
        """
        return self.get_value(input_data)
