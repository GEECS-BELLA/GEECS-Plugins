"""
ScanDataManager module for GEECS Scanner data acquisition.

This module provides the `ScanDataManager` class, which is responsible for managing
data preparation, organization, and saving during and after scans. It handles file
path setup, data format conversion, and integration with various data storage formats
including TDMS, HDF5, and text files.

Key Responsibilities
--------------------
- Initialize and manage scan data paths and output file structures
- Configure device save paths for non-scalar saving devices
- Convert scan results to standardized DataFrame format
- Save data in multiple formats (TDMS, TXT, HDF5)
- Handle data processing including NaN filling for asynchronous observables
- Generate scan information files and analysis-ready data files

Dependencies
------------
Standard Library
    time, logging, shutil, pathlib.Path, typing
Third-Party
    pandas, nptdms (TdmsWriter, ChannelObject)
Internal Modules
    geecs_scanner.data_acquisition (DeviceManager, DatabaseDictLookup)
    geecs_python_api.controls.interface.GeecsDatabase
    geecs_data_utils (ScanData, ScanMode, ScanConfig)

Usage
-----
This class is designed to be used primarily with the ScanManager during scan execution.

>>> from geecs_scanner.data_acquisition import ScanDataManager
>>> scan_data_mgr = ScanDataManager(device_manager, scan_data, database_dict)
>>> scan_data_mgr.initialize_scan_data_and_output_files()
>>> scan_data_mgr.configure_device_save_paths(save_local=True)
>>> # ... during scan execution ...
>>> results_df = scan_data_mgr.process_results(scan_results)

Notes
-----
The class works closely with DeviceManager to coordinate device configurations
and with ScanData to manage file paths and scan metadata.
"""

# Standard library imports
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Union
import time
import logging

# Third-party library imports
import pandas as pd
from nptdms import TdmsWriter, ChannelObject

# Internal project imports
from . import DeviceManager, DatabaseDictLookup
from geecs_python_api.controls.interface import GeecsDatabase
from geecs_data_utils import ScanData, ScanConfig

DeviceSavePaths = Dict[str, Dict[str, Union[Path, str]]]


class ScanDataManager:
    """Manage data preparation, organization, and saving during and after a scan.

    This class is responsible for setting up data paths, initializing writers for different
    formats (e.g., TDMS, HDF5), and handling the saving and processing of scan data. It works
    alongside DeviceManager and ScanData to ensure all relevant data is logged and stored
    appropriately.

    Parameters
    ----------
    device_manager : DeviceManager
        Manages the devices involved in the scan.
    scan_data : ScanData, optional
        Manages scan data paths and metadata. Defaults to None.
    database_dict : DatabaseDictLookup, optional
        Contains a dictionary of all experiment devices and variables.
        If None, a new instance will be created and reloaded. Defaults to None.

    Attributes
    ----------
    device_manager : DeviceManager
        Manages the devices involved in the scan.
    scan_data : ScanData
        Manages scan data paths and metadata.
    database_dict : DatabaseDictLookup
        Dictionary of all experiment devices and variables.
    tdms_writer : TdmsWriter, optional
        Writer for TDMS format data files. Initialized as None.
    data_txt_path : Path, optional
        Path to the main scan data text file. Initialized as None.
    data_h5_path : Path, optional
        Path to the HDF5 data file. Initialized as None.
    sFile_txt_path : Path, optional
        Path to the analysis-ready s-file. Initialized as None.
    tdms_output_path : Path, optional
        Path to the TDMS output file. Initialized as None.
    sFile_info_path : Path, optional
        Path to the s-file info file. Initialized as None.
    device_save_paths_mapping : DeviceSavePaths
        Mapping of device names to their save path configurations.
    scan_number_int : int, optional
        Integer scan number extracted from scan data. Initialized as None.
    parsed_scan_string : str, optional
        Formatted scan string (e.g., "Scan001"). Initialized as None.

    Notes
    -----
    This class is designed to be used primarily with the ScanManager during scan execution.
    It provides comprehensive data management for scientific scanning processes,
    supporting multiple data storage formats and handling complex device interactions.
    """

    DEPENDENT_SUFFIXES = ["-interp", "-interpSpec", "-interpDiv", "-Spatial"]

    def __init__(
        self,
        device_manager: DeviceManager,
        scan_data: Optional[ScanData] = None,
        database_dict: Optional[DatabaseDictLookup] = None,
    ):
        """Initialize the ScanDataManager with references to the ScanData and DeviceManager.

        Parameters
        ----------
        device_manager : DeviceManager
            Manages the devices involved in the scan. This is a required parameter.
        scan_data : ScanData, optional
            Manages scan data paths and metadata. Defaults to None.
        database_dict : DatabaseDictLookup, optional
            Contains a dictionary of all experiment devices and variables.
            If None, a new instance will be created and reloaded. Defaults to None.

        Notes
        -----
        If no database_dict is provided, a new DatabaseDictLookup instance will be
        created and reloaded automatically.
        """
        self.device_manager = device_manager  # Explicitly pass device_manager
        self.scan_data: ScanData = scan_data
        if database_dict is None:
            database_dict = DatabaseDictLookup().reload()
        self.database_dict = database_dict
        self.tdms_writer: Optional[TdmsWriter] = None
        self.data_txt_path: Optional[Path] = None
        self.data_h5_path: Optional[Path] = None
        self.sFile_txt_path: Optional[Path] = None
        self.tdms_output_path: Optional[Path] = None
        self.sFile_info_path: Optional[Path] = None

        self.device_save_paths_mapping: DeviceSavePaths = {}
        # self.scan_number_int = self.scan_data.get_tag().number
        # self.parsed_scan_string = f"Scan{self.scan_number_int:03}"

        self.scan_number_int: Optional[int] = None
        self.parsed_scan_string: Optional[str] = None

    def initialize_scan_data_and_output_files(self):
        """Initialize scan data metadata and set up file output paths.

        Prepares the file system for a new scan by:
        - Reloading paths configuration
        - Building next scan data instance
        - Extracting scan number and creating formatted scan string
        - Setting up file paths for various output formats (TDMS, HDF5, TXT)
        - Initializing TDMS writers

        Notes
        -----
        This method MUST be called before any other data operations to ensure:
        - Proper file path initialization
        - Correct scan number generation
        - TDMS writer setup

        Raises
        ------
        Exception
            If path configuration reload or scan data build fails
        """
        ScanData.reload_paths_config()
        self.scan_data = ScanData.build_next_scan_data()

        self.parsed_scan_string = self.scan_data.get_folder().parts[-1]
        self.scan_number_int = int(self.parsed_scan_string[-3:])

        folder = self.scan_data.get_folder()
        analysis_folder = self.scan_data.get_analysis_folder().parent

        self.tdms_output_path = folder / f"{self.parsed_scan_string}.tdms"
        self.data_txt_path = folder / f"ScanData{self.parsed_scan_string}.txt"
        self.data_h5_path = folder / f"ScanData{self.parsed_scan_string}.h5"
        self.sFile_txt_path = analysis_folder / f"s{self.scan_number_int}.txt"
        self.sFile_info_path = analysis_folder / f"s{self.scan_number_int}_info.txt"

        self.initialize_tdms_writers(str(self.tdms_output_path))
        time.sleep(1)

    def configure_device_save_paths(self, save_local: bool = True):
        """
        Configure save paths and enable saving for non-scalar devices.

        Parameters
        ----------
        save_local : bool, optional
            If True, configures local saving to a shared directory. If False, saves to scan folder.
        """
        for device_name in self.device_manager.non_scalar_saving_devices:
            logging.info(f"Configuring save paths for device: {device_name}")
            target_dir = self.scan_data.get_folder() / device_name
            target_dir.mkdir(parents=True, exist_ok=True)

            device = self.device_manager.devices.get(device_name)
            device_type = GeecsDatabase.find_device_type(device_name)

            if not device:
                logging.warning(f"Device {device_name} not found in DeviceManager.")
                continue

            dev_host_ip_string = device.dev_ip
            if save_local:
                source_dir = Path(f"//{dev_host_ip_string}/SharedData/{device_name}")
                self.purge_local_save_dir(source_dir)
                data_path_client_side = Path("C:/SharedData") / device_name
            else:
                source_dir = target_dir
                data_path_client_side = target_dir

            save_path = str(data_path_client_side).replace("/", "\\")
            device.set("localsavingpath", save_path, sync=False)
            time.sleep(0.1)
            device.set("save", "on", sync=False)

            self.device_save_paths_mapping[device_name] = {
                "target_dir": target_dir,
                "source_dir": source_dir,
                "device_type": device_type,
            }

    def purge_all_local_save_dir(self) -> None:
        """Purge all local save directories for non-scalar saving devices.

        Iterates through the SharedData folder of each device host and removes
        all subdirectories that contain the device name. This method helps clean
        up temporary data storage locations before or after a scan.

        Notes
        -----
        - Checks each non-scalar saving device's SharedData directory
        - Logs information about directories being purged
        - Logs a warning if the base directory does not exist
        - Uses purge_local_save_dir to remove files from individual subdirectories

        Raises
        ------
        OSError
            If there are permission issues accessing directories
        """
        for device_name in self.device_manager.non_scalar_saving_devices:
            device = self.device_manager.devices.get(device_name)
            if device:
                dev_name = device.get_name()
                dev_host_ip_string = device.dev_ip
                base_dir = Path(f"//{dev_host_ip_string}/SharedData")

                if base_dir.exists():
                    # Iterate over all directories in the SharedData folder.
                    for subdir in base_dir.iterdir():
                        if subdir.is_dir() and dev_name in subdir.name:
                            logging.info(f"Purging directory: {subdir}")
                            self.purge_local_save_dir(subdir)
                else:
                    logging.warning(f"Base directory {base_dir} does not exist.")

    @staticmethod
    def purge_local_save_dir(source_dir: Path) -> None:
        """
        Recursively purge all files from a given directory.

        Parameters
        ----------
        source_dir : Path
            Path to the directory whose contents will be deleted.
        """
        if source_dir.exists():
            for item in source_dir.rglob("*"):
                if item.is_file():
                    try:
                        item.unlink()
                        logging.info(f"Removed file: {item}")
                    except Exception as e:
                        logging.error(f"Error removing {item}: {e}")

    def initialize_tdms_writers(self, tdms_output_path: str) -> None:
        """Initialize TDMS writers for scalar data and index file generation.

        Creates a TdmsWriter instance to prepare for writing scan data to a TDMS file,
        with support for both data and index file generation.

        Parameters
        ----------
        tdms_output_path : str
            Full file path where the TDMS output file will be created.

        Notes
        -----
        - Creates a TdmsWriter with index_file=True to generate both data and index files
        - Logs the initialization of the TDMS writer with its output path
        - The index file helps in faster data retrieval and navigation

        Raises
        ------
        IOError
            If there are issues creating the TDMS file (e.g., permission issues, invalid path)
        OSError
            If the specified path is not writable

        Examples
        --------
        >>> scan_data_mgr.initialize_tdms_writers('/path/to/output/scan_data.tdms')
        """
        try:
            self.tdms_writer = TdmsWriter(tdms_output_path, index_file=True)
            logging.info(f"TDMS writer initialized with path: {tdms_output_path}")
        except (IOError, OSError) as e:
            logging.error(f"Failed to initialize TDMS writer: {e}")
            raise

    def write_scan_info_ini(self, scan_config: ScanConfig) -> None:
        """Generate a comprehensive scan configuration INI file for documentation and analysis.

        Creates a structured INI file capturing detailed metadata about the scan configuration,
        providing a human-readable record of experimental parameters and settings.

        Parameters
        ----------
        scan_config : ScanConfig
            Configuration object containing comprehensive details about the scan:
            - Device variable being scanned
            - Scan start, end, and step values
            - Scan mode and background settings
            - Additional descriptive information

        Notes
        -----
        INI File Generation Process:
        - Creates a standardized INI file with scan metadata
        - Uses predefined data_txt_path for file location
        - Generates filename based on scan folder
        - Handles default values for unspecified parameters
        - Converts boolean values to lowercase for INI compatibility
        - Creates parent directories if they don't exist

        File Contents Include:
        - Scan number
        - Scan start information
        - Scan parameter
        - Start and end values
        - Step size
        - Shots per step
        - Background flag
        - Scan mode

        Raises
        ------
        IOError
            If there are issues writing to the file (e.g., permission problems)
        ValueError
            If scan configuration is incomplete or invalid

        Examples
        --------
        >>> scan_config = ScanConfig(
        ...     device_var='LaserPower',
        ...     start=1.0,
        ...     end=10.0,
        ...     step=1.0,
        ...     wait_time=5,
        ...     background=False,
        ...     scan_mode=ScanMode.LINEAR,
        ...     additional_description='Laser power sweep experiment'
        ... )
        >>> scan_data_mgr.write_scan_info_ini(scan_config)
        # Creates ScanInfo{scan_folder}.ini with detailed scan metadata

        See Also
        --------
        initialize_scan_data_and_output_files : Method that sets up scan data paths
        process_results : Method that processes and saves scan results
        """
        # TODO: should probably add some exception handling here. the self.parsed_scan_string and
        # self.scan_number_int are set in create_and_set_data_paths. This method is only called
        # once immediately after that, so it should be ok, but if these variables aren't set
        # something sensible should happend...
        scan_folder = self.parsed_scan_string
        scan_number = self.scan_number_int

        filename = f"ScanInfo{scan_folder}.ini"

        scan_var = scan_config.device_var
        if not scan_var:
            scan_var = "Shotnumber"

        additional_description = scan_config.additional_description

        scan_info = f"{self.device_manager.scan_base_description}. scanning {scan_var}. {additional_description}"

        # Add the Scan Info section
        config_file_contents = [
            "[Scan Info]\n",
            f"Scan No = {scan_number}\n",
            f'ScanStartInfo = "{scan_info}"\n',
            f'Scan Parameter = "{scan_var}"\n',
            f"Start = {scan_config.start}\n",
            f"End = {scan_config.end}\n",
            f"Step size = {scan_config.step}\n",
            f"Shots per step = {scan_config.wait_time}\n",
            'ScanEndInfo = ""\n',
            f"Background = {str(scan_config.background).lower()}\n",  # INI-style lowercase booleans
            f'ScanMode = "{scan_config.scan_mode.value}"\n',
        ]

        # Create the full path for the file
        full_path = Path(self.data_txt_path).parent / filename
        full_path.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"Attempting to write to {full_path}")

        # Write to the .ini file
        with full_path.open("w") as configfile:
            for line in config_file_contents:
                configfile.write(line)

        logging.info(f"Scan info written to {full_path}")

    def save_to_txt_and_h5(self, df):
        """Save processed scan data to text file with tab-separated values.

        Converts a processed DataFrame to a tab-separated text file, providing
        a human-readable and easily parseable data representation. Currently,
        HDF5 (.h5) support is noted as not implemented.

        Parameters
        ----------
        df : pandas.DataFrame
            Processed DataFrame containing scan results.
            Expected to have standardized column headers and cleaned data.

        Notes
        -----
        Saving process details:
        - Uses tab-separated (.txt) format for maximum compatibility
        - Saves to predefined data_txt_path
        - Excludes index column to maintain clean data representation
        - Intended for data storage and sharing across different platforms
        - HDF5 support is currently not implemented (placeholder for future development)

        Raises
        ------
        IOError
            If file writing encounters permission or path-related issues
        ValueError
            If input DataFrame is empty or improperly formatted

        Examples
        --------
        >>> processed_df = scan_data_mgr.convert_to_dataframe(log_entries)
        >>> scan_data_mgr.save_to_txt_and_h5(processed_df)
        # Saves processed data to text file at data_txt_path

        See Also
        --------
        _make_sFile : Alternative method for saving analysis-ready data
        convert_to_dataframe : Method that prepares DataFrame for saving
        dataframe_to_tdms : Method for saving data in TDMS format
        """
        try:
            if df.empty:
                logging.warning("Attempting to save an empty DataFrame to text file")
                return

            # Save as .txt file (TSV format)
            df.to_csv(self.data_txt_path, sep="\t", index=False)
            logging.info(f"Scan data saved to {self.data_txt_path}")

            # Commented out alternative save path for potential future use
            # df.to_csv(self.sFile_txt_path, sep='\t', index=False)

        except (IOError, ValueError) as e:
            logging.error(f"Error saving text file: {e}")
            raise

    def _make_sFile(self, df):
        """Create an analysis-ready s-file from processed scan data.

        Saves a preprocessed DataFrame to a tab-separated text file specifically
        designed for scientific analysis. This method is typically used to generate
        a standardized output file for further data processing or archival.

        Parameters
        ----------
        df : pandas.DataFrame
            Processed DataFrame containing scan results.
            Expected to have standardized column headers and cleaned data.

        Notes
        -----
        Saving process details:
        - Uses tab-separated (.txt) format for compatibility
        - Saves to predefined sFile_txt_path
        - Excludes index column to maintain clean data representation
        - Intended for analysis-ready data storage and sharing

        Raises
        ------
        IOError
            If file writing encounters permission or path-related issues
        ValueError
            If input DataFrame is empty or improperly formatted

        Examples
        --------
        >>> processed_df = scan_data_mgr.convert_to_dataframe(log_entries)
        >>> scan_data_mgr._make_sFile(processed_df)
        # Saves processed data to s-file at sFile_txt_path

        See Also
        --------
        convert_to_dataframe : Method that prepares DataFrame for s-file creation
        save_to_txt_and_h5 : Alternative method for saving scan data
        """
        try:
            if df.empty:
                logging.warning("Attempting to save an empty DataFrame to s-file")
                return

            df.to_csv(self.sFile_txt_path, sep="\t", index=False)
            logging.info(f"Analysis-ready data saved to {self.sFile_txt_path}")
        except (IOError, ValueError) as e:
            logging.error(f"Error saving s-file: {e}")
            raise

    def dataframe_to_tdms(self, df):
        """Convert DataFrame to TDMS file with comprehensive device and variable metadata.

        Transforms a pandas DataFrame into a TDMS (Technical Data Management Streaming) file,
        preserving detailed device and variable information in the channel structure.

        Parameters
        ----------
        df : pandas.DataFrame
            Processed DataFrame containing scan data to be saved.
            Columns should represent device measurements with informative names.
            Expected column format: 'DeviceName VariableName' or 'DeviceName VariableName Alias:alias_name'

        Returns
        -------
        None
            Writes data directly to the TDMS file through the TdmsWriter.

        Notes
        -----
        TDMS Conversion Process:
        - Extracts device name from column header
        - Handles special columns like 'Bin #' and 'Elapsed Time'
        - Removes alias information from variable names
        - Creates ChannelObjects for each column
        - Writes data segments to TDMS file
        - Supports complex, multi-device data representation

        Detailed Transformation Steps:
        1. Iterate through DataFrame columns
        2. Determine device name (first word or special column)
        3. Extract variable name
        4. Remove alias information
        5. Create ChannelObject with device and variable metadata
        6. Write channel data to TDMS file segment

        Raises
        ------
        ValueError
            If column names do not follow expected naming conventions
        TypeError
            If data cannot be converted to TDMS format
        IOError
            If TDMS file writing encounters issues

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'Laser Power': [1.0, 2.0, 3.0],
        ...     'Detector Temperature Alias:temp_sensor': [22.3, 22.5, 22.7]
        ... })
        >>> scan_data_mgr.dataframe_to_tdms(df)
        # Saves DataFrame to TDMS file with device and alias information

        See Also
        --------
        convert_to_dataframe : Prepares DataFrame for TDMS conversion
        modify_headers : Handles header standardization before TDMS conversion
        """
        try:
            tdms_writer = self.tdms_writer
            with tdms_writer:
                for column in df.columns:
                    # Extract device name, handling special cases
                    if column == "Bin #" or column == "Elapsed Time":
                        device_name = column
                    else:
                        device_name = column.split(" ", 1)[0]

                    # Extract variable name
                    variable_name = column[len(device_name) :].strip()
                    variable_name = variable_name.split(" Alias:", 1)[0].strip()
                    variable_name = f"{device_name} {variable_name}".strip()

                    # Get data and create ChannelObject
                    data = df[column].values
                    ch_object = ChannelObject(device_name, variable_name, data)
                    tdms_writer.write_segment([ch_object])

            logging.info("TDMS file written successfully with comprehensive metadata.")
        except (ValueError, TypeError, IOError) as e:
            logging.error(f"Error converting DataFrame to TDMS: {e}")
            raise

    def convert_to_dataframe(self, log_entries):
        """Transform raw log entries into a structured, analysis-ready pandas DataFrame.

        This method is a critical data preprocessing function that converts
        unstructured log entries from the data acquisition system into a clean,
        standardized pandas DataFrame suitable for scientific analysis.

        Parameters
        ----------
        log_entries : dict
            A dictionary of log entries where:
            - Keys represent timestamps (float or numeric)
            - Values are dictionaries containing device measurements

            Example:
            {
                0.0: {'Device1': 10.5, 'Device2': 20.3},
                1.0: {'Device1': 11.2, 'Device2': 21.7}
            }

        Returns
        -------
        pandas.DataFrame
            A processed DataFrame with:
            - Rows sorted by elapsed time
            - Columns representing device measurements
            - Standardized headers
            - Filled asynchronous observable columns
            - Added 'Shotnumber' column for tracking

        Notes
        -----
        Transformation steps:
        1. Convert log entries to DataFrame
        2. Sort data by elapsed time
        3. Fill missing values in asynchronous observable columns
        4. Standardize column headers
        5. Add sequential shot number

        Raises
        ------
        ValueError
            If log_entries cannot be converted to a DataFrame
        TypeError
            If log_entries is not a dictionary or contains invalid data types

        Examples
        --------
        >>> log_entries = {
        ...     0.0: {'Laser:Power': 5.5, 'Detector:Temperature': 22.3},
        ...     1.0: {'Laser:Power': 6.0, 'Detector:Temperature': 22.5}
        ... }
        >>> processed_df = scan_data_mgr.convert_to_dataframe(log_entries)
        >>> print(processed_df)
           Laser Power  Detector Temperature  Shotnumber
        0          5.5                 22.3           1
        1          6.0                 22.5           2

        See Also
        --------
        fill_async_nans : Method used to handle missing values in async columns
        modify_headers : Method used to standardize column headers
        """
        try:
            # Validate input
            if not isinstance(log_entries, dict):
                raise TypeError("log_entries must be a dictionary")

            if not log_entries:
                logging.warning("Empty log entries provided")
                return pd.DataFrame()

            # Convert log entries to DataFrame
            log_df = pd.DataFrame.from_dict(log_entries, orient="index")

            # Ensure 'Elapsed Time' column exists and sort
            if "Elapsed Time" not in log_df.columns:
                logging.warning("No 'Elapsed Time' column found. Using index order.")
                log_df = log_df.sort_index().reset_index(drop=True)
            else:
                log_df = log_df.sort_values(by="Elapsed Time").reset_index(drop=True)

            # Handle asynchronous observables
            async_observables = self.device_manager.async_observables
            log_df = self.fill_async_nans(log_df, async_observables)

            # Modify and standardize headers
            new_headers = self.modify_headers(log_df.columns)
            log_df.columns = new_headers

            # Add shot number
            log_df["Shotnumber"] = log_df.index + 1

            return log_df

        except Exception as e:
            logging.error(f"Error converting log entries to DataFrame: {e}")
            raise

    def modify_headers(self, headers):
        """
        Modify the headers of a DataFrame by appending aliases or adjusting format.

        This method processes the column headers of a DataFrame. If a header contains
        a colon (e.g., 'device_name:variable'), it splits the header and checks if
        the variable has an alias in the device dictionary. The alias is then appended
        to the new header format.

        This works mostly as a little helper function to reformat header names exactly
        as they are written when using a Master Control scan

        Args:
            headers (list): List of column headers from the DataFrame.

        Returns
        -------
            list: A list of modified headers with aliases or adjusted formats.
        """
        new_headers = []
        for header in headers:
            if ":" in header:
                device_name, variable = header.split(":")
                new_header = f"{device_name} {variable}"
                # Check if alias exists
                device_dict = self.database_dict.get_database()
                alias = device_dict.get(device_name, {}).get(variable, {}).get("alias")
                if alias:
                    new_header = f"{new_header} Alias:{alias}"
            else:
                new_header = header  # Keep the header as is if no colon
            new_headers.append(new_header)
        return new_headers

    def process_results(self, results):
        """Process scan results and prepare data for storage and analysis.

        This method is a critical component of the data acquisition workflow. It takes
        raw scan results, transforms them into a structured format, and saves the data
        in multiple file formats for further analysis and record-keeping.

        Parameters
        ----------
        results : dict
            A dictionary containing scan results, where:
            - Keys represent timestamps
            - Values contain corresponding measurement data
            If empty, indicates no data was collected during the scan.

        Returns
        -------
        pandas.DataFrame
            A processed DataFrame containing:
            - Sorted scan data
            - Filled asynchronous observable columns
            - Added 'Shotnumber' column
            Returns an empty DataFrame if no results are provided.

        Notes
        -----
        Processing steps include:
        1. Convert raw results to a structured DataFrame
        2. Sort data by elapsed time
        3. Fill missing values in asynchronous observable columns
        4. Save data to multiple formats:
           - Tab-separated text file (.txt)
           - TDMS file (for detailed data storage)
        5. Log information about data logging process

        Logs
        ----
        - Info message when data logging is complete
        - Warning message if no data was collected

        Raises
        ------
        ValueError
            If results dictionary contains improperly formatted data
        TypeError
            If results is not a dictionary

        Examples
        --------
        >>> results = {
        ...     0.0: {'Device1': 10, 'Device2': 20},
        ...     1.0: {'Device1': 15, 'Device2': 25}
        ... }
        >>> processed_df = scan_data_mgr.process_results(results)
        """
        if not isinstance(results, dict):
            raise TypeError("Results must be a dictionary of timestamp-data pairs")

        if results:
            try:
                log_df = self.convert_to_dataframe(results)
                logging.info("Data logging complete. Returning DataFrame.")

                # Save results to .txt and .h5
                self.save_to_txt_and_h5(log_df)

                # Write TDMS files (data and index)
                self.dataframe_to_tdms(log_df)

                return log_df
            except Exception as e:
                logging.error(f"Error processing scan results: {e}")
                return pd.DataFrame()
        else:
            logging.warning("No data was collected during the logging period.")
            return pd.DataFrame()

    @staticmethod
    def fill_async_nans(log_df, async_observables, fill_value=0):
        """Handle missing values in asynchronous observable columns with advanced filling strategies.

        Applies sophisticated data imputation techniques to handle missing or inconsistent
        data in asynchronous observable columns, ensuring data continuity and preparedness
        for further analysis.

        Parameters
        ----------
        log_df : pandas.DataFrame
            Input DataFrame containing logged data with potential missing values.
            Columns represent different device measurements or observables.

        async_observables : list
            List of column names representing asynchronous observables that require
            special handling for missing data.

        fill_value : int or float, optional
            Default value to use for filling any remaining NaN entries after
            forward and backward filling. Defaults to 0.

        Returns
        -------
        pandas.DataFrame
            A processed DataFrame with:
            - Empty strings converted to NaN
            - Asynchronous observable columns filled using advanced strategies
            - Remaining NaN values replaced with specified fill_value
            - Object dtype arrays appropriately downcasted

        Notes
        -----
        Filling Strategy:
        1. Convert empty strings to NaN
        2. For each async observable column:
           - Skip columns that are entirely NaN
           - Apply forward fill to propagate last known value
           - Apply backward fill to handle leading NaNs
        3. Fill any remaining NaNs with specified fill_value
        4. Downcast object dtype arrays for memory efficiency

        Logging:
        - Warns about columns consisting entirely of NaN values
        - Logs the fill value used for remaining NaNs

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'Device1': [1, np.nan, np.nan, 4],
        ...     'Device2': ['', 20, 30, 40]
        ... })
        >>> async_cols = ['Device1', 'Device2']
        >>> processed_df = ScanDataManager.fill_async_nans(df, async_cols)
        # Fills NaNs in Device1 and Device2 columns

        Raises
        ------
        TypeError
            If input DataFrame or async_observables have unexpected types

        See Also
        --------
        convert_to_dataframe : Method that uses fill_async_nans during data preprocessing
        pandas.DataFrame.ffill : Forward fill method used internally
        pandas.DataFrame.bfill : Backward fill method used internally
        """
        try:
            # Validate input types
            if not isinstance(log_df, pd.DataFrame):
                raise TypeError("log_df must be a pandas DataFrame")

            if not isinstance(async_observables, list):
                raise TypeError("async_observables must be a list")

            # Convert empty strings to NaN
            log_df.replace("", pd.NA, inplace=True)

            for async_obs in async_observables:
                if async_obs in log_df.columns:
                    # Only process if the entire column is not NaN
                    if not log_df[async_obs].isna().all():
                        # First, apply forward fill (ffill) to propagate the last known value
                        log_df[async_obs] = log_df[async_obs].ffill()

                        # Then, apply backward fill (bfill) to fill leading NaNs with the first non-NaN value
                        log_df[async_obs] = log_df[async_obs].bfill()
                    else:
                        logging.warning(
                            f"Column {async_obs} consists entirely of NaN values and will be left unchanged."
                        )

            # Finally, fill any remaining NaN values with the specified fill_value
            log_df = log_df.fillna(fill_value)

            # Use infer_objects to downcast the object dtype arrays appropriately
            log_df = log_df.infer_objects(copy=False)
            logging.info(f"Filled remaining NaN and empty values with {fill_value}.")

            return log_df

        except Exception as e:
            logging.error(f"Error in fill_async_nans: {e}")
            raise
