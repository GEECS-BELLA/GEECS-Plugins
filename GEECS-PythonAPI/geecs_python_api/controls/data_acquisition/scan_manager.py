# Standard library imports
import os
import re
import time
import threading
import logging
import importlib
from pathlib import Path
from typing import Optional, List, Tuple, Any
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party library imports
import pandas as pd
import numpy as np
from nptdms import TdmsWriter, ChannelObject

# Internal project imports
from geecs_python_api.controls.data_acquisition import DeviceManager, ActionManager, DataLogger
from geecs_python_api.controls.data_acquisition.utils import ConsoleLogger
from geecs_python_api.controls.interface import load_config, GeecsDatabase
from geecs_python_api.controls.interface.geecs_paths_config import GeecsPathsConfig
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.analysis.scans.scan_data import ScanData
from image_analysis.utils import get_imaq_timestamp_from_png, get_picoscopeV2_timestamp, get_custom_imaq_timestamp

config = load_config()

if config and 'Experiment' in config and 'expt' in config['Experiment']:
    default_experiment = config['Experiment']['expt']
    logging.info(f"default experiment is: {default_experiment}")
else:
    logging.warning(
        "Configuration file not found or default experiment not defined. While use Undulator as experiment. Could be a problem for you.")
    default_experiment = 'Undulator'

try:
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info(default_experiment)
    device_dict = GeecsDevice.exp_info['devices']
except AttributeError:
    logging.warning("Could not retrieve dictionary of GEECS Devices")
    device_dict = None

ANALYSIS_CLASS_MAPPING = {
    'MagSpecStitcherAnalysis': 'scan_analysis.analyzers.Undulator.MagSpecStitcherAnalysis.MagSpecStitcherAnalysis',
    'CameraImageAnalysis': 'scan_analysis.analyzers.Undulator.CameraImageAnalysis.CameraImageAnalysis',
    'VisaEBeamAnalysis': 'scan_analysis.analyzers.Undulator.VisaEBeamAnalysis.VisaEBeamAnalysis'
}


class ScanDataManager:
    """
    Manages data preparation, organization and saving during and after a scan.

    This class is responsible for setting up data paths, initializing writers for different formats
    (e.g., TDMS, HDF5), and handling the saving and processing of scan data. It works alongside
    DeviceManager and ScanData to ensure all relevant data is logged and stored appropriately.
    This class is designed to be used primarily (or even exclusively) with the ScanMananger
    """
    
    DEPENDENT_SUFFIXES = ["-interp", "-interpSpec", "-interpDiv", "-Spatial"]
    
    def __init__(self, device_manager: 'DeviceManager', scan_data: Optional['ScanData'] = None):
        """
        Initialize the ScanDataManager with references to the ScanData and DeviceManager.

        Args:
            device_manager (DeviceManager): Manages the devices involved in the scan.
            scan_data (ScanData): manages scan data
        """
        self.device_manager = device_manager  # Explicitly pass device_manager
        self.scan_data = scan_data
        self.tdms_writer = None
        self.data_txt_path = None
        self.data_h5_path = None
        self.sFile_txt_path = None
        
        # self.scan_number_int = self.scan_data.get_tag().number
        # self.parsed_scan_string = f"Scan{self.scan_number_int:03}"
        
        self.scan_number_int = None
        self.parsed_scan_string = None

    def create_and_set_data_paths(self):
        """
        Create data paths for devices that need non-scalar saving, and initialize the TDMS writers.

        This method sets up the necessary directories and paths for saving device data,
        then initializes the TDMS writers for logging scalar and non-scalar data.
        """
        paths_config = GeecsPathsConfig()
        if not paths_config.is_default_server_address():
            raise NotADirectoryError("Unable to locate server address for saving data, unable to set paths")

        self.scan_data = ScanData.build_next_scan_data()

        for device_name in self.device_manager.non_scalar_saving_devices:
            data_path = self.scan_data.get_folder() / device_name
            data_path.mkdir(parents=True, exist_ok=True)
            
            device = self.device_manager.devices.get(device_name)
            if device:
                save_path = str(data_path).replace('/', "\\")
                logging.info(f"Setting save data path for {device_name} to {save_path}")
                device.set("localsavingpath", save_path, sync=False)
                time.sleep(.1)
                device.set('save', 'on', sync=False)
            else:
                logging.warning(f"Device {device_name} not found in DeviceManager.")
        
        analysis_save_path = self.scan_data.get_analysis_folder()
        
        self.parsed_scan_string = self.scan_data.get_folder().parts[-1]
        self.scan_number_int = int(self.parsed_scan_string[-3:])

        self.tdms_output_path = self.scan_data.get_folder() / f"{self.parsed_scan_string}.tdms"
        self.data_txt_path = self.scan_data.get_folder() / f"ScanData{self.parsed_scan_string}.txt"
        self.data_h5_path = self.scan_data.get_folder() /  f"ScanData{self.parsed_scan_string}.h5"

        self.sFile_txt_path = self.scan_data.get_analysis_folder().parent / f"s{self.scan_number_int}.txt"
        self.sFile_info_path = self.scan_data.get_analysis_folder().parent / f"s{self.scan_number_int}_info.txt"
        
        self.initialize_tdms_writers(str(self.tdms_output_path))

        time.sleep(1)
        
        return self.scan_data

    def initialize_tdms_writers(self, tdms_output_path):
        """
        Initialize the TDMS writers for scalar data and index data.

        Args:
            tdms_output_path (str): Path to the TDMS file for saving scalar data.
        """
        self.tdms_writer = TdmsWriter(tdms_output_path, index_file = True)
        logging.info(f"TDMS writer initialized with path: {tdms_output_path}")

    def write_scan_info_ini(self, scan_config):
        """
        Write the scan configuration to an .ini file.

        Args:
            scan_config (dict): Configuration dictionary containing details about the scan,
                                such as device variable, start, end, and step values.
        """
        # Check if scan_config is a dictionary
        if not isinstance(scan_config, dict):
            logging.error(f"scan_config is not a dictionary: {type(scan_config)}")
            return
        
        # TODO: should probably add some exception handling here. the self.parsed_scan_string and
        # self.scan_number_int are set in create_and_set_data_paths. This method is only called
        # once immediately after that, so it should be ok, but if these variables aren't set
        # something sensible should happend...
        scan_folder = self.parsed_scan_string
        scan_number = self.scan_number_int

        filename = f"ScanInfo{scan_folder}.ini"

        scan_var = scan_config.get('device_var', '')
        additional_description = scan_config.get('additional_description', '')

        scan_info = f'{self.device_manager.scan_base_description}. scanning {scan_var}. {additional_description}'

        # Add the Scan Info section
        config_file_contents = [
            "[Scan Info]\n",
            f"Scan No = \"{scan_number}\"\n",
            f"ScanStartInfo = \"{scan_info}\"\n",
            f"Scan Parameter = \"{scan_var}\"\n",
            f"Start = \"{scan_config.get('start', 0)}\"\n",
            f"End = \"{scan_config.get('end', 0)}\"\n",
            f"Step size = \"{scan_config.get('step', 1)}\"\n",
            f"Shots per step = \"{scan_config.get('wait_time', 1)}\"\n",
            f"ScanEndInfo = \"\""
        ]

        # Create the full path for the file
        full_path = Path(self.data_txt_path).parent / filename
        full_path.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"Attempting to write to {full_path}")

        # Write to the .ini file
        with full_path.open('w') as configfile:
            for line in config_file_contents:
                configfile.write(line)

        logging.info(f"Scan info written to {full_path}")

    def save_to_txt_and_h5(self, df):
        """
        Save the scan data to both .txt (TSV format). H5 support not yet included.

        Args:
            df (pandas.DataFrame): DataFrame containing the scan data to be saved.
        """

        # Save as .txt file (TSV format)
        df.to_csv(self.data_txt_path, sep='\t', index=False)
        # df.to_csv(self.sFile_txt_path, sep='\t', index=False)
        logging.info(f"Data saved to {self.data_txt_path}")
        
    def _make_sFile(self, df):
        """
        Save the scan data to sfile.txt.

        Args:
            df (pandas.DataFrame): DataFrame containing the scan data to be saved.
        """

        # Save as .txt file (TSV format)
        # df.to_csv(self.data_txt_path, sep='\t', index=False)
        df.to_csv(self.sFile_txt_path, sep='\t', index=False)
        logging.info(f"Data saved to {self.sFile_txt_path}")

    def dataframe_to_tdms(self, df):
        """
        Save the data from a DataFrame to a TDMS file.

        Args:
            df (pandas.DataFrame): DataFrame containing the data to be saved.

            output_path (Path): Path where the TDMS file will be saved.
        """
        # Initialize the TDMS writer

        tdms_writer = self.tdms_writer
        with tdms_writer:
            for column in df.columns:

                # Extract device name as the first word, handle special cases that don't conform
                # to that protocol first
                if column == 'Bin #' or column =='Elapsed Time':
                    device_name = column
                else:
                    device_name = column.split(' ', 1)[0]

                # print(device_name)
                # Extract the variable name by removing the device name and any leading space
                variable_name = column[len(device_name):].strip()

                # Remove alias information if "Alias:" appears in the variable name
                variable_name = variable_name.split(" Alias:", 1)[0].strip()
                variable_name = f'{device_name} {variable_name}'
                variable_name = variable_name.strip()
                # Get the data for this channel
                data = df[column].values
                # Create a ChannelObject and write it to the TDMS file
                ch_object = ChannelObject(device_name, variable_name, data)
                tdms_writer.write_segment([ch_object])

        logging.info(f"TDMS file written successfully.")

    def convert_to_dataframe(self, log_entries):
        """
        Convert the log entries into a pandas DataFrame. Log enetries generated by the
        data_logger are arranged in a different format than deisred. This method coerces
        the data in a more standard dataframe/tsv format with each variable as a header
        for a column.

        Args:
            log_entries (dict): Dictionary containing log entries, where keys are timestamps and values are data.

        Returns:
            pandas.DataFrame: DataFrame containing the logged data sorted by elapsed time.
        """

        log_df = pd.DataFrame.from_dict(log_entries, orient='index')
        log_df = log_df.sort_values(by='Elapsed Time').reset_index(drop=True)

        async_observables = self.device_manager.async_observables
        log_df = self.fill_async_nans(log_df, async_observables)

        # Modify the headers
        new_headers = self.modify_headers(log_df.columns)
        log_df.columns = new_headers

        log_df['Shotnumber'] = log_df.index + 1

        return log_df

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

        Returns:
            list: A list of modified headers with aliases or adjusted formats.
        """

        new_headers = []
        for header in headers:
            if ':' in header:
                device_name, variable = header.split(':')
                new_header = f"{device_name} {variable}"
                # Check if alias exists
                alias = device_dict.get(device_name, {}).get(variable, {}).get('alias')
                if alias:
                    new_header = f"{new_header} Alias:{alias}"
            else:
                new_header = header  # Keep the header as is if no colon
            new_headers.append(new_header)
        return new_headers

    def _process_results(self, results):

        """
        Process and save scan results to multiple formats.

        This method converts the results of a scan into a DataFrame, saves the data
        to both text and HDF5 formats, and writes the data to TDMS files.

        Args:
            results (dict): Dictionary containing scan results with timestamps and data.

        Returns:
            pandas.DataFrame: A DataFrame containing the processed scan data.
            If no data is collected, returns an empty DataFrame.
        """

        if results:
            log_df = self.convert_to_dataframe(results)
            logging.info("Data logging complete. Returning DataFrame.")

            # Save results to .txt and .h5
            self.save_to_txt_and_h5(log_df)

            # Write TDMS files (data and index)
            self.dataframe_to_tdms(log_df)

            return log_df
        else:
            logging.warning("No data was collected during the logging period.")
            return pd.DataFrame()

    def fill_async_nans(self, log_df, async_observables, fill_value=0):
        """
        Fill NaN values and empty strings in asynchronous observable columns with the most recent non-NaN value.
        If a column starts with NaN or empty strings, it will backfill with the first non-NaN value.
        After forward and backward filling, remaining NaN or empty entries are filled with `fill_value`. The
        back/front filling is meant to

        Args:
            log_df (pd.DataFrame): The DataFrame containing the logged data.
            async_observables (list): A list of asynchronous observables (columns) to process.
            fill_value (int, float): Value to fill remaining NaN and empty entries (default is 0).

        Returns:
            pandas.DataFrame: DataFrame with NaN values filled.
        """

        # Convert empty strings ('') to NaN
        log_df.replace('', pd.NA, inplace=True)

        for async_obs in async_observables:
            if async_obs in log_df.columns:
                # Only process if the entire column is not NaN
                if not log_df[async_obs].isna().all():
                    # First, apply forward fill (ffill) to propagate the last known value
                    log_df[async_obs] = log_df[async_obs].ffill()

                    # Then, apply backward fill (bfill) to fill leading NaNs with the first non-NaN value
                    log_df[async_obs] = log_df[async_obs].bfill()
                else:
                    logging.warning(f"Column {async_obs} consists entirely of NaN values and will be left unchanged.")

        # Finally, fill any remaining NaN values (including converted empty strings) with the specified fill_value
        log_df = log_df.fillna(fill_value)

        # Use infer_objects to downcast the object dtype arrays appropriately
        log_df = log_df.infer_objects(copy=False)
        logging.info(f"Filled remaining NaN and empty values with {fill_value}.")
        return log_df
    
    def process_and_rename(self, scan_number: Optional[int] = None) -> None:
        """
        Process the device directories and rename files based on timestamps and scan data.
        The device type is dynamically determined from the directory name, and the appropriate
        timestamp extraction method is used based on the device type.

        Args:
            scan_number (Optional[int]): Specific scan number to process. If None, uses 
                the scan number from `self.scan_number_int`.
        """

        if self.scan_data is None:
            logging.error("Called 'process_and_rename()' before 'scan_data' is set.")
            return

        directory_path = self.scan_data.get_folder()

        logging.info(f"Processing scan folder: {directory_path}")

        # Exclude directories with certain suffixes
        device_directories = [
            d for d in directory_path.iterdir() if d.is_dir() and not any(d.name.endswith(suffix) for suffix in self.DEPENDENT_SUFFIXES)
        ]

        # Load scan data  # TODO what happens if scan_number is given but scan_data is None?
        scan_num = self.scan_number_int if self.scan_number_int is not None else scan_number
        if scan_num is None:
            logging.error("Scan number is not provided.")
            return

        scan_folder_string = f"Scan{scan_num:03}"
        
        # sPath = self.scan_data.get_analysis_folder().parent / f's{scan_num}.txt'
        
        sDataPath = self.scan_data.get_folder() / f'ScanData{scan_folder_string}.txt'

        logging.info(f"Loading scan data from: {sDataPath}")

        try:
            df = pd.read_csv(sDataPath, sep='\t')
        except FileNotFoundError:
            logging.error(f"Scan data file {sDataPath} not found.")
            return

        max_workers = os.cpu_count() * 2  # Adjust multiplier based on your workload

        # Process each device directory concurrently using threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_device_files, device_dir, df, scan_folder_string) for device_dir in device_directories]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error during file processing: {e}")

    def process_device_files(self, device_dir: Path, df: pd.DataFrame, scan_number: str) -> None:
        """
        Generic method to process device files and rename them based on timestamps.

        Args:
            device_dir (Path): Path to the device directory.
            df (pd.DataFrame): DataFrame containing scan data.
            scan_number (str): Scan number in string format (e.g., 'Scan001').
        """
        device_name = device_dir.name
        logging.info(f"Processing device directory: {device_name}")

        # Get the device type and extract relevant data
        device_type = GeecsDatabase.find_device_type(device_name)
        if not device_type:
            logging.warning(f"Device type for {device_name} not found. Skipping.")
            return
            
        # Handle special case for FROG device type
        if device_type == "FROG":
            device_dir = device_dir.with_name(f"{device_dir.name}-Temporal")
            logging.info(f"Adjusted path for FROG device: {device_dir}")

        # Collect and match files with timestamps
        # device_files = sorted(device_dir.glob("*"), key=lambda x: int(str(x).split('_')[-1].split('.')[0]))
        device_files = sorted(device_dir.glob("*"), key=lambda x: int(x.stem.split('_')[-1]))
        logging.info(f"sorted device files to rename: {device_files}")
        matched_rows = self.process_and_match_files(device_files, df, device_name, device_type)

        # Rename master and dependent files
        self.rename_files(matched_rows, scan_number, device_name)
        self.process_dependent_directories(device_name, device_dir, matched_rows, scan_number)

    def process_and_match_files(
        self, device_files: List[Path], df: pd.DataFrame, device_name: str, device_type: str
    ) -> List[Tuple[Path, int]]:
        """
        Match device files with timestamps from the DataFrame.

        Args:
            device_files (List[Path]): List of device files to process.
            df (pd.DataFrame): DataFrame containing scan data.
            device_name (str): Name of the device.
            device_type (str): Type of the device.

        Returns:
            List[Tuple[Path, int]]: List of matched files and their corresponding row indices.
        """
        matched_rows = []
        device_timestamp_column = f'{device_name} timestamp'

        if device_timestamp_column not in df.columns:
            logging.warning(f"No matching timestamp column for {device_name} in scan data.")
            return matched_rows

        tolerance = 1
        rounded_df_timestamps = df[device_timestamp_column].round(tolerance)
        logging.info(f'rounded timestamps extracted from sFile: {rounded_df_timestamps}')
        for device_file in device_files:
            try:
                file_timestamp = self.extract_timestamp_from_file(device_file, device_type)
                file_timestamp_rounded = round(file_timestamp, tolerance)
                logging.info(f'rounded timestamp extracted for {device_file}: {file_timestamp_rounded}')
                match = rounded_df_timestamps[rounded_df_timestamps == file_timestamp_rounded]
                if not match.empty:
                    matched_rows.append((device_file, match.index[0]))
                    logging.info(f"Matched file {device_file} with row {match.index[0]}")
                else:
                    logging.warning(f"No match for {device_file} with timestamp {file_timestamp_rounded}")
            except Exception as e:
                logging.error(f"Error extracting timestamp from {device_file}: {e}")

        return matched_rows

    def extract_timestamp_from_file(self, device_file: Path, device_type: str) -> float:
        """
        Extract timestamp from a device file based on its type.

        Args:
            device_file (Path): Path to the device file.
            device_type (str): Type of the device.

        Returns:
            float: Extracted timestamp.
        """
        device_map = {
            "Point Grey Camera": get_imaq_timestamp_from_png,
            "MagSpecCamera": get_imaq_timestamp_from_png,
            "PicoscopeV2": get_picoscopeV2_timestamp,
            "MagSpecStitcher": get_custom_imaq_timestamp,
            "FROG": get_custom_imaq_timestamp,
        }

        if device_type in device_map:
            return device_map[device_type](device_file)
        else:
            raise ValueError(f"Unsupported device type: {device_type}")

    def process_dependent_directories(
        self, device_name: str, device_dir: Path, matched_rows: List[Tuple[Path, int]], scan_number: str
    ) -> None:
        """
        Process and rename files in dependent directories.

        Args:
            device_name (str): Name of the device.
            device_dir (Path): Path to the master device directory.
            matched_rows (List[Tuple[Path, int]]): List of matched master files and their indices.
            scan_number (str): Scan number in string format (e.g., 'Scan001').
        """
        for suffix in self.DEPENDENT_SUFFIXES:
            dependent_dir = device_dir.parent / f"{device_name}{suffix}"
            if dependent_dir.exists() and dependent_dir.is_dir():
                logging.info(f"Processing dependent directory: {dependent_dir}")
                dependent_files = sorted(dependent_dir.glob("*"), key=lambda x: int(str(x).split('_')[-1].split('.')[0]))
                self.rename_files_in_dependent_directory(dependent_files, matched_rows, scan_number, device_name, suffix)

    def rename_files_in_dependent_directory(
        self, dependent_files: List[Path], matched_rows: List[Tuple[Path, int]], scan_number: str, device_name: str, suffix: str
    ) -> None:
        """
        Rename dependent files based on matched rows from the master directory.

        Args:
            dependent_files (List[Path]): List of dependent files.
            matched_rows (List[Tuple[Path, int]]): List of matched master files and their indices.
            scan_number (str): Scan number in string format (e.g., 'Scan001').
            device_name (str): Name of the device.
            suffix (str): Suffix for the dependent directory.
        """
        for i, (master_file, row_index) in enumerate(matched_rows):
            if i < len(dependent_files):
                dependent_file = dependent_files[i]
                new_name = f"{scan_number}_{device_name}{suffix}_{str(row_index + 1).zfill(3)}{dependent_file.suffix}"
                new_path = dependent_file.parent / new_name
                dependent_file.rename(new_path)
                logging.info(f"Renamed {dependent_file} to {new_path}")
            else:
                logging.warning(f"Not enough files in dependent directory to match {master_file}")

    def rename_files(self, matched_rows: List[Tuple[Path, int]], scan_number: str, device_name: str) -> None:
        """
        Rename master files based on scan number, device name, and matched row index.

        Args:
            matched_rows (List[Tuple[Path, int]]): List of matched files and their indices.
            scan_number (str): Scan number in string format (e.g., 'Scan001').
            device_name (str): Name of the device.
        """
        for file_path, row_index in matched_rows:
            row_number = str(row_index + 1).zfill(3)
            new_file_name = f"{scan_number}_{device_name}_{row_number}{file_path.suffix}"
            new_file_path = file_path.parent / new_file_name

            if not new_file_path.exists():  # Check if the target file already exists
                logging.info(f"Renaming {file_path} to {new_file_path}")
                file_path.rename(new_file_path)

class ScanManager:
    """
    Manages the execution of scans, including configuration, device control,
    and data logging. This class handles the interaction between devices,
    data acquisition, and scanning logic. A 'save_devices' config file should be passed
    to the device_manager to initialize the desired saving configuration.
    """
    
    def __init__(self, experiment_dir=None, device_manager=None, shot_control_device="", MC_ip = None, scan_data=None):
        """
        Initialize the ScanManager and its components.

        Args:
            experiment_dir (str, optional): Directory where experiment data is stored.
            device_manager (DeviceManager, optional): DeviceManager instance for managing devices.
            shot_control_device (str, optional): GEECS Device that controls the shot timing
        """
        self.device_manager = device_manager or DeviceManager(experiment_dir=experiment_dir)
        self.action_manager = ActionManager(experiment_dir=experiment_dir)
        self.MC_ip = MC_ip
        
        # Initialize ScanDataManager with device_manager and scan_data
        self.scan_data_manager = ScanDataManager(self.device_manager, scan_data)

        self.data_logger = DataLogger(experiment_dir, self.device_manager)  # Initialize DataLogger
        self.save_data = True

        self.shot_control = GeecsDevice(shot_control_device)
        self.results = {}  # Store results for later processing

        self.stop_scanning_thread_event = threading.Event()  # Event to signal the logging thread to stop

        # Use the ConsoleLogger class
        self.console_logger = ConsoleLogger(log_file="scan_execution.log", level=logging.INFO, console=True)
        self.console_logger.setup_logging()

        self.virtual_variable_list = []
        self.virtual_variable_name = None

        self.acquisition_time = 0

        self.scanning_thread = None  # NEW: Separate thread for scanning

        self.scan_step_start_time = 0
        self.scan_step_end_time = 0


        self.initial_state = None
        self.scan_steps = []  # To store the precomputed scan steps

        self.pause_scan_event = threading.Event()  # Event to handle scan pausing
        self.pause_scan_event.set()  # Set to 'running' by default
        self.pause_time = 0
        
        self.enable_live_ECS_dump(client_ip = self.MC_ip)

    def pause_scan(self):
        """Pause the scanning process by clearing the pause event."""
        if self.pause_scan_event.is_set():
            self.pause_scan_event.clear()
            logging.info("Scanning paused.")

    def resume_scan(self):
        """Resume the scanning process by setting the pause event."""
        if not self.pause_scan_event.is_set():
            self.pause_scan_event.set()
            logging.info("Scanning resumed.")

    def get_database_dict(self):
        """TODO This should probably be a class variable"""
        return device_dict

    def reinitialize(self, config_path=None, config_dictionary=None):
        """
        Reinitialize the ScanManager with new configurations and reset the logging system.

        Args:
            config_path (str, optional): Path to the configuration file.
            config_dictionary (dict, optional): Dictionary containing configuration settings.
        """

        self.initial_state = None
        self.device_manager.reinitialize(config_path=config_path, config_dictionary=config_dictionary)
        self.data_logger.reinitialize_sound_player()
        self.data_logger.last_log_time_sync = {}
        self.console_logger.stop_logging()
        self.console_logger.setup_logging()

    def _set_trigger(self, state: str, amplitude: float):
        """
        Set the trigger state and amplitude.  # TODO make a "laser OFF" mode that has 'on': 'Internal'

        Args:
            state (str): Either 'on' or 'off' to control the trigger.
            amplitude (float): The amplitude value to set for the trigger.
        """

        valid_states = {
            'on': 'External rising edges',
            # 'on': 'Internal',
            'off': 'Single shot external rising edges'
        }
        if state in valid_states:
            self.shot_control.set('Trigger.Source', valid_states[state])
            self.shot_control.set('Amplitude.Ch AB', amplitude)
            logging.info(f"Trigger turned {state} with amplitude {amplitude}.")
        else:
            logging.error(f"Invalid trigger state: {state}")

    def trigger_off(self):
        """Turns off the trigger and sets the amplitude to 0.5."""
        self._set_trigger('off', 0.5)

    def trigger_on(self):
        """Turns on the trigger and sets the amplitude to 4.0."""
        self._set_trigger('on', 4.0)

    def is_scanning_active(self):
        """
        Check if a scan is currently active.

        Returns:
            bool: True if scanning is active, False otherwise.
        """

        return self.scanning_thread and self.scanning_thread.is_alive()

    def start_scan_thread(self, scan_config=None):
        """
        Start a new scan in a separate thread. Having it in a separate thread allows it to be interupted
        externally using the stop_scan_thread method

        Args:
            scan_config (dict, optional): Configuration settings for the scan, including variables, start, end, step, and wait times.
        """
        if self.is_scanning_active():
            logging.warning("scanning is already active, cannot start a new scan session.")
            return

        # Ensure the stop event is cleared before starting a new session
        self.stop_scanning_thread_event.clear()

        # Start a new thread for logging
        logging.info(f'scan config is this: {scan_config}')
        self.scanning_thread = threading.Thread(target=self.start_scan, args=(scan_config,))
        self.scanning_thread.start()
        logging.info("Scan thread started.")

    def stop_scanning_thread(self):
        """
        Stop the currently running scanning thread and clean up resources.
        """

        if not self.is_scanning_active():
            logging.warning("No active scanning thread to stop.")
            return

        # Stop the scan and wait for the thread to finish
        logging.info("Stopping the scanning thread...")

        # Set the event to signal the logging loop to stop
        logging.info("Stopping the scanning thread...")
        self.stop_scanning_thread_event.set()

        self.scanning_thread.join()  # Wait for the thread to finish
        self.scanning_thread = None  # Clean up the thread
        logging.info("scanning thread stopped and disposed.")

    def start_scan(self, scan_config):

        """
        Start a scan while dynamically performing actions
        or handling 'noscan' (or 'statistics') scans during the acquisition process.

        Args:
            scan_config (dict): Configuration for the scan, including device variables, start, end, step, and wait times.

        Returns:
            pandas.DataFrame: A DataFrame containing the results of the scan.
        """

        log_df = pd.DataFrame()  # Initialize in case of early exit

        try:
            # Pre-logging setup: Trigger devices off, initialize data paths, etc.
            logging.info(f'scan config getting sent to pre logging is this: {scan_config}')
            self.pre_logging_setup(scan_config)

            # Estimate acquisition time if necessary
            if scan_config:
                self.estimate_acquisition_time(scan_config)
                logging.info(f"Estimated acquisition time based on scan config: {self.acquisition_time} seconds.")

            # Start data logging
            if self.save_data:
                logging.info('add data saving here')
                self.results = self.data_logger.start_logging()
            else:
                logging.info('not doing any data saving')


            # start the acquisition loop
            self.scan_execution_loop()

        except Exception as e:
            logging.error(f"Error during scanning: {e}")

        logging.info("Stopping scanning.")
        time.sleep(1)
        log_df = self.stop_scan()

        return log_df  # Return the DataFrame with the logged data

    def stop_scan(self):
        """
        Stop the scan, save data, and restore the initial device states.

        Returns:
            pandas.DataFrame: DataFrame containing the saved scan data.
        """

        log_df = pd.DataFrame()

        if self.save_data:
            # Step 1: Stop data logging and handle device saving states
            self._stop_saving_devices()

        # Step 5: Restore the initial state of devices
        if self.initial_state is not None:
            self.restore_initial_state(self.initial_state)

        # Step 4: Turn the trigger back on
        self._set_trigger('on', 0.5)

        if self.device_manager.scan_closeout_action is not None:
            logging.info("Attempting to execute closeout actions.")
            logging.info(f"Action list {self.device_manager.scan_closeout_action}")

            self.action_manager.add_action({'closeout_action': self.device_manager.scan_closeout_action})
            self.action_manager.execute_action('closeout_action')

        if self.save_data:
            # Step 6: Process results, save to disk, and log data
            log_df = self.scan_data_manager._process_results(self.results)
            
            # Step 7: Process and rename data files
            self.scan_data_manager.process_and_rename()
            
            # Step 8: create sfile in analysis folder
            self.scan_data_manager._make_sFile(log_df)
            

        # Step 8: Stop the console logging
        self.console_logger.stop_logging()

        # Step 9: Move log file if data was saved (use paths from ScanDataManager)
        if self.save_data:
            # Access the path from ScanDataManager
            log_folder_path = Path(self.scan_data_manager.data_txt_path).parent
            self.console_logger.move_log_file(log_folder_path)

        self.scan_step_start_time = 0
        self.scan_step_end_time = 0
        self.data_logger.idle_time = 0

        # Step 10: Reset the device manager to clear up the current subscribers
        self.device_manager.reset()

        return log_df

    def _stop_saving_devices(self):
        """
        Stop the data logger and update the save paths for non-scalar devices.
        NOTE: this method should probably parsed with the device commands migrated elsewhere,
        maybe ScanDataManager or DeviceManager
        """

        # Stop data logging
        self.data_logger.stop_logging()

        # Handle device saving states
        for device_name in self.device_manager.non_scalar_saving_devices:
            device = self.device_manager.devices.get(device_name)
            if device:
                logging.info(f"Setting save to off for {device_name}")
                device.set('save', 'off', sync=False)
                logging.info(f"Setting save to off for {device_name} complete")
                device.set('localsavingpath', 'c:\\temp', sync=False)
                logging.info(f"Setting save path back to temp for {device_name} complete")
            else:
                logging.warning(f"Device {device_name} not found in DeviceManager.")

        time.sleep(2)  # Ensure asynchronous commands have time to finish
        logging.info("scanning has stopped for all devices.")

    def pre_logging_setup(self, scan_config):
        """
        Precompute all scan steps (including composite and normal variables),
        add scan devices to async_observables, and store the scan steps.
        Execute pre scan setup actions passed through

        Args:
            scan_config (dict): Configuration for the scan, including device variables and steps.
        """

        logging.info("Turning off the trigger.")
        self.trigger_off()

        time.sleep(2)

        if self.save_data:
            self.scan_data_manager.create_and_set_data_paths()
            self.scan_data_manager.write_scan_info_ini(scan_config)

        # Handle scan variables and ensure devices are initialized in DeviceManager
        logging.info(f'scan config in pre logging is this: {scan_config}')
        self.device_manager.handle_scan_variables(scan_config)

        time.sleep(1.5)

        device_var = scan_config['device_var']
        if not self.device_manager.is_statistic_noscan(device_var):
            self.initial_state = self.get_initial_state(scan_config)

        # Generate the scan steps
        self.scan_steps = self._generate_scan_steps(scan_config)

        if self.device_manager.scan_setup_action is not None:
            logging.info("Attempting to execute pre-scan actions.")
            logging.info(f"Action list {self.device_manager.scan_setup_action}")

            self.action_manager.add_action({'setup_action': self.device_manager.scan_setup_action})
            self.action_manager.execute_action('setup_action')
        
        logging.info(f'attempting to generate ECS live dump using {self.MC_ip}')
        if self.MC_ip is not None:
            logging.info(f'attempting to generate ECS live dump using {self.MC_ip}')            
            self.generate_live_ECS_dump(self.MC_ip)

        logging.info("Pre-logging setup completed.")
  
    def enable_live_ECS_dump(self, client_ip: str = '192.168.0.1'):
        steps = [
            "enable remote scan ECS dumps",
        ]
        
        for step in steps:
            success = self.shot_control.dev_udp.send_scan_cmd(step, client_ip=client_ip)
            time.sleep(.5)
            if not success:
                logging.warning(f"Failed to enable live ECS dumps on MC on computer: {client_ip}")
                break
    
    def generate_live_ECS_dump(self, client_ip: str = '192.168.0.1'):
        steps = [
            "Main: Check scans path>>None",
            "Save Live Expt Devices Configuration>>ScanStart"
        ]
        
        for step in steps:
            success = self.shot_control.dev_udp.send_scan_cmd(step, client_ip=client_ip)
            time.sleep(.5)
            if not success:
                logging.warning(f"Failed to generate an ECS live dump")
                break
    
    def _add_scan_devices_to_async_observables(self, scan_config):
        """
        Add the devices and variables involved in the scan to async_observables. This ensures their values
        are logged in the scan files

        Args:
            scan_config (dict): The scan configuration that includes the devices and variables.
        """

        device_var = scan_config['device_var']
        if self.device_manager.is_composite_variable(device_var):
            component_vars = self.device_manager.get_composite_components(device_var, scan_config['start'])
            for device_var in component_vars:
                if device_var not in self.device_manager.async_observables:
                    self.device_manager.async_observables.append(device_var)
        else:
            if device_var not in self.device_manager.async_observables:
                self.device_manager.async_observables.append(device_var)

        logging.info(f"Updated async_observables: {self.device_manager.async_observables}")

    def _generate_scan_steps(self, scan_config):
        """
        Generate the scan steps ahead of time, handling both normal and composite variables.

        Args:
            scan_config (dict): Configuration for the scan, including variables, start, end, step, and wait times.

        Returns:
            list: A list of scan steps, each containing the variables and their corresponding values.
        """

        self.data_logger.bin_num = 0
        steps = []

        device_var = scan_config['device_var']

        if self.device_manager.is_statistic_noscan(device_var):
            steps.append({
                'variables': device_var,
                'wait_time': scan_config.get('wait_time', 1),
                'is_composite': False
            })
        elif self.device_manager.is_composite_variable(device_var):
            self.data_logger.virtual_variable_name = device_var
            current_value = scan_config['start']
            while current_value <= scan_config['end']:
                component_vars = self.device_manager.get_composite_components(device_var, current_value)
                steps.append({
                    'variables': component_vars,
                    'wait_time': scan_config.get('wait_time', 1),
                    'is_composite': True
                })
                self.virtual_variable_list.append(current_value)
                current_value += scan_config['step']

        else:
            current_value = scan_config['start']
            while current_value <= scan_config['end']:
                steps.append({
                    'variables': {device_var: current_value},
                    'wait_time': scan_config.get('wait_time', 1),
                    'is_composite': False
                })
                current_value += scan_config['step']

        relative_flag = False
        # Check if it's a composite variable and if it has a relative flag in the YAML
        if self.device_manager.is_composite_variable(device_var):
            composite_variable_info = self.device_manager.composite_variables.get(device_var, {})
            relative_flag = composite_variable_info.get('relative', False)
            # self._apply_relative_adjustment(steps)

        # set relative flag for normal variables
        elif scan_config.get('relative', False):
            relative_flag = True

        if relative_flag and not self.device_manager.is_statistic_noscan(device_var):
            self._apply_relative_adjustment(steps)

        return steps

    def _apply_relative_adjustment(self, scan_steps):

        """
        Adjust the scan steps based on the initial state of the devices if relative scanning is enabled.
        This enables an easy way to start from a current state scan around it.

        Args:
            scan_steps (list): List of scan steps to be adjusted.
        """

        for step in scan_steps:
            for device_var, value in step['variables'].items():
                # Add the initial state value to the step value for each device
                if device_var in self.initial_state:
                    initial_value = self.initial_state[device_var]['value']
                    step['variables'][device_var] += initial_value
                else:
                    logging.warning(f"Initial state for {device_var} not found, skipping relative adjustment.")

    def scan_execution_loop(self):

        """
        Execute the precomputed scan steps in a loop, stopping if the stop event is triggered.

        Returns:
            pandas.DataFrame: A DataFrame containing the logged data from the scan.
        """

        log_df = pd.DataFrame()  # Initialize in case of early exit

        counter = 0
        while self.scan_steps:
            # Check if the stop event is set, and exit if so
            if self.stop_scanning_thread_event.is_set():
                logging.info("Scanning has been stopped externally.")
                break
            scan_step = self.scan_steps.pop(0)
            self._execute_step(scan_step['variables'], scan_step['wait_time'], scan_step['is_composite'])
            counter+=1

        logging.info("Stopping logging.")

        return log_df

    def _execute_step(self, component_vars, wait_time, is_composite, max_retries=3, retry_delay=0.5):
        """
        Execute a single step of the scan, handling both composite and normal variables.

        Args:
            component_vars (dict): Dictionary of variables and their values for the scan step.
            wait_time (float): The time to wait after devices have been changed. This is the acquisition time effectively.
            is_composite (bool): Flag indicating whether the step involves composite variables.
            max_retries (int): Maximum number of retries if setting the value is outside the tolerance.
            retry_delay (float): Delay in seconds between retries.
        """
        logging.info("Pausing logging. Turning trigger off before moving devices.")
        if self.data_logger.virtual_variable_name is not None:
            self.data_logger.virtual_variable_value = self.virtual_variable_list[self.data_logger.bin_num]
            logging.info(f"updating virtual value in data_logger from scan_manager to: {self.data_logger.virtual_variable_value}.")

        self.data_logger.bin_num += 1

        self.trigger_off()

        logging.info(f"shot control state: {self.shot_control.state}")

        if not self.device_manager.is_statistic_noscan(component_vars):
            for device_var, set_val in component_vars.items():
                device_name, var_name = device_var.split(':', 1)
                device = self.device_manager.devices.get(device_name)

                if device:
                    # Retrieve the tolerance for the variable
                    # TODO Better error handling when tolerance not defined in database editor
                    tol = float(GeecsDevice.exp_info['devices'][device_name][var_name]['tolerance'])

                    # Retry logic for setting device value
                    success = False
                    attempt = 0

                    while attempt < max_retries:
                        ret_val = device.set(var_name, set_val)  # Send the command to set the value
                        logging.info(f"Attempt {attempt + 1}: Setting {var_name} to {set_val} on {device_name}, returned {ret_val}")

                        # Check if the return value is within tolerance
                        if ret_val - tol <= set_val <= ret_val + tol:
                            logging.info(f"Success: {var_name} set to {ret_val} (within tolerance {tol}) on {device_name}")
                            success = True
                            break
                        else:
                            logging.warning(f"Attempt {attempt + 1}: {var_name} on {device_name} not within tolerance ({ret_val} != {set_val})")
                            attempt += 1
                            time.sleep(retry_delay)  # Wait before retrying

                    if not success:
                        logging.error(f"Failed to set {var_name} on {device_name} after {max_retries} attempts")
                else:
                    logging.warning(f"Device {device_name} not found in device manager.")

        logging.info("Resuming logging. Turning trigger on after all devices have been moved.")

        self.trigger_on()

        #code below used with data_logger to determine if a device is non responsive
        self.scan_step_start_time = time.time()
        self.data_logger.data_recording = True

        # first step of the scan, self.scan_step_end_time is equal to zero.
        # after that, it gets updated. As does the self.scan_step_start_time.
        # idle time should be the time between then end of the "previous" step
        # and the start of the next step.
        if self.scan_step_end_time > 0:
            self.data_logger.idle_time = self.scan_step_start_time - self.scan_step_end_time + self.pause_time
            logging.info(f'idle time between scan steps: {self.data_logger.idle_time}')

        logging.info(f"shot control state: {self.shot_control.state}")

        # Wait for acquisition time (or until scanning is externally stopped)
        current_time = 0
        start_time = time.time()
        interval_time = 0.1
        self.pause_time = 0
        while current_time < wait_time:
            if self.stop_scanning_thread_event.is_set():
                logging.info("Scanning has been stopped externally.")
                break

            # Check if the scan is paused, and wait until it’s resumed
            if not self.pause_scan_event.is_set():
                self.trigger_off()
                self.data_logger.data_recording = False
                t0 = time.time()
                logging.info("Scan is paused, waiting to resume...")
                self.pause_scan_event.wait()  # Blocks until the event is set (i.e., resumed)
                self.pause_time = time.time() - t0
                self.trigger_on()
                self.data_logger.data_recording = True


            time.sleep(interval_time)
            current_time = time.time() - start_time

            # TODO a few things here:
            # TODO 1. make `save_on_shot` a settable flag from the GUI so this is an optional process
            # TODO 2. ensure that we are only appending data rather than overwriting the file every shot.
            #  This could be done by storing the previously-written dataframe and subtracting the new one off of it
            save_on_shot = False
            if save_on_shot:
                if current_time % 1 < interval_time:
                    log_df = self.scan_data_manager.convert_to_dataframe(self.results)
                    self.scan_data_manager.dataframe_to_tdms(log_df)

        # Turn trigger off after waiting
        self.trigger_off()

        self.scan_step_end_time = time.time()
        self.data_logger.data_recording = False

        logging.info(f"shot control state: {self.shot_control.state}")

    def estimate_acquisition_time(self, scan_config):

        """
        Estimate the total acquisition time based on the scan configuration.

        Args:
            scan_config (dict): Configuration for the scan, including start, end, step, and wait times.
        """

        total_time = 0

        if self.device_manager.is_statistic_noscan(scan_config['device_var']):
            total_time += scan_config.get('wait_time', 1) - 0.5  # Default to 1 seconds if not provided
        else:
            start = scan_config['start']
            end = scan_config['end']
            step = scan_config['step']
            wait_time = scan_config.get('wait_time', 1)# - 0.5  # Default wait time between steps is 1 second

            # Calculate the number of steps and the total time for this device
            steps = ((end - start) / step) + 1
            total_time += steps * wait_time

        logging.info(f'Estimated scan time: {total_time}')

        self.acquisition_time = total_time

    def estimate_current_completion(self):
        """
        Estimate the current completion percentage of the ongoing scan.

        Returns:
            float: A percentage value (between 0.0 and 1.0) indicating the progress of the scan.
        """

        if self.acquisition_time == 0:
            return 0
        completion = self.data_logger.get_current_shot()/self.acquisition_time
        return 1 if completion > 1 else completion

    def get_initial_state(self, scan_config):

        """
        Retrieve the initial state of the devices involved in the scan.

        Args:
            scan_config (dict): Configuration for the scan, specifying the devices and variables.

        Returns:
            dict: A dictionary mapping each device variable to its initial state.
        """

        scan_variables = []

        if scan_config:
            device_var = scan_config['device_var']

            # Handle composite variables by retrieving each component's state
            if self.device_manager.is_composite_variable(device_var):
                component_vars = self.device_manager.get_composite_components(device_var, scan_config['start'])
                scan_variables.extend(component_vars.keys())
            else:
                scan_variables.append(device_var)

        logging.info(f"Scan variables for initial state: {scan_variables}")
        # Use the existing method to get the current state of all variables
        initial_state = self.device_manager.get_values(scan_variables)

        logging.info(f"Initial scan variable state: {initial_state}")
        return initial_state

    def restore_initial_state(self, initial_state):

        """
        Restore the devices to their initial states after the scan has completed.
        NOTE: this could be used more generally to restore any kind of state

        Args:
            initial_state (dict): A dictionary containing the initial states of the devices.
        """

        for device_var, value_dict in initial_state.items():
            device_name, var_name = device_var.split(':')
            device = self.device_manager.devices.get(device_name)

            if device:
                initial_value = value_dict['value']
                device.set(var_name, initial_value)
                logging.info(f"Restored {device_name}:{var_name} to initial value {initial_value}")
            else:
                logging.warning(f"Device {device_name} not found when trying to restore state.")

        logging.info("All devices restored to their initial states.")



if __name__ == '__main__':
    print("testing")
    module_and_class = 'scan_analysis.analyzers.Undulator.CameraImageAnalysis.CameraImageAnalysis'
    if module_and_class:
        module_name, class_name = module_and_class.rsplit('.', 1)
        print(module_name, class_name)
        module = importlib.import_module(module_name)
        analysis_class = getattr(module, class_name)
        print(analysis_class)