from __future__ import annotations
from typing import Optional, List, Tuple

# Standard library imports
import os
import time
import logging
from pathlib import Path
import concurrent.futures

# Third-party library imports
import pandas as pd
from nptdms import TdmsWriter, ChannelObject

# Internal project imports
from . import DeviceManager, DatabaseDictLookup

from geecs_python_api.controls.interface import GeecsDatabase
from geecs_python_api.analysis.scans.scan_data import ScanData

from image_analysis.utils import get_imaq_timestamp_from_png, get_picoscopeV2_timestamp, get_custom_imaq_timestamp, get_himg_timestamp


class ScanDataManager:
    """
    Manages data preparation, organization and saving during and after a scan.

    This class is responsible for setting up data paths, initializing writers for different formats
    (e.g., TDMS, HDF5), and handling the saving and processing of scan data. It works alongside
    DeviceManager and ScanData to ensure all relevant data is logged and stored appropriately.
    This class is designed to be used primarily (or even exclusively) with the ScanMananger
    """

    DEPENDENT_SUFFIXES = ["-interp", "-interpSpec", "-interpDiv", "-Spatial"]

    def __init__(self, device_manager: DeviceManager,
                 scan_data: Optional[ScanData] = None,
                 database_dict: Optional[DatabaseDictLookup] = None):
        """
        Initialize the ScanDataManager with references to the ScanData and DeviceManager.

        Args:
            device_manager (DeviceManager): Manages the devices involved in the scan.
            scan_data (ScanData): manages scan data
            database_dict (DatabaseDictLookup): contains a dictionary of all experiment devices and variables
        """
        self.device_manager = device_manager  # Explicitly pass device_manager
        self.scan_data = scan_data
        if database_dict is None:
            database_dict = DatabaseDictLookup().reload()
        self.database_dict = database_dict
        self.tdms_writer = None
        self.data_txt_path = None
        self.data_h5_path = None
        self.sFile_txt_path = None

        self.device_save_paths_mapping = {}
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

        ScanData.reload_paths_config()
        # if not ScanData.paths_config.is_default_server_address():
        #     raise NotADirectoryError("Unable to locate server address for saving data, unable to set paths")

        switch_paths = False
        
        self.scan_data = ScanData.build_next_scan_data()
        
        if not ScanData.paths_config.is_default_server_address():
            # raise NotADirectoryError("Unable to locate server address for saving data, unable to set paths")

            # something changed about how paths were working, which breaks how this part works on
            # non control room computers. Added this switch paths bit to make a solution

            # Define old and new root paths
            old_root = ScanData.paths_config.base_path
            new_root = ScanData.paths_config.get_default_server_address('Undulator')

            switch_paths = True
            
        data_path = self.scan_data.get_folder()
                
        for device_name in self.device_manager.non_scalar_saving_devices:
            logging.info(f'attemping to configure save paths for {device_name}')
            data_path = self.scan_data.get_folder() / device_name
            data_path.mkdir(parents=True, exist_ok=True)
            target_dir = data_path

            if switch_paths:
                data_path = new_root / data_path.relative_to(old_root)
            
            device = self.device_manager.devices.get(device_name)
            logging.info(f'device is {device}')
            if device:
                dev_host_ip_string = device.dev_ip
                path_str = f'//{dev_host_ip_string}/SharedData/{device_name}'

                source_dir = Path(path_str)

                data_path_client_side = Path('C:\\SharedData') / device_name
                #purge directory before starting
                for item in source_dir.iterdir():
                    if item.is_file():
                        item.unlink()

                logging.info(f'creating save path for {device_name}')

                save_path = str(data_path_client_side).replace('/', "\\")

                logging.info(f'save path created {save_path}')

                logging.info(f"Setting save data path for {device_name} to {save_path}")
                device.set("localsavingpath", save_path, sync=False)
                time.sleep(.1)
                device.set('save', 'on', sync=False)
            else:
                logging.warning(f"Device {device_name} not found in DeviceManager.")

            device_type = GeecsDatabase.find_device_type(device_name)

            # creating a dict here which contains all of the necessary information to
            # move and rename a saved file from a local directory to thec correct
            # scans directory on the data server

            self.device_save_paths_mapping[device_name] = {
                "target_dir": target_dir,
                "source_dir": source_dir,
                "device_type": device_type
            }

        analysis_save_path = self.scan_data.get_analysis_folder()

        self.parsed_scan_string = self.scan_data.get_folder().parts[-1]
        self.scan_number_int = int(self.parsed_scan_string[-3:])

        self.tdms_output_path = self.scan_data.get_folder() / f"{self.parsed_scan_string}.tdms"
        self.data_txt_path = self.scan_data.get_folder() / f"ScanData{self.parsed_scan_string}.txt"
        self.data_h5_path = self.scan_data.get_folder() / f"ScanData{self.parsed_scan_string}.h5"

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
        self.tdms_writer = TdmsWriter(tdms_output_path, index_file=True)
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
            f"ScanEndInfo = \"\"\n",
            f"Background = \"{scan_config.get('background', 'False')}\""
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
                if column == 'Bin #' or column == 'Elapsed Time':
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
                device_dict = self.database_dict.get_database()
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
            d for d in directory_path.iterdir() if
            d.is_dir() and not any(d.name.endswith(suffix) for suffix in self.DEPENDENT_SUFFIXES)
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
        
        try:
            # Process each device directory concurrently using threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.process_device_files, device_dir, df, scan_folder_string) for device_dir in
                           device_directories]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"Error during file processing: {e}")
        except Exception as e:
            logging.error(f'error during file renaming: {e}')

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
                # logging.info(f'rounded timestamp extracted for {device_file}: {file_timestamp_rounded}')
                match = rounded_df_timestamps[rounded_df_timestamps == file_timestamp_rounded]
                if not match.empty:
                    matched_rows.append((device_file, match.index[0]))
                    # logging.info(f"Matched file {device_file} with row {match.index[0]}")
                else:
                    logging.warning(f"No match for {device_file} with timestamp {file_timestamp_rounded}")
            except Exception as e:
                logging.error(f"Error extracting timestamp from {device_file}: {e}")

        return matched_rows

    def extract_timestamp_from_file(self, device_file: Path, device_type: str) -> float:
        """
        Extract timestamp from a device file based on its type.

        NOTE: deprecated, this method migrated to image_analysis.utils

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
            "HASO4_3": get_himg_timestamp,
            "Thorlabs CCS175 Spectrometer": get_picoscopeV2_timestamp,
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
                dependent_files = sorted(dependent_dir.glob("*"),
                                         key=lambda x: int(str(x).split('_')[-1].split('.')[0]))
                self.rename_files_in_dependent_directory(dependent_files, matched_rows, scan_number, device_name,
                                                         suffix)

    def rename_files_in_dependent_directory(
            self, dependent_files: List[Path], matched_rows: List[Tuple[Path, int]], scan_number: str, device_name: str,
            suffix: str
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
                # logging.info(f"Renamed {dependent_file} to {new_path}")
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
                # logging.info(f"Renaming {file_path} to {new_file_path}")
                file_path.rename(new_file_path)
