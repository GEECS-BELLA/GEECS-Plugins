from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Union

DeviceSavePaths = Dict[str, Dict[str, Union[Path,str]]]

# Standard library imports
import time
import logging
import shutil

# Third-party library imports
import pandas as pd
from nptdms import TdmsWriter, ChannelObject


# Internal project imports
from . import DeviceManager, DatabaseDictLookup

from geecs_python_api.controls.interface import GeecsDatabase
from geecs_python_api.analysis.scans.scan_data import ScanData

from image_analysis.utils import extract_timestamp_from_file

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

    def create_and_set_data_paths(self) -> ScanData:
        """
        Create data paths for devices that need non-scalar saving, and initialize the TDMS writers.

        This method sets up the necessary directories and paths for saving device data,
        then initializes the TDMS writers for logging scalar and non-scalar data.
        """

        ScanData.reload_paths_config()
        self.scan_data = ScanData.build_next_scan_data()
        
        # if not ScanData.paths_config.is_default_server_address():
        #     raise NotADirectoryError("Unable to locate server address for saving data, unable to set paths")

        for device_name in self.device_manager.non_scalar_saving_devices:
            logging.info(f'attemping to configure save paths for {device_name}')
            data_path = self.scan_data.get_folder() / device_name
            data_path.mkdir(parents=True, exist_ok=True)
            target_dir = data_path

            device = self.device_manager.devices.get(device_name)
            logging.info(f'device is {device}')

            if device:
                device_name = device.get_name()
                dev_host_ip_string = device.dev_ip
                source_dir = Path(f'//{dev_host_ip_string}/SharedData/{device_name}')

                self.purge_local_save_dir(source_dir)

                logging.info(f'creating save path for {device_name}')
                data_path_client_side = Path('C:\\SharedData') / device_name
                save_path = str(data_path_client_side).replace('/', "\\")

                logging.info(f'save path created {save_path}')

                logging.info(f"Setting save data path for {device_name} to {save_path}")
                device.set("localsavingpath", save_path, sync=False)
                time.sleep(.1)
                device.set('save', 'on', sync=False)

                device_type = GeecsDatabase.find_device_type(device_name)

                # creating a dict here which contains all of the necessary information to
                # move and rename a saved file from a local directory to thec correct
                # scans directory on the data server

                self.device_save_paths_mapping[device_name] = {
                    "target_dir": target_dir,
                    "source_dir": source_dir,
                    "device_type": device_type}

            else:
                logging.warning(f"Device {device_name} not found in DeviceManager.")

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

    def purge_all_local_save_dir(self) -> None:

        for device_name in self.device_manager.non_scalar_saving_devices:
            device = self.device_manager.devices.get(device_name)

            if device:
                device_name = device.get_name()
                dev_host_ip_string = device.dev_ip
                source_dir = Path(f'//{dev_host_ip_string}/SharedData/{device_name}')

                self.purge_local_save_dir(source_dir)

    @staticmethod
    def purge_local_save_dir(source_dir: Path) -> None:

        # Purge the source recursively (remove files only)
        if source_dir.exists():
            for item in source_dir.rglob('*'):
                if item.is_file():
                    try:
                        item.unlink()
                        logging.info(f"Removed file: {item}")
                    except Exception as e:
                        logging.error(f"Error removing {item}: {e}")

    def initialize_tdms_writers(self, tdms_output_path: str) -> None:
        """
        Initialize the TDMS writers for scalar data and index data.

        Args:
            tdms_output_path (str): Path to the TDMS file for saving scalar data.
        """
        self.tdms_writer = TdmsWriter(tdms_output_path, index_file=True)
        logging.info(f"TDMS writer initialized with path: {tdms_output_path}")

    def write_scan_info_ini(self, scan_config: DeviceSavePaths) -> None:
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

    def process_results(self, results):

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

    @staticmethod
    def fill_async_nans(log_df, async_observables, fill_value=0):
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

    def post_process_orphaned_files(self, log_df):
        """
        Process orphaned files for each device using the saved device mapping and logged timestamps.

        Args:
            log_df (pd.DataFrame): DataFrame with a column for each device's timestamp, e.g. "DeviceA timestamp".
        """
        for device_name, device_info in self.device_save_paths_mapping.items():
            # Extract the device-specific parameters from the mapping.
            source_dir = Path(device_info['source_dir'])
            target_dir = Path(device_info['target_dir'])
            device_type = device_info['device_type']
            # Get the logged timestamps for this device from the DataFrame.
            logged_timestamps = log_df[f'{device_name} timestamp'].tolist()

            # Call a helper function to post-process files for this device.
            self._post_process_device(source_dir, target_dir, device_name, device_type, logged_timestamps)

    def _post_process_device(self, source_dir: Path, target_dir: Path,
                             device_name: str, device_type: str,
                             logged_timestamps: list[float]):
        """
        For a given device, look for orphaned files in the source directory (and its subdirectories),
        extract their timestamp, match with logged timestamps, and then move them to the corresponding
        adjusted target directory (e.g. "Z:/data/Undulator/DeviceName-type1") using the appropriate naming convention.

        Args:
            source_dir (Path): The directory where the files are stored.
            target_dir (Path): The base directory for processed files.
            device_name (str): Name of the device.
            device_type (str): Type of the device.
            logged_timestamps (list[float]): List of timestamps logged for the device.
        """
        tolerance = 0.0011  # Adjust as needed

        # Look recursively for files that match the device name.
        files = [f for f in source_dir.rglob("*") if f.is_file() and device_name in f.name]
        for file in files:
            file_ts = extract_timestamp_from_file(file, device_type)
            logging.info(f"Post-processing file {file}: timestamp {file_ts}")

            match_found = False
            shot_index = None
            # Try to find a matching timestamp from the logged timestamps.
            for idx, ts in enumerate(logged_timestamps):
                if abs(file_ts - ts) < tolerance:
                    shot_index = idx + 1  # Determine shot index
                    match_found = True
                    break

            if match_found:
                # Use the file's parent directory name as the variant.
                variant = file.parent.name
                # Build the new file stem using the scan number, variant, and shot index.
                new_file_stem = self.rename_file(self.scan_number_int, variant, shot_index)
                new_filename = new_file_stem + file.suffix
                # Adjust the target directory: use target_dir.parent / variant.
                adjusted_target_dir = target_dir.parent / variant
                adjusted_target_dir.mkdir(parents=True, exist_ok=True)
                dest_file = adjusted_target_dir / new_filename

                try:
                    shutil.move(str(file), str(dest_file))
                    logging.info(f"Moved {file} to {dest_file}")
                except Exception as e:
                    logging.error(f"Error moving {file} to {dest_file}: {e}")
            else:
                logging.warning(f"No matching timestamp for file {file} (timestamp {file_ts})")

    # this method is copied directly from data_logger.py FileMover class. Maybe it should be made
    # a utility function of this project
    @staticmethod
    def rename_file(scan_number: int, device_name: str, shot_index: int) -> str:
        """
        Rename master files based on scan number, device name, and matched row index.

        Args:
            scan_number (str): Scan number in string format (e.g., 'Scan001').
            device_name (str): Name of the device.
            shot_index (int): shot number
        """

        scan_number_str = str(scan_number).zfill(3)
        shot_number_str = str(shot_index).zfill(3)
        file_name_stem = f'Scan{scan_number_str}_{device_name}_{shot_number_str}'

        return file_name_stem
