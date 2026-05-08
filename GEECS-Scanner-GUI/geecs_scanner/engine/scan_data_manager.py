"""Scan data path setup, format conversion, and file I/O for ScanManager."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
from nptdms import ChannelObject, TdmsWriter

from . import DatabaseDictLookup, DeviceManager
from geecs_data_utils import ScanConfig, ScanPaths
from geecs_python_api.controls.interface import GeecsDatabase
from geecs_scanner.engine.device_command_executor import DeviceCommandExecutor
from geecs_scanner.utils.exceptions import DataFileError, DeviceCommandError

logger = logging.getLogger(__name__)

DeviceSavePaths = Dict[str, Dict[str, Union[Path, str]]]


class ScanDataManager:
    """Set up scan output paths, convert results to DataFrame, and write data files.

    Attributes
    ----------
    device_manager : DeviceManager
    scan_paths : ScanPaths or None
    database_dict : DatabaseDictLookup
    tdms_writer : TdmsWriter or None
    data_txt_path : Path or None
    data_h5_path : Path or None
    sFile_txt_path : Path or None
    tdms_output_path : Path or None
    sFile_info_path : Path or None
    device_save_paths_mapping : DeviceSavePaths
    scan_number_int : int or None
    parsed_scan_string : str or None
    on_device_error : callable or None
        Injected by ScanManager after construction.
    """

    DEPENDENT_SUFFIXES = ["-interp", "-interpSpec", "-interpDiv", "-Spatial"]

    def __init__(
        self,
        device_manager: DeviceManager,
        scan_paths: Optional[ScanPaths] = None,
        database_dict: Optional[DatabaseDictLookup] = None,
    ):
        self.device_manager = device_manager
        self.scan_paths: ScanPaths = scan_paths
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
        self.on_device_error = None
        self.cmd_executor: Optional[DeviceCommandExecutor] = None
        self.scan_number_int: Optional[int] = None
        self.parsed_scan_string: Optional[str] = None

    def initialize_scan_data_and_output_files(self):
        """Allocate the next scan folder and set up all output file paths.

        Must be called before any data operations.
        """
        ScanPaths.reload_paths_config()
        self.scan_paths = ScanPaths.build_next_scan_data()

        self.parsed_scan_string = self.scan_paths.get_folder().parts[-1]
        self.scan_number_int = int(self.parsed_scan_string[-3:])

        folder = self.scan_paths.get_folder()
        analysis_folder = self.scan_paths.get_analysis_folder().parent

        self.tdms_output_path = folder / f"{self.parsed_scan_string}.tdms"
        self.data_txt_path = folder / f"ScanData{self.parsed_scan_string}.txt"
        self.data_h5_path = folder / f"ScanData{self.parsed_scan_string}.h5"
        self.sFile_txt_path = analysis_folder / f"s{self.scan_number_int}.txt"
        self.sFile_info_path = analysis_folder / f"s{self.scan_number_int}_info.txt"

        self.initialize_tdms_writers(str(self.tdms_output_path))
        time.sleep(1)

    def configure_device_save_paths(
        self,
        save_local: bool = True,
        stop_event: Optional[threading.Event] = None,
    ):
        """Set ``localsavingpath`` and ``save=on`` for every non-scalar device.

        Parameters
        ----------
        save_local : bool
            If True, point devices at their ``C:/SharedData`` folder (faster).
            If False, save directly into the scan folder.
        stop_event : threading.Event, optional
            If set, the pre-scan purge exits early and this method returns.
        """
        for device_name in self.device_manager.non_scalar_saving_devices:
            logger.debug("Configuring save paths for device: %s", device_name)
            target_dir = self.scan_paths.get_folder() / device_name
            target_dir.mkdir(parents=True, exist_ok=True)

            device = self.device_manager.devices.get(device_name)
            device_type = GeecsDatabase.find_device_type(device_name)

            if not device:
                logger.warning("Device %s not found in DeviceManager.", device_name)
                continue

            dev_host_ip_string = device.dev_ip
            if save_local:
                source_dir = Path(f"//{dev_host_ip_string}/SharedData/{device_name}")
                if not self.purge_local_save_dir(source_dir, stop_event=stop_event):
                    return
                data_path_client_side = Path("C:/SharedData") / device_name
            else:
                source_dir = target_dir
                data_path_client_side = target_dir

            save_path = str(data_path_client_side).replace("/", "\\")
            try:
                self.cmd_executor.set(device, "localsavingpath", save_path, sync=False)
                time.sleep(0.1)
                self.cmd_executor.set(device, "save", "on", sync=False)
            except DeviceCommandError as exc:
                abort = self.cmd_executor.escalate(exc)
                if abort:
                    return
                continue

            self.device_save_paths_mapping[device_name] = {
                "target_dir": target_dir,
                "source_dir": source_dir,
                "device_type": device_type,
            }

    def purge_all_local_save_dir(self) -> None:
        """Delete all files in each device's SharedData folder on its host."""
        for device_name in self.device_manager.non_scalar_saving_devices:
            device = self.device_manager.devices.get(device_name)
            if device:
                dev_name = device.get_name()
                dev_host_ip_string = device.dev_ip
                base_dir = Path(f"//{dev_host_ip_string}/SharedData")

                if base_dir.exists():
                    for subdir in base_dir.iterdir():
                        if subdir.is_dir() and dev_name in subdir.name:
                            logger.debug("Purging directory: %s", subdir)
                            self.purge_local_save_dir(subdir)
                else:
                    logger.warning("Base directory %s does not exist.", base_dir)

    @staticmethod
    def purge_local_save_dir(
        source_dir: Path,
        stop_event: Optional[threading.Event] = None,
    ) -> bool:
        """Delete all files under *source_dir*.

        Parameters
        ----------
        source_dir : Path
        stop_event : threading.Event, optional
            Checked between deletions. Returns False immediately when set.

        Returns
        -------
        bool
            True if completed; False if interrupted by *stop_event*.
        """
        if source_dir.exists():
            for item in source_dir.rglob("*"):
                if item.is_file():
                    if stop_event is not None and stop_event.is_set():
                        logger.debug(
                            "File purge interrupted by stop request at %s", item
                        )
                        return False
                    try:
                        item.unlink()
                        logger.debug("Removed file: %s", item)
                    except Exception:
                        logger.exception("Error removing %s", item)
        return True

    def initialize_tdms_writers(self, tdms_output_path: str) -> None:
        """Open a TdmsWriter at *tdms_output_path* (also writes an index file).

        Raises
        ------
        DataFileError
            If the file cannot be created.
        """
        try:
            self.tdms_writer = TdmsWriter(tdms_output_path, index_file=True)
            logger.info("TDMS writer initialized with path: %s", tdms_output_path)
        except (IOError, OSError) as exc:
            raise DataFileError(
                f"Failed to initialize TDMS writer at {tdms_output_path}"
            ) from exc

    def write_scan_info_ini(self, scan_config: ScanConfig) -> None:
        """Write ``ScanInfo{scan_folder}.ini`` with scan metadata."""
        scan_folder = self.parsed_scan_string
        scan_number = self.scan_number_int
        filename = f"ScanInfo{scan_folder}.ini"
        scan_var = scan_config.device_var or "Shotnumber"
        additional_description = scan_config.additional_description
        scan_info = f"{self.device_manager.scan_base_description}. scanning {scan_var}. {additional_description}"

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
            f"Background = {str(scan_config.background).lower()}\n",
            f'ScanMode = "{scan_config.scan_mode.value}"\n',
        ]

        full_path = Path(self.data_txt_path).parent / filename
        full_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Attempting to write to %s", full_path)

        with full_path.open("w") as configfile:
            for line in config_file_contents:
                configfile.write(line)

        logger.debug("Scan info written to %s", full_path)

    def save_to_txt_and_h5(self, df: pd.DataFrame) -> None:
        """Write *df* to ``ScanData{scan}.txt`` (TSV).

        Raises
        ------
        DataFileError
            On filesystem failure.
        """
        if df.empty:
            logger.warning("Attempting to save an empty DataFrame to text file")
            return

        try:
            df.to_csv(self.data_txt_path, sep="\t", index=False)
            logger.info("Scan data saved to %s", self.data_txt_path)
        except OSError as exc:
            raise DataFileError(
                f"Cannot write scan data to {self.data_txt_path}"
            ) from exc

    def _make_sFile(self, df: pd.DataFrame) -> None:
        """Write *df* to the analysis-ready ``s{N}.txt`` file.

        Raises
        ------
        DataFileError
            On filesystem failure.
        """
        if df.empty:
            logger.warning("Attempting to save an empty DataFrame to s-file")
            return

        try:
            df.to_csv(self.sFile_txt_path, sep="\t", index=False)
            logger.info("Analysis-ready data saved to %s", self.sFile_txt_path)
        except OSError as exc:
            raise DataFileError(
                f"Cannot write s-file to {self.sFile_txt_path}"
            ) from exc

    def dataframe_to_tdms(self, df: pd.DataFrame) -> None:
        """Write *df* to the scan TDMS file via the open TdmsWriter.

        Column format: ``"DeviceName VariableName"`` or
        ``"DeviceName VariableName Alias:alias"``.  Special columns
        ``"Bin #"`` and ``"Elapsed Time"`` are written as their own group.
        """
        try:
            tdms_writer = self.tdms_writer
            with tdms_writer:
                for column in df.columns:
                    if column in ("Bin #", "Elapsed Time"):
                        device_name = column
                    else:
                        device_name = column.split(" ", 1)[0]

                    variable_name = column[len(device_name) :].strip()
                    variable_name = variable_name.split(" Alias:", 1)[0].strip()
                    variable_name = f"{device_name} {variable_name}".strip()

                    data = df[column].values
                    ch_object = ChannelObject(device_name, variable_name, data)
                    tdms_writer.write_segment([ch_object])

            logger.info("TDMS file written successfully.")
        except (ValueError, TypeError, IOError) as e:
            logger.error("Error converting DataFrame to TDMS: %s", e)
            raise

    def convert_to_dataframe(self, log_entries: dict) -> pd.DataFrame:
        """Convert raw *log_entries* dict to a sorted, header-standardised DataFrame.

        Parameters
        ----------
        log_entries : dict
            ``{timestamp: {column: value, ...}, ...}`` from DataLogger.

        Returns
        -------
        pd.DataFrame
            Sorted by ``"Elapsed Time"``, async NaNs filled, ``Shotnumber`` added.
        """
        try:
            if not isinstance(log_entries, dict):
                raise TypeError("log_entries must be a dictionary")

            if not log_entries:
                logger.warning("Empty log entries provided")
                return pd.DataFrame()

            log_df = pd.DataFrame.from_dict(log_entries, orient="index")

            if "Elapsed Time" not in log_df.columns:
                logger.warning("No 'Elapsed Time' column found. Using index order.")
                log_df = log_df.sort_index().reset_index(drop=True)
            else:
                log_df = log_df.sort_values(by="Elapsed Time").reset_index(drop=True)

            log_df = self.fill_async_nans(log_df, self.device_manager.async_observables)
            log_df.columns = self.modify_headers(log_df.columns)
            log_df["Shotnumber"] = log_df.index + 1

            return log_df

        except Exception:
            logger.exception("Error converting log entries to DataFrame")
            raise

    def modify_headers(self, headers: list[str]) -> list[str]:
        """Reformat ``device:variable`` headers to ``device variable [Alias:alias]``.

        Matches the naming convention used by Master Control scans so downstream
        analysis tools see consistent column names regardless of scan origin.
        """
        new_headers: list[str] = []
        for header in headers:
            if ":" in header:
                device_name, variable = header.split(":", 1)
                new_header = f"{device_name} {variable}"
                device_dict = self.database_dict.get_database()
                alias = device_dict.get(device_name, {}).get(variable, {}).get("alias")
                if alias:
                    new_header = f"{new_header} Alias:{alias}"
            else:
                new_header = header
            new_headers.append(new_header)
        return new_headers

    def process_results(self, results: dict) -> pd.DataFrame:
        """Convert *results* to DataFrame and write all scan output files.

        Returns an empty DataFrame on file I/O failure so the scan shutdown
        sequence can complete even when the network share is unavailable.
        """
        if not isinstance(results, dict):
            raise TypeError("Results must be a dictionary of timestamp-data pairs")

        if results:
            try:
                log_df = self.convert_to_dataframe(results)
                logger.debug("Data logging complete. Returning DataFrame.")
                self.save_to_txt_and_h5(log_df)
                self.dataframe_to_tdms(log_df)
                return log_df
            except DataFileError:
                logger.exception("Failed to save scan data to disk")
                return pd.DataFrame()
            except Exception:
                logger.exception("Error processing scan results")
                return pd.DataFrame()
        else:
            logger.warning("No data was collected during the logging period.")
            return pd.DataFrame()

    @staticmethod
    def fill_async_nans(
        log_df: pd.DataFrame, async_observables: list, fill_value: int | float = 0
    ):
        """Forward-fill then backward-fill async observable columns, then fill 0.

        Async devices don't fire on every shot, so their columns have NaN gaps.
        ffill/bfill propagates the nearest real reading; any remaining NaNs
        (column entirely empty) are filled with *fill_value*.

        Parameters
        ----------
        log_df : pd.DataFrame
        async_observables : list
            Column names that need fill treatment.
        fill_value : int or float
            Fallback for columns that are entirely NaN.

        Returns
        -------
        pd.DataFrame
        """
        try:
            if not isinstance(log_df, pd.DataFrame):
                raise TypeError("log_df must be a pandas DataFrame")
            if not isinstance(async_observables, list):
                raise TypeError("async_observables must be a list")

            log_df.replace("", pd.NA, inplace=True)

            for async_obs in async_observables:
                if async_obs in log_df.columns:
                    if not log_df[async_obs].isna().all():
                        log_df[async_obs] = log_df[async_obs].ffill()
                        log_df[async_obs] = log_df[async_obs].bfill()
                    else:
                        logger.warning(
                            "Column %s consists entirely of NaN values and will be left unchanged.",
                            async_obs,
                        )

            log_df = log_df.fillna(fill_value)
            log_df = log_df.infer_objects(copy=False)
            logger.debug("Filled remaining NaN and empty values with %s.", fill_value)

            return log_df

        except Exception:
            logger.exception("Error in fill_async_nans")
            raise
