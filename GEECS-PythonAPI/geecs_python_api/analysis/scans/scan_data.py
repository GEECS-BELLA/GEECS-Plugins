from __future__ import annotations

import os
import re
import inspect
import numpy as np
import pandas as pd
import calendar as cal
from datetime import datetime
import logging
from pathlib import Path
from datetime import datetime as dtime, date
from typing import Optional, Union, NamedTuple, Tuple
from configparser import ConfigParser, NoSectionError
from geecs_python_api.tools.interfaces.tdms import read_geecs_tdms
from geecs_python_api.controls.interface.geecs_errors import api_error  # TODO this enforces loading the config...
from geecs_python_api.controls.api_defs import SysPath, ScanTag, month_to_int
from geecs_python_api.tools.distributions.binning import unsupervised_binning, BinningResults

# Create a module-level logger
logger = logging.getLogger(__name__)

# Set up default logging only if no handlers are present
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class GeecsPathsConfig:
    """
    Manages configuration for GEECS-related paths and experiment settings.

    This class loads pertinent path and experiment information from a configuration file.
    If the configuration file or the relevant entries are missing, predefined default values
    are used for the local base path, client base path, and experiment name.

    Attributes:
    ----------
    local_base_path : Path
        The base directory for storing GEECS data locally.
    client_base_path : Path
        The base directory for accessing GEECS data on the server.
    experiment : str
        The default experiment name.

    Methods:
    -------
    _load_paths(config_path, default_local_path, default_server_path, default_experiment)
        Loads configuration values for paths and experiment settings from a config file or
        falls back to predefined defaults.
    """
    
    def __init__(self):
        """
        Initializes the GeecsConfig object by loading configuration values.

        Loads the local base path, client base path, and experiment name from a configuration
        file. If the file or specific entries are missing, defaults are used. The resulting
        values are stored as instance attributes for consistent access throughout the application.
        """
        self.local_base_path, self.client_base_path, self.experiment = self._load_paths()

    @staticmethod
    def _load_paths(
        config_path: Path = Path('~/.config/geecs_python_api/config.ini').expanduser(),
        default_local_path: Path = Path('Z:/data'),
        default_server_path: Path = Path('Z:/data'),
        default_experiment: str = "Undulator"
    ) -> Tuple[Path, Path, str]:
        """
        Loads paths and experiment settings from a configuration file.

        If the configuration file is missing or specific entries are not defined, this method
        uses predefined default values for the local base path, client base path, and experiment name.

        Parameters:
        ----------
        config_path : Path, optional
            Path to the configuration file (default: ~/.config/geecs_python_api/config.ini).
        default_local_path : Path, optional
            Default path for storing local GEECS data (default: Z:/data).
        default_server_path : Path, optional
            Default path for server-based GEECS data (default: Z:/data).
        default_experiment : str, optional
            Default experiment name (default: "Undulator").

        Returns:
        -------
        Tuple[Path, Path, str]
            A tuple containing:
            - The resolved local base path.
            - The resolved client base path.
            - The experiment name.

        Raises:
        ------
        ValueError
            If the resolved local base path does not exist.
        """
        local_base_path = default_local_path
        server_base_path = default_server_path
        experiment = default_experiment

        config = ConfigParser()
        if config_path.exists():
            try:
                config.read(config_path)
                # Retrieve paths and experiment name from the configuration file
                local_base_path = Path(
                    config['Paths'].get('GEECS_DATA_LOCAL_BASE_PATH', str(default_local_path))
                )
                server_base_path = Path(
                    config['Paths'].get('GEECS_DEVICE_SERVER_DATA_BASE_PATH', str(default_server_path))
                )
                experiment = config['Experiment'].get('expt', default_experiment)
            except Exception as e:
                logger.error(f"Error reading config file {config_path}: {e}")
        else:
            logger.warning(f"Config file {config_path} not found. Using default paths.")

        # Validate that the local base path exists
        if not local_base_path.is_dir():
            raise ValueError(
                f"Data path {local_base_path} does not exist. Check the configuration file or default paths."
            )

        return local_base_path.resolve(), server_base_path, experiment
        
# Instantiate the configuration (singleton pattern)
CONFIG = GeecsPathsConfig()

class ScanData:
    """ Represents a GEECS experiment scan """

    def __init__(self, folder: Optional[SysPath] = None,
                 tag: Optional[Union[int, ScanTag, tuple]] = None,
                 experiment: Optional[str] = None,
                 base_path: Optional[Union[Path, str]] = None,
                 client_base_path: Optional[Union[Path, str]] = None,
                 load_scalars: bool = False,
                 read_mode: bool = True
                 ):
        """
        Parameter(s)
        ----------
        Either parameter can be provided. If both are, "folder" is used.
        Experiment name is retrieved from GeecsDevice static member and must match with folder name.

        folder : Union[str, bytes, PathLike]
            Data folder containing the scan data, e.g. "Z:/data/Undulator/Y2023/05-May/23_0501/scans/Scan002".
        tag : Union[int, tuple[int, int, int, int]]
            Either of:
                - Tuple with the scan identification information, e.g. (year, month, day, scan #) = (2023, 5, 1, 2)
                - scan number only, today's date is used
        experiment : str
              Experiment name, e.g. 'Undulator'.  Necessary if just given a tag
        """
                 
        self.scan_info: dict[str, str] = {}

        self.__folder: Optional[Path] = None
        self.__client_folder: Optional[Path] = None
        self.__tag: Optional[ScanTag] = None
        self.__tag_date: Optional[date] = None
        self.__analysis_folder: Optional[Path] = None 
               
        self.experiment = experiment or CONFIG.experiment
        self.local_base_path = Path(base_path or CONFIG.local_base_path)
        self.client_base_path = Path(client_base_path or CONFIG.client_base_path)
        

        self.data_dict = {}
        self.data_frame = None  # use tdms.geecs_tdms_dict_to_panda

        # Handle folder initialization
        if folder is None and tag is not None:
            folder, client_folder = self.build_scan_folder_path(tag)
        else:
            client_folder = None

        if folder:
            self._initialize_folders(folder, client_folder, read_mode)

        # Load scalar data if requested
        if load_scalars:
            self.load_scalar_data()

    def _initialize_folders(self, folder: Path, client_folder: Optional[Path], read_mode: bool):
        """
        Initialize and validate folder paths for the scan.

        Parameters:
        ----------
        folder : Path
            Local folder path for the scan.
        client_folder : Optional[Path]
            Client folder path for the scan.
        read_mode : bool
            If True, raise an error if the folder does not exist.

        Raises:
        ------
        ValueError
            If the folder path does not exist (in read mode) or does not follow the expected convention.
        """
        folder = Path(folder)
        client_folder = Path(client_folder) if client_folder else None

        # Extract the relevant parts of the folder path
        try:
            exp_name, year_folder_name, month_folder_name, date_folder_name, scans_literal, scan_folder_name = folder.parts[-6:]
        except ValueError:
            raise ValueError(f"Folder path {folder} does not contain the expected structure.")

        # Validate folder existence and create if necessary
        if not folder.exists():
            if read_mode:
                raise ValueError(f"Folder {folder} does not exist.")
            else:
                folder.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created folder: {folder}")

        # Validate folder naming conventions
        if (not re.match(r"Y\d{4}", year_folder_name)) or \
           (not re.match(r"\d{2}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", month_folder_name)) or \
           (not re.match(r"\d{2}_\d{4}", date_folder_name)) or \
           (not scans_literal == 'scans') or \
           (not re.match(r"Scan\d{3,}", scan_folder_name)):
            raise ValueError(f"Folder path {folder} does not appear to follow the expected naming convention.")

        # Infer the ScanTag and tag date from the folder name
        self.__tag_date = dtime.strptime(date_folder_name, "%y_%m%d").date()
        self.__tag = ScanTag(self.__tag_date.year, self.__tag_date.month, self.__tag_date.day, int(scan_folder_name[4:]))

        # Assign folder attributes
        self.__folder = folder
        self.__client_folder = client_folder
    
    @staticmethod
    def get_scan_tag(year, month, day, number, experiment_name: Optional[str]=None):
        year = int(year)
        if 10 <= year <= 99:
            year = year + 2000
        month = month_to_int(month)

        return ScanTag(year, month, int(day), int(number), experiment=experiment_name)
    
    @staticmethod
    def build_scan_folder_path(tag: NamedTuple,
                               base_directory: Optional[Union[Path, str]] = None,
                               client_base_directory: Optional[Union[Path, str]] = None,
                               experiment: Optional[str] = None) -> Tuple[Path, Path]:
        """
        Build scan folder paths for local and client directories.
        """
        base_directory = Path(base_directory or CONFIG.local_base_path)
        client_base_directory = Path(client_base_directory or CONFIG.client_base_path)
        experiment = experiment or CONFIG.experiment

        def build_path(base: Path) -> Path:
            folder = base / experiment
            folder = folder / f'Y{tag.year}' / f'{tag.month:02d}-{cal.month_name[tag.month][:3]}'
            folder = folder / f'{str(tag.year)[-2:]}_{tag.month:02d}{tag.day:02d}'
            folder = folder / 'scans' / f'Scan{tag.number:03d}'
            return folder

        return build_path(base_directory), build_path(client_base_directory)
    
    @staticmethod
    def build_device_shot_path(tag: ScanTag, device_name: str, shot_number: int, file_extension: str = 'png',
                               base_directory: Optional[Union[Path, str]] = None, experiment: Optional[str] = None) -> Path:
        """
        Builds the full path to a device's shot file based on the scan tag, device name, and shot number.

        Parameters:
        ----------
        tag : ScanTag
            The scan tag containing year, month, day, and scan number.
        device_name : str
            The name of the device.
        shot_number : int
            The shot number.
        file_extension : str, optional
            File extension for the shot file (default: 'png').
        base_directory : Optional[Union[Path, str]], optional
            Base directory for the scan (default: CONFIG.local_base_path).
        experiment : Optional[str], optional
            Experiment name (default: CONFIG.experiment).

        Returns:
        -------
        Path
            The full path to the device's shot file.
        """
        base_directory = base_directory or CONFIG.local_base_path
        experiment = experiment or CONFIG.experiment

        scan_path, _ = ScanData.build_scan_folder_path(tag=tag, base_directory=base_directory, experiment=experiment)
        file = scan_path / f'{device_name}' / f'Scan{tag.number:03d}_{device_name}_{shot_number:03d}.{file_extension}'
        return file


    @staticmethod
    def get_latest_scan_tag(experiment: Optional[str] = None, year: Optional[int] = None,
                            month: Optional[int] = None, day: Optional[int] = None) -> Optional[ScanTag]:
        """
        Locates the last generated scan for the given day or defaults to today if no date is provided.

        Parameters:
        ----------
        experiment : Optional[str], optional
            Experiment name (default: CONFIG.experiment).
        year : Optional[int], optional
            Year of the scan (4-digit, default: current year if not provided).
        month : Optional[int], optional
            Month of the scan (1-12, default: current month if not provided).
        day : Optional[int], optional
            Day of the scan (1-31, default: current day if not provided).

        Returns:
        -------
        Optional[ScanTag]
            The ScanTag representing the latest scan folder, or None if no scans exist for the given day.
        """
        experiment = experiment or CONFIG.experiment
        today = datetime.today()

        year = year or today.year
        month = month or today.month
        day = day or today.day

        i = 1
        while True:
            tag = ScanTag(year, month, day, i, experiment=experiment)
            try:
                ScanData(tag=tag, experiment=experiment, load_scalars=False, read_mode=True, base_path=CONFIG.local_base_path)
            except ValueError:
                break
            i += 1

        if i == 1:
            return None  # No scans exist for the given day
        return ScanTag(year, month, day, i - 1, experiment=experiment)

    @staticmethod
    def get_latest_scan_data(experiment: Optional[str] = None, year: Optional[int] = None,
                             month: Optional[int] = None, day: Optional[int] = None) -> 'ScanData':
        """
        Retrieves the ScanData object for the latest scan on the given day or today if no date is provided.

        Parameters:
        ----------
        experiment : Optional[str], optional
            Experiment name (default: CONFIG.experiment).
        year : Optional[int], optional
            Year of the scan (4-digit, default: current year if not provided).
        month : Optional[int], optional
            Month of the scan (1-12, default: current month if not provided).
        day : Optional[int], optional
            Day of the scan (1-31, default: current day if not provided).

        Returns:
        -------
        ScanData
            The ScanData object for the latest scan.
        """
        latest_tag = ScanData.get_latest_scan_tag(experiment, year, month, day)
        if not latest_tag:
            raise ValueError("No scans found for the specified date.")
        return ScanData(tag=latest_tag, experiment=experiment, load_scalars=True, read_mode=True, base_path=CONFIG.local_base_path)


    @staticmethod
    def get_next_scan_tag(experiment: Optional[str] = None, year: Optional[int] = None,
                          month: Optional[int] = None, day: Optional[int] = None) -> ScanTag:
        """
        Determines the next available scan tag for the given day or today if no date is provided.

        Parameters:
        ----------
        experiment : Optional[str], optional
            Experiment name (default: CONFIG.experiment).
        year : Optional[int], optional
            Year of the scan (4-digit, default: current year if not provided).
        month : Optional[int], optional
            Month of the scan (1-12, default: current month if not provided).
        day : Optional[int], optional
            Day of the scan (1-31, default: current day if not provided).

        Returns:
        -------
        ScanTag
            The ScanTag for the next available scan.
        """
        latest_tag = ScanData.get_latest_scan_tag(experiment, year, month, day)
        if not latest_tag:
            today = datetime.today()
            year = year or today.year
            month = month or today.month
            day = day or today.day
            return ScanTag(year, month, day, 1)

        return ScanTag(latest_tag.year, latest_tag.month, latest_tag.day, latest_tag.number + 1)


    @staticmethod
    def get_next_scan_folder(experiment: Optional[str] = None, year: Optional[int] = None,
                             month: Optional[int] = None, day: Optional[int] = None) -> Path:
        """
        Builds the folder path for the next scan on the given day or today if no date is provided.

        Parameters:
        ----------
        experiment : Optional[str], optional
            Experiment name (default: CONFIG.experiment).
        year : Optional[int], optional
            Year of the scan (4-digit, default: current year if not provided).
        month : Optional[int], optional
            Month of the scan (1-12, default: current month if not provided).
        day : Optional[int], optional
            Day of the scan (1-31, default: current day if not provided).

        Returns:
        -------
        Path
            The Path to the folder for the next scan.
        """
        next_tag = ScanData.get_next_scan_tag(experiment, year, month, day)
        return ScanData.build_scan_folder_path(tag=next_tag, experiment=experiment)[0]

    @staticmethod
    def build_next_scan_data(experiment: Optional[str] = None, year: Optional[int] = None,
                             month: Optional[int] = None, day: Optional[int] = None) -> 'ScanData':
        """
        Creates the ScanData object for the next scan and builds its folder.

        Parameters:
        ----------
        experiment : Optional[str], optional
            Experiment name (default: CONFIG.experiment).
        year : Optional[int], optional
            Year of the scan (4-digit, default: current year if not provided).
        month : Optional[int], optional
            Month of the scan (1-12, default: current month if not provided).
        day : Optional[int], optional
            Day of the scan (1-31, default: current day if not provided).

        Returns:
        -------
        ScanData
            The ScanData object for the next scan.
        """
        next_tag = ScanData.get_next_scan_tag(experiment, year, month, day)
        return ScanData(tag=next_tag, experiment=experiment, load_scalars=False, read_mode=False)

    def get_folder(self) -> Optional[Path]:
        return self.__folder

    def get_client_folder(self) -> Optional[Path]:
        return self.__client_folder
        
    def get_tag(self) -> Optional[ScanTag]:
        return self.__tag

    def get_tag_date(self) -> Optional[date]:
        return self.__tag_date

    def get_analysis_folder(self) -> Optional[Path]:
        if self.__analysis_folder is None:
            parts = list(Path(self.__folder).parts)
            parts[-2] = 'analysis'
            self.__analysis_folder = Path(*parts)
            if not self.__analysis_folder.is_dir():
                os.makedirs(self.__analysis_folder)

        return self.__analysis_folder

    def get_folders_and_files(self) -> dict[str, list[str]]:
        top_content = next(os.walk(self.__folder))
        return {'devices': top_content[1], 'files': top_content[2]}

    def get_device_data(self, device_name: str):
        if device_name in self.data_dict:
            return self.data_dict[device_name]
        else:
            return {}

    def load_scan_info(self):
        config_parser = ConfigParser()
        config_parser.optionxform = str

        try:
            config_parser.read(self.__folder / f'ScanInfoScan{self.__tag.number:03d}.ini')
            self.scan_info.update({key: value.strip("'\"")
                                   for key, value in config_parser.items("Scan Info")})
        except NoSectionError:
            temp_scan_data = inspect.stack()[0][3]
            api_error.warning(f'ScanInfo file does not have a "Scan Info" section',
                              f'ScanData class, method {temp_scan_data}')
        return self.scan_info

    def load_scalar_data(self) -> bool:
        tdms_path = self.__folder / f'Scan{self.__tag.number:03d}.tdms'
        if tdms_path.is_file():
            try:
                self.data_dict = read_geecs_tdms(tdms_path)
            except ValueError:
                api_error.warning(f'Could not read tdms file',
                                  f'{tdms_path}')
                self.data_dict = {}

        txt_path = self.__folder / f'ScanDataScan{self.__tag.number:03d}.txt'
        if txt_path.is_file():
            self.data_frame = pd.read_csv(txt_path, delimiter='\t')

        return tdms_path.is_file()

    def group_shots_by_step(self, device: str, variable: str) -> tuple[list[np.ndarray], Optional[np.ndarray], bool]:
        if not self.scan_info:
            self.load_scan_info()

        dev_data = self.get_device_data(device)
        if not dev_data:
            return [], None, False

        measured: BinningResults = unsupervised_binning(dev_data[variable], dev_data['shot #'])

        Expected = NamedTuple('Expected',
                              start=float,
                              end=float,
                              steps=int,
                              shots=int,
                              setpoints=np.ndarray,
                              indexes=list)
        parameter_start = float(self.scan_info['Start'])
        parameter_end = float(self.scan_info['End'])
        num_steps: int = 1 + round(np.abs(parameter_end - parameter_start) / float(self.scan_info['Step size']))
        num_shots_per_step: int = int(self.scan_info['Shots per step'])
        expected = Expected(start=parameter_start, end=parameter_end, steps=num_steps, shots=num_shots_per_step,
                            setpoints=np.linspace(parameter_start, parameter_end, num_steps),
                            indexes=[np.arange(p * num_shots_per_step, (p+1) * num_shots_per_step) for p in range(num_steps)])

        parameter_avgs_match_setpoints = all([inds.size == expected.shots for inds in measured.indexes])
        parameter_avgs_match_setpoints = parameter_avgs_match_setpoints and (len(measured.indexes) == expected.steps)
        if not parameter_avgs_match_setpoints:
            api_error.warning(f'Observed data binning does not match expected scan parameters (.ini)',
                              f'Function "{inspect.stack()[0][3]}"')

        if parameter_avgs_match_setpoints:
            indexes = expected.indexes
            setpoints = expected.setpoints
        else:
            indexes = measured.indexes
            setpoints = measured.avg_x

        return indexes, setpoints, parameter_avgs_match_setpoints
