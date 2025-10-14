"""
GEECS scan path management and navigation utilities.

This module provides comprehensive functionality for managing GEECS scan
folder structures, path generation, and scan metadata handling.

Contains the ScanPaths class which handles all aspects of GEECS scan
directory navigation, path construction, and scan identification.
"""

from __future__ import annotations

import os
import re
import inspect
import logging
import calendar as cal

from pathlib import Path
from datetime import datetime, date
from configparser import ConfigParser, NoSectionError
from typing import Optional, Union, Sequence
import pandas as pd

from geecs_data_utils.utils import SysPath, ConfigurationError, month_to_int
from geecs_data_utils.type_defs import ScanTag
from geecs_data_utils.geecs_paths_config import GeecsPathsConfig

# Acceptable extensions (lowercase, no dot)
_ACCEPTABLE_EXTS = {"png", "tif", "tiff", "h5", "dat", "tdms", "himg"}

# from geecs_data_utils.types import ScanConfig, ScanMode

# module‐level logger
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


class ScanPaths:
    """Represents a GEECS experiment scan.

    Attributes
    ----------
    scan_info : dict[str, str]
        Dictionary containing scan configuration information loaded from scan info file
    data_dict : dict
        Dictionary containing loaded device data
    data_frame : pandas.DataFrame or None
        DataFrame representation of the scan data
    paths_config : GeecsPathsConfig
        Class-level configuration object for managing GEECS data paths
    """

    paths_config = None

    def __init__(
        self,
        folder: Optional[SysPath] = None,
        tag: Optional[ScanTag] = None,
        base_directory: Union[Path, str, None] = None,
        read_mode: bool = True,
    ):
        """
        Initialize ScanPaths object.

        Either a folder or a tag+base_directory needs to be given in order to specify the location of a scan data folder

        Parameters
        ----------
        folder : Union[str, bytes, PathLike]
            Data folder containing the scan data, e.g. "Z:/data/Undulator/Y2023/05-May/23_0501/scans/Scan002".
        tag : Optional[ScanTag]
            NamedTuple with the experiment name, date, and scan number
        base_directory : Optional[Union[Path, str]]
            The base path for the data, e/g/ "Z:/data/"
            If not given, will default to the path located by GeecsPathsConfig
        read_mode: bool
            Flag that determines if ScanData should create the directory if it does not exist
        """
        self.scan_info: dict[str, str] = {}

        self._folder: Optional[Path] = None
        self._tag: Optional[ScanTag] = None
        self._tag_date: Optional[date] = None
        self._analysis_folder: Optional[Path] = None

        self.data_dict = {}
        self.data_frame = None  # use tdms.geecs_tdms_dict_to_panda

        # Handle folder initialization
        if folder is None and tag is not None:
            if base_directory is None or not Path(base_directory).exists():
                base_directory = ScanPaths.paths_config.base_path
            if not Path(base_directory).exists():
                raise NotADirectoryError(
                    f"Error setting base directory: '{base_directory}'"
                )
            folder = self.get_scan_folder_path(tag, base_directory=base_directory)

        self._initialize_folders(folder, read_mode)

    @classmethod
    def reload_paths_config(
        cls,
        config_path: Optional[Path] = None,
        default_experiment: Optional[str] = None,
        set_base_path: Optional[Union[Path, str]] = None,
        image_analysis_configs_path: Optional[Union[Path, str]] = None,
    ):
        """Use by GEECS Scanner to fix scan_data_manager in case experiment name has changed."""
        try:
            if (
                config_path is None
            ):  # Then don't explicitly pass config_path so that it uses the default location
                cls.paths_config = GeecsPathsConfig(
                    default_experiment=default_experiment,
                    set_base_path=set_base_path,
                    image_analysis_configs_path=image_analysis_configs_path,
                )
            else:
                cls.paths_config = GeecsPathsConfig(
                    config_path=config_path,
                    default_experiment=default_experiment,
                    set_base_path=set_base_path,
                    image_analysis_configs_path=image_analysis_configs_path,
                )
        except ConfigurationError as e:
            logger.error(f"Configuration Error in ScanData: {e}")
            cls.paths_config = None

    def _initialize_folders(self, folder: Path, read_mode: bool):
        """
        Initialize and validate folder paths for the scan.

        Parameters
        ----------
        folder : Path
            Folder path for the scan.
        read_mode : bool
            If True, raise an error if the folder does not exist.

        Raises
        ------
        ValueError
            If the folder path does not exist (in read mode) or does not follow the expected convention.
        """
        folder = Path(folder)

        # Extract the relevant parts of the folder path
        try:
            (
                exp_name,
                year_folder_name,
                month_folder_name,
                date_folder_name,
                scans_literal,
                scan_folder_name,
            ) = folder.parts[-6:]
        except ValueError:
            raise ValueError(
                f"Folder path {folder} does not contain the expected structure."
            )

        # Validate folder existence and create if necessary
        if not folder.exists():
            if read_mode:
                raise ValueError(f"Folder {folder} does not exist.")
            else:
                folder.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created folder: {folder}")

        # Validate folder naming conventions
        if (
            (not re.match(r"Y\d{4}", year_folder_name))
            or (
                not re.match(
                    r"\d{2}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
                    month_folder_name,
                )
            )
            or (not re.match(r"\d{2}_\d{4}", date_folder_name))
            or (not scans_literal == "scans")
            or (not re.match(r"Scan\d{3,}", scan_folder_name))
        ):
            raise ValueError(
                f"Folder path {folder} does not appear to follow the expected naming convention."
            )

        # Infer the ScanTag and tag date from the folder name
        self._tag_date = datetime.strptime(date_folder_name, "%y_%m%d").date()
        self._tag = self.get_scan_tag(
            self._tag_date.year,
            self._tag_date.month,
            self._tag_date.day,
            int(scan_folder_name[4:]),
            experiment=exp_name,
        )

        # Assign folder attributes
        self._folder = folder

    @staticmethod
    def get_scan_tag(
        year: Union[int, str],
        month: Union[int, str],
        day: Union[int, str],
        number: Union[int, str],
        experiment: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> ScanTag:
        """
        Return a ScanTag tuple given the appropriate information, formatted correctly.

        Ideally one should only build ScanTag objects using this function.

        Parameters
        ----------
        year : Union[int, str]
            Target scan year
        month : Union[int, str]
            Target scan month
        day : Union[int, str]
            Target scan day
        number : Union[int, str]
            Target scan number
        experiment : str
            Target scan's experiment name
        experiment_name : str
            Target scan's experiment name (deprecated)

        Returns
        -------
        ScanTag
            properly formatted information to describe the target scan
        """
        year = int(year)
        if 0 <= year <= 99:
            year += 2000
        month = month_to_int(month)

        exp = experiment or experiment_name or ScanPaths.paths_config.experiment
        if experiment_name is not None:
            logger.warning(
                "Recommended to use 'experiment' instead of 'experiment_name' for 'get_scan_tag'..."
            )

        return ScanTag(
            year=year, month=month, day=int(day), number=int(number), experiment=exp
        )

    @staticmethod
    def get_scan_folder_path(
        tag: ScanTag, base_directory: Optional[Union[Path, str]] = None
    ) -> Path:
        """Build scan folder paths for local and client directories."""
        return (
            ScanPaths.get_daily_scan_folder(tag=tag, base_directory=base_directory)
            / f"Scan{tag.number:03d}"
        )

    @staticmethod
    def get_daily_scan_folder(
        experiment: str = None,
        tag: ScanTag = None,
        base_directory: Optional[Union[Path, str]] = None,
    ) -> Path:
        """Build path to the daily scan folder. If no tag given but experiment name given, uses the current day."""
        base = base_directory or ScanPaths.paths_config.base_path

        if tag is None and experiment is None:
            raise ValueError(
                "Need to give experiment name or Scan Tag to `get_daily_scan_folder`"
            )

        if tag is None:
            today = datetime.today()
            tag = ScanPaths.get_scan_tag(
                today.year,
                month=today.month,
                day=today.day,
                number=0,
                experiment=experiment,
            )

        folder = Path(base) / tag.experiment
        folder = (
            folder / f"Y{tag.year}" / f"{tag.month:02d}-{cal.month_name[tag.month][:3]}"
        )
        folder /= f"{str(tag.year)[-2:]}_{tag.month:02d}{tag.day:02d}"
        folder = folder / "scans"

        return folder

    @staticmethod
    def get_scan_analysis_folder_path(
        tag: ScanTag, base_directory: Optional[Union[Path, str]] = None
    ) -> Path:
        """Build analysis folder path using the scan folder path as a baseline."""
        scan_folder_path = ScanPaths.get_scan_folder_path(
            tag=tag, base_directory=base_directory
        )

        parts = list(scan_folder_path.parts)
        parts[-2] = "analysis"
        return Path(*parts)

    @staticmethod
    def get_device_shot_path(
        tag: ScanTag,
        device_name: str,
        shot_number: int,
        file_extension: str = "png",
        base_directory: Optional[Union[Path, str]] = None,
    ) -> Path:
        """
        Build the full path to a device's shot file based on the scan tag, device name, and shot number.

        Parameters
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

        Returns
        -------
        Path
            The full path to the device's shot file.
        """
        scan_path = ScanPaths.get_scan_folder_path(
            tag=tag, base_directory=base_directory
        )
        extension = (
            "." + file_extension if "." not in file_extension else file_extension
        )
        file = (
            scan_path
            / f"{device_name}"
            / f"Scan{tag.number:03d}_{device_name}_{shot_number:03d}{extension}"
        )
        return file

    @staticmethod
    def get_latest_scan_tag(
        experiment: Optional[str] = None,
        year: Optional[int] = None,
        month: Optional[int] = None,
        day: Optional[int] = None,
        base_directory: Union[str, Path, None] = None,
    ) -> Optional[ScanTag]:
        """
        Locates the last generated scan for the given day or defaults to today if no date is provided.

        Parameters
        ----------
        experiment : Optional[str], optional
            Experiment name (default: CONFIG.experiment).
        year : Optional[int], optional
            Year of the scan (4-digit, default: current year if not provided).
        month : Optional[int], optional
            Month of the scan (1-12, default: current month if not provided).
        day : Optional[int], optional
            Day of the scan (1-31, default: current day if not provided).

        Returns
        -------
        Optional[ScanTag]
            The ScanTag representing the latest scan folder, or None if no scans exist for the given day.
        """
        today = datetime.today()
        year = year or today.year
        month = month or today.month
        day = day or today.day

        i = 1
        while True:
            tag = ScanPaths.get_scan_tag(year, month, day, i, experiment=experiment)
            try:
                ScanPaths(tag=tag, read_mode=True, base_directory=base_directory)
            except ValueError:
                break
            i += 1

        if i == 1:
            return None  # No scans exist for the given day
        return ScanPaths.get_scan_tag(year, month, day, i - 1, experiment=experiment)

    @staticmethod
    def get_next_scan_tag(
        experiment: Optional[str] = None,
        year: Optional[int] = None,
        month: Optional[int] = None,
        day: Optional[int] = None,
        base_directory: Union[str, Path, None] = None,
    ) -> ScanTag:
        """
        Determine the next available scan tag for the given day or today if no date is provided.

        Parameters
        ----------
        experiment : Optional[str], optional
            Experiment name (default: CONFIG.experiment).
        year : Optional[int], optional
            Year of the scan (4-digit, default: current year if not provided).
        month : Optional[int], optional
            Month of the scan (1-12, default: current month if not provided).
        day : Optional[int], optional
            Day of the scan (1-31, default: current day if not provided).

        Returns
        -------
        ScanTag
            The ScanTag for the next available scan.
        """
        latest_tag = ScanPaths.get_latest_scan_tag(
            experiment, year, month, day, base_directory=base_directory
        )
        if not latest_tag:
            today = datetime.today()
            year = year or today.year
            month = month or today.month
            day = day or today.day
            return ScanPaths.get_scan_tag(year, month, day, 1, experiment=experiment)

        return ScanPaths.get_scan_tag(
            latest_tag.year,
            latest_tag.month,
            latest_tag.day,
            latest_tag.number + 1,
            experiment=experiment,
        )

    @staticmethod
    def get_next_scan_folder(
        experiment: Optional[str] = None,
        year: Optional[int] = None,
        month: Optional[int] = None,
        day: Optional[int] = None,
        base_directory: Union[str, Path, None] = None,
    ) -> Path:
        """
        Build the folder path for the next scan on the given day or today if no date is provided.

        Parameters
        ----------
        experiment : Optional[str], optional
            Experiment name (default: CONFIG.experiment).
        year : Optional[int], optional
            Year of the scan (4-digit, default: current year if not provided).
        month : Optional[int], optional
            Month of the scan (1-12, default: current month if not provided).
        day : Optional[int], optional
            Day of the scan (1-31, default: current day if not provided).

        Returns
        -------
        Path
            The Path to the folder for the next scan.
        """
        next_tag = ScanPaths.get_next_scan_tag(
            experiment, year, month, day, base_directory=base_directory
        )
        return ScanPaths.get_scan_folder_path(
            tag=next_tag, base_directory=base_directory
        )

    @staticmethod
    def build_next_scan_data(
        experiment: Optional[str] = None,
        year: Optional[int] = None,
        month: Optional[int] = None,
        day: Optional[int] = None,
        base_directory: Union[str, Path, None] = None,
    ) -> ScanPaths:
        """
        Create the ScanData object for the next scan and builds its folder.

        Parameters
        ----------
        experiment : Optional[str], optional
            Experiment name (default: CONFIG.experiment).
        year : Optional[int], optional
            Year of the scan (4-digit, default: current year if not provided).
        month : Optional[int], optional
            Month of the scan (1-12, default: current month if not provided).
        day : Optional[int], optional
            Day of the scan (1-31, default: current day if not provided).

        Returns
        -------
        ScanData
            The ScanData object for the next scan.
        """
        next_tag = ScanPaths.get_next_scan_tag(
            experiment, year, month, day, base_directory=base_directory
        )
        return ScanPaths(tag=next_tag, read_mode=False, base_directory=base_directory)

    @staticmethod
    def is_background_scan(
        tag: ScanTag, base_directory: Optional[Union[Path, str]] = None
    ) -> bool:
        """
        Check if the given scan tag references a scan that was designated as a background.

        Parameters
        ----------
        tag : ScanTag
            The scan tag containing year, month, day, and scan number.
        base_directory : Optional[Union[Path, str]], optional
            Base directory for the scan (default: CONFIG.local_base_path).

        Returns
        -------
        bool
            True if scan was explictly set as a Background scan, False otherwise
        """
        scan_folder = ScanPaths.get_scan_folder_path(
            tag=tag, base_directory=base_directory
        )
        config_filename = scan_folder / f"ScanInfoScan{tag.number:03d}.ini"

        config = ConfigParser()
        config.read(config_filename)

        if config.has_section("Scan Info") and config.has_option(
            "Scan Info", "Background"
        ):
            return config.get("Scan Info", "Background").strip().lower() == '"true"'
        return False

    def get_folder(self) -> Optional[Path]:
        """Get the scan folder path."""
        return self._folder

    def get_tag(self) -> Optional[ScanTag]:
        """Get the scan tag."""
        return self._tag

    def get_tag_date(self) -> Optional[date]:
        """Get the scan date."""
        return self._tag_date

    def get_analysis_folder(self) -> Optional[Path]:
        """Get the analysis folder path, creating it if necessary."""
        if self._analysis_folder is None:
            parts = list(Path(self._folder).parts)
            parts[-2] = "analysis"
            self._analysis_folder = Path(*parts)
            if not self._analysis_folder.is_dir():
                os.makedirs(self._analysis_folder)

        return self._analysis_folder

    def get_folders_and_files(self) -> dict[str, list[str]]:
        """Get lists of device folders and files in the scan directory."""
        top_content = next(os.walk(self._folder))
        return {"devices": top_content[1], "files": top_content[2]}

    def get_device_data(self, device_name: str):
        """Get data for a specific device from the loaded data dictionary."""
        if device_name in self.data_dict:
            return self.data_dict[device_name]
        else:
            return {}

    def load_scan_info(self):
        """Load scan configuration information from the scan info file."""
        config_parser = ConfigParser()
        config_parser.optionxform = str

        try:
            config_parser.read(self._folder / f"ScanInfoScan{self._tag.number:03d}.ini")
            self.scan_info.update(
                {
                    key: value.strip("'\"")
                    for key, value in config_parser.items("Scan Info")
                }
            )
        except NoSectionError:
            temp_scan_data = inspect.stack()[0][3]
            logging.warning(
                f'ScanInfo file does not have a "Scan Info" section (in {temp_scan_data})'
            )

        return self.scan_info

    def get_ecs_dump_file(self) -> Optional[Path]:
        """
        Get the ECS Live Dump file corresponding to this scan.

        Returns
        -------
        Optional[Path]
            Path to the ECS dump file if it exists, else None.
        """
        if not self._folder:
            return None

        ecs_folder = self._folder.parent.parent / "ECS Live dumps"
        filename = f"Scan{self._tag.number}.txt"
        ecs_file = ecs_folder / filename

        return ecs_file if ecs_file.exists() else None

    def build_device_file_map(self, device: str, file_tail: str) -> dict[int, Path]:
        """
        Build a mapping from shot number to file path for a given device.

        Parameters
        ----------
        device : str
            Device name (subfolder of scan directory).
        file_tail : str
            Suffix and extension, e.g., '.png', '_avg.h5'.

        Returns
        -------
        dict[int, Path]
            Mapping from shot number to file path.
        """
        base_path = self.get_folder()
        if not base_path:
            raise ValueError("Scan folder is not set.")

        device_folder = base_path / device
        if not device_folder.exists():
            logger.warning(f"Device folder missing: {device_folder}")
            return {}

        pattern = re.compile(
            rf"Scan\d{{3,}}_{re.escape(device)}_(\d{{3,}}){re.escape(file_tail)}$"
        )

        file_map = {}
        for file in device_folder.iterdir():
            if not file.is_file():
                continue
            match = pattern.match(file.name)
            if match:
                shot_number = int(match.group(1))
                file_map[shot_number] = file

        return file_map

    def get_common_shot_dataframe(
        self, device_file_specs: Sequence[tuple[str, str]]
    ) -> pd.DataFrame:
        """
        Generate a DataFrame containing file paths for all devices with common shot number.

        This method identifies shot numbers that are common (present in all specified
        devices' subfolders) and returns a table where each row corresponds to a shot
        number, and each column contains the full path to the file for that device.

        Parameters
        ----------
        device_file_specs : Sequence[tuple[str, str]]
            A sequence of (device_name, file_tail) pairs.
            - `device_name` is the name of the subdirectory inside the scan folder.
            - `file_tail` is the suffix used in the filename, including extension,
              such as '.png', '_avg.h5', or '.tdms'.

        Returns
        -------
        pd.DataFrame
            A DataFrame with one row per shot number that exists for all devices.
            Columns:
            - 'shot_number': The shot number (int).
            - One column per device name, with each entry as a `Path` object to the matching file.
            If no common shots are found, an empty DataFrame with appropriate columns is returned.

        Examples
        --------
        >>> tag = ScanTag(year=2025, month=8, day=7, number=5, experiment='Undulator')
        >>> sd = ScanPaths(tag=tag)
        >>> dev_list = [
        ...     ('Z_Test_Scope', '.dat'),
        ...     ('Z_Test_Scope_2', '.dat'),
        ...     ('UC_ALineEBeam3', '.png')
        ... ]
        >>> common_shots = sd.get_common_shot_dataframe(dev_list)
        """
        device_maps = {
            device: self.build_device_file_map(device, file_tail)
            for device, file_tail in device_file_specs
        }

        # Find common shot numbers across all devices
        common_shots = set.intersection(*(set(m.keys()) for m in device_maps.values()))
        if not common_shots:
            logger.warning("No common shots found across specified devices.")
            return pd.DataFrame(
                columns=["shot_number"] + [device for device, _ in device_file_specs]
            )

        # Build rows: one per shot
        rows = []
        for shot in sorted(common_shots):
            row = {"shot_number": shot}
            for device, file_map in device_maps.items():
                row[device] = file_map[shot]
            rows.append(row)

        return pd.DataFrame(rows)

    # ---- Minimal naming + discovery primitives (no shot loops) -------------------

    @staticmethod
    def _shot_str(shot: int) -> str:
        """Zero-pad to at least 3 digits; expand naturally beyond 999."""
        return f"{shot:03d}" if shot < 1000 else str(shot)

    def list_device_folders(self) -> list[str]:
        """Return device subfolder names from this scan folder."""
        try:
            return self.get_folders_and_files().get("devices", [])
        except Exception:
            root = self.get_folder()
            return (
                [p.name for p in root.iterdir() if p.is_dir()]
                if root and root.exists()
                else []
            )

    def device_folder(self, device: str) -> Path:
        """Resolve '<scan>/<device>'."""
        return self.get_folder() / device

    @staticmethod
    def build_asset_filename(
        *, scan: int, shot: int, device: str, ext: str, variant: Optional[str] = None
    ) -> str:
        """Build canonical expected file naming."""
        ext = ext.lstrip(".").lower()
        shot_str = ScanPaths._shot_str(shot)
        variant_seg = "" if not variant else f"{variant}"
        return f"Scan{scan:03d}_{device}_{shot_str}{variant_seg}.{ext}"

    def build_asset_path(
        self, *, shot: int, device: str, ext: str, variant: Optional[str] = None
    ) -> Path:
        """Full expected path for one asset."""
        tag = self.get_tag()
        fname = self.build_asset_filename(
            scan=tag.number, shot=shot, device=device, ext=ext, variant=variant
        )
        return self.device_folder(device) / fname

    def infer_device_ext(self, device: str, *, max_files: int = 5) -> str:
        """Peek at up to `max_files` files to find proper file extension."""
        from collections import Counter

        dpath = self.device_folder(device)
        if not dpath.exists():
            return "png"

        counts = Counter()
        seen = 0
        for f in dpath.iterdir():
            if f.is_file():
                ext = f.suffix.lower().lstrip(".")
                if ext in _ACCEPTABLE_EXTS:
                    counts[ext] += 1
                    seen += 1
                    if seen >= max_files:
                        break
        return counts.most_common(1)[0][0] if counts else "png"


ScanPaths.reload_paths_config()

if __name__ == "__main__":
    test_tag = ScanPaths.get_scan_tag(2025, 6, 5, number=31, experiment="Undulator")
    sp = ScanPaths(tag=test_tag)
    print(sp.load_scan_info())
