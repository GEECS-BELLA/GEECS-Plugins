"""
GEECS scan data loading and management utilities.

This module provides functionality for loading and manipulating GEECS
experimental scan data, including TDMS file reading, scalar data loading,
and data format conversions.

Contains the ScanData class which extends ScanPaths with data loading
capabilities for GEECS experimental scans.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import nptdms as tdms

from geecs_data_utils.scan_paths import ScanPaths

# Module-level logger
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def read_geecs_tdms(file_path: Path) -> Optional[dict[str, dict[str, np.ndarray]]]:
    """
    Read a GEECS TDMS file and return nested dictionary structure.

    Parameters
    ----------
    file_path : Path
        Path to the TDMS file to read

    Returns
    -------
    Optional[dict[str, dict[str, np.ndarray]]]
        Nested dictionary with structure device -> variable -> ndarray,
        None if file is not valid TDMS format

    Examples
    --------
    >>> data = read_geecs_tdms(Path("scan001.tdms"))
    >>> if data:
    ...     print(data.keys())  # Device names
    """
    if not file_path.is_file() or file_path.suffix.lower() != ".tdms":
        return None

    with tdms.TdmsFile.open(str(file_path)) as f_tdms:

        def convert(channel: tdms.TdmsChannel):
            arr = channel[:]
            try:
                return arr.astype("float64")
            except ValueError:
                return arr

        return {
            group.name: {
                var.name.split(group.name)[1].lstrip("_"): convert(var)
                for var in group.channels()
            }
            for group in f_tdms.groups()
        }


def geecs_tdms_dict_to_panda(
    data_dict: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """
    Convert nested TDMS dictionary into a multi-indexed pandas DataFrame.

    Parameters
    ----------
    data_dict : dict[str, dict[str, np.ndarray]]
        Nested dictionary from read_geecs_tdms with device -> variable -> data structure

    Returns
    -------
    pd.DataFrame
        Multi-indexed DataFrame with devices as top-level columns,
        indexed by shot number

    Examples
    --------
    >>> data = read_geecs_tdms(Path("scan001.tdms"))
    >>> df = geecs_tdms_dict_to_panda(data)
    >>> print(df.columns.levels[0])  # Device names
    """
    return pd.concat(
        map(pd.DataFrame, data_dict.values()), keys=data_dict.keys(), axis=1
    ).set_index("Shotnumber")


class ScanData(ScanPaths):
    """GEECS experiment scan with scalar loading.

    Attributes
    ----------
    scan_info : dict[str, str]
        Dictionary containing scan configuration information loaded from scan info file
    data_dict : dict[str, dict[str, np.ndarray]]
        Nested dictionary containing TDMS data with device -> variable -> data structure
    data_frame : pandas.DataFrame or None
        DataFrame containing scalar data loaded from text files
    paths_config : GeecsPathsConfig
        Class-level configuration object for managing GEECS data paths
    """

    def __init__(self, *args, load_scalars: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if load_scalars:
            self.load_scalar_data()

    @staticmethod
    def get_latest_scan_data(
        experiment: Optional[str] = None,
        year: Optional[int] = None,
        month: Optional[int] = None,
        day: Optional[int] = None,
        base_directory: Union[str, Path, None] = None,
    ) -> ScanData:
        """
        Get the latest scan data for a given date.

        Parameters
        ----------
        experiment : str, optional
            Experiment name (default is None)
        year : int, optional
            Year (default is None, uses current year)
        month : int, optional
            Month (default is None, uses current month)
        day : int, optional
            Day (default is None, uses current day)
        base_directory : Union[str, Path, None], optional
            Base directory for data search (default is None)

        Returns
        -------
        ScanData
            ScanData object with scalar data loaded

        Raises
        ------
        ValueError
            If no scans found for the specified date
        """
        tag = ScanData.get_latest_scan_tag(
            experiment, year, month, day, base_directory=base_directory
        )
        if not tag:
            raise ValueError("No scans found for the specified date.")
        return ScanData(
            tag=tag, load_scalars=True, read_mode=True, base_directory=base_directory
        )

    def load_scalar_data(self) -> bool:
        """
        Load scalar data from TDMS and text files.

        Loads both TDMS data (into self.data_dict) and text data
        (into self.data_frame) if available for this scan.

        Returns
        -------
        bool
            True if TDMS file was successfully loaded, False otherwise
        """
        tdms_path = self._folder / f"Scan{self._tag.number:03d}.tdms"
        if tdms_path.is_file():
            try:
                self.data_dict = read_geecs_tdms(tdms_path) or {}
            except ValueError:
                logger.warning("Could not read TDMS file %s", tdms_path)
                self.data_dict = {}
        txt_path = self._folder / f"ScanDataScan{self._tag.number:03d}.txt"
        if txt_path.is_file():
            self.data_frame = pd.read_csv(txt_path, delimiter="\t")
        return tdms_path.is_file()

    def get_sfile_data(self) -> pd.DataFrame:
        """
        Load scan data from analysis sfile.

        This is the preferred method for loading the scalar data as it uses
        the data stored in the 'analysis' directory rather than the 'raw'
        data in the 'scans' directory.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the scan data from the analysis sfile

        Raises
        ------
        FileNotFoundError
            If the sfile does not exist for this scan
        """
        analysis_sfile = self.get_analysis_folder().parent / f"s{self._tag.number}.txt"
        if not analysis_sfile.exists():
            raise FileNotFoundError(f"No sfile for scan {self._tag}")
        return pd.read_csv(analysis_sfile, delimiter="\t")

    def copy_fresh_sfile_to_analysis(self) -> None:
        """Replace the analysis sfile with a fresh copy from scans."""
        analysis_sfile = self.get_analysis_folder().parent / f"s{self._tag.number}.txt"
        scan_sfile = self.get_folder() / f"ScanDataScan{self._tag.number:03d}.txt"

        if not scan_sfile.exists():
            raise FileNotFoundError(f"Original s file '{scan_sfile}' not found.")
        if analysis_sfile.exists():
            analysis_sfile.unlink()
        shutil.copy2(src=scan_sfile, dst=analysis_sfile)

    def load_ecs_live_dump(self) -> dict[str, dict[str, str]]:
        """
        Load and parse the ECS Live Dump file for this scan.

        Returns
        -------
        dict
            Parsed ECS dump structured by device name.

        Raises
        ------
        FileNotFoundError
            If ECS dump file does not exist.
        """
        path = self.get_ecs_dump_file()
        if not path:
            raise FileNotFoundError(f"No ECS dump file found for scan {self._tag}")
        return self.parse_ecs_dump(path)


if __name__ == "__main__":
    tag = ScanData.get_scan_tag(2025, 5, 7, number=10, experiment="Undulator")
    sd = ScanData(tag=tag)
    print(sd.get_sfile_data())
