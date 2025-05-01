from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import nptdms as tdms

from geecs_paths_utils.scan_paths import ScanPaths

# Module-level logger
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

def read_geecs_tdms(file_path: Path) -> Optional[dict[str, dict[str, np.ndarray]]]:
    """
    Read a GEECS TDMS file and return nested dict of device -> variable -> ndarray.
    """
    if not file_path.is_file() or file_path.suffix.lower() != '.tdms':
        return None

    with tdms.TdmsFile.open(str(file_path)) as f_tdms:
        def convert(channel: tdms.TdmsChannel):
            arr = channel[:]
            try:
                return arr.astype('float64')
            except ValueError:
                return arr

        return {
            group.name: {
                var.name.split(group.name)[1].lstrip('_'): convert(var)
                for var in group.channels()
            }
            for group in f_tdms.groups()
        }

def geecs_tdms_dict_to_panda(data_dict: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    """
    Convert nested TDMS dict into a multi-indexed pandas DataFrame.
    """
    return (
        pd.concat(
            map(pd.DataFrame, data_dict.values()),
            keys=data_dict.keys(),
            axis=1
        )
        .set_index('Shotnumber')
    )


class ScanData(ScanPaths):
    """GEECS experiment scan with scalar loading."""

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
        tag = ScanData.get_latest_scan_tag(experiment, year, month, day, base_directory=base_directory)
        if not tag:
            raise ValueError("No scans found for the specified date.")
        return ScanData(tag=tag, load_scalars=True, read_mode=True, base_directory=base_directory)

    def load_scalar_data(self) -> bool:
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
        analysis_sfile = self.get_analysis_folder().parent / f"s{self._tag.number}.txt"
        if not analysis_sfile.exists():
            raise FileNotFoundError(f"No sfile for scan {self._tag}")
        return pd.read_csv(analysis_sfile, delimiter="\t")

    def copy_fresh_sfile_to_analysis(self) -> None:
        """
        Replace the analysis sfile with a fresh copy from scans.
        """
        analysis_sfile = self.get_analysis_folder().parent / f"s{self._tag.number}.txt"
        scan_sfile = self.get_folder() / f"ScanDataScan{self._tag.number:03d}.txt"

        if not scan_sfile.exists():
            raise FileNotFoundError(f"Original s file '{scan_sfile}' not found.")
        if analysis_sfile.exists():
            analysis_sfile.unlink()
        shutil.copy2(src=scan_sfile, dst=analysis_sfile)

# Initialize paths config
ScanData.reload_paths_config()

if __name__ == "__main__":
    tag = ScanData.get_scan_tag(2025, 4, 3, number=2, experiment="Undulator")
    sd = ScanData(tag=tag)
    latest = sd.get_latest_scan_data()
    print(latest.data_frame)
