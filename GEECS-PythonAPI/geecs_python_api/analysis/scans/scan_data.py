from __future__ import annotations

import os
import re
import inspect
import numpy as np
import pandas as pd
import calendar as cal
from datetime import datetime
from pathlib import Path
from datetime import datetime as dtime, date
from typing import Optional, Union, NamedTuple
from configparser import ConfigParser, NoSectionError
from geecs_python_api.tools.interfaces.tdms import read_geecs_tdms
from geecs_python_api.controls.interface.geecs_errors import api_error  # TODO this enforces loading the config...
from geecs_python_api.controls.api_defs import SysPath, ScanTag, month_to_int
from geecs_python_api.tools.distributions.binning import unsupervised_binning, BinningResults


class ScanData:
    """ Represents a GEECS experiment scan """

    def __init__(self, folder: Optional[SysPath] = None,
                 tag: Optional[Union[int, ScanTag, tuple]] = None,
                 experiment: Optional[str] = None,
                 base_path: Optional[Union[Path, str]] = r'Z:\data',
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
        self.__tag: Optional[ScanTag] = None
        self.__tag_date: Optional[date] = None
        self.__analysis_folder: Optional[Path] = None

        self.data_dict = {}
        self.data_frame = None  # use tdms.geecs_tdms_dict_to_panda

        if folder is None:
            experiment = tag.experiment if tag.experiment is not None else experiment
            if tag and experiment:
                folder = self.build_scan_folder_path(tag, base_directory=base_path, experiment=experiment)

        if folder:
            folder = Path(folder)

            (exp_name, year_folder_name, month_folder_name, date_folder_name,
             scans_literal, scan_folder_name) = folder.parts[-6:]

            if not folder.exists():
                if read_mode:
                    raise ValueError("Folder does not exist")
                else:
                    folder.mkdir(parents=True, exist_ok=True)

            if (not re.match(r"Y\d{4}", year_folder_name)) or \
               (not re.match(r"\d{2}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", month_folder_name)) or \
               (not re.match(r"\d{2}_\d{4}", date_folder_name)) or \
               (not scans_literal == 'scans') or \
               (not re.match(r"Scan\d{3,}", scan_folder_name)):
                raise ValueError("Folder path does not appear to follow convention")

            self.__tag_date = dtime.strptime(date_folder_name, "%y_%m%d").date()
            self.__tag = \
                ScanTag(self.__tag_date.year, self.__tag_date.month, self.__tag_date.day, int(scan_folder_name[4:]))
            self.__folder = folder

        if load_scalars:
            self.load_scalar_data()

    @staticmethod
    def get_scan_tag(year, month, day, number, experiment_name: Optional[str]=None):
        year = int(year)
        if 10 <= year <= 99:
            year = year + 2000
        month = month_to_int(month)

        return ScanTag(year, month, int(day), int(number), experiment=experiment_name)

    @staticmethod
    def build_scan_folder_path(tag: ScanTag, base_directory: Union[Path, str] = r'Z:\data',
                               experiment: str = 'Undulator') -> Path:
        base_directory = Path(base_directory)
        experiment = tag.experiment if tag.experiment is not None else experiment

        folder: Path = base_directory / experiment
        folder = folder / f'Y{tag[0]}' / f'{tag[1]:02d}-{cal.month_name[tag[1]][:3]}'
        folder = folder / f'{str(tag[0])[-2:]}_{tag[1]:02d}{tag[2]:02d}'
        folder = folder / 'scans' / f'Scan{tag[3]:03d}'

        return folder

    @staticmethod
    def build_device_shot_path(tag: ScanTag, device_name: str, shot_number: int, file_extension: str = 'png',
                               base_directory: Union[Path, str] = r'Z:\data', experiment: str = 'Undulator') -> Path:
        experiment = tag.experiment if tag.experiment is not None else experiment

        scan_path = ScanData.build_scan_folder_path(tag=tag, base_directory=base_directory, experiment=experiment)
        file = scan_path / f'{device_name}' / f'Scan{tag[3]:03d}_{device_name}_{shot_number:03d}.{file_extension}'
        return file

    @staticmethod
    def get_latest_scan_tag(experiment: str, year: Optional[int] = None,
                            month: Optional[int] = None, day: Optional[int] = None) -> Optional[ScanTag]:
        """
        Locates the last generated scan of the given day.  If no day is given or info incomplete, then assume it's today
        :param experiment: The name of the experiment
        :param year: Optional, the year as a 4-digit int
        :param month: Optional, the month as a 1/2-digit int
        :param day: Optional, the day as a 1/2-digit int
        :return: The ScanTag tuple representing the last, existing scan folder
        """
        old_date = True
        if year is None or month is None or day is None:
            today = datetime.today()
            year = today.year
            month = today.month
            day = today.day
            old_date = False

        i = 1
        new_scan_flag = True
        while new_scan_flag:
            tag = ScanTag(year, month, day, i, experiment=experiment)
            try:
                ScanData(tag=tag, experiment=experiment, load_scalars=False, read_mode=True)
            except ValueError:
                break
            i = i+1
        if old_date and (i-1 == 0):
            return None  # In this case, there were no scans performed on the day in question.
        else:
            return ScanTag(year, month, day, i-1, experiment=experiment)

    @staticmethod
    def get_latest_scan_data(experiment: str, year: Optional[int] = None,
                             month: Optional[int] = None, day: Optional[int] = None) -> 'ScanData':
        """ :return: the ScanData class of the latest scan on the given day (or today if no date given) """
        latest_tag = ScanData.get_latest_scan_tag(experiment, year, month, day)
        return ScanData(tag=latest_tag, experiment=experiment, load_scalars=True, read_mode=True)

    @staticmethod
    def get_next_scan_folder(experiment: str, year: Optional[int] = None,
                             month: Optional[int] = None, day: Optional[int] = None) -> Path:
        """ :return: the Path to the folder of the next scan on the given day (or today if no date given) """
        latest_tag = ScanData.get_latest_scan_tag(experiment, year, month, day)
        next_tag = ScanTag(latest_tag.year, latest_tag.month, latest_tag.day,
                           latest_tag.number + 1, experiment=latest_tag.experiment)
        return ScanData.build_scan_folder_path(tag=next_tag, experiment=experiment)

    @staticmethod
    def build_next_scan_data(experiment: str, year: Optional[int] = None,
                             month: Optional[int] = None, day: Optional[int] = None) -> 'ScanData':
        """ :return: the ScanData the next scan and builds the folder for the given day (or today if no date given) """
        next_scan_folder = ScanData.get_next_scan_folder(experiment, year, month, day)
        return ScanData(folder=next_scan_folder, load_scalars=False, read_mode=False)

    def get_folder(self) -> Optional[Path]:
        return self.__folder

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
