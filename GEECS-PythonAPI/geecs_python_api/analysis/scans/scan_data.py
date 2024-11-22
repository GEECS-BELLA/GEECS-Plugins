import os
import re
import inspect
import pandas as pd
import calendar as cal
from pathlib import Path
from datetime import datetime as dtime, date
from typing import Optional, Union
from configparser import ConfigParser, NoSectionError
from geecs_python_api.tools.interfaces.tdms import read_geecs_tdms
from geecs_python_api.controls.interface import api_error
from geecs_python_api.controls.api_defs import SysPath, ScanTag


class ScanData:
    """ Represents a GEECS experiment scan """

    def __init__(self, folder: Optional[SysPath] = None,
                 tag: Optional[Union[int, ScanTag, tuple]] = None,
                 experiment: Optional[str] = None,
                 base_path: Optional[Union[Path, str]] = r'Z:\data',
                 load_scalars: bool = True
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

        self.identified = False
        self.scan_info: dict[str, str] = {}

        self.__folder: Optional[Path] = None
        self.__tag: Optional[ScanTag] = None
        self.__tag_date: Optional[date] = None
        self.__analysis_folder: Optional[Path] = None

        if folder is None:
            if tag and experiment:
                folder = self.build_scan_folder_path(tag, base_directory=base_path, experiment=experiment)

        if folder:
            try:
                folder = Path(folder)

                (exp_name, year_folder_name, month_folder_name, date_folder_name,
                 scans_literal, scan_folder_name) = folder.parts[-6:]

                if (not re.match(r"Y\d{4}", year_folder_name)) or \
                   (not re.match(r"\d{2}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", month_folder_name)) or \
                   (not re.match(r"\d{2}_\d{4}", date_folder_name)) or \
                   (not scans_literal == 'scans') or \
                   (not re.match(r"Scan\d{3,}", scan_folder_name)):
                    raise ValueError("Folder path does not appear to follow convention")
                elif not folder.exists():
                    raise ValueError("Folder does not exist")

                self.__tag_date = dtime.strptime(date_folder_name, "%y_%m%d").date()
                self.__tag = \
                    ScanTag(self.__tag_date.year, self.__tag_date.month, self.__tag_date.day, int(scan_folder_name[4:]))

                self.identified = folder.is_dir()
                if self.identified:
                    self.__folder = folder

            except Exception:
                raise

        if not self.identified:
            raise ValueError

        # scan info
        self.load_scan_info()

        # folders & files
        top_content = next(os.walk(self.__folder))
        self.files = {'devices': top_content[1], 'files': top_content[2]}

        parts = list(Path(self.__folder).parts)
        parts[-2] = 'analysis'
        self.__analysis_folder = Path(*parts)
        if not self.__analysis_folder.is_dir():
            os.makedirs(self.__analysis_folder)

        # scalar data
        self.data_frame = None  # use tdms.geecs_tdms_dict_to_panda
        if load_scalars:
            self.load_scalar_data()
        else:
            self.data_dict = {}

        # TODO needed for child classes to have access to these protected variables
        self.folder = self.__folder
        self.tag = self.__tag
        self.tag_date = self.__tag_date
        self.analysis_folder = self.__analysis_folder

    @staticmethod
    def build_scan_folder_path(tag: ScanTag, base_directory: Union[Path, str] = r'Z:\data',
                               experiment: str = 'Undulator') -> Path:
        base_directory = Path(base_directory)
        folder: Path = base_directory / experiment
        folder = folder / f'Y{tag[0]}' / f'{tag[1]:02d}-{cal.month_name[tag[1]][:3]}'
        folder = folder / f'{str(tag[0])[-2:]}_{tag[1]:02d}{tag[2]:02d}'
        folder = folder / 'scans' / f'Scan{tag[3]:03d}'

        return folder

    @staticmethod
    def build_device_shot_path(tag: ScanTag, device_name: str, shot_number: int, file_extension: str = 'png',
                               base_directory: Union[Path, str] = r'Z:\data', experiment: str = 'Undulator') -> Path:
        scan_path = ScanData.build_scan_folder_path(tag=tag, base_directory=base_directory, experiment=experiment)
        file = scan_path / f'{device_name}' / f'Scan{tag[3]:03d}_{device_name}_{shot_number:03d}.{file_extension}'
        return file

    def get_folder(self) -> Optional[Path]:
        return self.__folder

    def get_tag(self) -> Optional[ScanTag]:
        return self.__tag

    def get_analysis_folder(self) -> Optional[Path]:
        return self.__analysis_folder

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


if __name__ == '__main__':
    print("First test, with first building the scan folder")

    experiment_name = 'Undulator'
    scan_tag = ScanTag(2023, 8, 9, 4)

    scan_folder = ScanData.build_scan_folder_path(scan_tag, experiment=experiment_name)
    scan_data = ScanData(scan_folder)

    print(scan_data.files['devices'])
    print(scan_data.files['files'])
    print(scan_data.get_folder())
    print(scan_data.get_analysis_folder())
    print(ScanData.build_device_shot_path(scan_tag, 'UC_Device', 5))

    print()
    print("Another test, this time using a scan with a corrupted .tdms and with the tag+exp name")
    print()

    experiment_name = 'Undulator'
    scan_tag = ScanTag(2024, 11, 19, 18)

    scan_data = ScanData(tag=scan_tag, experiment=experiment_name)
    print(scan_data.files['devices'])
    print(scan_data.files['files'])
    print(scan_data.get_folder())
    print(scan_data.get_analysis_folder())
    print(ScanData.build_device_shot_path(scan_tag, scan_data.files['devices'][0], 5))
