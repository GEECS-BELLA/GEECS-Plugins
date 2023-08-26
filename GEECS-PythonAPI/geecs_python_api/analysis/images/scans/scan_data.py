import os
import re
import inspect
import numpy as np
import calendar as cal
from pathlib import Path
from datetime import datetime as dtime, date
from typing import Optional, Union, Any
from configparser import ConfigParser, NoSectionError
from geecs_python_api.controls.api_defs import SysPath, ScanTag
import geecs_python_api.controls.experiment.htu as htu
from geecs_python_api.tools.images.batches import list_files
from geecs_python_api.controls.interface import api_error
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.tools.interfaces.tdms import read_geecs_tdms


class ScanData:
    """ Represents a GEECS experiment scan """

    def __init__(self, folder: Optional[SysPath] = None,
                 tag: Optional[Union[int, ScanTag, tuple]] = None,
                 load_scalars: bool = True,
                 ignore_experiment_name: bool = False,
                 experiment_base_path: Optional[SysPath] = None):
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
        ignore_experiment_name : bool
              Allows working offline with local copy of the data, when specifying a folder
        experiment_base_path : SysPath
              Allows working offline with local copy of the data, when specifying a tag
              e.g. experiment_base_path='C:/Users/GuillaumePlateau/Documents/LBL/Data/Undulator'
        """

        self.identified = False
        self.scan_info: dict[str, str] = {}

        self.__folder: Optional[Path] = None
        self.__tag: Optional[ScanTag] = None
        self.__tag_date: Optional[date] = None
        self.__analysis_folder: Optional[Path] = None

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

                self.identified = ignore_experiment_name or (exp_name == GeecsDevice.exp_info['name'])
                if self.identified:
                    self.__folder = folder

            except Exception:
                raise

        if not self.identified and tag:
            if isinstance(tag, int):
                self.__tag_date = dtime.now().date()
                tag = ScanTag(self.__tag_date.year, self.__tag_date.month, self.__tag_date.day, tag)

            if isinstance(tag, tuple):
                try:
                    if not isinstance(tag, ScanTag):
                        tag = ScanTag(*tag)

                    if experiment_base_path is None:
                        exp_path = Path(GeecsDevice.exp_info['data_path'])
                    else:
                        exp_path = Path(experiment_base_path)

                    if not exp_path.is_dir():
                        raise ValueError("Experiment base folder does not exist")

                    if self.__tag_date is None:
                        self.__tag_date = date(tag.year, tag.month, tag.day)

                    folder = (exp_path /
                              self.__tag_date.strftime("Y%Y") /
                              self.__tag_date.strftime("%m-%b") /
                              self.__tag_date.strftime("%y_%m%d") /
                              'scans'/f'Scan{tag.number:03d}')
                    self.identified = folder.is_dir()
                    if self.identified:
                        self.__tag = tag
                        self.__folder = folder
                    else:
                        raise OSError

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
        if load_scalars:
            self.load_scalar_data()
        else:
            self.data_dict = None
            self.data_frame = None

    def get_folder(self) -> Optional[Path]:
        return self.__folder

    def get_tag(self) -> Optional[ScanTag]:
        return self.__tag

    def get_analysis_folder(self) -> Optional[Path]:
        return self.__analysis_folder

    def load_scan_info(self):
        config_parser = ConfigParser()
        config_parser.optionxform = str

        try:
            config_parser.read(self.__folder / f'ScanInfoScan{self.__tag.number:03d}.ini')
            self.scan_info.update({key: value.strip("'\"")
                                   for key, value in config_parser.items("Scan Info")})
        except NoSectionError:
            api_error.warning(f'ScanInfo file does not have a "Scan Info" section',
                              f'ScanData class, method "{inspect.stack()[0][3]}"')

    def load_scalar_data(self) -> bool:
        tdms_path = self.__folder / f'Scan{self.__tag.number:03d}.tdms'
        if tdms_path.is_file():
            self.data_dict, self.data_frame = read_geecs_tdms(tdms_path)

        return tdms_path.is_file()

    def load_mag_spec_data(self) -> dict[str, Any]:
        magspec_dict = {'full': {}, 'hi_res': {}}
        magspec_dict['full']['paths'] = list_files(self.__folder / 'U_BCaveMagSpec-interpSpec', -1, '.txt')
        magspec_dict['hi_res']['paths'] = list_files(self.__folder / 'U_HiResMagCam-interpSpec', -1, '.txt')

        for key in ['full', 'hi_res']:
            specs = []
            shots = []
            for path in magspec_dict[key]['paths']:
                try:
                    specs.append(np.loadtxt(path, skiprows=1))
                    shots.append(int(path.name[-7:-4]))
                except Exception:
                    continue

            magspec_dict[key]['specs'] = np.array(specs)
            magspec_dict[key]['shots'] = np.array(shots)

        return magspec_dict

    @staticmethod
    def build_folder_path(tag: ScanTag, base_directory: Union[Path, str] = r'Z:\data', experiment: str = 'Undulator')\
            -> Path:
        base_directory = Path(base_directory)
        folder: Path = base_directory / experiment
        folder = folder / f'Y{tag[0]}' / f'{tag[1]:02d}-{cal.month_name[tag[1]][:3]}'
        folder = folder / f'{str(tag[0])[-2:]}_{tag[1]:02d}{tag[2]:02d}'
        folder = folder / 'scans' / f'Scan{tag[3]:03d}'

        return folder


if __name__ == '__main__':
    _base_path, is_local = htu.initialize()
    _base_tag = ScanTag(2023, 7, 6, 4)

    _folder = ScanData.build_folder_path(_base_tag, _base_path)
    _scan_data = ScanData(_folder, ignore_experiment_name=is_local)

    print('Loading mag spec data...')
    _magspec_data = _scan_data.load_mag_spec_data()
    # plt.figure()
    # for x, ind in zip(measured.avg_x, measured.indexes):
    #     plt.plot(x * np.ones(ind.shape), ind, '.', alpha=0.3)
    # plt.xlabel('Current [A]')
    # plt.ylabel('Indexes')
    # plt.show(block=True)

    print('Done')
