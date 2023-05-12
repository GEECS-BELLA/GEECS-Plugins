import os
from pathlib import Path
import re
from datetime import datetime as dtime, date
from typing import Optional, Union
from geecs_api.api_defs import SysPath
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.tools.scans.tdms import read_geecs_tdms


class Scan:
    """ Represents a GEECS experiment scan """

    def __init__(self, folder: Optional[SysPath] = None, tag: Optional[Union[int, tuple[int, int, int, int]]] = None):
        """
        Parameter(s)
        ----------
        Either parameter can be provided. If both are, "folder" is used.
        Experiment name is retrieved from GeecsDevice static member and must match with folder name.

        folder : Union[str, bytes, PathLike]
            Data folder containing the scan data, e.g. "Z:/data/Undulator/Y2023/05-May/23_0501/scans/Scan002".
        info : Union[int, tuple[int, int, int, int]]
            Either of:
                - Tuple with the scan identification information, e.g. (year, month, day, scan #) = (2023, 5, 1, 2)
                - scan number only, today's date is used
        """

        self.identified = False
        self.__folder: Optional[Path] = None
        self.__tag: Optional[tuple[int, int, int, int]] = None
        self.__tag_date: Optional[date] = None

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
                    ValueError("Folder does not exist")

                self.__tag_date = dtime.strptime(date_folder_name, "%y_%m%d").date()
                self.__tag = \
                    (self.__tag_date.year, self.__tag_date.month, self.__tag_date.day, int(scan_folder_name[4:]))

                self.identified = (exp_name == GeecsDevice.exp_info['name'])
                if self.identified:
                    self.__folder = folder

            except Exception:
                raise

        if not self.identified and tag:
            if isinstance(tag, int):
                self.__tag_date = dtime.now().date()
                tag = (self.__tag_date.year, self.__tag_date.month, self.__tag_date.day, tag)

            if isinstance(tag, tuple):
                try:
                    exp_path = Path(GeecsDevice.exp_info['data_path'])
                    folder = (exp_path /
                              self.__tag_date.strftime("Y%Y") /
                              self.__tag_date.strftime("%m-%b") /
                              self.__tag_date.strftime("%y_%m%d") /
                              'scans'/f'Scan{tag[3]:03d}')
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

        # folders & files
        top_content = next(os.walk(self.__folder))
        self.files = {'devices': top_content[1], 'files': top_content[2]}

        parts = list(Path(self.__folder).parts)
        parts[-2] = 'analysis'
        self.__analysis_folder = Path(*parts)
        if not self.__analysis_folder.is_dir():
            os.makedirs(self.__analysis_folder)

        # scalar data
        tdms_path = self.__folder / f'Scan{self.__tag[3]:03d}.tdms'
        self.data_dict, self.data_frame = read_geecs_tdms(tdms_path)

    def get_folder(self):
        return self.__folder

    def get_tag(self):
        return self.__tag

    def get_analysis_folder(self):
        return self.__analysis_folder


if __name__ == '__main__':
    # GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

    # scan_path = os.path.normpath(r'Z:\data\Undulator\Y2022\08-Aug\22_0819\scans\Scan008')
    scan_path = Path(r'C:\Users\GuillaumePlateau\Documents\LBL\Data\Undulator\Y2023\05-May\23_0509\scans\Scan008')
    print(Path.exists(scan_path))
    print(scan_path.exists())
    print(os.path.isdir(scan_path))

    # scan = Scan(folder=scan_path)
    # scan = Scan(tag=(2023, 4, 27, 12))
    print('Done')
