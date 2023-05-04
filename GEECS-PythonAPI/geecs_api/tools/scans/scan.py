import os
import re
import calendar as cal
from datetime import datetime as dtime
from typing import Optional, Union
from geecs_api.api_defs import SysPath
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.tools.scans.utility import read_geecs_tdms


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
        self.__folder: Optional[SysPath] = None
        self.__tag: Optional[tuple[int, int, int, int]] = None

        if folder:
            try:
                folder = os.path.normpath(folder)
                if os.sep == '\\':
                    info_str = re.search(
                        r'Y[0-9]{4}\\[0-9]{2}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
                        r'\\[_0-9]+\\scans\\Scan[0-9]{3}', folder)
                else:
                    info_str = re.search(
                        r'Y[0-9]{4}/[0-9]{2}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
                        r'/[_0-9]+/scans/Scan[0-9]{3}', folder)

                if info_str and os.path.isdir(folder):
                    exp_name: SysPath = os.path.basename(os.path.normpath(folder.split(info_str[0])[0]))
                    info_str = info_str[0].split(os.sep)
                    self.__tag = (int(info_str[0][1:]),
                                  int(info_str[1][:2]),
                                  int(info_str[2][-2:]),
                                  int(info_str[4][-3:]))
                    self.identified = (exp_name == GeecsDevice.exp_info['name'])
                    if self.identified:
                        self.__folder = folder

            except Exception:
                raise

        if not self.identified and tag:
            if isinstance(tag, int):
                stamp = dtime.now()
                tag = (int(stamp.strftime('%Y')),
                       int(stamp.strftime('%m')),
                       int(stamp.strftime('%d')),
                       tag)

            if isinstance(tag, tuple):
                try:
                    exp_path = GeecsDevice.exp_info['data_path']
                    folder = os.path.join(exp_path,
                                          f'Y{tag[0]}',
                                          f'{tag[1]:02d}-{cal.month_name[tag[1]][:3]}',
                                          f'{str(tag[0])[-2:]}_{tag[1]:02d}{tag[2]:02d}',
                                          'scans', f'Scan{tag[3]:03d}')
                    self.identified = os.path.isdir(folder)
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

        # scalar data
        tdms_path = os.path.join(self.__folder, f'Scan{self.__tag[3]:03d}.tdms')
        self.data_dict, self.data_frame = read_geecs_tdms(tdms_path)

    def get_folder(self):
        return self.__folder

    def get_tag(self):
        return self.__tag


if __name__ == '__main__':
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

    # scan_path = os.path.normpath(r'Z:\data\Undulator\Y2022\08-Aug\22_0819\scans\Scan008')
    # scan = Scan(folder=scan_path)
    scan = Scan(tag=(2023, 4, 27, 12))
    print('Done')
