import os
import pandas as pd
import nptdms as tdms
import numpy.typing as npt
from typing import Optional
from geecs_api.api_defs import SysPath


def read_geecs_tdms(file_path: SysPath) \
        -> tuple[Optional[dict[str, dict[str, npt.ArrayLike]]], Optional[pd.DataFrame]]:
    file_extension: str = os.path.splitext(file_path)[1].lower()

    if os.path.isfile(file_path) and (file_extension == '.tdms'):
        data_dict = {}

        with tdms.TdmsFile.open(file_path) as f_tdms:
            # data_dict = {device_name: {variable.name.split(device_name)[1][1:]: variable[:].astype('float64')
            #                            for variable in variables}
            #              for device_name, variables
            #              in [(group.name, group.channels()) for group in f_tdms.groups()]}
            for device_name, variables in [(group.name, group.channels()) for group in f_tdms.groups()]:
                dev_data_dict = {}

                for variable in variables:
                    var_name = variable.name.split(device_name)[1][1:]
                    try:
                        dev_data_dict[var_name] = variable[:].astype('float64')
                    except Exception:
                        continue

                if dev_data_dict:
                    data_dict[device_name] = dev_data_dict

        data_frame: pd.DataFrame = pd.DataFrame(data_dict)
        return data_dict, data_frame

    else:
        return None, None


if __name__ == '__main__':
    _file_path = r'Z:\data\Undulator\Y2023\04-Apr\23_0427\scans\Scan012\Scan012.tdms'
    _data_dict, _data_frame = read_geecs_tdms(_file_path)
    print('Done')
