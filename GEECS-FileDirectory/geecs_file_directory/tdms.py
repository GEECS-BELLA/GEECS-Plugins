import numpy as np
import nptdms as tdms
from typing import Optional
from pathlib import Path


def read_geecs_tdms(file_path: Path) -> Optional[dict[str, dict[str, np.ndarray]]]:
    file_extension: str = file_path.suffix.lower()

    if file_path.is_file() and (file_extension == '.tdms'):
        with tdms.TdmsFile.open(str(file_path)) as f_tdms:
            def convert_channel_to_float(channel: tdms.TdmsChannel):
                try:
                    return channel[:].astype('float64')
                except ValueError:
                    return channel[:]

            data_dict = {device_name: {variable.name.split(device_name)[1][1:]: convert_channel_to_float(variable)
                                       for variable in variables}
                         for device_name, variables
                         in [(group.name, group.channels()) for group in f_tdms.groups()]}

        return data_dict

    else:
        return None


if __name__ == '__main__':
    _file_path = Path(r'Z:\data\Undulator\Y2023\04-Apr\23_0427\scans\Scan012\Scan012.tdms')
    _data_dict = read_geecs_tdms(_file_path)
    print('Done')
