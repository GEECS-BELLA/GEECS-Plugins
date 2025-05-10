from __future__ import annotations

import logging
import nptdms as tdms
import numpy as np

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