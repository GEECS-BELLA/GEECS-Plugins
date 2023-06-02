import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from geecs_api.tools.scans.scan_data import ScanData
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.tools.distributions.binning import bin_scan


if __name__ == '__main__':
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

    # _base = Path(r'C:\Users\GuillaumePlateau\Documents\LBL\Data')
    _base: Path = Path(r'Z:\data')
    _base_tag = (2023, 4, 20, 48)
    _camera_tag = 'A3'

    _key_device = 'U_S4H'

    _scan = ScanData(tag=(2023, 4, 13, 26), experiment_base_path=_base / 'Undulator')
    _key_data = _scan.data_dict[_key_device]

    bin_x, avg_y, std_x, std_y, near_ix, indexes = bin_scan(_key_data['Current'], _key_data['shot #'])

    plt.figure()
    plt.plot(_key_data['shot #'], _key_data['Current'], '.b', alpha=0.3)
    plt.xlabel('Shot #')
    plt.ylabel('Current [A]')
    plt.show(block=False)

    plt.figure()
    for x, ind in zip(bin_x, indexes):
        plt.plot(x * np.ones(ind.shape), ind, '.', alpha=0.3)
    plt.xlabel('Current [A]')
    plt.ylabel('Shot #')
    plt.show(block=True)

    print('done')
