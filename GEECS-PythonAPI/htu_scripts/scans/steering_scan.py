import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Union
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.transport.magnets.steering import SteeringSupply
from geecs_api.tools.distributions.binning import unsupervised_binning
from geecs_api.tools.images.scan_images import ScanImages
from geecs_api.tools.scans.scan_data import ScanData


class SteeringScan:
    def __init__(self, scan_images: ScanImages, steering_supply: Union[SteeringSupply, str]):
        """
        Container for data analysis of a set of images collected at an undulator station.

        scan (ScanImages object): analysis object for the relevant scan
        steering_supply (SteeringSupply | str), either of:
            - SteeringSupply object
            - GEECS device name of the relevant steering magnet supply
        """

        self.scan_images: ScanImages = scan_images
        self.supply: Optional[SteeringSupply] = None

        # Steering supply
        if isinstance(steering_supply, SteeringSupply):
            self.supply = steering_supply
            self.supply_name: str = steering_supply.get_name()
            self.axis = 'x' if self.supply_name[-1] == 'H' else 'y'
        elif isinstance(steering_supply, str) and re.match(r'U_S[1-4][HV]', steering_supply):  # device name
            self.supply_name = steering_supply
            self.axis = 'x' if self.supply_name[-1] == 'H' else 'y'
        else:
            self.supply_name = steering_supply


# def steering_scan(e_transport: , scan_images: ScanImages, steering_supply: Union[SteeringSupply, str]):
#     """
#     Container for data analysis of a set of images collected at an undulator station.
#
#     scan (ScanImages object): analysis object for the relevant scan
#     steering_supply (SteeringSupply | str), either of:
#         - SteeringSupply object
#         - GEECS device name of the relevant steering magnet supply
#     """
#
#     self.scan_images: ScanImages = scan_images
#     self.supply: Optional[SteeringSupply] = None
#
#     # Steering supply
#     if isinstance(steering_supply, SteeringSupply):
#         self.supply = steering_supply
#         self.supply_name: str = steering_supply.get_name()
#         self.axis = 'x' if self.supply_name[-1] == 'H' else 'y'
#     elif isinstance(steering_supply, str) and re.match(r'U_S[1-4][HV]', steering_supply):  # device name
#         self.supply_name = steering_supply
#         self.axis = 'x' if self.supply_name[-1] == 'H' else 'y'
#     else:
#         self.supply_name = steering_supply


if __name__ == '__main__':
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

    # _base = Path(r'C:\Users\GuillaumePlateau\Documents\LBL\Data')
    _base: Path = Path(r'Z:\data')
    _base_tag = (2023, 4, 20, 48)
    _camera_tag = 'A3'

    _key_device = 'U_S4H'

    _scan = ScanData(tag=(2023, 4, 13, 26), experiment_base_path=_base / 'Undulator')
    _key_data = _scan.data_dict[_key_device]

    bin_x, avg_y, std_x, std_y, near_ix, indexes, bins = unsupervised_binning(_key_data['Current'], _key_data['shot #'])

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
