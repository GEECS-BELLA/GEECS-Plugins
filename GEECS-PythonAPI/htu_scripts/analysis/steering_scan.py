import re
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Any, Union
from geecs_api.tools.scans.scan import Scan
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.diagnostics.cameras import Camera
from geecs_api.devices.HTU.transport.magnets.steering import SteeringSupply
from geecs_api.tools.distributions.binning import bin_scan
import geecs_api.tools.images.ni_vision as ni


class SteeringScan:
    def __init__(self, scan: Scan, steering_supply: Union[SteeringSupply, str], camera: Union[int, Camera, str]):
        """
        Container for data analysis of a set of images collected at an undulator station.

        scan (Scan object): analysis object for the relevant scan
        steering_supply (SteeringSupply | str), either of:
            - SteeringSupply object
            - GEECS device name of the relevant steering magnet supply
        camera (int | Camera | str), either of:
            - Camera object
            - GEECS device name of the relevant camera
            - relevant screen shorthand label (A1-3, U1-9)
        """

        self.scan: Scan = scan
        self.supply: Optional[SteeringSupply] = None
        self.camera: Optional[Camera] = None

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

        # Camera
        if isinstance(camera, Camera):  # Camera class object
            self.camera = camera
            self.camera_name: str = camera.get_name()
            self.camera_roi: Optional[np.ndarray] = camera.roi
            self.camera_r90 = camera.rot_90
        elif isinstance(camera, str) and (camera in Camera.ROIs):  # device name
            self.camera_name = camera
            self.camera_roi = np.array(Camera.ROIs[camera])
            self.camera_r90 = Camera.get_rot_90(Camera.label_from_name(camera))
        elif isinstance(camera, str) and re.match(r'(U[1-9]|A[1-3]|Rad2|P1)', camera):  # shorthand label ('A1','U3',)
            self.camera_name = Camera.name_from_label(camera)
            self.camera_roi = np.array(Camera.ROIs[self.camera_name])
            self.camera_r90 = Camera.get_rot_90(camera)
        else:
            self.camera_name = camera
            self.camera_roi = None
            self.camera_r90 = 0

        self.image_folder: Path = self.scan.get_folder() / self.supply_name
        self.save_folder: Path = self.scan.get_analysis_folder() / self.supply_name / 'Profiles Images'
        if not self.save_folder.is_dir():
            os.makedirs(self.save_folder)

        self.image_analyses: Optional[list[dict[str, Any]]] = None
        self.analyses_summary: Optional[dict[str, Any]] = None

        self.average_image: Optional[np.ndarray] = None
        self.average_analysis: Optional[dict[str, Any]] = None

    def read_image_as_float(self, image_path: Path) -> np.ndarray:
        image = ni.read_imaq_image(image_path)
        if isinstance(self.camera_roi, np.ndarray) and (self.camera_roi.size >= 4):
            image = image[self.camera_roi[-2]:self.camera_roi[-1], self.camera_roi[0]:self.camera_roi[1]]
        image = np.rot90(image, self.camera_r90)
        return image.astype('float64')


if __name__ == '__main__':
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

    # _base = Path(r'C:\Users\GuillaumePlateau\Documents\LBL\Data')
    _base: Path = Path(r'Z:\data')
    _base_tag = (2023, 4, 20, 48)
    _camera_tag = 'A3'

    _key_device = 'U_S4H'

    _scan = Scan(tag=(2023, 4, 13, 26), experiment_base_path=_base/'Undulator')
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
