from __future__ import annotations
import os
import glob
import time
import shutil
import cv2
from typing import Any
from datetime import datetime as dtime
from geecs_api.api_defs import VarAlias, SysPath
from geecs_api.devices.geecs_device import GeecsDevice, api_error
import geecs_api.tools.images.ni_vision as ni


class Camera(GeecsDevice):
    def __init__(self, device_name: str):
        super().__init__(device_name)

        # self.gui_path: SysPath = GeecsDevice.exp_info['GUIs'][device_name]

        self.bkg_folder: SysPath = os.path.join(self.data_root_path, 'backgrounds', f'{device_name}')
        if not os.path.isdir(self.bkg_folder):
            os.makedirs(self.bkg_folder)

        self.__variables = {VarAlias('BackgroundPath'): (None, None),
                            VarAlias('localsavingpath'): (None, None),
                            VarAlias('exposure'): (None, None),
                            VarAlias('triggerdelay'): (None, None)}
        self.build_var_dicts(tuple(self.__variables.keys()))
        self.var_bkg_path: str = self.var_names_by_index.get(0)[0]
        self.var_save_path: str = self.var_names_by_index.get(1)[0]
        self.var_exposure: str = self.var_names_by_index.get(2)[0]

    def get_variables(self):
        return self.__variables

    def interpret_value(self, var_alias: VarAlias, val_string: str) -> Any:
        if var_alias in self.__variables.keys():
            if (var_alias == VarAlias('BackgroundPath')) or (var_alias == VarAlias('localsavingpath')):
                return val_string
            else:
                try:
                    return float(val_string)
                except Exception:
                    return 0.
        else:
            return val_string

    def save_local_background(self, n_images: int = 15):
        # background images folder
        source_path: SysPath = os.path.join(self.bkg_folder, 'tmp_images')
        if os.path.isdir(source_path):
            shutil.rmtree(source_path, ignore_errors=True)
        while True:
            if not os.path.isdir(source_path):
                break
        os.makedirs(source_path)

        self.set(self.var_save_path, value=source_path, exec_timeout=10., sync=True)

        # file name
        stamp = dtime.now()
        file_name = stamp.strftime('%y%m%d_%H%M') + f'_{self.get_name()}_x{n_images}_Local.png'

        # save images
        if n_images > 0:
            self.set('save', 'on', exec_timeout=5.)
            time.sleep(n_images + 2)
            self.set('save', 'off', exec_timeout=5.)

            # wait to write files to disk
            t0 = time.monotonic()
            while True:
                if time.monotonic() - t0 > 10. or len(next(os.walk(source_path))[2]) >= n_images:
                    break

        # average image
        self.calculate_average_image(source_path, self.bkg_folder, file_name, n_images)

    def save_background(self, exec_timeout: float = 30.0):
        next_scan_folder, _ = self.next_scan_folder()
        scan_name = os.path.basename(next_scan_folder)

        # background file name
        stamp = dtime.now()
        file_name = stamp.strftime('%y%m%d_%H%M') + f'_{self.get_name()}_{scan_name}.png'

        # save images
        GeecsDevice.run_no_scan(monitoring_device=self,
                                comment=f'{self.get_name()}: background collection',
                                timeout=exec_timeout)

        # average image
        source_path: SysPath = os.path.join(next_scan_folder, self.get_name())
        self.calculate_average_image(images_folder=os.path.join(next_scan_folder,self.get_name()),
                                     target_folder=self.bkg_folder,
                                     file_name=file_name)

    @staticmethod
    def save_multiple_backgrounds(cameras: list[Camera], exec_timeout: float = 30.0):
        if not cameras:
            return
        next_scan_folder, _ = cameras[0].next_scan_folder()

        # background file name
        stamp = dtime.now()
        file_stamp = stamp.strftime('%y%m%d_%H%M')
        scan_name = os.path.basename(next_scan_folder)

        # save images
        GeecsDevice.run_no_scan(monitoring_device=cameras[0],
                                comment=', '.join([cam.get_name() for cam in cameras]) + ': background collection',
                                timeout=exec_timeout)

        # average images
        for camera in cameras:
            try:
                camera.calculate_average_image(images_folder=os.path.join(next_scan_folder, camera.get_name()),
                                               target_folder=camera.bkg_folder,
                                               file_name=f'{file_stamp}_{camera.get_name()}_{scan_name}.png')
            except Exception:
                continue

    def calculate_average_image(self, images_folder: SysPath, target_folder: SysPath,
                                file_name: str = 'avg_bkg.png', n_images: int = 0):
        images = sorted(glob.glob(os.path.join(images_folder, '*.png')), key=lambda x: x[0].split('_')[-1][:-4])
        if n_images > 0:
            images = images[-n_images:]

        if images:
            try:
                avg_image = ni.read_imaq_image(images[0])
                data_type: str = avg_image.dtype.name
                avg_image = avg_image.astype('float64')

                if len(images) > 1:
                    for it, image_path in enumerate(images[1:]):
                        image_data = ni.read_imaq_image(image_path)
                        image_data = image_data.astype('float64')
                        alpha = 1.0 / (it + 2)
                        beta = 1.0 - alpha
                        avg_image = cv2.addWeighted(image_data, alpha, avg_image, beta, 0.0)

                if not os.path.isdir(target_folder):
                    os.makedirs(target_folder)
                bkg_filepath: SysPath = os.path.join(target_folder, file_name)

                cv2.imwrite(bkg_filepath, avg_image.round().astype(data_type))
                time.sleep(1.)  # buffer to write file to disk
                self.set(self.var_bkg_path, value=bkg_filepath, exec_timeout=10., sync=True)

            except Exception as ex:
                api_error.error(str(ex), 'Failed to calculate average image')
