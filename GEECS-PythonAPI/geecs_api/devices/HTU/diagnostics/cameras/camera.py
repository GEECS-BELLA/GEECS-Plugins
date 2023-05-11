from __future__ import annotations
import os
import time
import shutil
import cv2
from typing import Any
<<<<<<< HEAD
<<<<<<< HEAD
=======
from datetime import datetime as dtime
from geecs_api.tools.images.batch_analyses import average_images
>>>>>>> parent of 7a19ff8 (Merge branch 'htu-labview-python-bridge')
=======
>>>>>>> parent of 4e1a7d3 (Lots of rearranging and cleaning to eventually merge with Reinier's approach. Added a bunch of image processing stuff)
from geecs_api.api_defs import VarAlias, SysPath
from geecs_api.devices.geecs_device import GeecsDevice


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

<<<<<<< HEAD
<<<<<<< HEAD
    def save_background(self, exec_timeout: float = 30.0):
        # background folder
=======
    def save_local_background(self, n_images: int = 15, set_as_background: bool = True):
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
        file_path: SysPath = os.path.join(self.bkg_folder, file_name)

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
        avg_image = average_images(source_path, n_images)

        if avg_image:
            cv2.imwrite(file_path, avg_image)
            if set_as_background:
                time.sleep(1.)  # buffer to write file to disk
                self.set(self.var_bkg_path, value=file_path, exec_timeout=10., sync=True)

    def save_background(self, exec_timeout: float = 30.0, set_as_background: bool = True):
>>>>>>> parent of 7a19ff8 (Merge branch 'htu-labview-python-bridge')
=======
    def save_background(self, exec_timeout: float = 30.0):
        # background folder
>>>>>>> parent of 4e1a7d3 (Lots of rearranging and cleaning to eventually merge with Reinier's approach. Added a bunch of image processing stuff)
        next_scan_folder, _ = self.next_scan_folder()
        scan_name = os.path.basename(next_scan_folder)

        # background file name
        stamp = dtime.now()
        file_name: str = stamp.strftime('%y%m%d_%H%M') + f'_{self.get_name()}_{scan_name}.png'
        file_path: SysPath = os.path.join(self.bkg_folder, file_name)

        # save images
        GeecsDevice.run_no_scan(monitoring_device=self,
                                comment=f'{self.get_name()}: background collection',
                                timeout=exec_timeout)

        # average image
<<<<<<< HEAD
<<<<<<< HEAD
        saving_path: SysPath = os.path.join(next_scan_folder, self.get_name())
        self._calculate_average_image(self, saving_path, bkg_folder)
=======
        avg_image = average_images(images_folder=os.path.join(next_scan_folder, self.get_name()))
>>>>>>> parent of 7a19ff8 (Merge branch 'htu-labview-python-bridge')
=======
        saving_path: SysPath = os.path.join(next_scan_folder, self.get_name())
        self._calculate_average_image(self, saving_path, bkg_folder)
>>>>>>> parent of 4e1a7d3 (Lots of rearranging and cleaning to eventually merge with Reinier's approach. Added a bunch of image processing stuff)

        if avg_image:
            cv2.imwrite(file_path, avg_image)
            if set_as_background:
                time.sleep(1.)  # buffer to write file to disk
                self.set(self.var_bkg_path, value=file_path, exec_timeout=10., sync=True)

    @staticmethod
    def save_multiple_backgrounds(cameras: list[Camera], exec_timeout: float = 30.0, set_as_background: bool = True):
        if not cameras:
            return
        next_scan_folder, _ = cameras[0].next_scan_folder()

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> parent of 4e1a7d3 (Lots of rearranging and cleaning to eventually merge with Reinier's approach. Added a bunch of image processing stuff)
                if not os.path.isdir(bkg_folder):
                    os.makedirs(bkg_folder)
                bkg_filepath: SysPath = os.path.join(bkg_folder, 'avg_bkg.png')
                cv2.imwrite(bkg_filepath, avg_image)
                camera.set(camera.var_bkg_path, value=bkg_filepath, exec_timeout=10., sync=True)
<<<<<<< HEAD
=======
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
                file_name = f'{file_stamp}_{camera.get_name()}_{scan_name}.png'
                file_path: SysPath = os.path.join(camera.bkg_folder, file_name)

                avg_image = average_images(images_folder=os.path.join(next_scan_folder, camera.get_name()))

                if avg_image:
                    cv2.imwrite(file_path, avg_image)
                    if set_as_background:
                        time.sleep(1.)  # buffer to write file to disk
                        camera.set(camera.var_bkg_path, value=file_path, exec_timeout=10., sync=True)
>>>>>>> parent of 7a19ff8 (Merge branch 'htu-labview-python-bridge')
=======
>>>>>>> parent of 4e1a7d3 (Lots of rearranging and cleaning to eventually merge with Reinier's approach. Added a bunch of image processing stuff)

            except Exception:
                continue
