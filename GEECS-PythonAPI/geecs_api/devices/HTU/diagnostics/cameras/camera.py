from __future__ import annotations
import os
import glob
import cv2
from typing import Any
<<<<<<< Updated upstream
=======
from datetime import datetime as dtime
from geecs_api.tools.images.batches import average_images
>>>>>>> Stashed changes
from geecs_api.api_defs import VarAlias, SysPath
from geecs_api.devices.geecs_device import GeecsDevice


class Camera(GeecsDevice):
    def __init__(self, device_name: str):
        super().__init__(device_name)

        # self.gui_path: SysPath = GeecsDevice.exp_info['GUIs'][device_name]

        self.__variables = {VarAlias('BackgroundPath'): (None, None),
                            VarAlias('localsavingpath'): (None, None),
                            VarAlias('exposure'): (None, None),
                            VarAlias('triggerdelay'): (None, None)}
        self.build_var_dicts(tuple(self.__variables.keys()))
        self.var_bkg_path: str = self.var_names_by_index.get(0)[0]
        self.var_save_path: str = self.var_names_by_index.get(1)[0]
        self.var_exposure: str = self.var_names_by_index.get(2)[0]

        # self.register_cmd_executed_handler()
        # self.register_var_listener_handler()

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

<<<<<<< Updated upstream
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
        avg_image, _ = average_images(source_path, n_images)

        if avg_image:
            cv2.imwrite(file_path, avg_image)
            if set_as_background:
                time.sleep(1.)  # buffer to write file to disk
                self.set(self.var_bkg_path, value=file_path, exec_timeout=10., sync=True)

    def save_background(self, exec_timeout: float = 30.0, set_as_background: bool = True):
>>>>>>> Stashed changes
        next_scan_folder, _ = self.next_scan_folder()
        bkg_folder: SysPath = os.path.join(next_scan_folder, f'{self.get_name()}_Background')

        # save images
        self.run_no_scan(f'{self.get_name()}: background collection', timeout=exec_timeout)

        # average image
<<<<<<< Updated upstream
        saving_path: SysPath = os.path.join(next_scan_folder, self.get_name())
        self._calculate_average_image(self, saving_path, bkg_folder)
=======
        avg_image, _ = average_images(images_folder=os.path.join(next_scan_folder, self.get_name()))
>>>>>>> Stashed changes

    def save_multiple_backgrounds(self, cameras: list[Camera], exec_timeout: float = 30.0):
        if not cameras:
            return

        # background folders
        next_scan_folder, _ = cameras[0].next_scan_folder()
        bkg_folders: list[SysPath] = [os.path.join(next_scan_folder, f'{cam.get_name()}_Background') for cam in cameras]

        # save images
        self.run_no_scan(', '.join([cam.get_name() for cam in cameras]) + ': background collection', exec_timeout)

        # image averages
        saving_paths: list[SysPath] = [os.path.join(next_scan_folder, cam.get_name()) for cam in cameras]
        for it in range(len(cameras)):
            Camera._calculate_average_image(cameras[it], saving_paths[it], bkg_folders[it])

    @staticmethod
    def _calculate_average_image(camera: Camera, saving_path: SysPath, bkg_folder: SysPath):
        images = glob.glob(os.path.join(saving_path, '*.png'))
        if images:
            try:
                avg_image = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
                if len(images) > 1:
                    for it in range(len(images) - 1):
                        image_data = cv2.imread(images[it + 1], cv2.IMREAD_GRAYSCALE)
                        alpha = 1.0 / (it + 2)
                        beta = 1.0 - alpha
                        avg_image = cv2.addWeighted(image_data, alpha, avg_image, beta, 0.0)

<<<<<<< Updated upstream
                if not os.path.isdir(bkg_folder):
                    os.makedirs(bkg_folder)
                bkg_filepath: SysPath = os.path.join(bkg_folder, 'avg_bkg.png')
                cv2.imwrite(bkg_filepath, avg_image)
                camera.set(camera.var_bkg_path, value=bkg_filepath, exec_timeout=10., sync=True)
=======
                avg_image, _ = average_images(images_folder=os.path.join(next_scan_folder, camera.get_name()))

                if avg_image:
                    cv2.imwrite(file_path, avg_image)
                    if set_as_background:
                        time.sleep(1.)  # buffer to write file to disk
                        camera.set(camera.var_bkg_path, value=file_path, exec_timeout=10., sync=True)
>>>>>>> Stashed changes

            except Exception:
                pass
