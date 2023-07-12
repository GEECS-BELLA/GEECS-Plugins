from __future__ import annotations
import os
import time
import shutil
import cv2
import numpy as np
from typing import Any
from pathlib import Path
from datetime import datetime as dtime
from geecs_api.tools.images.batches import average_images
from geecs_api.api_defs import VarAlias, SysPath
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.interface.geecs_database import GeecsDatabase


class Camera(GeecsDevice):
    # ROIs with [left, right, top, bottom] (x_lim = [:1], y_lim = [-2:])
    ROIs = {'UC_Phosphor1': [166, 1283, 179, 898],
            'UC_ALineEbeam1': [174, 1021, 209, 777],
            'UC_ALineEBeam2': [275, 1114, 154, 778],
            'UC_ALineEBeam3': [183, 858, 187, 740],
            'UC_VisaEBeam1': [589, 918, 336, 619],  # [290, 605, 286, 651],
            'UC_VisaEBeam2': [95, 439, 101, 448],  # [102, 470, 153, 406],
            'UC_VisaEBeam3': [157, 517, 130, 476],  # [163, 483, 140, 472],
            'UC_VisaEBeam4': [186, 548, 204, 554],
            'UC_VisaEBeam5': [150, 497, 96, 444],  # [142, 457, 128, 416],
            'UC_VisaEBeam6': [175, 525, 89, 396],  # [157, 490, 116, 410],
            'UC_VisaEBeam7': [123, 474, 117, 492],  # [121, 475, 129, 478],
            'UC_VisaEBeam8': [111, 486, 90, 494],  # [133, 490, 112, 513],
            'UC_VisaEBeam9': [709, 1125, 269, 681],
            'UC_UndulatorRad2': [1364, 2233, 482, 1251]}  # [545, 2292, 368, 1292]}  # [276, 2515, 204, 1483]}

    def __init__(self, device_name: str):
        super().__init__(device_name)

        # self.gui_path: SysPath = GeecsDevice.exp_info['GUIs'][device_name]

        self.bkg_folder: SysPath = os.path.join(self.data_root_path, 'backgrounds', f'{device_name}')
        if not os.path.isdir(self.bkg_folder):
            os.makedirs(self.bkg_folder)

        self.var_spans = {VarAlias('BackgroundPath'): (None, None),
                          VarAlias('localsavingpath'): (None, None),
                          VarAlias('exposure'): (None, None),
                          VarAlias('triggerdelay'): (None, None)}
        self.build_var_dicts()
        self.var_bkg_path: str = self.var_names_by_index.get(0)[0]
        self.var_save_path: str = self.var_names_by_index.get(1)[0]
        self.var_exposure: str = self.var_names_by_index.get(2)[0]

        if self.get_name() in Camera.ROIs:
            self.roi = np.array(Camera.ROIs[self.get_name()])
        else:
            self.roi = None

        self.label = Camera.label_from_name(self.get_name())
        self.rot_90 = Camera.get_rot_90(self.label)

    @staticmethod
    def get_rot_90(label: str) -> int:
        if label in ['A1', 'A2']:
            return 90
        else:
            return 0

    def state_background_path(self) -> Path:
        return Path(self._state_value(self.var_bkg_path))

    def interpret_value(self, var_alias: VarAlias, val_string: str) -> Any:
        if var_alias in self.var_spans.keys():
            if (var_alias == VarAlias('BackgroundPath')) or (var_alias == VarAlias('localsavingpath')):
                return val_string
            else:
                try:
                    return float(val_string)
                except Exception:
                    return 0.
        else:
            return val_string

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

        if avg_image is not None and avg_image.any():
            cv2.imwrite(file_path, avg_image)
            if set_as_background:
                time.sleep(1.)  # buffer to write file to disk
                self.set(self.var_bkg_path, value=file_path, exec_timeout=10., sync=True)

    def save_background(self, exec_timeout: float = 30.0, set_as_background: bool = True):
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
        avg_image, _ = average_images(images_folder=os.path.join(next_scan_folder, self.get_name()))

        if avg_image is not None and avg_image.any():
            cv2.imwrite(file_path, avg_image)
            if set_as_background:
                time.sleep(1.)  # buffer to write file to disk
                self.set(self.var_bkg_path, value=file_path, exec_timeout=10., sync=True)

    @staticmethod
    def label_from_name(name: str):
        if name[3] == 'A':
            return f'A{name[-1]}'
        elif name[3] == 'V':
            return f'U{name[-1]}'
        elif name[3] == 'U':
            return name[-4:]
        elif name[3] == 'D':
            return 'DP'
        elif name[3] == 'P':
            return 'P1'
        else:
            return name

    @staticmethod
    def name_from_label(label: str):
        labels = [Camera.label_from_name(name) for name in Camera.ROIs.keys()]
        if label in labels:
            return list(Camera.ROIs.keys())[labels.index(label)]
        else:
            return label

    @staticmethod
    def save_multiple_backgrounds(cameras: list[Camera], exec_timeout: float = 30.0, set_as_background: bool = True):
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
                file_name = f'{file_stamp}_{camera.get_name()}_{scan_name}.png'
                file_path: SysPath = os.path.join(camera.bkg_folder, file_name)

                avg_image, _ = average_images(images_folder=os.path.join(next_scan_folder, camera.get_name()))

                if avg_image is not None and avg_image.any():
                    cv2.imwrite(file_path, avg_image)
                    if set_as_background:
                        time.sleep(1.)  # buffer to write file to disk
                        camera.set(camera.var_bkg_path, value=file_path, exec_timeout=10., sync=True)

            except Exception:
                continue


if __name__ == '__main__':
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

    cam = Camera('UC_ALineEBeam3')
    cam.save_local_background(10, set_as_background=True)

    cam.close()
