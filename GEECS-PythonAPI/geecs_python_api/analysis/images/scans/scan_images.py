""" @author: Guillaume Plateau, TAU Systems """

import cv2
import os
import re
import math
import numpy as np
import screeninfo
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from progressbar import ProgressBar
from typing import Optional, Any, Union
from geecs_python_api.controls.api_defs import ScanTag
import geecs_python_api.controls.experiment.htu as htu
from geecs_python_api.tools.images.batches import list_files
from geecs_python_api.controls.devices.geecs_device import api_error
import geecs_python_api.tools.images.ni_vision as ni
from geecs_python_api.tools.interfaces.exports import save_py
from geecs_python_api.analysis.images.scans.scan_data import ScanData
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.controls.devices.HTU.diagnostics.cameras import Camera
from geecs_python_api.tools.images.filtering import FiltersParameters
from geecs_python_api.tools.images.spot import spot_analysis, fwhm
from geecs_python_api.tools.interfaces.prompts import text_input

from image_analysis.analyzers.UC_BeamSpot import UC_BeamSpotImageAnalyzer
from image_analysis.utils import ROI as ImageAnalyzerROI


class ScanImages:
    fig_size = (int(round(screeninfo.get_monitors()[0].width / 540. * 10) / 10),
                int(round(screeninfo.get_monitors()[0].height / 450. * 10) / 10))

    def __init__(self, scan: ScanData, camera: Union[int, Camera, str], angle: Optional[int] = None):
        """
        Container for data analysis of a set of images collected at an undulator station.

        scan (Scan object): analysis object for the relevant scan
        camera (int | Camera | str), either of:
            - station number (1-9)
            - Camera object
            - GEECS device name of the relevant camera
            - relevant screen shorthand label (U1 - U9)
        angle (int): rotation angle to apply (multiples of +/-90 deg only). Ignored if camera object is provided.
        """

        self.scan_scalar_data = scan.data_dict
        self.scan_data_folder: Path = scan.get_folder()
        self.scan_analysis_folder: Path = scan.get_analysis_folder()

        self.camera: Optional[Camera] = None
        self.camera_name: str
        self.camera_roi: Optional[np.ndarray]
        self.camera_r90: int

        if isinstance(camera, Camera):  # Camera class object
            self.camera = camera
            self.camera_name = camera.get_name()
            self.camera_roi: Optional[np.ndarray] = camera.roi
            self.camera_r90 = camera.rot_90
        elif isinstance(camera, str) and (camera in Camera.ROIs):  # device name
            self.camera_name = camera
            self.camera_roi = np.array(Camera.ROIs[camera])
            self.camera_r90 = Camera.get_rot_90(Camera.label_from_name(camera))
        elif isinstance(camera, str) and re.match(r'(U[1-9]|A[1-3]|Rad2|P1|DP)', camera):  # shorthand label ('A1', )
            self.camera_name = Camera.name_from_label(camera)
            if (self.camera_name in Camera.ROIs) and (len(Camera.ROIs[self.camera_name]) == 4):
                self.camera_roi = np.array(Camera.ROIs[self.camera_name])
            else:
                self.camera_roi = None
            self.camera_r90 = Camera.get_rot_90(camera)
        elif isinstance(camera, int) and (1 <= camera <= 9):  # undulator screen number
            self.camera_name = Camera.name_from_label(f'U{camera}')
            self.camera_roi = np.array(Camera.ROIs[self.camera_name])
            self.camera_r90 = Camera.get_rot_90(f'U{camera}')
        else:
            self.camera_name = camera
            self.camera_roi = None
            self.camera_r90 = 0

        if angle:
            self.camera_r90: int = int(round(angle / 90.))
        self.raw_shape_ij: tuple[int, int] = (0, 0)  # to be updated by latest opened image

        self.camera_label: str = Camera.label_from_name(self.camera_name)

        # self.image_folder: Path = self.scan_obj.get_folder() / self.camera_name
        # self.save_folder: Path = self.scan_obj.get_analysis_folder() / self.camera_name
        self.image_folder: Path = self.scan_data_folder / self.camera_name
        self.save_folder: Path = self.scan_analysis_folder / self.camera_name
        self.set_save_folder()  # default no-scan analysis folder

        self.analysis: dict[str, Any] = ScanImages._new_analysis_dict()
        self.analyses: Optional[list[dict[str, Any]]] = None
        self.summary: Optional[dict[str, Any]] = None

        self.average_image: Optional[np.ndarray] = None
        self.average_analysis: Optional[dict[str, Any]] = None

    @staticmethod
    def _new_analysis_dict() -> dict[str, Any]:
        return {'paths': {}, 'camera': {}, 'filters': {}, 'arrays': {}, 'positions': {}, 'flags': {}, 'metrics': {}}

    def set_save_folder(self, path: Optional[Path] = None):
        if path and path.is_dir():
            self.save_folder = path
        if path is None:
            # self.save_folder = self.scan_obj.get_analysis_folder() / self.camera_name
            self.save_folder = self.scan_analysis_folder / self.camera_name
            if not self.save_folder.is_dir():
                os.makedirs(self.save_folder)

    def read_image_as_float(self, image_path: Path) -> np.ndarray:
        image = ni.read_imaq_image(image_path)
        self.raw_shape_ij = image.shape
        if isinstance(self.camera_roi, np.ndarray) and (self.camera_roi.size >= 4):
            image = image[self.camera_roi[-2]:self.camera_roi[-1]+1, self.camera_roi[0]:self.camera_roi[1]+1]
        image = np.rot90(image, self.camera_r90 // 90)
        return image.astype(float)

    def run_analysis_with_checks(self, images: Union[int, list[Path]] = -1, initial_filtering=FiltersParameters(),
                                 profiles: tuple = ('com',), plots: bool = False, store_images: bool = True,
                                 save_plots: bool = False, save: bool = False) -> tuple[Optional[Path], dict[str, Any]]:

        export_file_path: Optional[Path] = None
        data_dict: dict[str, Any] = {}
        filtering = FiltersParameters(initial_filtering.contrast,
                                      initial_filtering.hp_median, initial_filtering.hp_threshold,
                                      initial_filtering.denoise_cycles, initial_filtering.gauss_filter,
                                      initial_filtering.com_threshold, initial_filtering.bkg_image,
                                      initial_filtering.box, initial_filtering.ellipse)

        while True:
            try:
                export_file_path, data_dict = \
                    self.analyze_image_batch(images, filtering, store_images, plots, profiles, save_plots, save)
            except Exception as ex:
                api_error(str(ex), f'Failed to analyze {self.scan_data_folder.name}')
                pass

            repeat = text_input(f'Repeat analysis (adjust contrast/threshold)? : ',
                                accepted_answers=['y', 'yes', 'n', 'no'])
            if repeat.lower()[0] == 'n':
                break
            else:
                while True:
                    try:
                        filtering.contrast = \
                            float(text_input(f'New contrast value (old: {filtering.contrast:.3f}) : '))
                        filtering.com_threshold = \
                            float(text_input(f'New threshold value (old: {filtering.com_threshold:.3f}) : '))
                        break
                    except Exception:
                        print('Contrast value must be a positive number (e.g. 1.3)')
                        continue

        return export_file_path, data_dict

    def analyze_image_batch(self, images: Union[int, list[Path]] = -1, filtering=FiltersParameters(),
                            store_images: bool = True, plots: bool = False, profiles: tuple = ('com',),
                            save_plots: bool = False, save: bool = False) -> tuple[Optional[Path], dict[str, Any]]:
        export_file_path: Optional[Path] = None
        data_dict: dict[str, Any] = {}

        if isinstance(images, int):
            paths = list_files(self.image_folder, images, '.png')
        else:
            paths = images

        # paths = paths[:10]  # tmp
        skipped_files = []
        self.analyses = []

        # run analysis
        if paths:
            try:
                with ProgressBar(max_value=len(paths)) as pb:
                    count = 0
                    avg_image: Optional[np.ndarray] = None
                    for image_path in paths:
                        avg_image, count = \
                            self.analyze_average_render(image_path, filtering, count, avg_image, profiles,
                                                        plots, block_plot=False, save_plots=save_plots)
                        if not self.analysis['flags']['is_valid']:
                            skipped_files.append(image_path.name)

                        if not store_images:
                            list_keys = list(self.analysis['arrays'].keys())
                            for k in list_keys:
                                if k != 'denoised':
                                    self.analysis['arrays'].pop(k)

                        self.analyses.append(self.analysis)
                        pb.increment()

                # report skipped files
                print(f'Successful analyses: {count}')
                if skipped_files:
                    print(f'Skipped:')
                    for file in skipped_files:
                        print(f'\t{file}')
                else:
                    print('No file skipped.')

                # analyze the average image
                try:
                    self.analyze_image(avg_image, filtering)
                    self.analyze_profiles()

                    if plots:
                        # always save average image
                        ScanImages.render_image_analysis(self.analysis, tag='average_image', profiles=profiles,
                                                         block=True, save_folder=self.save_folder)

                    if not store_images:
                        list_keys = list(self.analysis['arrays'].keys())
                        for k in list_keys:
                            if k != 'denoised':
                                self.analysis['arrays'].pop(k)

                except Exception as ex:
                    api_error.error(str(ex), 'Failed to analyze average image')

                self.average_image = avg_image
                self.average_analysis = self.analysis
                self.average_analysis['positions']['raw'] = {}
                for pos in ['max_ij', 'com_ij']:
                    self.average_analysis['positions']['raw'][pos] = \
                        ScanImages.processed_to_original_ij(self.average_analysis['positions'][pos],
                                                            self.average_analysis['filters']['roi'],
                                                            self.average_analysis['camera']['r90'])
                self.summarize_analyses()

                data_dict = {
                    'camera_r90': self.camera_r90,
                    'camera_name': self.camera_name,
                    'camera_roi': self.camera_roi,
                    'camera_label': self.camera_label,
                    'image_folder': self.image_folder,
                    'save_folder': self.save_folder,
                    'image_analyses': self.analyses if self.analyses is not None else {},
                    'summary': self.summary if self.summary is not None else {},
                    'average_image': self.average_image if self.average_image is not None else {},
                    'average_analysis': self.average_analysis if self.average_analysis is not None else {}}

                if save:
                    print(f'Figures saved in:\n\t{self.save_folder}')

                    # export_file_path: Path = self.save_folder.parent / 'profiles_analysis'
                    export_file_path = self.save_folder / 'profiles_analysis'
                    save_py(file_path=export_file_path, data=data_dict)
                    print(f'Data exported to:\n\t{export_file_path}.dat')

            except Exception as ex:
                api_error.error(str(ex), 'Failed to analyze image')
                pass

        return export_file_path, data_dict

    def analyze_average_render(self, image: Path, filtering=FiltersParameters(), count: int = 0,
                               avg_image: Optional[np.ndarray] = None, profiles: tuple = ('com',),
                               plots: bool = False, block_plot: bool = False, save_plots: bool = False)\
            -> tuple[np.ndarray, int]:
        self.analyze_image(image, filtering)

        if self.analysis['flags']['is_valid']:
            # profiles
            count += 1
            self.analyze_profiles()
            if plots:
                save_folder = self.save_folder if save_plots else None
                ScanImages.render_image_analysis(self.analysis, profiles=profiles,
                                                 block=block_plot, save_folder=save_folder)

            # average
            if avg_image is None:
                avg_image = self.analysis['arrays']['denoised'].copy()
            else:
                # alpha = 1.0 / (it + 2)
                alpha = 1.0 / count
                beta = 1.0 - alpha
                avg_image = cv2.addWeighted(self.analysis['arrays']['denoised'], alpha, avg_image, beta, 0.0)

        return avg_image, count

    def analyze_image(self, image: Union[Path, np.ndarray], filtering=FiltersParameters(), sigma_radius: float = 5):
        self.analysis = ScanImages._new_analysis_dict()

        # raw image
        if isinstance(image, np.ndarray):
            image_raw = image
        else:
            self.analysis['paths']['image'] = image
            image_raw = self.read_image_as_float(image)

            if isinstance(filtering.bkg_image, Path):
                self.analysis['paths']['background'] = filtering.bkg_image
                image_raw -= self.read_image_as_float(filtering.bkg_image)
            if isinstance(filtering.bkg_image, np.ndarray):
                image_raw -= filtering.bkg_image

        # camera info
        self.analysis['camera']['name'] = self.camera_name
        self.analysis['camera']['r90'] = self.camera_r90
        self.analysis['camera']['raw_shape_ij'] = self.raw_shape_ij

        try:
            self.analysis['camera']['um_per_pix'] = \
                float(GeecsDevice.exp_info['devices'][self.camera_name]['SpatialCalibration']['defaultvalue'])
        except Exception:
            self.analysis['camera']['um_per_pix'] = 1.

        try:
            self.analysis['camera']['target_raw_ij'] = (
                int(float(GeecsDevice.exp_info['devices'][self.camera_name]['Target.Y']['defaultvalue'])),
                int(float(GeecsDevice.exp_info['devices'][self.camera_name]['Target.X']['defaultvalue'])))
        except Exception:
            self.analysis['camera']['target_raw_ij'] = (np.nan, np.nan)

        # now we're ready to run the analysis!
        image_analyzer = UC_BeamSpotImageAnalyzer(
            # image is already cropped to self.camera_roi but this is currently
            # passed to save in the analysis dict
            camera_roi=ImageAnalyzerROI(left=self.camera_roi[0], right=self.camera_roi[1] + 1,
                                        top=self.camera_roi[2], bottom=self.camera_roi[3] + 1),
            contrast=filtering.contrast,
            hp_median=filtering.hp_median, 
            hp_threshold=filtering.hp_threshold,
            denoise_cycles=filtering.denoise_cycles,
            gauss_filter=filtering.gauss_filter,
            com_threshold=filtering.com_threshold,
            background=None,  # background already subtracted above
            box=filtering.box,
            sigma_radius=sigma_radius,
        )
        image_analysis = image_analyzer.analyze_image(image_raw)
        # image_analysis contains keys filter_pars, positions, filters, arrays, 
        # metrics, flags. These should all be empty in self.analysis, so the 
        # update action shouldn't overwrite anything. Assert that this is the case.
        assert all((k not in self.analysis) or (not self.analysis[k]) for k in image_analysis)
        self.analysis.update(image_analysis)

    def analyze_profiles(self):
        try:
            positions = [(*self.analysis['positions']['max_ij'], 'max'),
                         (*self.analysis['positions']['com_ij'], 'com')]

            if 'box_ij' in self.analysis['positions']:
                positions.append((*self.analysis['positions']['box_ij'], 'box'))

            if 'ellipse_ij' in self.analysis['positions']:
                positions.append((*self.analysis['positions']['ellipse_ij'], 'ell'))

            x_win = (self.analysis['metrics']['roi_com'][0], self.analysis['metrics']['roi_com'][1]) \
                if 'roi_com' in self.analysis['metrics'] else None
            y_win = (self.analysis['metrics']['roi_com'][2], self.analysis['metrics']['roi_com'][3]) \
                if 'roi_com' in self.analysis['metrics'] else None
            # noinspection PyTypeChecker
            profiles = spot_analysis(self.analysis['arrays']['denoised'], positions, x_window=x_win, y_window=y_win)

            if profiles:
                self.analysis['metrics']['profiles'] = profiles

        except Exception as ex:
            api_error.error(str(ex), 'Failed to analyze profiles')
            pass

    def summarize_analyses(self):
        self.summary = {'positions_roi': {'data': {}, 'fit': {}},
                        'positions_raw': {'data': {}, 'fit': {}},
                        'deltas': {'data': {}, 'fit': {}},
                        'fwhms': {'pix_ij': {}, 'um_xy': {}},
                        'um_per_pix': float}

        # for each image analysis
        for analysis in self.analyses:
            if (analysis is not None) and analysis['flags']['is_valid'] and ('profiles' in analysis['metrics']):
                # collect values from ROI analysis
                for pos in analysis['positions']['short_names']:
                    pos_ij = analysis['positions'][f'{pos}_ij']
                    profile = analysis['metrics']['profiles'][pos]
                    fit_mean_ij = (profile['y']['opt'][2], profile['x']['opt'][2])
                    fit_fwhm_ij = (fwhm(profile['y']['opt'][3]), fwhm(profile['x']['opt'][3]))

                    if f'{pos}_ij' in self.summary['positions_roi']['data']:
                        self.summary['positions_roi']['data'][f'{pos}_ij'] = \
                            np.concatenate([self.summary['positions_roi']['data'][f'{pos}_ij'], [pos_ij]])
                        self.summary['positions_roi']['fit'][f'{pos}_ij'] = \
                            np.concatenate([self.summary['positions_roi']['fit'][f'{pos}_ij'], [fit_mean_ij]])
                        self.summary['fwhms']['pix_ij'][pos] = \
                            np.concatenate([self.summary['fwhms']['pix_ij'][pos], [fit_fwhm_ij]])
                    else:
                        self.summary['positions_roi']['data'][f'{pos}_ij'] = np.array([pos_ij])
                        self.summary['positions_roi']['fit'][f'{pos}_ij'] = np.array([fit_mean_ij])
                        self.summary['fwhms']['pix_ij'][pos] = np.array([fit_fwhm_ij])

        um_per_pix: float = self.analyses[0]['camera']['um_per_pix']
        self.summary['um_per_pix'] = um_per_pix

        # convert to raw image positions, um FWHMs, and calculate target deltas
        for k in self.summary['positions_roi']['data'].keys():
            # self.summary['positions_raw']['data'][k] = \
            #     ScanImages.processed_to_original_ij(self.summary['positions_roi']['data'][k],
            #                                         self.analyses[0]['camera']['raw_shape_ij'],
            #                                         self.analyses[0]['camera']['r90'])
            # self.summary['positions_raw']['fit'][k] = \
            #     ScanImages.processed_to_original_ij(self.summary['positions_roi']['fit'][k],
            #                                         self.analyses[0]['camera']['raw_shape_ij'],
            #                                         self.analyses[0]['camera']['r90'])
            self.summary['positions_raw']['data'][k] = \
                ScanImages.processed_to_original_ij(self.summary['positions_roi']['data'][k],
                                                    self.analyses[0]['filters']['roi'],
                                                    self.analyses[0]['camera']['r90'])
            self.summary['positions_raw']['fit'][k] = \
                ScanImages.processed_to_original_ij(self.summary['positions_roi']['fit'][k],
                                                    self.analyses[0]['filters']['roi'],
                                                    self.analyses[0]['camera']['r90'])

            self.summary['fwhms']['um_xy'][k[:-3]] = np.fliplr(self.summary['fwhms']['pix_ij'][k[:-3]]) * um_per_pix

            self.summary['deltas']['data']['pix_ij'], _ = \
                ScanImages.calculate_target_offset(self.analyses[0], self.summary['positions_roi']['data'][k])
            self.summary['deltas']['data']['um_xy'] = \
                np.fliplr(self.summary['deltas']['data']['pix_ij']) * um_per_pix

            self.summary['deltas']['fit']['pix_ij'], _ = \
                ScanImages.calculate_target_offset(self.analyses[0], self.summary['positions_roi']['fit'][k])
            self.summary['deltas']['fit']['um_xy'] = \
                np.fliplr(self.summary['deltas']['fit']['pix_ij']) * um_per_pix

    @staticmethod  # tmp
    def old_processed_to_original_ij(coords_ij: Union[tuple[int, int], np.ndarray], raw_shape_ij: tuple[int, int],
                                     rot_deg: int = 0) -> Union[tuple[int, int], np.ndarray]:
        rots: int = divmod(rot_deg, 90)[0] % 4
        raw_coords: Union[tuple[int, int], np.ndarray]
        is_tuple: bool = isinstance(coords_ij, tuple)
        is_1d: bool = isinstance(coords_ij, np.ndarray) and (coords_ij.ndim == 1)

        if is_tuple or is_1d:
            coords_ij = np.array([coords_ij])

        if rots == 1:
            # 90deg: [j, j_max - i - 1]
            raw_coords = np.stack([coords_ij[:, 1],
                                  raw_shape_ij[1] - coords_ij[:, 0] - 1]).transpose()
        elif rots == 2:
            # 180deg: [i_max - i - 1, j_max - j - 1]
            raw_coords = np.stack([raw_shape_ij[0] - coords_ij[:, 0] - 1,
                                  raw_shape_ij[1] - coords_ij[:, 1] - 1]).transpose()
        elif rots == 3:
            # 270deg: [i_max - j - 1, i]
            raw_coords = np.stack([raw_shape_ij[0] - coords_ij[:, 1] - 1,
                                  coords_ij[:, 0]]).transpose()
        else:
            # 0deg: [i, j]
            raw_coords = coords_ij

        if is_tuple:
            raw_coords = tuple(raw_coords[0])

        if is_1d:
            raw_coords = raw_coords[0]

        return raw_coords

    @staticmethod
    def processed_to_original_ij(coords_ij: Union[tuple[int, int], np.ndarray],
                                 roi_ij: list[int, int, int, int], rot_deg: int = 0) \
            -> Union[tuple[int, int], np.ndarray]:
        rots: int = divmod(rot_deg, 90)[0] % 4
        sub_coords: Union[tuple[int, int], np.ndarray]
        is_tuple: bool = isinstance(coords_ij, tuple)
        is_1d: bool = isinstance(coords_ij, np.ndarray) and (coords_ij.ndim == 1)
        sub_shape_ij: tuple[int, int]

        if is_tuple or is_1d:
            coords_ij = np.array([coords_ij])

        if rots == 1:
            # 90deg: [j, j_max - i - 1]
            sub_shape_ij: tuple[int, int] = (roi_ij[1] - roi_ij[0] + 1,
                                             roi_ij[3] - roi_ij[2] + 1)
            sub_coords = np.stack([coords_ij[:, 1],
                                  sub_shape_ij[0] - coords_ij[:, 0] - 1]).transpose()
        elif rots == 2:
            # 180deg: [i_max - i - 1, j_max - j - 1]
            sub_shape_ij: tuple[int, int] = (roi_ij[3] - roi_ij[2] + 1,
                                             roi_ij[1] - roi_ij[0] + 1)
            sub_coords = np.stack([sub_shape_ij[0] - coords_ij[:, 0] - 1,
                                  sub_shape_ij[1] - coords_ij[:, 1] - 1]).transpose()
        elif rots == 3:
            # 270deg: [i_max - j - 1, i]
            sub_shape_ij: tuple[int, int] = (roi_ij[1] - roi_ij[0] + 1,
                                             roi_ij[3] - roi_ij[2] + 1)
            sub_coords = np.stack([sub_shape_ij[1] - coords_ij[:, 1] - 1,
                                  coords_ij[:, 0]]).transpose()
        else:
            # 0deg: [i, j]
            sub_coords = coords_ij

        raw_coords = sub_coords + np.array([roi_ij[2], roi_ij[0]])

        if is_tuple:
            raw_coords = tuple(raw_coords[0])

        if is_1d:
            raw_coords = raw_coords[0]

        return raw_coords

    @staticmethod
    def original_to_processed_ij(coords_ij: Union[tuple[int, int], np.ndarray], raw_shape_ij: tuple[int, int],
                                 rot_deg: int = 0) -> tuple[int, int]:
        rots: int = divmod(rot_deg, 90)[0] % 4
        raw_coords: Union[tuple[int, int], np.ndarray]
        is_tuple: bool = isinstance(coords_ij, tuple)
        is_1d: bool = isinstance(coords_ij, np.ndarray) and (coords_ij.ndim == 1)

        if is_tuple or is_1d:
            coords_ij = np.array([coords_ij])

        if rots == 1:
            # 90deg: [j_max - j - 1, i]
            raw_coords = np.stack([raw_shape_ij[1] - coords_ij[:, 1] - 1,
                                   coords_ij[:, 0]]).transpose()
        elif rots == 2:
            # 180deg: [i_max - i - 1, j_max - j - 1]
            raw_coords = np.stack([raw_shape_ij[0] - coords_ij[:, 0] - 1,
                                   raw_shape_ij[1] - coords_ij[:, 1] - 1]).transpose()
        elif rots == 3:
            # 270deg: [j, i_max - i - 1]
            raw_coords = np.stack([coords_ij[:, 1],
                                  raw_shape_ij[0] - coords_ij[:, 0] - 1]).transpose()
        else:
            # 0deg: [i, j]
            raw_coords = coords_ij

        if is_tuple:
            raw_coords = tuple(raw_coords[0])

        if is_1d:
            raw_coords = raw_coords[0]

        return raw_coords

    @staticmethod
    def calculate_target_offset(analysis: dict[str, Any], pos_roi_ij: Union[tuple[int, int], np.ndarray]) \
            -> tuple[Union[tuple[int, int], np.ndarray], float]:
        rot_deg: int = analysis['camera']['r90']
        # raw_shape_ij: tuple[int, int] = analysis['camera']['raw_shape_ij']
        roi_ij: list[int, int, int, int] = analysis['filters']['roi']
        um_per_pix: float = analysis['camera']['um_per_pix']
        target_raw_ij: tuple[int, int] = analysis['camera']['target_raw_ij']

        is_tuple: bool = isinstance(pos_roi_ij, tuple)
        is_1d: bool = isinstance(pos_roi_ij, np.ndarray) and (pos_roi_ij.ndim == 1)

        if is_tuple or is_1d:
            pos_roi_ij = np.array([pos_roi_ij])

        # pos_raw_ij = ScanImages.processed_to_original_ij(pos_roi_ij, raw_shape_ij, rot_deg)
        pos_raw_ij = ScanImages.processed_to_original_ij(pos_roi_ij, roi_ij, rot_deg)
        delta_raw_ij = pos_raw_ij - np.array(target_raw_ij)

        if is_tuple:
            delta_raw_ij = tuple(delta_raw_ij[0])

        if is_1d:
            delta_raw_ij = delta_raw_ij[0]

        return delta_raw_ij, um_per_pix

    @staticmethod
    def render_image_analysis(analysis: dict[str, Any], tag: str = '', profiles: tuple[str] = ('com',),
                              comparison: bool = True, block: bool = False, save_folder: Optional[Path] = None):
        if not block and save_folder is None:
            return

        try:
            um_per_pix: float = analysis['camera']['um_per_pix']
            v_max = None
            fig_1 = None

            if not tag:
                image_path: Path = Path(analysis['paths']['image'])
                tag = image_path.name.split(".")[0].split("_")[-1]

            # raw roi
            roi_raw: np.ndarray
            if 'roi' in analysis['filters']:
                roi_raw = analysis['filters']['roi']
            else:
                roi_raw = np.zeros((4,))

            for profile in profiles:
                if ('profiles' in analysis['metrics']) and (profile in analysis['metrics']['profiles']):
                    fig_1 = plt.figure(figsize=(ScanImages.fig_size[0] * 1.5, ScanImages.fig_size[1]))

                    grid = plt.GridSpec(2, 4, hspace=0.3, wspace=0.3)
                    ax_i = fig_1.add_subplot(grid[:, :2])
                    ax_x = fig_1.add_subplot(grid[0, 2:])
                    ax_y = fig_1.add_subplot(grid[1, 2:], sharey=ax_x)

                    # raw image
                    if 'arrays' in analysis:
                        v_max = 1.5 * analysis['arrays']['denoised'][analysis['positions']['max_ij'][0],
                                                                     analysis['positions']['max_ij'][1]]
                        ax_i.imshow(analysis['arrays']['denoised'],
                                    cmap='hot', aspect='equal', origin='upper', vmin=0, vmax=v_max)
                        ax_i.set_title(r'ROI (x$_{min/max}$, y$_{min/max}$): ' + f'{roi_raw}')

                        if 'edges' in analysis['arrays']:
                            edges = np.where(analysis['arrays']['edges'] != 0)
                            ax_i.scatter(edges[1], edges[0], s=0.3, c='b', alpha=0.2)

                    # roi com
                    if 'roi_com' in analysis['metrics']:
                        roi: np.ndarray = analysis['metrics']['roi_com']
                        rect = mpatches.Rectangle((roi[0], roi[2]), roi[1] - roi[0], roi[3] - roi[2],
                                                  fill=False, edgecolor='cyan', linestyle='--', linewidth=0.5)
                        ax_i.add_patch(rect)

                    # roi edges
                    if 'roi_edges' in analysis['metrics']:
                        roi: np.ndarray = analysis['metrics']['roi_edges']
                        rect = mpatches.Rectangle((roi[0], roi[2]), roi[1] - roi[0], roi[3] - roi[2],
                                                  fill=False, edgecolor='yellow', linestyle='--', linewidth=0.5)
                        ax_i.add_patch(rect)

                    # ellipse
                    if 'ellipse' in analysis['metrics']:
                        ell = mpatches.Ellipse(
                            xy=(analysis['metrics']['ellipse'][1], analysis['metrics']['ellipse'][0]),
                            width=2*analysis['metrics']['ellipse'][3], height=2*analysis['metrics']['ellipse'][2],
                            angle=math.degrees(analysis['metrics']['ellipse'][4]))
                        ax_i.add_artist(ell)
                        ell.set_alpha(0.66)
                        ell.set_edgecolor('g')
                        ell.set_linewidth(1)
                        # noinspection PyArgumentList
                        ell.set_fill(False)

                    # profile lineouts
                    ax_i.axvline(analysis['positions'][f'{profile}_ij'][1], color='w', linestyle='--', linewidth=0.5)
                    ax_i.axhline(analysis['positions'][f'{profile}_ij'][0], color='w', linestyle='--', linewidth=0.5)
                    ax_i.plot(analysis['positions'][f'{profile}_ij'][1],
                              analysis['positions'][f'{profile}_ij'][0], 'w.', markersize=3)
                    if ax_i.xaxis_inverted():
                        ax_i.invert_xaxis()
                    if not ax_i.yaxis_inverted():
                        ax_i.invert_yaxis()

                    # lineouts
                    ax_x.plot(analysis['metrics']['profiles'][profile]['x']['axis'],
                              analysis['metrics']['profiles'][profile]['x']['data'], 'b-',
                              label=rf'X-data ({um_per_pix:.2f} $\mu$m/pix)')
                    ax_x.plot(analysis['metrics']['profiles'][profile]['x']['axis'],
                              analysis['metrics']['profiles'][profile]['x']['fit'], 'm-',
                              label=f"FWHM: {fwhm(analysis['metrics']['profiles'][profile]['x']['opt'][3]):.1f}")
                    ax_x.legend(loc='best', prop={'size': 8})

                    ax_y.plot(analysis['metrics']['profiles'][profile]['y']['axis'],
                              analysis['metrics']['profiles'][profile]['y']['data'], 'b-',
                              label=rf'Y-data ({um_per_pix:.2f} $\mu$m/pix)')
                    ax_y.plot(analysis['metrics']['profiles'][profile]['y']['axis'],
                              analysis['metrics']['profiles'][profile]['y']['fit'], 'm-',
                              label=f"FWHM: {fwhm(analysis['metrics']['profiles'][profile]['y']['opt'][3]):.1f}")
                    ax_y.legend(loc='best', prop={'size': 8})

                    # plot & save
                    if save_folder:
                        image_path: Path = save_folder / f'{profile}_profiles_{tag}.png'
                        plt.savefig(image_path, dpi=300)

                    # plt.show(block=False)

                    # try:
                    #     plt.close(fig)
                    # except Exception:
                    #     pass

            # ___________________________________________
            if comparison and ('positions' in analysis) and ('profiles' in analysis['metrics']):
                profiles_fig_size = (ScanImages.fig_size[0] * 1.5,
                                     ScanImages.fig_size[1] * math.ceil(len(analysis['positions']) / 3))

                fig_2, axs = plt.subplots(ncols=3, nrows=len(analysis['positions']['short_names']),
                                          figsize=profiles_fig_size, sharex='col', sharey='col')

                for it, pos in enumerate(analysis['positions']['short_names']):
                    pos_ij = analysis['positions'][f'{pos}_ij']
                    profile = analysis['metrics']['profiles'][pos]

                    if isinstance(pos_ij, tuple) and not np.isnan(pos_ij[:2]).any():
                        if 'arrays' in analysis:
                            axs[it, 0].imshow(analysis['arrays']['denoised'],
                                              cmap='hot', aspect='equal', origin='upper', vmin=0, vmax=v_max)
                        axs[it, 0].axvline(pos_ij[1], color='w', linestyle='--', linewidth=0.5)
                        axs[it, 0].axhline(pos_ij[0], color='w', linestyle='--', linewidth=0.5)
                        axs[it, 0].plot(pos_ij[1], pos_ij[0], '.w', markersize=2)
                        axs[it, 0].set_ylabel(analysis['positions']['long_names'][it])
                        axs[it, 1].plot(profile['x']['axis'], profile['x']['data'], 'b-',
                                        label=rf'y$_0$ [pix] = {pos_ij[0]}, $\delta$x = {roi_raw[0]}')
                        axs[it, 1].plot(profile['x']['axis'], profile['x']['fit'], 'm-',
                                        label=r'$\mu_x$ = ' + f"{profile['x']['opt'][2]:.1f}, " +
                                              r'$\sigma_x$ = ' + f"{fwhm(profile['x']['opt'][3]):.1f}")
                        axs[it, 1].legend(loc='best', prop={'size': 8})
                        if it == 0:
                            axs[it, 1].set_title(rf'{um_per_pix:.2f} $\mu$m/pix')
                        axs[it, 2].plot(profile['y']['axis'], profile['y']['data'], 'b-',
                                        label=rf'x$_0$ [pix] = {pos_ij[1]}, $\delta$y = {roi_raw[2]}')
                        axs[it, 2].plot(profile['y']['axis'], profile['y']['fit'], 'm-',
                                        label=r'$\mu_y$ = ' + f"{profile['y']['opt'][2]:.1f}, " +
                                              r'$\sigma_y$ = ' + f"{fwhm(profile['y']['opt'][3]):.1f}")
                        axs[it, 2].legend(loc='best', prop={'size': 8})
                        if it == 0:
                            axs[it, 2].set_title(rf'{um_per_pix:.2f} $\mu$m/pix')

                if save_folder:
                    image_path: Path = save_folder / f'all_profiles_{tag}.png'
                    plt.savefig(image_path, dpi=300)

                if block:
                    plt.show(block=block)

                try:
                    plt.close(fig_1)
                    plt.close(fig_2)
                except Exception:
                    pass

        except Exception as ex:
            api_error.error(str(ex), f'Failed to plot analysis for {Path(analysis["image_path"]).name}')
            pass


if __name__ == '__main__':
    test = False

    if not test:
        _base_path, is_local = htu.HtuExp.initialize('Undulator')
        _base_tag = ScanTag(2023, 8, 9, 25)
        _camera = 'A3'

        _folder = ScanData.build_folder_path(_base_tag, _base_path)
        _scan_data = ScanData(_folder, ignore_experiment_name=is_local)
        _scan_images = ScanImages(_scan_data, _camera)

        _export_file_path, _data_dict = _scan_images.run_analysis_with_checks(
            images=-1,
            initial_filtering=FiltersParameters(contrast=1.333, hp_median=3, hp_threshold=3., denoise_cycles=0,
                                                gauss_filter=5., com_threshold=0.8, bkg_image=None, box=True,
                                                ellipse=False),
            plots=True, store_images=False, save=True)
    else:
        _rots = 1
        raw = np.zeros((7, 5))
        _roi = [2, 3, 1, 4]
        raw[_roi[2]:_roi[3] + 1, _roi[0]:_roi[1] + 1] = \
            np.reshape(np.arange((_roi[1] - _roi[0] + 1) * (_roi[3] - _roi[2] + 1)) + 1,
                       (_roi[3] - _roi[2] + 1, _roi[1] - _roi[0] + 1))
        print(f'raw:\n{raw}')

        sub = raw[_roi[2]:_roi[3] + 1, _roi[0]:_roi[1] + 1]
        print(f'sub:\n{sub}')

        rot = np.rot90(sub, _rots)
        print(f'rot:\n{rot}')

        p_subs = [(2, 1), (0, 2), (1, 0), (1, 1)]
        p_sub = p_subs[_rots]
        print(f'point in rot {p_sub}: {rot[p_sub]}')

        p_raw = ScanImages.processed_to_original_ij(p_sub, _roi, _rots * 90)
        print(f'point in raw {p_raw}: {raw[p_raw]}')

    print('done')
