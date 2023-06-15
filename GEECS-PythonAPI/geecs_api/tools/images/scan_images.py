""" @author: Guillaume Plateau, TAU Systems """

import cv2
import os
import re
import math
import numpy as np
import screeninfo
from pathlib import Path
import calendar as cal
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.transform import hough_ellipse
from skimage.feature import canny
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import savgol_filter
from progressbar import ProgressBar
from typing import Optional, Any, Union
from geecs_api.tools.images.batches import list_images
from geecs_api.devices.geecs_device import api_error
import geecs_api.tools.images.ni_vision as ni
from geecs_api.tools.interfaces.exports import save_py
from geecs_api.tools.scans.scan_data import ScanData
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.diagnostics.cameras import Camera
from geecs_api.tools.images.filtering import basic_filter, FiltersParameters
from geecs_api.tools.images.spot import spot_analysis, fwhm
from geecs_api.tools.interfaces.prompts import text_input


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

        self.scan: ScanData = scan
        self.camera: Optional[Camera] = None

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
        self.raw_size: tuple[int, int] = (0, 0)  # to be updated by latest opened image

        self.camera_label: str = Camera.label_from_name(self.camera_name)

        self.image_folder: Path = self.scan.get_folder() / self.camera_name
        # self.save_folder: Path = self.scan.get_analysis_folder() / self.camera_name / 'Profiles Images'
        self.save_folder: Path = self.scan.get_analysis_folder() / self.camera_name
        self.set_save_folder()  # default no-scan analysis folder

        self.image_analyses: Optional[list[dict[str, Any]]] = None
        self.analyses_summary: Optional[dict[str, Any]] = None

        self.average_image: Optional[np.ndarray] = None
        self.average_analysis: Optional[dict[str, Any]] = None

    def set_save_folder(self, path: Optional[Path] = None):
        if path and path.is_dir():
            self.save_folder = path
        if path is None:
            # self.save_folder = self.scan.get_analysis_folder() / self.camera_name / 'Profiles Images'
            self.save_folder = self.scan.get_analysis_folder() / self.camera_name
            if not self.save_folder.is_dir():
                os.makedirs(self.save_folder)

    def read_image_as_float(self, image_path: Path) -> np.ndarray:
        image = ni.read_imaq_image(image_path)
        self.raw_size = image.shape
        if isinstance(self.camera_roi, np.ndarray) and (self.camera_roi.size >= 4):
            image = image[self.camera_roi[-2]:self.camera_roi[-1]+1, self.camera_roi[0]:self.camera_roi[1]+1]
        image = np.rot90(image, self.camera_r90 // 90)
        return image.astype('float64')

    def run_analysis_with_checks(self, images: Union[int, list[Path]] = -1, initial_filtering=FiltersParameters(),
                                 trim_collection: bool = False, new_targets: bool = False, plots: bool = False,
                                 save: bool = False) -> tuple[Optional[Path], dict[str, Any]]:
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
                    self.analyze_image_batch(images, filtering, trim_collection, new_targets, plots, save)
            except Exception as ex:
                api_error(str(ex), f'Failed to analyze {self.scan.get_folder().name}')
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
                            trim_collection: bool = False, new_targets: bool = False, plots: bool = False,
                            save: bool = False) -> tuple[Optional[Path], dict[str, Any]]:
        export_file_path: Optional[Path] = None
        data_dict: dict[str, Any] = {}

        if isinstance(images, int):
            paths = list_images(self.image_folder, images, '.png')
        else:
            paths = images

        # paths = paths[25:30]  # tmp
        skipped_files = []
        analyses: list[dict[str, Any]] = []

        # run analysis
        if paths:
            try:
                with ProgressBar(max_value=len(paths)) as pb:
                    # analyze one at a time until valid image found to initialize average
                    for skipped, image_path in enumerate(paths):
                        analysis = self.analyze_image(image_path, filtering)
                        analysis['image_path'] = image_path

                        if analysis['is_valid']:
                            analysis = ScanImages.profiles_analysis(analysis)
                            if plots:
                                self.render_image_analysis(analysis, block=False, save=save)
                            avg_image: np.ndarray = analysis['image_raw'].copy()
                        else:
                            skipped_files.append(Path(image_path).name)

                        if trim_collection:
                            analysis.pop('image_denoised')
                            analysis.pop('image_blurred')
                            analysis.pop('image_thresholded')
                        analyses.append(analysis)
                        pb.increment()

                        if analysis['is_valid']:
                            break

                    # analyze the rest of the images
                    if len(paths) > (skipped + 1):
                        for it, image_path in enumerate(paths[skipped+1:]):
                            analysis = self.analyze_image(image_path, filtering)
                            analysis['image_path'] = image_path

                            if analysis['is_valid']:
                                # profiles
                                analysis = ScanImages.profiles_analysis(analysis)
                                if plots:
                                    self.render_image_analysis(analysis, block=False, save=save)

                                # average
                                alpha = 1.0 / (it + 2)
                                beta = 1.0 - alpha
                                avg_image = cv2.addWeighted(analysis['image_raw'], alpha, avg_image, beta, 0.0)
                            else:
                                skipped_files.append(Path(image_path).name)

                            # collect
                            if trim_collection:
                                analysis.pop('image_denoised')
                                analysis.pop('image_blurred')
                                analysis.pop('image_thresholded')
                            analyses.append(analysis)
                            pb.increment()

                # report skipped files
                if skipped_files:
                    print(f'Skipped:')
                    for file in skipped_files:
                        print(f'\t{file}')
                else:
                    print('No file skipped.')

                # analyze the average image
                try:
                    self.average_analysis = self.analyze_image(avg_image, filtering)
                    self.average_analysis = \
                        ScanImages.profiles_analysis(self.average_analysis)
                    if trim_collection:
                        self.average_analysis.pop('image_denoised')
                        self.average_analysis.pop('image_blurred')
                        self.average_analysis.pop('image_thresholded')
                    if plots:
                        self.render_image_analysis(self.average_analysis, tag='average_image', block=True, save=save)
                except Exception as ex:
                    api_error.error(str(ex), 'Failed to analyze average image')

                self.image_analyses = analyses
                self.average_image = avg_image
                self.analyses_summary = self._summarize_image_analyses()
                self.analyses_summary['targets'] = self._target_analysis(new_targets)

                data_dict = {
                    'scan': self.scan,
                    'camera_r90': self.camera_r90,
                    'camera_name': self.camera_name,
                    'camera_roi': self.camera_roi,
                    'camera_label': self.camera_label,
                    'image_folder': self.image_folder,
                    'save_folder': self.save_folder,
                    'image_analyses': self.image_analyses if self.image_analyses is not None else {},
                    'analyses_summary': self.analyses_summary if self.analyses_summary is not None else {},
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

    def analyze_image(self, image: Union[Path, np.ndarray], filtering=FiltersParameters()) -> dict[str, Any]:
        # raw image
        if isinstance(image, np.ndarray):
            image_raw = image
        else:
            image_raw = self.read_image_as_float(image)
            if isinstance(filtering.bkg_image, Path):
                image_raw -= self.read_image_as_float(filtering.bkg_image)
            if isinstance(filtering.bkg_image, np.ndarray):
                image_raw -= filtering.bkg_image

        # initial filtering
        analysis = basic_filter(image_raw, filtering.hp_median, filtering.hp_threshold, filtering.denoise_cycles,
                                filtering.gauss_filter, filtering.com_threshold)
        if self.camera_roi is None:
            analysis['image_roi'] = \
                np.array([0, image_raw.shape[1] - 1, 0, image_raw.shape[0] - 1])  # left, right, top, bottom
        else:
            analysis['image_roi'] = self.camera_roi.copy()

        # check image
        analysis = ScanImages.is_image_valid(analysis, filtering.contrast)

        # stop if low-contrast image
        if not analysis['is_valid']:
            return analysis

        try:
            # com roi
            line_x = analysis['image_blurred'][analysis['position_com'][0], :]
            line_x = savgol_filter(line_x, max(int(len(line_x) / 6), 2), 3)
            c_x = np.argmax(line_x)
            left = np.argmin(np.abs(line_x[:c_x] - line_x[c_x]/2.))
            right = np.argmin(np.abs(line_x[c_x:] - line_x[c_x]/2.)) + c_x
            sigma_x_x4 = 4 * (right - left) / (2 * math.sqrt(2 * math.log(2.)))

            line_y = analysis['image_blurred'][:, analysis['position_com'][1]]
            line_y = savgol_filter(line_y, max(int(len(line_y) / 6), 2), 3)
            c_y = np.argmax(line_y)
            top = np.argmin(np.abs(line_y[:c_y] - line_y[c_y]/2.))
            bottom = np.argmin(np.abs(line_x[c_y:] + line_y[c_y]/2.)) + c_y
            sigma_y_x4 = 4 * (bottom - top) / (2 * math.sqrt(2 * math.log(2.)))

            analysis['roi_com'] = np.array([c_x - sigma_x_x4, c_x + sigma_x_x4,
                                            c_y - sigma_y_x4, c_y + sigma_y_x4])  # left, right, top, bottom

            # beam edges
            image_edges: np.ndarray = canny(analysis['image_thresholded'], sigma=3)
            image_edges = closing(image_edges.astype(float), square(3))

            # boxed image
            if filtering.box:
                label_image = label(image_edges)
                areas = [box.area for box in regionprops(label_image)]
                roi = regionprops(label_image)[areas.index(max(areas))]

                if roi:
                    gain_factor: float = 0.2
                    roi = np.array([roi.bbox[1], roi.bbox[3], roi.bbox[0], roi.bbox[2]])  # left, right, top, bottom

                    width_gain = int(round((roi[1] - roi[0]) * gain_factor))
                    left_gain = min(roi[0], width_gain)
                    right_gain = min(image_edges.shape[1] - 1 - roi[1], width_gain)

                    height_gain = int(round((roi[3] - roi[2]) * gain_factor))
                    top_gain = min(roi[2], height_gain)
                    bottom_gain = min(image_edges.shape[0] - 1 - roi[3], height_gain)

                    gain_pixels = max(left_gain, right_gain, top_gain, bottom_gain)
                    left_gain = min(roi[0], gain_pixels)
                    right_gain = min(image_edges.shape[1] - 1 - roi[1], gain_pixels)
                    top_gain = min(roi[2], gain_pixels)
                    bottom_gain = min(image_edges.shape[0] - 1 - roi[3], gain_pixels)

                    analysis['roi_edges'] = np.array([roi[0] - left_gain, roi[1] + right_gain,
                                                      roi[2] - top_gain, roi[3] + bottom_gain])
                    analysis['box_left_gain'] = left_gain
                    analysis['box_right_gain'] = right_gain
                    analysis['box_top_gain'] = top_gain
                    analysis['box_bottom_gain'] = bottom_gain

                    pos_box = np.array([(analysis['roi_edges'][2] + analysis['roi_edges'][3]) / 2.,
                                        (analysis['roi_edges'][0] + analysis['roi_edges'][1]) / 2.])
                    pos_box = np.round(pos_box).astype(int)
                    analysis['position_box'] = tuple(pos_box)  # i, j

                    # update edges image
                    image_edges[:, :analysis['roi_edges'][0]] = 0
                    image_edges[:, analysis['roi_edges'][1]+1:] = 0
                    image_edges[:analysis['roi_edges'][2], :] = 0
                    image_edges[analysis['roi_edges'][3]+1:, :] = 0

            analysis['image_edges'] = image_edges

            # ellipse fit (min_size: min of major axis, max_size: max of minor axis)
            if filtering.ellipse:
                h_ellipse = hough_ellipse(image_edges, min_size=10, max_size=60, accuracy=8)
                h_ellipse.sort(order='accumulator')
                best = list(h_ellipse[-1])
                yc, xc, a, b = (int(round(x)) for x in best[1:5])
                ellipse = (yc, xc, a, b, best[5])
                analysis['ellipse'] = ellipse  # (i, j, major, minor, orientation)

                # cy, cx = ellipse_perimeter(*ellipse)
                pos_ellipse = np.array([yc, xc])
                analysis['position_ellipse'] = tuple(pos_ellipse)  # i, j

        except Exception:
            pass

        return analysis

    @staticmethod
    def rotated_to_original_ij(coords_ij: tuple[int, int], raw_shape_ij: tuple[int, int], rot_deg: int = 0) \
            -> tuple[int, int]:
        rots: int = divmod(rot_deg, 90)[0] % 4
        raw_coords: tuple[int, int]
        if rots == 1:
            # 90deg: [j, j_max - i - 1]
            raw_coords = (coords_ij[1], raw_shape_ij[1] - coords_ij[0] - 1)
        elif rots == 2:
            # 180deg: [i_max - i - 1, j_max - j - 1]
            raw_coords = (raw_shape_ij[0] - coords_ij[0] - 1, raw_shape_ij[1] - coords_ij[1] - 1)
        elif rots == 3:
            # 270deg: [i_max - j - 1, i]
            raw_coords = (raw_shape_ij[0] - coords_ij[1] - 1, coords_ij[0])
        else:
            # 0deg: [i, j]
            raw_coords = coords_ij

        return raw_coords

    @staticmethod
    def original_to_rotated_ij(coords_ij: tuple[int, int], raw_shape_ij: tuple[int, int], rot_deg: int = 0) \
            -> tuple[int, int]:
        rots: int = divmod(rot_deg, 90)[0] % 4
        raw_coords: tuple[int, int]
        if rots == 1:
            # 90deg: [j_max - j - 1, i]
            raw_coords = (raw_shape_ij[1] - coords_ij[1] - 1, coords_ij[0])
        elif rots == 2:
            # 180deg: [i_max - i - 1, j_max - j - 1]
            raw_coords = (raw_shape_ij[0] - coords_ij[0] - 1, raw_shape_ij[1] - coords_ij[1] - 1)
        elif rots == 3:
            # 270deg: [j, i_max - i - 1]
            raw_coords = (coords_ij[1], raw_shape_ij[0] - coords_ij[0] - 1)
        else:
            # 0deg: [i, j]
            raw_coords = coords_ij

        return raw_coords

    def _target_analysis(self, new_targets: bool = False) -> dict[str, Any]:
        positions = ['max', 'com', 'box', 'ellipse']
        target_analysis: dict[str, Any] = {}
        target_found: bool = True

        try:
            target_analysis['target_um_pix'] = \
                float(GeecsDevice.exp_info['devices'][self.camera_name]['SpatialCalibration']['defaultvalue'])
            if new_targets:
                target_analysis['target_ij_raw'] = (0, 0)
            else:
                target_analysis['target_ij_raw'] = \
                    (int(GeecsDevice.exp_info['devices'][self.camera_name]['Target.Y']['defaultvalue']),
                     int(GeecsDevice.exp_info['devices'][self.camera_name]['Target.X']['defaultvalue']))
        except Exception:
            target_found = False
            target_analysis['target_um_pix'] = 1.
            target_analysis['target_ij_raw'] = (0, 0)

        if self.average_analysis and self.analyses_summary:
            rots: int = divmod(self.camera_r90, 90)[0] % 4  # quotient modulo 4: [0-3]
            target_ij = np.array(target_analysis['target_ij_raw'])  # defined in raw image

            if rots == 1:
                # 90deg: [j_raw - j_max - 1, i_min]
                roi_ij_offset = np.array([self.raw_size[1] - self.camera_roi[1] - 1, self.camera_roi[2]])
                if target_found and not new_targets:
                    target_ij = np.array([self.raw_size[1] - target_ij[1] - 1, target_ij[0]])

            elif rots == 2:
                # 180deg: [i_raw - i_max - 1, j_raw - j_max - 1]
                roi_ij_offset = np.array([self.raw_size[0] - self.camera_roi[3] - 1,
                                          self.raw_size[1] - self.camera_roi[1] - 1])
                if target_found and not new_targets:
                    target_ij = np.array([self.raw_size[0] - target_ij[0] - 1, self.raw_size[1] - target_ij[1] - 1])

            elif rots == 3:
                # 270deg: [j_min, i_raw - i_max - 1]
                roi_ij_offset = np.array([self.camera_roi[0], self.raw_size[0] - self.camera_roi[3] - 1])
                if target_found and not new_targets:
                    target_ij = np.array([target_ij[1], self.raw_size[0] - target_ij[0] - 1])

            else:
                # 0deg: [i_min, j_min]
                roi_ij_offset = self.camera_roi[[2, 0]]

            for pos in positions:
                if f'position_{pos}' in self.average_analysis and self.average_analysis[f'position_{pos}']:
                    target_analysis[f'avg_img_{pos}_delta_pix'] = \
                        np.array(self.average_analysis[f'position_{pos}']) + roi_ij_offset - target_ij
                    target_analysis[f'avg_img_{pos}_delta_mm'] = \
                        np.roll(target_analysis[f'avg_img_{pos}_delta_pix'], 1) \
                        * target_analysis['target_um_pix'] / 1000.

                if f'scan_pos_{pos}' in self.analyses_summary and self.analyses_summary[f'scan_pos_{pos}'].any():
                    target_analysis[f'scan_pos_{pos}_delta_pix'] = \
                        np.array(self.analyses_summary[f'scan_pos_{pos}']) + roi_ij_offset - target_ij
                    target_analysis[f'scan_pos_{pos}_delta_mm'] = \
                        np.roll(target_analysis[f'scan_pos_{pos}_delta_pix'], 1) \
                        * target_analysis['target_um_pix'] / 1000.

                    target_analysis[f'target_deltas_pix_{pos}_mean'] = \
                        np.mean(target_analysis[f'scan_pos_{pos}_delta_pix'], axis=0)
                    target_analysis[f'target_deltas_pix_{pos}_std'] = \
                        np.std(target_analysis[f'scan_pos_{pos}_delta_pix'], axis=0)

                    target_analysis[f'target_deltas_mm_{pos}_mean'] = \
                        np.mean(target_analysis[f'scan_pos_{pos}_delta_mm'], axis=0)
                    target_analysis[f'target_deltas_mm_{pos}_std'] = \
                        np.std(target_analysis[f'scan_pos_{pos}_delta_mm'], axis=0)

                    target_analysis['roi_ij_offset'] = roi_ij_offset
                    target_analysis['target_ij'] = target_ij
                    target_analysis['raw_shape_ij'] = self.raw_size
                    target_analysis['camera_roi'] = self.camera_roi
                    target_analysis['camera_r90'] = rots * 90

        return target_analysis

    def _summarize_image_analyses(self) -> dict[str, Any]:
        positions = ['max', 'com', 'box', 'ellipse']
        summary: dict[str, Any] = {}

        for pos in positions:
            scan_pos = []
            scan_pos_fwhm_x = []
            scan_pos_fwhm_y = []

            for analysis in self.image_analyses:
                if analysis is not None and analysis['is_valid'] and f'opt_x_{pos}' in analysis:
                    scan_pos.append(analysis[f'position_{pos}'])
                    scan_pos_fwhm_x.append(fwhm(analysis[f'opt_x_{pos}'][3]))
                    scan_pos_fwhm_y.append(fwhm(analysis[f'opt_y_{pos}'][3]))

            if scan_pos:
                summary[f'scan_pos_{pos}'] = np.array(scan_pos)
                summary[f'scan_pos_{pos}_fwhm_x'] = np.array(scan_pos_fwhm_x)
                summary[f'scan_pos_{pos}_fwhm_y'] = np.array(scan_pos_fwhm_y)

                summary[f'mean_pos_{pos}'] = np.mean(summary[f'scan_pos_{pos}'], axis=0)
                summary[f'mean_pos_{pos}_fwhm_x'] = np.mean(summary[f'scan_pos_{pos}_fwhm_x'])
                summary[f'mean_pos_{pos}_fwhm_y'] = np.mean(summary[f'scan_pos_{pos}_fwhm_y'])

                summary[f'std_pos_{pos}'] = np.std(summary[f'scan_pos_{pos}'], axis=0)
                summary[f'std_pos_{pos}_fwhm_x'] = np.std(summary[f'scan_pos_{pos}_fwhm_x'])
                summary[f'std_pos_{pos}_fwhm_y'] = np.std(summary[f'scan_pos_{pos}_fwhm_y'])

        return summary

    @staticmethod
    def profiles_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
        try:
            positions = [(*analysis['position_max'], 'max'),
                         (*analysis['position_com'], 'com')]
            if 'position_box' in analysis:
                positions.append((*analysis['position_box'], 'box'))
            if 'position_ellipse' in analysis:
                positions.append((*analysis['position_ellipse'], 'ell'))
            labels = ['maximum', 'center of mass', 'box', 'ellipse']

            x_win = (analysis['roi_edges'][0], analysis['roi_edges'][1]) if 'roi' in analysis else None
            y_win = (analysis['roi_edges'][2], analysis['roi_edges'][3]) if 'roi' in analysis else None
            # noinspection PyTypeChecker
            profiles = spot_analysis(analysis['image_raw'], positions, x_window=x_win, y_window=y_win)

            if profiles:
                for k, v in profiles.items():
                    analysis[k] = v

            analysis['positions'] = positions
            analysis['positions_labels'] = labels

        except Exception as ex:
            api_error.error(str(ex), 'Failed to analyze profiles')
            pass

        return analysis

    @staticmethod
    def is_image_valid(analysis: dict[str, Any], contrast: float = 2.) -> dict[str, Any]:
        counts, bins = np.histogram(analysis['image_denoised'],
                                    bins=max(10, 2 * int(np.std(analysis['image_blurred']))))
        analysis['image_hist'] = (counts, bins)
        analysis['bkg_level']: float = bins[np.where(counts == np.max(counts))[0][0]]  # most common value
        analysis['contrast']: float = contrast
        analysis['is_valid']: bool = np.std(bins) > contrast * analysis['bkg_level']

        return analysis

    def render_image_analysis(self, analysis: dict[str, Any], tag: str = '', profiles: tuple[str] = ('max',),
                              comparison: bool = True, block: bool = False, save: bool = False):
        try:
            if not tag:
                image_path: Path = Path(analysis['image_path'])
                tag = image_path.name.split(".")[0].split("_")[-1]

            for profile in profiles:
                if f'axis_x_{profile}' in analysis:
                    fig = plt.figure(figsize=(ScanImages.fig_size[0] * 1.5, ScanImages.fig_size[1]))
                    grid = plt.GridSpec(2, 4, hspace=0.3, wspace=0.3)
                    ax_i = fig.add_subplot(grid[:, :2])
                    ax_x = fig.add_subplot(grid[0, 2:])
                    ax_y = fig.add_subplot(grid[1, 2:], sharey=ax_x)

                    # raw image
                    ax_i.imshow(analysis['image_raw'], cmap='hot', aspect='equal', origin='upper')

                    if 'image_edges' in analysis:
                        edges = np.where(analysis['image_edges'] != 0)
                        ax_i.scatter(edges[1], edges[0], s=0.3, c='b', alpha=0.2)

                    # roi edges
                    if 'roi_edges' in analysis:
                        roi_color = '--y'
                        roi_line = 0.66
                        # left
                        ax_i.plot(analysis['roi_edges'][0] *
                                  np.ones((analysis['roi_edges'][3] - analysis['roi_edges'][2] + 1)),
                                  np.arange(analysis['roi_edges'][2], analysis['roi_edges'][3] + 1),
                                  roi_color, linewidth=roi_line)
                        # right
                        ax_i.plot(analysis['roi_edges'][1] *
                                  np.ones((analysis['roi_edges'][3] - analysis['roi_edges'][2] + 1)),
                                  np.arange(analysis['roi_edges'][2], analysis['roi_edges'][3] + 1),
                                  roi_color, linewidth=roi_line)
                        # top
                        ax_i.plot(np.arange(analysis['roi_edges'][0], analysis['roi_edges'][1] + 1),
                                  analysis['roi_edges'][2] *
                                  np.ones((analysis['roi_edges'][1] - analysis['roi_edges'][0] + 1)),
                                  roi_color, linewidth=roi_line)
                        # bottom
                        ax_i.plot(np.arange(analysis['roi_edges'][0], analysis['roi_edges'][1] + 1),
                                  analysis['roi_edges'][3] *
                                  np.ones((analysis['roi_edges'][1] - analysis['roi_edges'][0] + 1)),
                                  roi_color, linewidth=roi_line)

                    # ellipse
                    if 'ellipse' in analysis:
                        ell = mpatches.Ellipse(xy=(analysis['ellipse'][1], analysis['ellipse'][0]),
                                               width=2*analysis['ellipse'][3], height=2*analysis['ellipse'][2],
                                               angle=math.degrees(analysis['ellipse'][4]))
                        ax_i.add_artist(ell)
                        ell.set_alpha(0.66)
                        ell.set_edgecolor('g')
                        ell.set_linewidth(1)
                        # noinspection PyArgumentList
                        ell.set_fill(False)

                    # profile lineouts
                    ax_i.axvline(analysis[f'position_{profile}'][1], color='w', linestyle='--', linewidth=0.5)
                    ax_i.axhline(analysis[f'position_{profile}'][0], color='w', linestyle='--', linewidth=0.5)
                    ax_i.plot(analysis[f'position_{profile}'][1],
                              analysis[f'position_{profile}'][0], 'w.', markersize=3)
                    if ax_i.xaxis_inverted():
                        ax_i.invert_xaxis()
                    if not ax_i.yaxis_inverted():
                        ax_i.invert_yaxis()

                    # lineouts
                    ax_x.plot(analysis[f'axis_x_{profile}'], analysis[f'data_x_{profile}'], 'b-', label='data (x)')
                    ax_x.plot(analysis[f'axis_x_{profile}'], analysis[f'fit_x_{profile}'], 'm-',
                              label=f'FWHM: {fwhm(analysis[f"opt_x_{profile}"][3]):.1f}')
                    ax_x.legend(loc='best', prop={'size': 8})

                    ax_y.plot(analysis[f'axis_y_{profile}'], analysis[f'data_y_{profile}'], 'b-', label='data (y)')
                    ax_y.plot(analysis[f'axis_y_{profile}'], analysis[f'fit_y_{profile}'], 'm-',
                              label=f'FWHM: {fwhm(analysis[f"opt_y_{profile}"][3]):.1f}')
                    ax_y.legend(loc='best', prop={'size': 8})

                    # plot & save
                    file_name: str = f'{profile}_profiles_{tag}.png'
                    image_path: Path = self.save_folder / file_name
                    if save:
                        plt.savefig(image_path, dpi=300)

                    if block:
                        plt.show(block=block)

                    try:
                        plt.close(fig)
                    except Exception:
                        pass

            # ___________________________________________
            if comparison and ('positions' in analysis) and ('axis_x_max' in analysis):
                profiles_fig_size = (ScanImages.fig_size[0] * 1.5,
                                     ScanImages.fig_size[1] * math.ceil(len(analysis['positions']) / 3))
                fig, axs = plt.subplots(ncols=3, nrows=len(analysis['positions']), figsize=profiles_fig_size,
                                        sharex='col', sharey='col')
                for it, pos in enumerate(analysis['positions']):
                    if not np.isnan(pos[:2]).any():
                        axs[it, 0].imshow(analysis['image_raw'], cmap='hot', aspect='equal', origin='upper')
                        axs[it, 0].axvline(pos[1], color='w', linestyle='--', linewidth=0.5)
                        axs[it, 0].axhline(pos[0], color='w', linestyle='--', linewidth=0.5)
                        axs[it, 0].plot(pos[1], pos[0], '.w', markersize=2)
                        axs[it, 0].set_ylabel(analysis['positions_labels'][it])
                        axs[it, 1].plot(analysis[f'axis_x_{pos[2]}'], analysis[f'data_x_{pos[2]}'], 'b-', label='data')
                        axs[it, 1].plot(analysis[f'axis_x_{pos[2]}'], analysis[f'fit_x_{pos[2]}'], 'm-', label='fit(x)')
                        axs[it, 1].legend(loc='best', prop={'size': 8})
                        axs[it, 2].plot(analysis[f'axis_y_{pos[2]}'], analysis[f'data_y_{pos[2]}'], 'b-', label='data')
                        axs[it, 2].plot(analysis[f'axis_y_{pos[2]}'], analysis[f'fit_y_{pos[2]}'], 'm-', label='fit(y)')
                        axs[it, 2].legend(loc='best', prop={'size': 8})

                file_name: str = f'all_profiles_{tag}.png'
                image_path: Path = self.save_folder / file_name

                if save:
                    plt.savefig(image_path, dpi=300)

                if block:
                    plt.show(block=block)

                try:
                    plt.close(fig)
                except Exception:
                    pass

        except Exception as ex:
            api_error.error(str(ex), f'Failed to plot analysis for {Path(analysis["image_path"]).name}')
            pass


if __name__ == '__main__':
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

    # _base = Path(r'C:\Users\GuillaumePlateau\Documents\LBL\Data')
    _base: Path = Path(r'Z:\data')
    _base_tag = (2023, 4, 20, 48)
    _camera_tag = 'U3'

    _folder = _base/'Undulator'/f'Y{_base_tag[0]}'/f'{_base_tag[1]:02d}-{cal.month_name[_base_tag[1]][:3]}'
    _folder = _folder/f'{str(_base_tag[0])[-2:]}_{_base_tag[1]:02d}{_base_tag[2]:02d}'/'scans'/f'Scan{_base_tag[3]:03d}'

    _scan = ScanData(_folder, ignore_experiment_name=False)
    _images = ScanImages(_scan, _camera_tag)
    _images.run_analysis_with_checks(
        initial_filtering=FiltersParameters(contrast=1.333, hp_median=2, hp_threshold=3., denoise_cycles=0,
                                            gauss_filter=5., com_threshold=0.66, bkg_image=None, box=True,
                                            ellipse=False),
        plots=True, save=True)
    print('done')
