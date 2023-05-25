""" @author: Guillaume Plateau, TAU Systems """

import cv2
import os
import re
import math
import numpy as np
import screeninfo
from pathlib import Path
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.transform import hough_ellipse
from skimage.feature import canny
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from progressbar import ProgressBar
from typing import Optional, Any, Union
from geecs_api.api_defs import SysPath
from geecs_api.tools.images.batches import list_images
from geecs_api.devices.geecs_device import api_error
import geecs_api.tools.images.ni_vision as ni
from geecs_api.tools.interfaces.exports import save_py
from geecs_api.tools.scans.scan import Scan
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.diagnostics.cameras import Camera
from geecs_api.tools.images.filtering import filter_image
from geecs_api.tools.images.spot import spot_analysis, fwhm
from geecs_api.tools.interfaces.prompts import text_input


class UndulatorNoScan:
    fig_size = (int(round(screeninfo.get_monitors()[0].width / 540. * 10) / 10),
                int(round(screeninfo.get_monitors()[0].height / 450. * 10) / 10))

    def __init__(self, scan: Scan, camera: Union[int, Camera, str], angle: Optional[int] = None):
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

        self.scan: Scan = scan
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
            self.camera_roi = np.array(Camera.ROIs[self.camera_name])
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

        self.camera_label: str = Camera.label_from_name(self.camera_name)

        self.image_folder: SysPath = self.scan.get_folder() / self.camera_name
        self.save_folder: Path = Path(self.scan.get_analysis_folder()) / self.camera_name / 'Profiles Images'
        if not self.save_folder.is_dir():
            os.makedirs(self.save_folder)

        self.image_analyses: Optional[list[dict[str, Any]]] = None
        self.analyses_summary: Optional[dict[str, Any]] = None

        self.average_image: Optional[np.ndarray] = None
        self.average_analysis: Optional[dict[str, Any]] = None

    def read_image_as_float(self, image_path: SysPath) -> np.ndarray:
        image = ni.read_imaq_image(image_path)
        if isinstance(self.camera_roi, np.ndarray) and (self.camera_roi.size >= 4):
            image = image[self.camera_roi[-2]:self.camera_roi[-1], self.camera_roi[0]:self.camera_roi[1]]
        image = np.rot90(image, self.camera_r90)
        return image.astype('float64')

    def run_analysis_with_checks(self, n_images: int = 0, initial_contrast: float = 1.333,
                                 hp_median: int = 2, hp_threshold: float = 3., denoise_cycles: int = 0,
                                 gauss_filter: float = 5., com_threshold: float = 0.5, plots: bool = False,
                                 bkg_image: Optional[Path] = None, skip_ellipse: bool = True):
        contrast: float = initial_contrast
        while True:
            try:
                self.analyze_images(n_images, contrast, hp_median, hp_threshold, denoise_cycles,
                                    gauss_filter, com_threshold, plots, bkg_image, skip_ellipse)
            except Exception as ex:
                api_error(str(ex), f'Failed to analyze {self.scan.get_folder().name}')
                pass

            repeat = text_input(f'Repeat analysis (adjust contrast)? : ',
                                accepted_answers=['y', 'yes', 'n', 'no'])
            if repeat.lower()[0] == 'n':
                break
            else:
                while True:
                    try:
                        contrast = float(text_input(f'New contrast value (old: {contrast:.3f}) : '))
                        break
                    except Exception:
                        print('Contrast value must be a positive number (e.g. 1.3)')
                        continue

    def analyze_images(self, n_images: int = 0, contrast: float = 2, hp_median: int = 2, hp_threshold: float = 3.,
                       denoise_cycles: int = 0, gauss_filter: float = 5., com_threshold: float = 0.5,
                       plots: bool = False, bkg_image: Optional[Path] = None, skip_ellipse: bool = True):
        paths = list_images(self.image_folder, n_images, '.png')
        # paths = paths[-5:]  # tmp
        # paths = paths[25:30]  # tmp
        skipped_files = []
        analyses: list[dict[str, Any]] = []

        # run analysis
        if paths:
            try:
                with ProgressBar(max_value=len(paths)) as pb:
                    # analyze one at a time until valid image found to initialize average
                    for skipped, image_path in enumerate(paths):
                        analysis = self._analyze_image(image_path, bkg_image, contrast, hp_median, hp_threshold,
                                                       denoise_cycles, gauss_filter, com_threshold, skip_ellipse)
                        analysis['image_path'] = image_path

                        if analysis['is_valid']:
                            analysis = UndulatorNoScan.profiles_analysis(analysis)
                            if plots:
                                self.plot_save_image_analysis(analysis, block=False)
                            avg_image: np.ndarray = analysis['image_raw'].copy()
                        else:
                            skipped_files.append(Path(image_path).name)

                        analyses.append(analysis)
                        pb.increment()

                        if analysis['is_valid']:
                            break

                    # analyze the rest of the images
                    if len(paths) > (skipped + 1):
                        for it, image_path in enumerate(paths[skipped+1:]):
                            analysis = self._analyze_image(image_path, bkg_image, contrast, hp_median, hp_threshold,
                                                           denoise_cycles, gauss_filter, com_threshold, skip_ellipse)
                            analysis['image_path'] = image_path

                            if analysis['is_valid']:
                                # profiles
                                analysis = UndulatorNoScan.profiles_analysis(analysis)
                                if plots:
                                    self.plot_save_image_analysis(analysis, block=False)

                                # average
                                alpha = 1.0 / (it + 2)
                                beta = 1.0 - alpha
                                avg_image = cv2.addWeighted(analysis['image_raw'], alpha, avg_image, beta, 0.0)
                            else:
                                skipped_files.append(Path(image_path).name)

                            # collect
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
                self.average_analysis = self._analyze_image(avg_image, bkg_image, contrast, hp_median, hp_threshold,
                                                            denoise_cycles, gauss_filter, com_threshold, skip_ellipse)
                try:
                    self.average_analysis = \
                        UndulatorNoScan.profiles_analysis(self.average_analysis)
                    if plots:
                        self.plot_save_image_analysis(self.average_analysis, tag='average_image', block=True)
                except Exception as ex:
                    api_error.error(str(ex), 'Failed to analyze average image')

                self.image_analyses = analyses
                self.average_image = avg_image
                self.analyses_summary = self._summarize_image_analyses()
                self.analyses_summary['targets'] = self._target_analysis()
                print(f'Figures saved in:\n\t{self.save_folder}')

                export_file_path: Path = self.save_folder.parent / 'profiles_analysis'
                save_py(file_path=export_file_path,
                        data={'scan': self.scan,
                              'camera_r90': self.camera_r90,
                              'camera_name': self.camera_name,
                              'camera_roi': self.camera_roi,
                              'camera_label': self.camera_label,
                              'image_folder': self.image_folder,
                              'save_folder': self.save_folder,
                              'image_analyses': self.image_analyses if self.image_analyses is not None else {},
                              'analyses_summary': self.analyses_summary if self.analyses_summary is not None else {},
                              'average_image': self.average_image if self.average_image is not None else {},
                              'average_analysis': self.average_analysis if self.average_analysis is not None else {}})
                print(f'Data exported to:\n\t{export_file_path}.dat')

            except Exception as ex:
                api_error.error(str(ex), 'Failed to analyze image')
                pass

    def _analyze_image(self, image: Union[SysPath, np.ndarray], bkg_image: Optional[Path] = None, contrast: float = 2,
                       hp_median: int = 2, hp_threshold: float = 3., denoise_cycles: int = 0, gauss_filter: float = 5.,
                       com_threshold: float = 0.5, skip_ellipse: bool = True) -> dict[str, Any]:
        # raw image
        if isinstance(image, np.ndarray):
            image_raw = image
        else:
            image_raw = self.read_image_as_float(image)
            if bkg_image:
                image_raw -= self.read_image_as_float(bkg_image)

        # initial filtering
        analysis = filter_image(image_raw, hp_median, hp_threshold, denoise_cycles, gauss_filter, com_threshold)

        # check image
        analysis = UndulatorNoScan.is_image_valid(analysis, contrast)

        # stop if low-contrast image
        if not analysis['is_valid']:
            return analysis

        try:
            # beam edges
            image_edges: np.ndarray = canny(analysis['image_thresholded'], sigma=3)
            image_edges = closing(image_edges.astype(float), square(3))
            # image_edges = clear_border(image_edges)

            # edges_binary = image_edges.astype('uint8') * 255
            # contours = cv2.findContours(edges_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # hierarchy = contours[1] if len(contours) == 2 else contours[2]
            # contours = contours[0] if len(contours) == 2 else contours[1]
            # hierarchy = hierarchy[0]
            #
            # count = 0
            # result = edges_binary.copy()
            # result = cv2.merge([result, result, result])
            # for component in zip(contours, hierarchy):
            #     cntr = component[0]
            #     hier = component[1]
            #     # discard outermost no parent contours and keep innermost no child contours
            #     # hier = indices for next, previous, child, parent
            #     # no parent or no child indicated by negative values
            #     if (hier[3] > -1) & (hier[2] < 0):
            #         count = count + 1
            #         cv2.drawContours(result, [cntr], 0, (0, 0, 255), 2)

            # boxed image
            label_image = label(image_edges)
            areas = [box.area for box in regionprops(label_image)]
            roi = regionprops(label_image)[areas.index(max(areas))]

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

            analysis['roi'] = np.array([roi[0] - left_gain, roi[1] + right_gain,
                                        roi[2] - top_gain, roi[3] + bottom_gain])
            analysis['box_left_gain'] = left_gain
            analysis['box_right_gain'] = right_gain
            analysis['box_top_gain'] = top_gain
            analysis['box_bottom_gain'] = bottom_gain

            pos_box = np.array([(analysis['roi'][2] + analysis['roi'][3]) / 2.,
                                (analysis['roi'][0] + analysis['roi'][1]) / 2.])
            pos_box = np.round(pos_box).astype(int)
            analysis['position_box'] = tuple(pos_box)  # i, j

            # update edges image
            image_edges[:, :analysis['roi'][0]] = 0
            image_edges[:, analysis['roi'][1]+1:] = 0
            image_edges[:analysis['roi'][2], :] = 0
            image_edges[analysis['roi'][3]+1:, :] = 0
            analysis['image_edges'] = image_edges

            # ellipse fit (min_size: min of major axis, max_size: max of minor axis)
            if not skip_ellipse:
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

    def _target_analysis(self) -> dict[str, Any]:
        positions = ['max', 'com', 'box', 'ellipse']
        target_analysis: dict[str, Any] = {}

        try:
            target_analysis['target_um_pix'] = \
                float(GeecsDevice.exp_info['devices'][self.camera_name]['SpatialCalibration']['defaultvalue'])
            target_analysis['target_xy'] = \
                (float(GeecsDevice.exp_info['devices'][self.camera_name]['Target.Y']['defaultvalue']),
                 float(GeecsDevice.exp_info['devices'][self.camera_name]['Target.X']['defaultvalue']))
        except Exception:
            target_analysis['target_um_pix'] = 1.
            target_analysis['target_xy'] = (0., 0.)

        if self.average_analysis and self.analyses_summary:
            for pos in positions:
                if f'position_{pos}' in self.average_analysis and self.average_analysis[f'position_{pos}']:
                    target_analysis[f'avg_img_{pos}_delta'] = \
                        (np.array(self.average_analysis[f'position_{pos}']) - np.array(target_analysis['target_xy'])) \
                        * target_analysis['target_um_pix'] / 1000.

                if f'scan_pos_{pos}' in self.analyses_summary and self.analyses_summary[f'scan_pos_{pos}'].any():
                    target_analysis[f'scan_pos_{pos}_delta'] = \
                        (np.array(self.analyses_summary[f'scan_pos_{pos}']) - np.array(target_analysis['target_xy'])) \
                        * target_analysis['target_um_pix'] / 1000.
                    target_analysis[f'target_deltas_{pos}_mean'] = \
                        np.mean(target_analysis[f'scan_pos_{pos}_delta'], axis=0)
                    target_analysis[f'target_deltas_{pos}_std'] = \
                        np.std(target_analysis[f'scan_pos_{pos}_delta'], axis=0)

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
                         (*analysis['position_com'], 'com'),
                         (*analysis['position_box'], 'box')]
            if 'position_ellipse' in analysis:
                positions.append((*analysis['position_ellipse'], 'ell'))
            labels = ['maximum', 'center of mass', 'box', 'ellipse']

            # noinspection PyTypeChecker
            profiles = spot_analysis(analysis['image_raw'], positions,
                                     x_window=(analysis['roi'][0], analysis['roi'][1]),
                                     y_window=(analysis['roi'][2], analysis['roi'][3]))

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

    def plot_save_image_analysis(self, analysis: dict[str, Any], tag: str = '', block: bool = False):
        try:
            if not tag:
                image_path: Path = Path(analysis['image_path'])
                tag = image_path.name.split(".")[0].split("_")[-1]

            if 'axis_x_max' in analysis:
                fig = plt.figure(figsize=(UndulatorNoScan.fig_size[0] * 1.5, UndulatorNoScan.fig_size[1]))
                grid = plt.GridSpec(2, 4, hspace=0.3, wspace=0.3)
                ax_i = fig.add_subplot(grid[:, :2])
                ax_x = fig.add_subplot(grid[0, 2:])
                ax_y = fig.add_subplot(grid[1, 2:], sharey=ax_x)

                # raw image
                edges = np.where(analysis['image_edges'] != 0)
                ax_i.imshow(analysis['image_raw'], cmap='hot', aspect='equal', origin='upper')
                ax_i.scatter(edges[1], edges[0], s=0.3, c='b', alpha=0.2)

                # roi box
                roi_color = '--y'
                roi_line = 0.66
                ax_i.plot(analysis['roi'][0] * np.ones((analysis['roi'][3] - analysis['roi'][2] + 1)),
                          np.arange(analysis['roi'][2], analysis['roi'][3] + 1), roi_color, linewidth=roi_line)  # left
                ax_i.plot(analysis['roi'][1] * np.ones((analysis['roi'][3] - analysis['roi'][2] + 1)),
                          np.arange(analysis['roi'][2], analysis['roi'][3] + 1), roi_color, linewidth=roi_line)  # right
                ax_i.plot(np.arange(analysis['roi'][0], analysis['roi'][1] + 1),
                          analysis['roi'][2] * np.ones((analysis['roi'][1] - analysis['roi'][0] + 1)),
                          roi_color, linewidth=roi_line)  # top
                ax_i.plot(np.arange(analysis['roi'][0], analysis['roi'][1] + 1),
                          analysis['roi'][3] * np.ones((analysis['roi'][1] - analysis['roi'][0] + 1)),
                          roi_color, linewidth=roi_line)  # bottom

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

                # max
                ax_i.axvline(analysis['position_max'][1], color='k', linestyle='--', linewidth=0.5)
                ax_i.axhline(analysis['position_max'][0], color='k', linestyle='--', linewidth=0.5)
                ax_i.plot(analysis['position_max'][1], analysis['position_max'][0], 'k.', markersize=3)
                if ax_i.xaxis_inverted():
                    ax_i.invert_xaxis()
                if not ax_i.yaxis_inverted():
                    ax_i.invert_yaxis()

                # lineouts
                ax_x.plot(analysis['axis_x_max'], analysis['data_x_max'], 'b-', label='data (x)')
                ax_x.plot(analysis['axis_x_max'], analysis['fit_x_max'], 'm-',
                          label=f'FWHM: {fwhm(analysis["opt_x_max"][3]):.1f}')
                ax_x.legend(loc='best', prop={'size': 8})

                ax_y.plot(analysis['axis_y_max'], analysis['data_y_max'], 'b-', label='data (y)')
                ax_y.plot(analysis['axis_y_max'], analysis['fit_y_max'], 'm-',
                          label=f'FWHM: {fwhm(analysis["opt_y_max"][3]):.1f}')
                ax_y.legend(loc='best', prop={'size': 8})

                # plot & save
                file_name: str = f'max_profiles_{tag}.png'
                image_path: Path = self.save_folder / file_name
                plt.savefig(image_path, dpi=300)
                if block:
                    plt.show(block=block)
                try:
                    plt.close(fig)
                except Exception:
                    pass

            # ___________________________________________
            if 'positions' in analysis and 'axis_x_max' in analysis:
                profiles_fig_size = (UndulatorNoScan.fig_size[0] * 1.5,
                                     UndulatorNoScan.fig_size[1] * math.ceil(len(analysis['positions']) / 3))
                fig, axs = plt.subplots(ncols=3, nrows=len(analysis['positions']), figsize=profiles_fig_size,
                                        sharex='col', sharey='col')
                for it, pos in enumerate(analysis['positions']):
                    axs[it, 0].imshow(analysis['image_raw'], cmap='hot', aspect='equal', origin='upper')
                    axs[it, 0].axvline(pos[1], color='k', linestyle='--', linewidth=0.5)
                    axs[it, 0].axhline(pos[0], color='k', linestyle='--', linewidth=0.5)
                    axs[it, 0].plot(pos[1], pos[0], '.k', markersize=2)
                    axs[it, 0].set_ylabel(analysis['positions_labels'][it])
                    axs[it, 1].plot(analysis[f'axis_x_{pos[2]}'], analysis[f'data_x_{pos[2]}'], 'b-', label='data')
                    axs[it, 1].plot(analysis[f'axis_x_{pos[2]}'], analysis[f'fit_x_{pos[2]}'], 'm-', label='fit(x)')
                    axs[it, 1].legend(loc='best', prop={'size': 8})
                    axs[it, 2].plot(analysis[f'axis_y_{pos[2]}'], analysis[f'data_y_{pos[2]}'], 'b-', label='data')
                    axs[it, 2].plot(analysis[f'axis_y_{pos[2]}'], analysis[f'fit_y_{pos[2]}'], 'm-', label='fit(y)')
                    axs[it, 2].legend(loc='best', prop={'size': 8})

                file_name: str = f'all_profiles_{tag}.png'
                image_path: Path = self.save_folder / file_name
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
    _base = Path(r'Z:\data')
    _folder = _base / r'Undulator\Y2023\05-May\23_0509\scans\Scan028'

    _scan = Scan(_folder, ignore_experiment_name=False)
    images = UndulatorNoScan(_scan, 'U7')
    images.analyze_images(contrast=1.333, hp_median=2, hp_threshold=3., denoise_cycles=0,
                          gauss_filter=5., com_threshold=0.66, plots=True, skip_ellipse=True)
    print('done')
