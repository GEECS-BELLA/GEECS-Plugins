import cv2
import re
import math
import numpy as np
import screeninfo
import scipy.ndimage as simg
from skimage.segmentation import clear_border
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
from geecs_api.tools.scans.scan import Scan
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.diagnostics.cameras import Camera
from geecs_api.tools.images.filtering import clip_hot_pixels, filter_image
from geecs_api.tools.images.spot import spot_analysis, fwhm


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
        if angle is None:
            self.camera_r90: int = 0
        else:
            self.camera_r90: int = int(round(angle / 90.))

        if isinstance(camera, Camera):
            self.camera = camera
            self.camera_name: str = camera.get_name()
            self.camera_roi: Optional[np.ndarray] = camera.roi
            self.camera_r90 = camera.rot_90
        elif isinstance(camera, str) and (camera in Camera.ROIs):
            self.camera_name = camera
            self.camera_roi = np.array(Camera.ROIs[camera])
        elif isinstance(camera, str) and re.match(r'(U[1-9]|A[1-3]|Rad2)', camera):
            self.camera_name = Camera.name_from_label(camera)
            self.camera_roi = np.array(Camera.ROIs[self.camera_name])
        elif isinstance(camera, int) and (1 <= camera <= 9):
            self.camera_name = Camera.name_from_label(f'U{camera}')
            self.camera_roi = np.array(Camera.ROIs[self.camera_name])
        else:
            self.camera_name = camera
            self.camera_roi = None

        self.camera_label: str = Camera.label_from_name(self.camera_name)

        self.image_folder: SysPath = self.scan.get_folder() / self.camera_name
        self.image_analyses: Optional[list[dict[str, Any]]] = None
        self.average_image: Optional[np.ndarray] = None
        self.average_analysis: Optional[dict[str, Any]] = None
        self.analyses_summary: Optional[dict[str, Union[float, np.ndarray]]] = None
        self.target_analysis: Optional[dict[str, Union[float, np.ndarray]]] = None

    @staticmethod
    def find_roi(image: np.ndarray, threshold: Optional[float] = None, plots: bool = False):
        roi = None
        roi_box = np.array([0, image.shape[1] - 1, 0, image.shape[0]])

        try:
            # filter and smooth
            blur = clip_hot_pixels(image, median_filter_size=2, threshold_factor=3)
            blur = simg.gaussian_filter(blur, sigma=5.)

            # threshold
            if threshold is None:
                counts, bins = np.histogram(blur, bins=10)
                threshold = bins[np.where(counts == np.max(counts))[0][0] + 1]
            bw = closing(blur > threshold, square(3))

            # remove artifacts connected to image border
            cleared = clear_border(bw)
            # cleared = bw

            # label image regions
            label_image = label(cleared)
            areas = [box.area for box in regionprops(label_image)]
            roi = regionprops(label_image)[areas.index(max(areas))]
            roi_box = roi.bbox

        except Exception:
            pass

        finally:
            if plots:
                fig, ax = plt.subplots(figsize=UndulatorNoScan.fig_size)
                ax.imshow(image)
                rect = mpatches.Rectangle((roi_box[1], roi_box[0]), roi_box[3] - roi_box[1], roi_box[2] - roi_box[0],
                                          fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
                ax.set_axis_off()
                plt.tight_layout()
                plt.show(block=True)

        return roi_box, roi

    def read_image_as_float(self, image_path: SysPath) -> np.ndarray:
        image = ni.read_imaq_image(image_path)
        if isinstance(self.camera_roi, np.ndarray) and (self.camera_roi.size >= 4):
            image = image[self.camera_roi[-2]:self.camera_roi[-1], self.camera_roi[0]:self.camera_roi[1]]
        image = np.rot90(image, self.camera_r90)
        return image.astype('float64')

    def analyze_images(self, n_images: int = 0, hp_median: int = 2, hp_threshold: float = 3., denoise_cycles: int = 0,
                       gauss_filter: float = 5., com_threshold: float = 0.5, plots: bool = False):
        paths = list_images(self.image_folder, n_images, '.png')
        paths = paths[24:]  # tmp
        analyses: list[dict[str, Any]] = []

        # run analysis
        if paths:
            try:
                with ProgressBar(max_value=len(paths)) as pb:
                    # analyze one at a time until valid image found to initialize average
                    for skipped, image_path in enumerate(paths):
                        analysis = self.analyze_image(image_path, hp_median, hp_threshold, denoise_cycles,
                                                      gauss_filter, com_threshold, plots)

                        if analysis['is_valid']:
                            # profiles
                            pos_max: tuple[int, int] = analysis['position_max']
                            pos_com: tuple[int, int] = analysis['position_com']
                            profiles = spot_analysis(analysis['image_raw'], [(pos_max[0], pos_max[1], 'max'),
                                                                             (pos_com[0], pos_com[1], 'com')])
                            for k, v in profiles.items():
                                analysis[k] = v

                            # average
                            avg_image: np.ndarray = analysis['image_raw'].copy()

                            break

                    analyses.append(analysis)
                    pb.increment()

                    # analyze the rest of the images
                    if len(paths) > (skipped + 1):
                        for it, image_path in enumerate(paths[skipped+1:]):
                            analysis = self.analyze_image(image_path, hp_median, hp_threshold, denoise_cycles,
                                                          gauss_filter, com_threshold, plots)

                            if analysis['is_valid']:
                                # profiles
                                pos_max: tuple[int, int] = analysis['position_max']
                                pos_com: tuple[int, int] = analysis['position_com']
                                profiles = spot_analysis(analysis['image_raw'], [(pos_max[0], pos_max[1], 'max'),
                                                                                 (pos_com[0], pos_com[1], 'com')])
                                for k, v in profiles.items():
                                    analysis[k] = v

                                # average
                                alpha = 1.0 / (it + 2)
                                beta = 1.0 - alpha
                                avg_image = cv2.addWeighted(analysis['image_raw'], alpha, avg_image, beta, 0.0)

                            # collect
                            analyses.append(analysis)
                            pb.increment()

                # analyze the average image
                self.average_analysis = self.analyze_image(avg_image, hp_median, hp_threshold, denoise_cycles,
                                                           gauss_filter, com_threshold, plots)
                if self.average_analysis['is_valid']:
                    # profiles
                    pos_max: tuple[int, int] = self.average_analysis['position_max']
                    pos_com: tuple[int, int] = self.average_analysis['position_com']
                    profiles = spot_analysis(self.average_analysis['image_raw'], [(pos_max[0], pos_max[1], 'max'),
                                                                                  (pos_com[0], pos_com[1], 'com')])
                    for k, v in profiles.items():
                        self.average_analysis[k] = v

                self.image_analyses = analyses
                self.average_image = avg_image
                # self.analyses_summary = self.summarize_image_analyses()

            except Exception as ex:
                api_error.error(str(ex), 'Failed to analyze image')
                pass

    def analyze_image(self, image: Union[SysPath, np.ndarray], hp_median: int = 2, hp_threshold: float = 3.,
                      denoise_cycles: int = 0, gauss_filter: float = 5., com_threshold: float = 0.5,
                      plots: bool = False) -> dict[str, Any]:
        # raw mage
        if isinstance(image, np.ndarray):
            image_raw = image
        else:
            image_raw = self.read_image_as_float(image)

        if plots:
            plt.figure(figsize=UndulatorNoScan.fig_size)
            plt.imshow(image_raw)
            plt.show(block=False)

        # initial filtering
        analysis = filter_image(image_raw, hp_median, hp_threshold, denoise_cycles, gauss_filter, com_threshold)

        # check image (contrast)
        counts, bins = np.histogram(analysis['image_denoised'],
                                    bins=max(10, 2 * int(np.std(analysis['image_blurred']))))
        analysis['bkg_level']: float = bins[np.where(counts == np.max(counts))[0][0]]
        analysis['is_valid']: bool = np.std(bins) > analysis['bkg_level']

        # stop if low-contrast image
        if not analysis['is_valid']:
            return analysis

        try:
            # beam edges
            image_edges: np.ndarray = canny(analysis['image_thresholded'], sigma=3)
            image_edges = closing(image_edges.astype(float), square(3))
            image_edges = clear_border(image_edges)
            analysis['image_edges'] = image_edges
            if plots:
                plt.figure(figsize=UndulatorNoScan.fig_size)
                plt.imshow(image_edges)
                plt.show(block=False)

            # boxed image
            bw = closing(analysis['image_blurred'] > 0.5 * np.max(analysis['image_blurred']), square(3))
            cleared = clear_border(bw)
            label_image = label(cleared)
            areas = [box.area for box in regionprops(label_image)]
            roi = regionprops(label_image)[areas.index(max(areas))]
            analysis['roi'] = np.array([roi.bbox[1], roi.bbox[3], roi.bbox[0], roi.bbox[2]])  # left, right, top, bottom

            pos_box = np.array([(roi.bbox[0] + roi.bbox[2]) / 2., (roi.bbox[1] + roi.bbox[3]) / 2.])
            pos_box = np.round(pos_box).astype(int)
            analysis['position_box'] = pos_box  # i, j

            if plots:
                plt.figure(figsize=UndulatorNoScan.fig_size)
                plt.imshow(image_raw)
                plt.plot(roi.bbox[1] * np.ones((roi.bbox[2] - roi.bbox[0] + 1)),
                         np.arange(roi.bbox[0], roi.bbox[2] + 1), '-r')  # left edge
                plt.plot(roi.bbox[3] * np.ones((roi.bbox[2] - roi.bbox[0] + 1)),
                         np.arange(roi.bbox[0], roi.bbox[2] + 1), '-r')  # right edge
                plt.plot(np.arange(roi.bbox[1], roi.bbox[3] + 1),
                         roi.bbox[0] * np.ones((roi.bbox[3] - roi.bbox[1] + 1)), '-r')  # top edge
                plt.plot(np.arange(roi.bbox[1], roi.bbox[3] + 1),
                         roi.bbox[2] * np.ones((roi.bbox[3] - roi.bbox[1] + 1)), '-r')  # bottom edge
                plt.plot(pos_box[1], pos_box[0], 'k.')
                plt.show(block=False)

            # ellipse fit (min_size: min of major axis, max_size: max of minor axis)
            h_ellipse = hough_ellipse(image_edges, min_size=10, max_size=60, accuracy=8)
            h_ellipse.sort(order='accumulator')
            best = list(h_ellipse[-1])
            yc, xc, a, b = (int(round(x)) for x in best[1:5])
            ellipse = (yc, xc, a, b, best[5])
            analysis['ellipse'] = ellipse  # (i, j, major, minor, orientation)

            # cy, cx = ellipse_perimeter(*ellipse)
            pos_ellipse = np.array([yc, xc])
            analysis['position_box'] = pos_ellipse  # i, j

            if plots:
                ell = mpatches.Ellipse(xy=(xc, yc), width=2*ellipse[3], height=2*ellipse[2],
                                       angle=math.degrees(ellipse[4]))
                plt.figure(figsize=UndulatorNoScan.fig_size)
                plt.imshow(image_raw)
                plt.gca().add_artist(ell)
                ell.set_alpha(1)
                ell.set_edgecolor('r')
                # noinspection PyArgumentList
                ell.set_fill(False)
                plt.plot(pos_ellipse[1], pos_ellipse[0], 'k.')
                plt.show(block=True)

        except Exception:
            pass

        return analysis

    # make private
    # add distance to box/ellipse centers
    def target_analysis(self) -> dict[str, Union[float, np.ndarray]]:
        target_analysis = {}
        try:
            target_analysis['target_um_pix'] = \
                float(GeecsDevice.exp_info['devices'][self.camera_name]['SpatialCalibration']['defaultvalue'])
            target_analysis['target_xy'] = \
                (float(GeecsDevice.exp_info['devices'][self.camera_name]['Target.Y']['defaultvalue']),
                 float(GeecsDevice.exp_info['devices'][self.camera_name]['Target.X']['defaultvalue']))
        except Exception:
            target_analysis['target_um_pix'] = 1.
            target_analysis['target_xy'] = (0., 0.)

        # from max
        target_analysis['avg_img_max_delta'] = \
            (np.array(self.average_analysis['position_max']) - np.array(target_analysis['target'])) \
            * target_analysis['target_um_pix'] / 1000.
        target_analysis['scan_pos_max_delta'] = \
            (np.array(self.analyses_summary['scan_pos_max']) - np.array(target_analysis['target'])) \
            * target_analysis['target_um_pix'] / 1000.
        target_analysis['target_deltas_max_mean'] = np.array([np.mean(target_analysis['scan_pos_max_delta'], axis=0)])
        target_analysis['target_deltas_max_std'] = np.array([np.std(target_analysis['scan_pos_max_delta'], axis=0)])

        # from CoM
        target_analysis['avg_img_com_delta'] = \
            (np.array(self.average_analysis['position_com']) - np.array(target_analysis['target'])) \
            * target_analysis['target_um_pix'] / 1000.
        target_analysis['scan_pos_com_delta'] = \
            (np.array(self.analyses_summary['scan_pos_com']) - np.array(target_analysis['target'])) \
            * target_analysis['target_um_pix'] / 1000.
        target_analysis['target_deltas_com_mean'] = np.array([np.mean(target_analysis['scan_pos_com_delta'], axis=0)])
        target_analysis['target_deltas_com_std'] = np.array([np.std(target_analysis['scan_pos_com_delta'], axis=0)])

        return target_analysis

    # make private
    # add width/height of box mean/std; same for ellipse
    def summarize_image_analyses(self) -> dict[str, Union[float, np.ndarray]]:
        scan_pos_max = np.array([analysis['position_max'] for analysis in self.image_analyses
                                 if analysis is not None and 'position_max' in analysis])
        scan_pos_max_fwhm_x = np.array([fwhm(analysis['opt_x_max'][3]) for analysis in self.image_analyses
                                        if analysis is not None and 'opt_x_max' in analysis])
        scan_pos_max_fwhm_y = np.array([fwhm(analysis['opt_y_max'][3]) for analysis in self.image_analyses
                                        if analysis is not None and 'opt_y_max' in analysis])

        scan_pos_com = np.array([analysis['position_com'] for analysis in self.image_analyses
                                 if analysis is not None and 'position_com' in analysis])
        scan_pos_com_fwhm_x = np.array([fwhm(analysis['opt_x_com'][3]) for analysis in self.image_analyses
                                        if analysis is not None and 'opt_x_com' in analysis])
        scan_pos_com_fwhm_y = np.array([fwhm(analysis['opt_y_com'][3]) for analysis in self.image_analyses
                                        if analysis is not None and 'opt_y_com' in analysis])

        mean_pos_max = np.mean(scan_pos_max, axis=0)
        mean_pos_max_fwhm_x = np.mean(scan_pos_max_fwhm_x)
        mean_pos_max_fwhm_y = np.mean(scan_pos_max_fwhm_y)
        std_pos_max = np.std(scan_pos_max, axis=0)
        std_pos_max_fwhm_x = np.std(scan_pos_max_fwhm_x)
        std_pos_max_fwhm_y = np.std(scan_pos_max_fwhm_y)

        mean_pos_com = np.mean(scan_pos_com, axis=0)
        mean_pos_com_fwhm_x = np.mean(scan_pos_com_fwhm_x)
        mean_pos_com_fwhm_y = np.mean(scan_pos_com_fwhm_y)
        std_pos_com = np.std(scan_pos_com, axis=0)
        std_pos_com_fwhm_x = np.std(scan_pos_com_fwhm_x)
        std_pos_com_fwhm_y = np.std(scan_pos_com_fwhm_y)

        return {'scan_pos_max': scan_pos_max,
                'scan_pos_max_fwhm_x': scan_pos_max_fwhm_x,
                'scan_pos_max_fwhm_y': scan_pos_max_fwhm_y,
                'scan_pos_com': scan_pos_com,
                'scan_pos_com_fwhm_x': scan_pos_com_fwhm_x,
                'scan_pos_com_fwhm_y': scan_pos_com_fwhm_y,
                'mean_pos_max': mean_pos_max,
                'mean_pos_max_fwhm_x': mean_pos_max_fwhm_x,
                'mean_pos_max_fwhm_y': mean_pos_max_fwhm_y,
                'std_pos_max': std_pos_max,
                'std_pos_max_fwhm_x': std_pos_max_fwhm_x,
                'std_pos_max_fwhm_y': std_pos_max_fwhm_y,
                'mean_pos_com': mean_pos_com,
                'mean_pos_com_fwhm_x': mean_pos_com_fwhm_x,
                'mean_pos_com_fwhm_y': mean_pos_com_fwhm_y,
                'std_pos_com': std_pos_com,
                'std_pos_com_fwhm_x': std_pos_com_fwhm_x,
                'std_pos_com_fwhm_y': std_pos_com_fwhm_y}


if __name__ == '__main__':
    _folder = r'C:\Users\GuillaumePlateau\Documents\LBL\Data\Undulator\Y2023\05-May\23_0509\scans\Scan028'

    _scan = Scan(_folder, ignore_experiment_name=True)
    images = UndulatorNoScan(_scan, 'U7', angle=0)
    images.analyze_images(hp_median=2, hp_threshold=3., denoise_cycles=0,
                          gauss_filter=5., com_threshold=0.5, plots=True)

    plt.figure(figsize=UndulatorNoScan.fig_size)
    plt.imshow(images.average_image)
    plt.show(block=True)

    # plt.ion()
    # ax, _ = show_one(images.average_image, size_factor=1., colormap='hot', hide_ticks=False, show_colorbar=True,
    #                  show_contours=True, contours=2, contours_colors='w', contours_labels=False,
    #                  show=False, block_execution=False)
    # if not ax.yaxis_inverted():
    #     ax.invert_yaxis()
    # plt.show(block=True)

    # _image, _ = average_images(folder)
    # HtuImages.find_roi(_image, None, display_factor=0.5)
