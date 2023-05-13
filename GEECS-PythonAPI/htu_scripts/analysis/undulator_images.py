import cv2
import numpy as np
import numpy.typing as npt
import scipy.ndimage as simg
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from progressbar import ProgressBar
from typing import Optional, Any, Union
from geecs_api.api_defs import SysPath
from geecs_api.tools.images.batches import list_images, average_images
from geecs_api.devices.geecs_device import api_error
import geecs_api.tools.images.ni_vision as ni
from geecs_api.tools.scans.scan import Scan
from geecs_api.devices.HTU.diagnostics.cameras import Camera
from geecs_api.tools.images.filtering import clip_hot_pixels
from htu_scripts.analysis.spot_analysis import spot_analysis, fwhm


class UndulatorImage:
    def __init__(self, scan: Scan, label: str, camera: Union[Camera, str]):
        self.scan: Scan = scan
        self.camera_label: str = label

        if isinstance(camera, Camera):
            self.camera_name: str = camera.get_name()
            self.camera_roi: Optional[np.ndarray] = camera.roi
        elif isinstance(camera, str) and (camera in Camera.ROIs):
            self.camera_name = camera
            self.camera_roi = np.array(Camera.ROIs[camera])
        else:
            self.camera_roi = None

        self.image_folder: SysPath = self.scan.get_folder() / self.camera_name
        self.image_analyses: Optional[list[dict[str, Any]]] = None
        self.average_image: Optional[np.ndarray] = None
        self.target_analysis: Optional[dict[str, tuple[np.ndarray, np.ndarray]]] = None


def analyze_images(images_folder: SysPath, n_images: int = 0, file_extension: str = '.png', rotate_deg: int = 0,
                   screen_label: str = '', hp_median: int = 2, hp_threshold: float = 3., denoise_cycles: int = 0,
                   gauss_filter: float = 5., com_threshold: float = 0.5) \
        -> tuple[list[dict[str, Any]], Optional[np.ndarray]]:
    paths = list_images(images_folder, n_images, file_extension)
    analyses: list[dict[str, Any]] = []
    avg_image: Optional[np.ndarray] = None
    rot_90deg: int = int(round(rotate_deg / 90.))

    # run analysis
    if paths:
        try:
            with ProgressBar(max_value=len(paths)) as pb:
                avg_image = np.rot90(ni.read_imaq_image(paths[0]), rot_90deg)
                avg_image = avg_image.astype('float64')
                analyses.append(spot_analysis(avg_image, hp_median, hp_threshold,
                                              denoise_cycles, gauss_filter, com_threshold))
                pb.increment()

                if len(paths) > 1:
                    for it, image_path in enumerate(paths[1:]):
                        image_data = np.rot90(ni.read_imaq_image(image_path), rot_90deg)
                        image_data = image_data.astype('float64')
                        analyses.append(spot_analysis(image_data, hp_median, hp_threshold,
                                                      denoise_cycles, gauss_filter, com_threshold))
                        alpha = 1.0 / (it + 2)
                        beta = 1.0 - alpha
                        avg_image = cv2.addWeighted(image_data, alpha, avg_image, beta, 0.0)
                        pb.increment()

        except Exception as ex:
            api_error.error(str(ex), 'Failed to analyze image')
            pass

    return analyses, avg_image


def summarize_image_analyses(analyses: list[dict[str, Any]]) -> dict[str, Union[float, npt.ArrayLike]]:
    scan_pos_max = np.array([analysis['position_max'] for analysis in analyses if analysis is not None])
    scan_pos_max_fwhm_x = np.array([fwhm(analysis['opt_x_max'][3]) for analysis in analyses if analysis is not None])
    scan_pos_max_fwhm_y = np.array([fwhm(analysis['opt_y_max'][3]) for analysis in analyses if analysis is not None])

    scan_pos_com = np.array([analysis['position_com'] for analysis in analyses if analysis is not None])
    scan_pos_com_fwhm_x = np.array([fwhm(analysis['opt_x_com'][3]) for analysis in analyses if analysis is not None])
    scan_pos_com_fwhm_y = np.array([fwhm(analysis['opt_y_com'][3]) for analysis in analyses if analysis is not None])

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


def find_roi(image: np.ndarray, threshold: Optional[float] = None, display: bool = False):
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
        if display:
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            ax.imshow(image)
            rect = mpatches.Rectangle((roi_box[1], roi_box[0]), roi_box[3] - roi_box[1], roi_box[2] - roi_box[0],
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.set_axis_off()
            plt.tight_layout()
            plt.show(block=True)

    return roi_box, roi


if __name__ == '__main__':
    folder = r'C:\Users\GuillaumePlateau\Documents\LBL\Data\Undulator\Y2023\05-May\23_0509\scans\Scan030\UC_VisaEBeam9'

    _image, _ = average_images(folder)
    find_roi(_image, None, True)

    # plt.figure()
    # plt.imshow(_image)
    # plt.show(block=True)
