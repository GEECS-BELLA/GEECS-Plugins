import numpy as np
import scipy.ndimage as simg
from pathlib import Path
from typing import Any, Optional, Union
from dataclasses import dataclass
from tkinter import filedialog
from scipy.ndimage import median_filter
from skimage.restoration import denoise_wavelet, cycle_spin
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import RectangleSelector
from geecs_python_api.tools.images.batches import average_images
import geecs_python_api.tools.images.ni_vision as ni
from geecs_python_api.controls.devices.HTU.diagnostics.cameras import Camera


@dataclass
class FiltersParameters:
    contrast: float = 1.333
    hp_median: int = 2
    hp_threshold: float = 3.
    denoise_cycles: int = 0
    gauss_filter: float = 5.
    com_threshold: float = 0.5
    bkg_image: Optional[Union[Path, np.ndarray]] = None
    box: bool = True
    ellipse: bool = False


def mad(x):
    """ Median absolute deviation
    """
    return np.median(np.abs(x - np.median(x)))


def get_mad_threshold(x: np.ndarray, mad_multiplier: float = 5.2):
    return np.median(x) + mad_multiplier * mad(x)


def clip_outliers(image: np.ndarray,
                  mad_multiplier: float = 5.2,
                  nonzero_only: bool = True):
    """ Clips image values above a MAD-derived threshold.

    The threshold is median + mad_multiplier * MAD.

    if nonzero_only (default True), the median and MAD only take nonzero values
    into account, because otherwise in many pointing images the median and MAD are
    simply 0.

    """

    image_clipped = image.copy()
    thresh = (get_mad_threshold(image[image > 1e-9], mad_multiplier) if nonzero_only
              else get_mad_threshold(image, mad_multiplier))
    image_clipped[image_clipped > thresh] = thresh

    return image_clipped


def clip_hot_pixels(image: np.ndarray, median_filter_size: int = 2, threshold_factor: float = 3) -> np.ndarray:
    data_type = image.dtype
    image = image.astype('float64')

    # median filtering and hot pixels list
    blurred = median_filter(image, size=median_filter_size)
    difference = image - blurred
    threshold = threshold_factor * np.std(difference)
    hot_pixels = np.nonzero(np.abs(difference) > threshold)

    # fix image with median values
    for (y, x) in zip(hot_pixels[0], hot_pixels[1]):
        image[y, x] = blurred[y, x]

    return image.astype(data_type)


def denoise(image: np.ndarray, max_shifts: int = 3):
    return cycle_spin(image, func=denoise_wavelet, max_shifts=max_shifts)


def basic_filter(image: np.ndarray, analysis_dict: Optional[dict[str, Any]] = None,
                 hp_median: int = 2, hp_threshold: float = 3., denoise_cycles: int = 3,
                 gauss_filter: float = 5., com_threshold: float = 0.5) -> dict[str, Any]:
    """
    Applies a basic set of filters to an image

    hp_median:      hot pixels median filter size
    hp_threshold:   hot pixels threshold in # of sta. dev. of the median deviation
    gauss_filter:   gaussian filter size
    com_threshold:  image threshold for center-of-mass calculation
    """
    # clip hot pixels
    if hp_median > 0:
        image_clipped = clip_hot_pixels(image, median_filter_size=hp_median, threshold_factor=hp_threshold)
    else:
        image_clipped: np.ndarray = image.copy()

    # denoise
    if denoise_cycles > 0:
        image_denoised = denoise(image_clipped, max_shifts=denoise_cycles)
    else:
        image_denoised: np.ndarray = image_clipped.copy()

    # gaussian filter
    if gauss_filter > 0:
        image_blurred = simg.gaussian_filter(image, sigma=gauss_filter)
    else:
        image_blurred: np.ndarray = image_denoised.copy()

    # thresholding
    if com_threshold > 0:
        # peak location
        i_max, j_max = np.where(image_blurred == image_blurred.max(initial=0))
        i_max, j_max = i_max[0].item(), j_max[0].item()

        # center of mass of thresholded image
        val_max = image_blurred[i_max, j_max]
        binary = image_blurred > (com_threshold * val_max)
        image_thresholded = image_blurred * binary.astype(float)
        i_com, j_com = simg.center_of_mass(image_thresholded)
    else:
        i_max, j_max, i_com, j_com = -1
        image_thresholded: np.ndarray = image_blurred.copy()

    filters = {'hp_median': hp_median,
               'hp_threshold': hp_threshold,
               'denoise_cycles': denoise_cycles,
               'gauss_filter': gauss_filter,
               'com_threshold': com_threshold}

    positions = {'max_ij': (int(i_max), int(j_max)),
                 'com_ij': (int(i_com), int(j_com)),
                 'long_names': ['maximum', 'center of mass'],
                 'short_names': ['max', 'com']}

    arrays = {'raw_roi': image,
              'denoised': image_denoised,
              'blurred': image_blurred,
              'thresholded': image_thresholded}

    if analysis_dict is None:
        analysis_dict = {'filter_pars': filters,
                         'positions': positions,
                         'arrays': arrays}
    else:
        if 'filters' in analysis_dict:
            for k, v in filters.items():
                analysis_dict['filters'][k] = v
        else:
            analysis_dict['filters'] = filters

        if 'positions' in analysis_dict:
            for k, v in positions.items():
                analysis_dict['positions'][k] = v
        else:
            analysis_dict['positions'] = positions

        if 'arrays' in analysis_dict:
            for k, v in arrays.items():
                analysis_dict['arrays'][k] = v
        else:
            analysis_dict['arrays'] = arrays

    return analysis_dict


class ROI:
    """ Specify a region of interest for an ImageAnalyzer to crop with.
    
    This is given a class so that there can be no confusion about the order
    of indices.

    Initialzed with top, bottom, left, right indices. Cropping is done with 
    slices containing these indices, which means: 
        * None is valid, meaning go to the edge
        * Negative integers are valid, which means up to edge - value.
        * The top and left indices are inclusive, and bottom and right indices
          are exclusive.

    The convention of (0, 0) at the top left corner is used, which means 
    top should be less than bottom. 

    If not given as kwargs, the coordinates are in order that they would be 
    used in slicing: 
        image[i0:i1, i2:i3]

    """
    def __init__(self, top: Optional[int] = None, 
                       bottom: Optional[int] = None, 
                       left: Optional[int] = None, 
                       right: Optional[int] = None, 

                       bad_index_order = 'raise',
                ):

        """
        Parameters
        ----------
        top, bottom, left, right : Optional[int]
            indices of ROI. Cropping is done with slices containing these indices,
            which means: 
                * None is valid, meaning go to the edge
                * Negative integers are valid, which means up to edge - value.
                * The top and left indices are inclusive, and bottom and right indices
                  are exclusive.

            The convention of (0, 0) at the top left corner is used, which means 
            top should be less than bottom. 

        bad_index_order : one of 'raise', 'invert', 'invert_warn'
            what to do if top > bottom, or right > left
                'raise': raise ValueError
                'invert': silently switch top/bottom, or left/right indices
                'invert_warn': switch top/bottom or left/right indices with warning.

        """

        def check_index_order(low_name, low_index, high_name, high_index):
            """ Checks whether low_index < high_index, and takes appropriate action.

            Returns
            -------
            low_index, high_index : int
                possibly inverted.

            """
            if low_index is None or high_index is None:
                return low_index, high_index

            if low_index > high_index:
                if bad_index_order == 'raise': 
                    raise ValueError(f"{low_index} should be less than {high_index} ((0, 0) is at the top left corner)")
                elif bad_index_order == 'invert':
                    low_index, high_index = high_index, low_index
                elif bad_index_order == 'invert_warn':
                    low_index, high_index = high_index, low_index
                    warn(f"Inverting {low_index} and {high_index}.")
                else:
                    raise ValueError(f"Unknown action for bad_index_order: {bad_index_order}")

            return low_index, high_index

        top, bottom = check_index_order('top', top, 'bottom', bottom)
        left, right = check_index_order('left', left, 'right', right)

        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def crop(self, image: np.ndarray):
        return image[self.top:self.bottom, self.left:self.right]



def check_roi(images: Path, initial_roi: Optional[np.ndarray] = None, camera_name: Optional[str] = None,
              disp_range: Optional[tuple[float, float]] = None, color_map: str = 'jet'):
    # image
    if images.is_dir():
        avg_image, _ = average_images(images)
    elif images.is_file():
        avg_image = ni.read_imaq_image(images)
    else:
        return

    # ROIs
    if camera_name and (camera_name in Camera.ROIs):
        default_roi: Optional[np.ndarray] = np.array(Camera.ROIs[camera_name])
        if default_roi.size < 4:
            default_roi = None
    else:
        default_roi = None

    # if initial_roi is None or (initial_roi.size < 4):
    #     initial_roi = find_roi(avg_image, threshold=None, plots=False)

    if disp_range:
        avg_image[np.where(avg_image < disp_range[0])] = disp_range[0]
        avg_image[np.where(avg_image > disp_range[1])] = disp_range[1]

    # figure
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.imshow(avg_image, cmap=color_map, aspect='equal', origin='upper')
    if default_roi is not None:
        rect_def = mpatches.Rectangle((default_roi[0], default_roi[2]),
                                      default_roi[1] - default_roi[0], default_roi[3] - default_roi[2],
                                      fill=False, edgecolor='gray', linewidth=1, linestyle='--')
        ax.add_patch(rect_def)

    # if initial_roi is not None:
    #     rect_ini = mpatches.Rectangle((initial_roi[0], initial_roi[2]),
    #                                   initial_roi[1] - initial_roi[0], initial_roi[3] - initial_roi[2],
    #                                   fill=False, edgecolor='m', linewidth=1, linestyle='-')
    #     ax.add_patch(rect_ini)

    ax.set_axis_off()
    plt.tight_layout()
    if ax.xaxis_inverted():
        ax.invert_xaxis()
    if not ax.yaxis_inverted():
        ax.invert_yaxis()

    def line_select_callback(e_click, e_release):
        x1, y1 = e_click.xdata, e_click.ydata
        x2, y2 = e_release.xdata, e_release.ydata

        [p.remove() for p in reversed(ax.patches)]
        rect = plt.Rectangle((min(x1, x2), min(y1, y2)), np.abs(x1 - x2), np.abs(y1 - y2),
                             fill=False, edgecolor='k', linewidth=1, linestyle='-')
        ax.add_patch(rect)

    rs = RectangleSelector(ax, line_select_callback, useblit=False, button=[1],
                           minspanx=5, minspany=5, spancoords='pixels', interactive=True)
    plt.show()
    selected_roi = [int(round(v)) for v in rs.extents]
    print(f'Selected ROI (left, right, top, bottom):\n\t{selected_roi}')


def find_roi(image: np.ndarray, threshold: Optional[float] = None, plots: bool = False):
    roi_box = np.array([0, image.shape[1] - 1, 0, image.shape[0]])  # left, right, top, bottom

    try:
        # filter and smooth
        blur = clip_hot_pixels(image, median_filter_size=2, threshold_factor=3)
        blur = simg.gaussian_filter(blur, sigma=5.)

        # threshold
        if threshold is None:
            counts, bins = np.histogram(blur, bins=10)
            threshold = bins[np.where(counts == np.max(counts))[0][0] + 1]
        binary = closing(blur > threshold, square(3))

        # label image regions
        label_image = label(binary)
        areas = [box.area for box in regionprops(label_image)]
        roi = regionprops(label_image)[areas.index(max(areas))]
        roi_box = np.array([roi.bbox[1], roi.bbox[3], roi.bbox[0], roi.bbox[2]])

    except Exception:
        pass

    finally:
        if plots:
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            ax.imshow(image)
            rect = mpatches.Rectangle((roi_box[0], roi_box[2]), roi_box[1] - roi_box[0], roi_box[3] - roi_box[2],
                                      fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)
            ax.set_axis_off()
            plt.tight_layout()
            plt.show(block=True)

    return roi_box


if __name__ == '__main__':
    # _base = Path(r'C:\Users\GuillaumePlateau\Documents\LBL\Data')
    _base = Path(r'Z:\data')

    # _camera = 'UC_UndulatorRad2'
    # _camera = 'UC_VisaEBeam9'
    # _folder = _base / fr'Undulator\Y2023\04-Apr\23_0420\scans\Scan075\{_camera}'
    _folder = filedialog.askdirectory(initialdir=_base/r'Undulator\Y2023', title='Directory:')

    # _range = None
    _range = (0, 80.)

    if _folder:
        _folder = Path(_folder)
        # check_roi(_folder, camera_name=Camera.name_from_label(_camera), disp_range=_range, color_map='hot')
        check_roi(_folder, camera_name=_folder.name, disp_range=_range, color_map='jet')
