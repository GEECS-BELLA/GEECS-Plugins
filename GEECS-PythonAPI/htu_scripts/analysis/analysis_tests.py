import cv2
import numpy as np
import numpy.typing as npt
import scipy.ndimage as simg
from progressbar import ProgressBar
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from typing import Optional, Any, Union
from geecs_api.api_defs import SysPath
from geecs_api.tools.images.batches import list_images, average_images
from geecs_api.devices.geecs_device import api_error
import geecs_api.tools.images.ni_vision as ni
from geecs_api.tools.images.filtering import clip_hot_pixels
from geecs_api.tools.images.spot import spot_analysis, fwhm

# base_path = Path(r'Z:\data')
base_path = Path(r'C:\Users\GuillaumePlateau\Documents\LBL\Data')
folder = base_path / Path(r'Undulator\Y2023\05-May\23_0509\scans\Scan022\UC_VisaEBeam1')

# average
image, _ = average_images(folder)
# image = (image / np.max(image)) * 65535

# filter and smooth
blur = clip_hot_pixels(image, median_filter_size=2, threshold_factor=3)
blur = simg.gaussian_filter(blur, sigma=5.)

# threshold
bw = closing(blur > 1.1 * np.min(blur), square(3))

# remove artifacts connected to image border
cleared = clear_border(bw)

# label image regions
label_image = label(cleared)

# to make the background transparent, pass the value of `bg_label`,
# and leave `bg_color` as `None` and `kind` as `overlay`
image_label_overlay = label2rgb(label_image, image=image, bg_label=0)
areas = [box.area for box in regionprops(label_image)]
roi = regionprops(label_image)[areas.index(max(areas))]

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(blur)
rect = mpatches.Rectangle((roi.bbox[1], roi.bbox[0]), roi.bbox[3] - roi.bbox[1], roi.bbox[2] - roi.bbox[0],
                          fill=False, edgecolor='red', linewidth=1)
ax.add_patch(rect)
ax.set_axis_off()
plt.tight_layout()
plt.show(block=True)

# plt.figure()
# plt.imshow(image_label_overlay, cmap='gray')
# plt.show(block=True)

# thresh = cv2.inRange(blur, 1.1 * np.min(image), np.max(image))

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
# clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# edges = canny(thresh, sigma=3, low_threshold=0, high_threshold=255)

# contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# contours = contours[0] if len(contours) == 2 else contours[1]

# image = cv2.bilateralFilter(image.astype('float32'), 9, 150, 150)

# sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
# image = cv2.filter2D(image, -1, sharpen_kernel)

# image, contours, _hierarchy = cv2.findContours(image.astype('uint16'), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)
# counts = cv2.findContours(image.astype('uint16'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# counts = counts[0] if len(counts) == 2 else counts[1]

# get external contours
# contours = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours = contours[0] if len(contours) == 2 else contours[1]

# result1 = img.copy()
# result2 = img.copy()
# for c in contours:
#     cv2.drawContours(result1,[c],0,(0,0,0),2)
#     get rotated rectangle from contour
    # rot_rect = cv2.minAreaRect(c)
    # box = cv2.boxPoints(rot_rect)
    # box = np.int0(box)
    # draw rotated rectangle on copy of img
    # cv2.drawContours(result2,[box],0,(0,0,0),2)

# plt.figure()
# plt.imshow(image, cmap='jet')
# plt.show(block=True)

# blur = cv2.medianBlur(image, 5)

# plt.figure()
# plt.imshow(blur, cmap='jet')
# plt.show(block=True)

# Threshold and morph close
# thresh = cv2.threshold(sharpen, 160, 255, cv2.THRESH_BINARY_INV)[1]
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find contours and filter using threshold area
# counts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# counts = counts[0] if len(counts) == 2 else counts[1]


# image = clip_hot_pixels(image, median_filter_size=2, threshold_factor=1)
# plt.figure()
# plt.imshow(image, cmap='jet', vmin=0, vmax=np.mean(image))
# plt.show(block=True)
# min_area = 100
# max_area = 1500
# image_number = 0
#
# for c in counts:
#     area = cv2.contourArea(c)
#
#     if min_area < area < max_area:
#         x, y, w, h = cv2.boundingRect(c)
#         ROI = image[y:y+h, x:x+w]
#         cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
#         cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
#         image_number += 1
#
# cv2.imshow('sharpen', sharpen)
# cv2.imshow('close', close)
# cv2.imshow('thresh', thresh)
# cv2.imshow('image', image)
# cv2.waitKey()

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

