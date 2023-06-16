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
from progressbar import ProgressBar
from typing import Optional, Any, Union
from geecs_api.tools.images.batches import list_images
from geecs_api.devices.geecs_device import api_error
import geecs_api.tools.images.ni_vision as ni
from geecs_api.tools.interfaces.exports import save_py, load_py
from geecs_api.tools.scans.scan_data import ScanData
from geecs_api.tools.images.scan_images import ScanImages
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.diagnostics.cameras import Camera
from geecs_api.tools.images.filtering import basic_filter, FiltersParameters
from geecs_api.tools.distributions.fit_utility import fit_distribution
from geecs_api.tools.images.spot import spot_analysis, fwhm, n_sigma_window
from geecs_api.tools.interfaces.prompts import text_input


# base_path = Path(r'C:\Users\GuillaumePlateau\Documents\LBL\Data')
base_path = Path(r'Z:\data')

is_local = (str(base_path)[0] == 'C')
if not is_local:
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')


# Parameters
# -----------------------------------------------------------------
_save: bool = True
_images: int = -1

_base_tag = (2023, 4, 20, 75)
# _screen_label = 'U9'

_scan_path: Path = base_path/'Undulator'/f'Y{_base_tag[0]}'/f'{_base_tag[1]:02d}-{cal.month_name[_base_tag[1]][:3]}'\
                   / f'{str(_base_tag[0])[-2:]}_{_base_tag[1]:02d}{_base_tag[2]:02d}'/'scans'/f'Scan{_base_tag[3]:03d}'

_save_dir = _scan_path.parents[1]/'analysis'/_scan_path.parts[-1]
_image_path: Path = _save_dir / 'analysis.png'

_export_file_path: Path = _save_dir / 'charge_vs_radiation'
dat_file = _export_file_path.parent / f'{_export_file_path.name}.dat'
image_path: Path = _save_dir / 'radiation_vs_charge.png'


# Values
# -----------------------------------------------------------------
analyze: str = 'y'
if dat_file.is_file():
    analyze = text_input('\nRe-run the analysis? : ', accepted_answers=['y', 'yes', 'n', 'no'])

if (analyze.lower()[0] == 'y') or (not dat_file.is_file()):
    _scan = ScanData(_scan_path, ignore_experiment_name=is_local)

    # Scope
    q_scope = np.array(_scan.data_frame[('U-picoscope5245D', 'charge pC')])
    if _images > 0:
        q_scope = q_scope[-_images:]

    # U9
    _scan_images_u9 = ScanImages(_scan, 'U9')
    analysis_file_u9, analysis_u9 = \
        _scan_images_u9.run_analysis_with_checks(images=_images,
                                                 initial_filtering=FiltersParameters(com_threshold=0.66,
                                                                                     contrast=1.333),
                                                 trim_collection=True, new_targets=True, plots=True, save=_save)
    mean_counts_u9 = [np.sum(im_analysis['image_raw']) / im_analysis['image_raw'].size
                      for im_analysis in analysis_u9['image_analyses']]
    mean_counts_u9 = np.array(mean_counts_u9)

    poly_u9_to_q, _, fit_u9_to_q = fit_distribution(mean_counts_u9, q_scope, fit_type='linear')

    # Rad2
    _scan_images_rad2 = ScanImages(_scan, 'Rad2')
    analysis_file_rad2, analysis_rad2 = \
        _scan_images_rad2.run_analysis_with_checks(images=_images,
                                                   initial_filtering=FiltersParameters(com_threshold=0.66,
                                                                                       contrast=1.333),
                                                   trim_collection=True, new_targets=True, plots=True, save=_save)
    mean_counts_rad = [np.sum(im_analysis['image_raw']) / im_analysis['image_raw'].size
                       for im_analysis in analysis_rad2['image_analyses']]
    mean_counts_rad = np.array(mean_counts_rad)

    # Save
    data_dict: dict[str, Any] = {'scan_path': _scan_path,
                                 'analysis_file_u9': analysis_file_u9,
                                 'analysis_u9': analysis_u9,
                                 'mean_counts_u9': mean_counts_u9,
                                 'analysis_file_rad2': analysis_file_rad2,
                                 'analysis_rad2': analysis_rad2,
                                 'mean_counts_rad': mean_counts_rad,
                                 'q_scope': q_scope,
                                 'poly_u9_to_q': poly_u9_to_q,
                                 'fit_u9_to_q': fit_u9_to_q}
    if _save:
        if not _save_dir.is_dir():
            os.makedirs(_save_dir)

        save_py(file_path=_export_file_path, data=data_dict)
        print(f'Data exported to:\n\t{_export_file_path}.dat')

else:
    print('Loading analysis...')
    analysis, analysis_file = load_py(_export_file_path, as_dict=True)
    _scan = ScanData(analysis['scan_path'], ignore_experiment_name=is_local)
    q_scope = analysis['q_scope']
    mean_counts_rad = analysis['mean_counts_rad']


# Analysis
# -----------------------------------------------------------------
q_sort = np.argsort(q_scope)
q_scope = q_scope[q_sort]
mean_counts_rad = mean_counts_rad[q_sort]

# bins = np.linspace(np.floor(q_scope[0]), np.ceil(q_scope[-1]), 20+1)
bins = np.linspace(q_scope[0], q_scope[-1], 19+1)
half_bin = np.mean(bins[1:] - bins[:-1]) / 2
bins = np.insert(bins + half_bin, 0, bins[0] - half_bin)

lowest_count = []
for it in range(bins.size-1):
    cts = mean_counts_rad[np.where((bins[it] <= q_scope) & (q_scope < bins[it + 1]))[0]]
    lowest_count.append(np.mean(cts[np.argsort(cts)[:2]]))
lowest_count = np.array(lowest_count)

bin_pos = (bins[1:] + bins[:-1]) / 2
poly_low_rad, _, fit_low_rad = fit_distribution(bin_pos, lowest_count, fit_type='linear')


# Plot
# -----------------------------------------------------------------
fig = plt.figure(figsize=(ScanImages.fig_size[0] * 1.5, ScanImages.fig_size[1]))
plt.plot(q_scope, mean_counts_rad, '.', alpha=0.3, markersize=12)
plt.plot(q_scope, poly_low_rad[0] * q_scope + poly_low_rad[1], linewidth=1)
plt.xlabel('Charge [pC]')
plt.ylabel('Mean Counts [a.u.]')

if _save:
    plt.savefig(image_path, dpi=300)

# plt.show(block=True)

try:
    plt.close(fig)
except Exception:
    pass

print('done')
