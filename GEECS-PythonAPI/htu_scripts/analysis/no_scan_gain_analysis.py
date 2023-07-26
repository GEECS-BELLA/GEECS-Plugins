import os
import numpy as np
from pathlib import Path
import calendar as cal
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from typing import Any
from geecs_api.tools.images.batches import list_images
from geecs_api.tools.interfaces.exports import save_py, load_py
from geecs_api.tools.scans.scan_data import ScanData
from geecs_api.tools.scans.scan_images import ScanImages
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.tools.distributions.fit_utility import fit_distribution


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

_export_file_path: Path = _save_dir / 'radiation_vs_charge'
dat_file = _export_file_path.parent / f'{_export_file_path.name}.dat'
image_path_rad: Path = _save_dir / 'radiation_vs_charge.png'
image_path_gain: Path = _save_dir / 'gain_vs_charge.png'
image_path_rad_and_gain: Path = _save_dir / 'rad_and_gain_vs_charge.png'


# Values
# -----------------------------------------------------------------
analyze: str = 'y'
if dat_file.is_file():
    analyze = 'n'
    # analyze = text_input('\nRe-run the analysis? : ', accepted_answers=['y', 'yes', 'n', 'no'])

if (analyze.lower()[0] == 'y') or (not dat_file.is_file()):
    _scan = ScanData(_scan_path, ignore_experiment_name=is_local)

    # Scope
    q_scope = np.array(_scan.data_frame[('U-picoscope5245D', 'charge pC')])
    if _images > 0:
        q_scope = q_scope[-_images:]

    # U9
    _scan_images_u9 = ScanImages(_scan, 'U9')
    paths_u9 = list_images(_scan_images_u9.image_folder, -1, '.png')

    u9_mean = np.zeros((len(paths_u9),))
    with ProgressBar(max_value=len(paths_u9)) as pb:
        for it, image_path_rad in enumerate(paths_u9):
            image_raw = _scan_images_u9.read_image_as_float(image_path_rad)
            u9_mean[it] = np.mean(image_raw)
            pb.increment()

    # Rad2
    _scan_images_rad2 = ScanImages(_scan, 'Rad2')
    _scan_images_rad2.camera_roi = np.array([1364, 2233, 482, 1251])
    paths_rad2 = list_images(_scan_images_rad2.image_folder, -1, '.png')

    rad2_mean = np.zeros((len(paths_rad2),))
    with ProgressBar(max_value=len(paths_rad2)) as pb:
        for it, image_path_rad in enumerate(paths_rad2):
            image_raw = _scan_images_rad2.read_image_as_float(image_path_rad)
            rad2_mean[it] = np.mean(image_raw)
            pb.increment()

    # Save
    data_dict: dict[str, Any] = {'scan_path': _scan_path,
                                 'q_scope': q_scope,
                                 'paths_u9': paths_u9,
                                 'roi_u9': _scan_images_u9.camera_roi,
                                 'u9_mean': u9_mean,
                                 'paths_rad2': paths_rad2,
                                 'roi_rad2': _scan_images_rad2.camera_roi,
                                 'rad2_mean': rad2_mean}

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
    u9_mean = analysis['u9_mean']
    rad2_mean = analysis['rad2_mean']

    print(f'ROI U9: {analysis["roi_u9"]}')
    print(f'ROI Rad2: {analysis["roi_rad2"]}')


# Analysis
# -----------------------------------------------------------------
q_sort = np.argsort(q_scope)
q_scope = q_scope[q_sort]
u9_mean = u9_mean[q_sort]
rad2_mean = rad2_mean[q_sort]

# baseline u9
bins_u = np.linspace(u9_mean[0], u9_mean[-1], 9 + 1)
half_bin_u = np.mean(bins_u[1:] - bins_u[:-1]) / 2
bins_u = np.insert(bins_u + half_bin_u, 0, bins_u[0] - half_bin_u)

base_count_u9 = []
for it in range(bins_u.size - 1):
    cts = rad2_mean[np.where((bins_u[it] <= u9_mean) & (u9_mean < bins_u[it + 1]))[0]]
    base_count_u9.append(np.min(cts))
base_count_u9 = np.array(base_count_u9)

bin_pos_u = (bins_u[1:] + bins_u[:-1]) / 2
poly_u9, _, fit_u9 = fit_distribution(bin_pos_u, base_count_u9, fit_type='linear')
intercept_u9 = poly_u9[1]

# baseline q-scope
bins_q = np.linspace(q_scope[0], q_scope[-1], 14 + 1)
half_bin_q = np.mean(bins_q[1:] - bins_q[:-1]) / 2
bins_q = np.insert(bins_q + half_bin_q, 0, bins_q[0] - half_bin_q)

base_count_rad2 = []
for it in range(bins_q.size - 1):
    cts = rad2_mean[np.where((bins_q[it] <= q_scope) & (q_scope < bins_q[it + 1]))[0]]
    base_count_rad2.append(np.min(cts))
base_count_rad2 = np.array(base_count_rad2)

bin_pos_q = (bins_q[1:] + bins_q[:-1]) / 2
poly_rad2, _, fit_rad2 = fit_distribution(bin_pos_q, base_count_rad2, fit_type='linear')
intercept_rad2 = poly_rad2[1]


# Gains
# -----------------------------------------------------------------
rad2_mean_corrected_u = rad2_mean - intercept_u9
gain_u = rad2_mean_corrected_u / (poly_u9[0] * u9_mean)

rad2_mean_corrected_q = rad2_mean - intercept_rad2
gain_q = rad2_mean_corrected_q / (poly_rad2[0] * q_scope)


# Plot radiation vs. charge
# -----------------------------------------------------------------
fig_rc = plt.figure(figsize=(ScanImages.fig_size[0] * 1.5, ScanImages.fig_size[1]))
grid = plt.GridSpec(1, 2, hspace=0.3, wspace=0.3)
ax_rq = fig_rc.add_subplot(grid[0, 0])
ax_ru = fig_rc.add_subplot(grid[0, 1], sharey=ax_rq)

ax_rq.plot(q_scope, rad2_mean_corrected_q, '.', alpha=0.3, markersize=12)
ax_rq.plot(np.insert(q_scope, 0, 0.), poly_rad2[0] * np.insert(q_scope, 0, 0.), linewidth=1)
# ax_q.plot(bin_pos_q, base_count_rad2 - intercept_rad2, '.k')
ax_rq.set_xlabel('Charge [pC]')
ax_rq.set_ylabel('Mean Counts [a.u.]')
ax_rq_xlim = ax_rq.get_xlim()

ax_ru.plot(u9_mean, rad2_mean_corrected_u, '.', alpha=0.3, markersize=12)
ax_ru.plot(np.insert(u9_mean, 0, 0.), poly_u9[0] * np.insert(u9_mean, 0, 0.), linewidth=1)
# ax_u.plot(bin_pos_u, base_count_u9 - intercept_u9, '.k')
ax_ru.set_xlabel('End-Station Screen Counts [a.u.]')
ax_ru_xlim = ax_ru.get_xlim()

if _save:
    plt.savefig(image_path_rad, dpi=300)

plt.show(block=False)


# Plot gain vs. charge
# -----------------------------------------------------------------
fig_gc = plt.figure(figsize=(ScanImages.fig_size[0] * 1.5, ScanImages.fig_size[1]))
grid = plt.GridSpec(1, 2, hspace=0.3, wspace=0.3)
ax_gu = fig_gc.add_subplot(grid[0, 1])
ax_gq = fig_gc.add_subplot(grid[0, 0], sharey=ax_gu)

ax_gq.plot(q_scope, gain_q, '.', alpha=0.3, markersize=12)
ax_gq.axhline(1., c='orange', linestyle='--', linewidth=1)
ax_gq.set_xlim(ax_rq_xlim[0], ax_rq_xlim[1])
ax_gq.set_xlabel('Charge [pC]')
ax_gq.set_ylabel('Gain')

ax_gu.plot(u9_mean, gain_u, '.', alpha=0.3, markersize=12)
ax_gu.axhline(1., c='orange', linestyle='--', linewidth=1)
ax_gu.set_ylim(0, ax_gu.get_ylim()[1])
ax_gu.set_xlim(ax_ru_xlim[0], ax_ru_xlim[1])
ax_gu.set_xlabel('End-Station Screen Counts [a.u.]')

if _save:
    plt.savefig(image_path_gain, dpi=300)

plt.show(block=False)


# Plot combined
# -----------------------------------------------------------------
fig_all = plt.figure(figsize=(ScanImages.fig_size[0] * 1.5, ScanImages.fig_size[1]))
grid = plt.GridSpec(2, 2, hspace=0.3, wspace=0.3)

ax_rq = fig_all.add_subplot(grid[0, 0])
ax_ru = fig_all.add_subplot(grid[0, 1], sharey=ax_rq)

ax_gu = fig_all.add_subplot(grid[1, 1], sharex=ax_ru)
ax_gq = fig_all.add_subplot(grid[1, 0], sharex=ax_rq, sharey=ax_gu)

# -
ax_rq.plot(q_scope, rad2_mean_corrected_q, '.', alpha=0.3, markersize=12)
ax_rq.plot(np.insert(q_scope, 0, 0.), poly_rad2[0] * np.insert(q_scope, 0, 0.), c='orange', linewidth=1)
ax_rq.set_ylabel('Mean Counts [a.u.]')
ax_rq_xlim = ax_rq.get_xlim()

ax_ru.plot(u9_mean, rad2_mean_corrected_u, '.g', alpha=0.3, markersize=12)
ax_ru.plot(np.insert(u9_mean, 0, 0.), poly_u9[0] * np.insert(u9_mean, 0, 0.), c='orange', linewidth=1)
ax_ru_xlim = ax_ru.get_xlim()

# -
ax_gq.plot(q_scope, gain_q, '.', alpha=0.3, markersize=12, label=rf'max $\simeq$ {np.max(gain_q):.1f}')
ax_gq.axhline(1., c='orange', linestyle='--', linewidth=1)
# ax_gq.set_ylim(0, ax_gq.get_ylim()[1])
ax_gq.set_xlim(ax_rq_xlim[0], ax_rq_xlim[1])
ax_gq.set_xlabel('Charge [pC]')
ax_gq.set_ylabel('Gain')
ax_gq.legend(loc='best', prop={'size': 8})

ax_gu.plot(u9_mean, gain_u, '.g', alpha=0.3, markersize=12, label=rf'max $\simeq$ {np.max(gain_u):.1f}')
ax_gu.axhline(1., c='orange', linestyle='--', linewidth=1)
ax_gu.set_ylim(0, ax_gu.get_ylim()[1])
ax_gu.set_xlim(ax_ru_xlim[0], ax_ru_xlim[1])
ax_gu.set_xlabel('End-Station Screen Counts [a.u.]')
ax_gu.legend(loc='best', prop={'size': 8})

if _save:
    plt.savefig(image_path_rad_and_gain, dpi=300)

plt.show(block=True)


# Close
# -----------------------------------------------------------------
try:
    plt.close(fig_rc)
    plt.close(fig_gc)
    plt.close(fig_all)
except Exception:
    pass

print('done')
