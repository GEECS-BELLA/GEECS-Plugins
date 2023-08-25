import os
import time
import inspect
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from typing import Union, NamedTuple, Any, Optional
from geecs_python_api.controls.interface import api_error
from geecs_python_api.controls.api_defs import ScanTag
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.controls.devices.HTU.diagnostics.cameras import Camera
from geecs_python_api.tools.distributions.binning import unsupervised_binning, BinningResults
from geecs_python_api.analysis.images.scans.scan_images import ScanImages
from geecs_python_api.analysis.images.scans.scan_data import ScanData
from geecs_python_api.tools.images.batches import average_images
from geecs_python_api.tools.images.filtering import FiltersParameters
from geecs_python_api.tools.interfaces.exports import load_py, save_py
from geecs_python_api.tools.interfaces.prompts import text_input
from geecs_python_api.tools.distributions.fit_utility import fit_distribution
from htu_scripts.analysis.beam_analyses_collector import add_beam_analysis


def steering_scan_analysis(dat_paths: list[Path]):



    # export to .dat
    data_dict: dict[str, Any] = {
        'indexes': indexes,
        'setpoints': setpoints,
        'analysis_files': analysis_files,
        'beam_analysis': beam_analysis,
        'device_name': device_name,
        'scan_folder': scan_images.scan_obj.get_folder(),
        'camera_name': scan_images.camera_name,
        'pos_short_names': pos_short_names,
        'pos_long_names': pos_long_names}
    export_file_path = scan_data.get_analysis_folder() / f'steering_analysis_{device_name}'
    save_py(file_path=export_file_path, data=data_dict, as_bulk=False)
    print(f'Data exported to:\n\t{export_file_path}.dat')

    return export_file_path, data_dict


def render_steering_scan_analysis(data_dict: dict[str, Any]):
    x_axis: np.ndarray = data_dict['setpoints']
    beam_analysis: dict[str, Any] = data_dict['beam_analysis']

    fig, axs = plt.subplots(ncols=len(data_dict['pos_short_names']), nrows=2,
                            figsize=(ScanImages.fig_size[0] * 1.5, ScanImages.fig_size[1] * 1.5),
                            sharex='col', sharey='row')
    for it, pos in enumerate(data_dict['pos_short_names']):
        # X(dx), Y(dx)
        axs[0, it].plot(data_dict['setpoints'], data_dict['beam_analysis']['max_mean_pos_pix'][:, 0],
                        '.k', markersize=10)
        opt = data_dict['beam_analysis']['x_fit']['opt']
        opt_sign = '-' if opt[1] < 0 else ''
        axs[0, it].plot(data_dict['setpoints'], data_dict['beam_analysis']['x_fit']['fit'], 'gray',
                        label=rf"$X \simeq {opt[0]} \cdot dx {opt_sign}{abs(opt[1])} \sigma$")

        axs[0, it].plot(data_dict['setpoints'], data_dict['beam_analysis']['max_mean_pos_pix'][:, 0],
                        '.k', markersize=10)
        opt = data_dict['beam_analysis']['x_fit']['opt']
        opt_sign = '-' if opt[1] < 0 else ''
        axs[0, it].plot(data_dict['setpoints'], data_dict['beam_analysis']['x_fit']['fit'], 'gray',
                        label=rf"$X \simeq {opt[0]} \cdot dx {opt_sign}{abs(opt[1])} \sigma$")

        axs[0, it].legend(loc='best', prop={'size': 8})
        axs[0, it].set_xticks([])
        axs[0, it].set_title(pos_long_names[it])

        # X(dy), Y(dy)
        axs[1, it].fill_between(
            x_axis,
            f_deltas * (beam_analysis[f'{pos}_deltas_means'][:, 0] - beam_analysis[f'{pos}_deltas_stds'][:, 0]),
            f_deltas * (beam_analysis[f'{pos}_deltas_means'][:, 0] + beam_analysis[f'{pos}_deltas_stds'][:, 0]),
            label=r'$D_y \pm \sigma$', color='m', alpha=0.33)
        axs[1, it].plot(x_axis, f_deltas * beam_analysis[f'{pos}_deltas_avg_imgs'][:, 0], 'ob-',
                        label=r'$D_y$ $(\mu_{image})$', linewidth=1, markersize=3)
        axs[1, it].legend(loc='best', prop={'size': 8})
        axs[1, it].set_xticks([])

        # FWHM X
        axs[2, it].fill_between(
            x_axis,
            f_fwhms * beam_analysis[f'{pos}_fwhm_means'][:, 1] - beam_analysis[f'{pos}_fwhm_stds'][:, 1],
            f_fwhms * beam_analysis[f'{pos}_fwhm_means'][:, 1] + beam_analysis[f'{pos}_fwhm_stds'][:, 1],
            label=r'$FWHM_x \pm \sigma$', color='y', alpha=0.33)
        axs[2, it].plot(x_axis, f_fwhms * beam_analysis[f'{pos}_fwhm_means'][:, 1], 'og-',
                        label=r'$FWHM_x$ $(\mu_{image})$', linewidth=1, markersize=3)
        axs[2, it].legend(loc='best', prop={'size': 8})
        axs[2, it].set_xticks([])

        # FWHM Y
        axs[3, it].fill_between(
            x_axis,
            f_fwhms * beam_analysis[f'{pos}_fwhm_means'][:, 0] - beam_analysis[f'{pos}_fwhm_stds'][:, 0],
            f_fwhms * beam_analysis[f'{pos}_fwhm_means'][:, 0] + beam_analysis[f'{pos}_fwhm_stds'][:, 0],
            label=r'$FWHM_y \pm \sigma$ [$\mu$m]', color='y', alpha=0.33)
        axs[3, it].plot(x_axis, f_fwhms * beam_analysis[f'{pos}_fwhm_means'][:, 0], 'og-',
                        label=r'$FWHM_y$ $(\mu_{image})$', linewidth=1, markersize=3)
        axs[3, it].legend(loc='best', prop={'size': 8})
        axs[3, it].set_xlabel('Screen')
        axs[3, it].set_xticks(x_axis, screen_labels)

    axs[0, 0].set_ylabel(f'X-Offsets [{units_deltas}]')
    axs[1, 0].set_ylabel(f'Y-Offsets [{units_deltas}]')
    axs[2, 0].set_ylabel(f'X-FWHM [{units_fwhms}]')
    axs[3, 0].set_ylabel(f'Y-FWHM [{units_fwhms}]')

    # set matching vertical limits for deltas/FWHMs
    y_lim = (min(axs[0, 0].get_ylim()[0], axs[1, 0].get_ylim()[0]),
             max(axs[0, 0].get_ylim()[1], axs[1, 0].get_ylim()[1]))
    [axs[0, j].set_ylim(y_lim) for j in range(len(pos_short_names))]
    [axs[1, j].set_ylim(y_lim) for j in range(len(pos_short_names))]

    y_lim = (min(axs[2, 0].get_ylim()[0], axs[3, 0].get_ylim()[0]),
             max(axs[2, 0].get_ylim()[1], axs[3, 0].get_ylim()[1]))
    [axs[2, j].set_ylim(y_lim) for j in range(len(pos_short_names))]
    [axs[3, j].set_ylim(y_lim) for j in range(len(pos_short_names))]

    if save_dir:
        save_path = save_dir / 'beam_analysis.png'
        plt.savefig(save_path, dpi=300)

    plt.show(block=True)


if __name__ == '__main__':
    # initialization
    # --------------------------------------------------------------------------
    _base_path, is_local = htu.initialize()
    _base_tag = ScanTag(2023, 8, 3, 19)
    _bkg_tag = ScanTag(2023, 8, 3, 18)

    _device = 'U_S1H'
    _camera = 'P1'

    _folder = ScanData.build_folder_path(_base_tag, _base_path)
    _bkg_folder = ScanData.build_folder_path(_bkg_tag, _base_path)

    _scan_data = ScanData(_folder, ignore_experiment_name=is_local)
    _scan_images = ScanImages(_scan_data, _camera)

    _filters = FiltersParameters(contrast=1., hp_median=2, hp_threshold=3., denoise_cycles=0, gauss_filter=6.,
                                 com_threshold=0.9, bkg_image=None, box=True, ellipse=False)

    # background
    # --------------------------------------------------------------------------
    avg_image, _ = average_images(_bkg_folder)
    _filters.bkg_image = avg_image

    # scan analysis
    # --------------------------------------------------------------------------
    _quad_analysis = steering_scan_analysis(_scan_data, _scan_images, _quad, fwhms_metric='median', quad_2_screen=_quad_2_screen)

    print('done')
