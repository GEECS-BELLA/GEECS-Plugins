import os
import numpy as np
import calendar as cal
from pathlib import Path
from typing import Optional, Union, Any
from geecs_api.api_defs import SysPath
from geecs_api.tools.scans.scan import Scan
from geecs_api.tools.interfaces.exports import load_py, save_py
from geecs_api.tools.images.spot import fwhm
from geecs_api.tools.images.displays import show_one
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.diagnostics.cameras.camera import Camera
from htu_scripts.analysis.undulator_no_scan import UndulatorNoScan
import matplotlib.pyplot as plt
from tkinter import filedialog
from progressbar import ProgressBar


def screen_scan_analysis(no_scans: dict[str, tuple[Union[Path, str], Union[Path, str]]], screen_labels: list[str],
                         save_dir: Optional[SysPath] = None) -> dict[str, Any]:
    """
    no_scans = dict of tuples (analysis file paths, scan data directory paths)
    """
    beam_analysis: dict[str, Any] = {}

    # save analysis
    if not save_dir:
        save_dir = filedialog.askdirectory(title='Save directory:')
        if save_dir:
            save_dir = Path(save_dir)
    elif not isinstance(save_dir, Path):
        save_dir = Path(save_dir)

    analysis_files: list[Path] = []
    scan_paths: list[Path] = []
    pos_short_names: list[str] = []
    pos_long_names: list[str] = []

    with ProgressBar(max_value=len(no_scans)) as pb:
        for it, (analysis_file, scan_path) in enumerate(no_scans.values()):
            if not analysis_file.is_file():
                continue

            analysis_files.append(analysis_file)
            scan_paths.append(scan_path)

            analysis = load_py(analysis_file, as_dict=True)

            # noinspection PyUnboundLocalVariable
            label = Camera.label_from_name(analysis['camera_name'])
            index = screen_labels.index(label)
            summary: dict[str, Any] = analysis['analyses_summary']
            targets: dict[str, Any] = summary['targets']

            if not pos_short_names:
                for im_analysis in analysis['image_analyses']:
                    if 'positions' in im_analysis:
                        pos_short_names = [pos[-1] for pos in im_analysis['positions']]
                        pos_long_names = [pos for pos in im_analysis['positions_labels']]
                        break

            for pos in pos_short_names:
                if f'{pos}_deltas_avg_imgs' not in beam_analysis:
                    tmp = np.zeros((len(screen_labels), 2))
                    tmp[:] = np.nan
                    beam_analysis[f'{pos}_deltas_avg_imgs'] = tmp.copy()  # Dx, Dy [mm]
                    beam_analysis[f'{pos}_deltas_means'] = tmp.copy()
                    beam_analysis[f'{pos}_deltas_stds'] = tmp.copy()
                    beam_analysis[f'{pos}_fwhm_means'] = tmp.copy()
                    beam_analysis[f'{pos}_fwhm_stds'] = tmp.copy()
                    beam_analysis['target_um_pix'] = np.ones((len(screen_labels),), dtype=float)

                if targets and (f'avg_img_{pos}_delta' in targets):
                    beam_analysis[f'{pos}_deltas_avg_imgs'][index, :] = targets[f'avg_img_{pos}_delta']
                    beam_analysis[f'{pos}_deltas_means'][index, :] = targets[f'target_deltas_{pos}_mean']
                    beam_analysis[f'{pos}_deltas_stds'][index, :] = targets[f'target_deltas_{pos}_std']
                    beam_analysis['target_um_pix'][index] = targets['target_um_pix']

                if summary and (f'mean_pos_{pos}_fwhm_x' in summary):
                    beam_analysis[f'{pos}_fwhm_means'][index, 1] = \
                        summary[f'mean_pos_{pos}_fwhm_x'] * beam_analysis['target_um_pix'][index]
                    beam_analysis[f'{pos}_fwhm_means'][index, 0] = \
                        summary[f'mean_pos_{pos}_fwhm_y'] * beam_analysis['target_um_pix'][index]
                    beam_analysis[f'{pos}_fwhm_stds'][index, 1] = \
                        summary[f'std_pos_{pos}_fwhm_x'] * beam_analysis['target_um_pix'][index]
                    beam_analysis[f'{pos}_fwhm_stds'][index, 0] = \
                        summary[f'std_pos_{pos}_fwhm_y'] * beam_analysis['target_um_pix'][index]

            pb.increment()

    data_dict: dict[str, Any] = {'screen_labels': screen_labels,
                                 'analysis_files': analysis_files,
                                 'scan_paths': scan_paths,
                                 'beam_analysis': beam_analysis}
    if save_dir:
        if not save_dir.is_dir():
            os.makedirs(save_dir)

        export_file_path: Path = save_dir / 'beam_analysis'
        save_py(file_path=export_file_path, data=data_dict)
        print(f'Data exported to:\n\t{export_file_path}.dat')

    x_axis = np.arange(1, len(screen_labels) + 1, dtype='int')

    fig, axs = plt.subplots(ncols=len(pos_short_names), nrows=4,
                            figsize=(UndulatorNoScan.fig_size[0] * 1.5, UndulatorNoScan.fig_size[1] * 1.5),
                            sharex='col', sharey='row')
    for it, pos in enumerate(pos_short_names):
        # Deltas X
        axs[0, it].fill_between(
            x_axis,
            1000 * (beam_analysis[f'{pos}_deltas_means'][:, 1] - beam_analysis[f'{pos}_deltas_stds'][:, 1]),
            1000 * (beam_analysis[f'{pos}_deltas_means'][:, 1] + beam_analysis[f'{pos}_deltas_stds'][:, 1]),
            label=r'$D_x \pm \sigma$', color='m', alpha=0.33)
        axs[0, it].plot(x_axis, 1000 * beam_analysis[f'{pos}_deltas_avg_imgs'][:, 1], 'ob-',
                        label=r'$D_x$ $(\mu_{image})$', linewidth=1, markersize=3)
        axs[0, it].legend(loc='best', prop={'size': 8})
        axs[0, it].set_xticks([])
        axs[0, it].set_title(pos_long_names[it])

        # Deltas Y
        axs[1, it].fill_between(
            x_axis,
            1000 * (beam_analysis[f'{pos}_deltas_means'][:, 0] - beam_analysis[f'{pos}_deltas_stds'][:, 0]),
            1000 * (beam_analysis[f'{pos}_deltas_means'][:, 0] + beam_analysis[f'{pos}_deltas_stds'][:, 0]),
            label=r'$D_y \pm \sigma$', color='m', alpha=0.33)
        axs[1, it].plot(x_axis, 1000 * beam_analysis[f'{pos}_deltas_avg_imgs'][:, 0], 'ob-',
                        label=r'$D_y$ $(\mu_{image})$', linewidth=1, markersize=3)
        axs[1, it].legend(loc='best', prop={'size': 8})
        axs[1, it].set_xticks([])

        # FWHM X
        axs[2, it].fill_between(
            x_axis,
            beam_analysis[f'{pos}_fwhm_means'][:, 1] - beam_analysis[f'{pos}_fwhm_stds'][:, 1],
            beam_analysis[f'{pos}_fwhm_means'][:, 1] + beam_analysis[f'{pos}_fwhm_stds'][:, 1],
            label=r'$FWHM_x \pm \sigma$', color='y', alpha=0.33)
        axs[2, it].plot(x_axis, beam_analysis[f'{pos}_fwhm_means'][:, 1], 'og-',
                        label=r'$FWHM_x$ $(\mu_{image})$', linewidth=1, markersize=3)
        axs[2, it].legend(loc='best', prop={'size': 8})
        axs[2, it].set_xticks([])

        # FWHM Y
        axs[3, it].fill_between(
            x_axis,
            beam_analysis[f'{pos}_fwhm_means'][:, 0] - beam_analysis[f'{pos}_fwhm_stds'][:, 0],
            beam_analysis[f'{pos}_fwhm_means'][:, 0] + beam_analysis[f'{pos}_fwhm_stds'][:, 0],
            label=r'$FWHM_y \pm \sigma$ [$\mu$m]', color='y', alpha=0.33)
        axs[3, it].plot(x_axis, beam_analysis[f'{pos}_fwhm_means'][:, 0], 'og-',
                        label=r'$FWHM_y$ $(\mu_{image})$', linewidth=1, markersize=3)
        axs[3, it].legend(loc='best', prop={'size': 8})
        axs[3, it].set_xlabel('Screen')
        axs[3, it].set_xticks(x_axis, screen_labels)

    axs[0, 0].set_ylabel(r'X-Offsets [$\mu$m]')
    axs[1, 0].set_ylabel(r'Y-Offsets [$\mu$m]')
    axs[2, 0].set_ylabel(r'X-FWHM [$\mu$m]')
    axs[3, 0].set_ylabel(r'Y-FWHM [$\mu$m]')

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
    return data_dict


def plot_scan_image(scan: Scan, block_execution: bool = False):
    # display limits
    box_factor: float = 2.
    # [func(*pars) for func, pars in [(min, p_list) for p_list in ['opt_x_com', 'opt_x_max']]]
    left = min(scan.avg_img_analysis['opt_x_com'][2] - box_factor * fwhm(scan.avg_img_analysis['opt_x_com'][3]),
               scan.avg_img_analysis['opt_x_max'][2] - box_factor * fwhm(scan.avg_img_analysis['opt_x_max'][3]))
    right = max(scan.avg_img_analysis['opt_x_com'][2] + box_factor * fwhm(scan.avg_img_analysis['opt_x_com'][3]),
                scan.avg_img_analysis['opt_x_max'][2] + box_factor * fwhm(scan.avg_img_analysis['opt_x_max'][3]))
    top = min(scan.avg_img_analysis['opt_y_com'][2] - box_factor * fwhm(scan.avg_img_analysis['opt_y_com'][3]),
              scan.avg_img_analysis['opt_y_max'][2] - box_factor * fwhm(scan.avg_img_analysis['opt_y_max'][3]))
    bottom = max(scan.avg_img_analysis['opt_y_com'][2] + box_factor * fwhm(scan.avg_img_analysis['opt_y_com'][3]),
                 scan.avg_img_analysis['opt_y_max'][2] + box_factor * fwhm(scan.avg_img_analysis['opt_y_max'][3]))

    x_lim = (round(left / 10.) * 10, round(right / 10.) * 10)
    y_lim = (round(top / 10.) * 10, round(bottom / 10.) * 10)

    # centers distributions
    avg_img_pos_max = scan.avg_img_analysis['position_max']

    x_hist = scan.analyses_summary['scan_pos_max'][:, 1]
    y_hist = scan.analyses_summary['scan_pos_max'][:, 0]
    n_bins = int(round(len(x_hist) / 5.))

    x_counts, x_bins = np.histogram(x_hist, bins=n_bins)
    y_counts, y_bins = np.histogram(y_hist, bins=n_bins)

    x_pos = (x_bins[1:] + x_bins[:-1]) / 2.
    x_step = np.mean(x_pos[1:] - x_pos[:-1])
    x_pos = np.insert(np.append(x_pos, x_pos[-1] + x_step), 0, x_pos[0] - x_step)
    x_counts = np.insert(np.append(x_counts, 0.), 0, 0.)
    x_counts_plot = x_counts / max(x_counts) * ((y_lim[1] - y_lim[0]) / 6) + y_lim[0]

    y_pos = (y_bins[1:] + y_bins[:-1]) / 2.
    y_step = np.mean(y_pos[1:] - y_pos[:-1])
    y_pos = np.insert(np.append(y_pos, y_pos[-1] + y_step), 0, y_pos[0] - y_step)
    y_counts = np.insert(np.append(y_counts, 0.), 0, 0.)
    y_counts_plot = y_counts / max(y_counts) * ((x_lim[1] - x_lim[0]) / 6) + x_lim[0]

    # plot
    plt.ion()
    ax, _ = show_one(scan.avg_img, x_lim=x_lim, y_lim=y_lim, colormap='hot',
                     markers_ij=scan.analyses_summary['scan_pos_max'],
                     markers_color='c.', hide_ticks=False, show_colorbar=True,
                     show_contours=True, contours=2, contours_colors='w', contours_labels=True,
                     show=False, block_execution=False)
    ax.scatter(avg_img_pos_max[1], avg_img_pos_max[0], c='k')
    ax.plot(x_pos, x_counts_plot, '-c')
    ax.plot(y_counts_plot, y_pos, '-c')
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    plt.show(block=block_execution)


if __name__ == '__main__':
    # base_path = Path(r'C:\Users\GuillaumePlateau\Documents\LBL\Data')
    base_path = Path(r'Z:\data')

    is_local = (str(base_path)[0] == 'C')
    if not is_local:
        GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

    _base_tag = (2023, 5, 9, 0)
    _scans_screens = [(22, 'U1'),
                      (23, 'U2'),
                      (24, 'U3'),
                      (25, 'U4'),
                      (26, 'U5'),
                      (27, 'U6'),
                      (28, 'U7')]

    _analysis_files: list[Path] = \
        [base_path/'Undulator'/f'Y{_base_tag[0]}'/f'{_base_tag[1]:02d}-{cal.month_name[_base_tag[1]][:3]}' /
         f'{str(_base_tag[0])[-2:]}_{_base_tag[1]:02d}{_base_tag[2]:02d}'/'analysis' /
         f'Scan{number[0]:03d}'/f'UC_VisaEBeam{number[1][-1]}'/'profiles_analysis.dat'
         for number in _scans_screens]
    _scan_paths = [file.parents[3]/'scans'/file.parts[-3] for file in _analysis_files]
    _no_scans: dict[str, tuple[Path, Path]] = \
        {key[1]: (analysis, data) for key, analysis, data in zip(_scans_screens, _analysis_files, _scan_paths)}

    _save_dir = _analysis_files[0].parents[2] / \
        f'{_scan_paths[0].name}_Screens_{_scans_screens[0][1]}_{_scans_screens[-1][1]}'
    _labels = [label[1] for label in _scans_screens]

    _data_dict = screen_scan_analysis(no_scans=_no_scans, screen_labels=_labels, save_dir=_save_dir)

    print('done')
