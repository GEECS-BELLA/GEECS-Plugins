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


# no_scans = dict of tuples (analysis file paths, scan paths)
def screen_scan_analysis(no_scans: dict[str, tuple[Union[Path, str], Union[Path, str]]], screen_labels: list[str],
                         save_dir: Optional[SysPath] = None):
    # target_deltas_avg_imgs: np.ndarray = np.zeros((len(screen_labels), 2))
    # target_deltas_avg_imgs[:] = np.nan
    # target_deltas_mean: np.ndarray = np.zeros((len(screen_labels), 2))  # Dx, Dy [mm]
    # target_deltas_mean[:] = np.nan
    # target_deltas_std: np.ndarray = np.zeros((len(screen_labels), 2))
    # target_deltas_std[:] = np.nan
    #
    # beam_fwhm_mean: np.ndarray = np.zeros((len(screen_labels), 2))  # Dx, Dy [mm]
    # beam_fwhm_mean[:] = np.nan
    # beam_fwhm_std: np.ndarray = np.zeros((len(screen_labels), 2))
    # beam_fwhm_std[:] = np.nan

    beam_analysis: dict[str, Any] = {}

    # save analysis
    if not save_dir:
        save_dir = Path(filedialog.askdirectory(title='Save directory:'))
    elif not isinstance(save_dir, Path):
        save_dir = Path(save_dir)

    analysis_files: list[Path] = []
    scan_paths: list[Path] = []
    pos_labels: list[str] = []

    for it, (analysis_file, scan_path) in enumerate(no_scans):
        if not analysis_file.is_file():
            continue

        analysis_files.append(analysis_file)
        scan_paths.append(scan_path)

        analysis = load_py(analysis_file, as_dict=True)

        # noinspection PyUnboundLocalVariable
        label = Camera.label_from_name(analysis['camera_name'])
        index = screen_labels.index(label)
        targets: dict[str, Any] = analysis['analyses_summary']['targets']

        if not pos_labels:
            for im_analysis in analysis['image_analyses']:
                if 'positions' in im_analysis:
                    positions = im_analysis['positions']
                    pos_labels = [pos[-1] for pos in positions]
                    break

        for pos in pos_labels:
            if f'{pos}_deltas_avg_imgs' not in beam_analysis:
                tmp = np.zeros((len(screen_labels), 2))
                tmp[:] = np.nan
                beam_analysis[f'{pos}_deltas_avg_imgs'] = tmp  # Dx, Dy [mm]
                beam_analysis[f'{pos}_deltas_means'] = tmp
                beam_analysis[f'{pos}_deltas_stds'] = tmp

            if targets and (f'avg_img_{pos}_delta' in targets):
                beam_analysis[f'{pos}_deltas_avg_imgs'][index, :] = targets[f'avg_img_{pos}_delta']
                beam_analysis[f'{pos}_deltas_means'] = targets[f'target_deltas_{pos}_mean']
                beam_analysis[f'{pos}_deltas_stds'] = targets[f'target_deltas_{pos}_std']

    if save_dir:
        export_file_path: Path = save_dir / 'beam_analysis'
        save_py(file_path=export_file_path,
                data={'screen_labels': screen_labels,
                      'analysis_files': analysis_files,
                      'scan_paths': scan_paths,
                      'beam_analysis': beam_analysis})
        print(f'Data exported to:\n\t{export_file_path}.dat')

    x_axis = np.arange(1, len(screen_labels) + 1, dtype='int')

    fig, axs = plt.subplots(ncols=len(pos_labels), nrows=2,
                            figsize=(UndulatorNoScan.fig_size[0] * 1.5, UndulatorNoScan.fig_size[1]),
                            sharex='all', sharey='all')
    for it, pos in enumerate(pos_labels):
        axs[0, it].fill_between(x_axis,
                                beam_analysis[f'{pos}_deltas_means'][:, 1] - beam_analysis[f'{pos}_deltas_stds'][:, 1],
                                beam_analysis[f'{pos}_deltas_means'][:, 1] + beam_analysis[f'{pos}_deltas_stds'][:, 1],
                                label='Dx [mm]', color='m')
        axs[0, it].plot(x_axis, beam_analysis[f'{pos}_deltas_avg_imgs'][:, 1], 'ob-', label='Dx <img> [mm]')
        axs[0, it].legend(loc='best', prop={'size': 8})
        axs[0, it].xticks([])

        axs[1, it].fill_between(x_axis,
                                beam_analysis[f'{pos}_deltas_means'][:, 0] - beam_analysis[f'{pos}_deltas_stds'][:, 0],
                                beam_analysis[f'{pos}_deltas_means'][:, 0] + beam_analysis[f'{pos}_deltas_stds'][:, 0],
                                label='Dy [mm]', color='m')
        axs[1, it].plot(x_axis, beam_analysis[f'{pos}_deltas_avg_imgs'][:, 0], 'ob-', label='Dy <img> [mm]')
        axs[1, it].legend(loc='best', prop={'size': 8})
        axs[1, it].xlabel('Screen')
        axs[1, it].xticks(x_axis, screen_labels)

    axs[0, 0].set_ylabel('X-Offsets [mm]')
    axs[1, 0].set_ylabel('Y-Offsets [mm]')

    if save_dir:
        save_path = save_dir / 'beam_analysis.png'
        plt.savefig(save_path, dpi=300)

    plt.show(block=True)


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

    # _tags = [(2023, 5, 9, 12, 'UC_ALineEbeam1', 'A1', 90),
    #          (2023, 5, 9, 14, 'UC_ALineEBeam2', 'A2', 90),
    #          (2023, 5, 9, 15, 'UC_ALineEBeam3', 'A3', 0)]

    _tags = [(2023, 5, 9, 22, 'UC_VisaEBeam1', 'U1', 0)]

    _no_scans = [(base_path/'Undulator'/f'Y{tag[0]}'/f'{tag[1]:02d}-{cal.month_name[tag[1]][:3]}' /
                  f'{str(tag[0])[-2:]}_{tag[1]:02d}{tag[2]:02d}'/'scans'/f'Scan{tag[3]:03d}', tag[4])
                 for tag in _tags]
    _labels = [tag[5] for tag in _tags]
    _rotations = [tag[6] for tag in _tags]
    # screen_scan_analysis(_no_scans, _labels, ignore_experiment_name=is_local)
    print('done')
