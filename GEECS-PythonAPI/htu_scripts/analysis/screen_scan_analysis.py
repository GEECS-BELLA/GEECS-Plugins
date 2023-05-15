import os
import shelve
import numpy as np
import calendar as cal
from pathlib import Path
from typing import Optional
from geecs_api.api_defs import SysPath
from geecs_api.tools.scans.scan import Scan
from htu_scripts.analysis.undulator_no_scan import analyze_images, summarize_image_analyses
from geecs_api.tools.images.spot import spot_analysis, fwhm
from geecs_api.tools.images.displays import show_one
import matplotlib.pyplot as plt
from tkinter import filedialog


# no_scans = list of (scan path, scan number, camera name)
def screen_scan_analysis(no_scans: list[tuple[SysPath, str]], screen_labels: list[str],
                         rotate_deg: list[int], save_dir: Optional[SysPath] = None,
                         ignore_experiment_name: bool = False) \
        -> tuple[list[Scan], np.ndarray, np.ndarray, np.ndarray]:
    scans: list[Scan] = []
    target_deltas_mean: np.ndarray = np.zeros((len(no_scans), 2))  # Dx, Dy [pix]
    target_deltas_std: np.ndarray = np.zeros((len(no_scans), 2))
    target_deltas_avg_imgs: np.ndarray = np.zeros((len(no_scans), 2))

    # save analysis
    if not save_dir:
        save_dir = Path(filedialog.askdirectory(title='Save directory:'))
    elif not isinstance(save_dir, Path):
        save_dir = Path(save_dir)

    for it, (no_scan, rot_deg, label) in enumerate(zip(no_scans, rotate_deg, screen_labels)):
        # create Scan object
        scan = Scan(folder=no_scan[0], ignore_experiment_name=ignore_experiment_name)
        scan.screen_label = label
        scan.camera = no_scan[1]

        # average/analyze camera images
        images_folder: SysPath = os.path.join(scan.get_folder(), scan.camera)
        scan.analyses, scan.avg_img = analyze_images(images_folder, rotate_deg=rot_deg, screen_label=label,
                                                     hp_median=2, hp_threshold=3.,
                                                     denoise_cycles=0, gauss_filter=5., com_threshold=0.5)
        scan.analyses_summary = summarize_image_analyses(scan.analyses)
        scan.avg_img_analysis = spot_analysis(scan.avg_img)

        # distance [pix] from target
        scan, deltas_mean, deltas_std = target_analysis(scan)
        target_deltas_mean[it] = deltas_mean
        target_deltas_std[it] = deltas_std
        target_deltas_avg_imgs[it] = scan.target_analysis['avg_img_delta']

        # image plot
        plot_scan_image(scan, block_execution=False)
        if save_dir:
            image_path = Path(save_dir) / f'avg_img_{scan.camera}.png'
            plt.savefig(image_path, dpi=300)

        # store scan
        scans.append(scan)

    analysis_label: str = f'{screen_labels[0]}-{screen_labels[-1]}'
    if save_dir:
        with shelve.open(str(Path(save_dir) / f'analysis_{analysis_label}')) as shelve_file:
            shelve_file['scans'] = scans
            shelve_file['target_deltas_mean'] = target_deltas_mean
            shelve_file['target_deltas_std'] = target_deltas_std
            shelve_file['target_deltas_avg_imgs'] = target_deltas_avg_imgs

    # target plot
    x_axis = np.arange(1, target_deltas_mean.shape[0] + 1, dtype='int')
    plt.figure()
    plt.fill_between(x_axis,
                     target_deltas_mean[:, 0] - target_deltas_std[:, 0],
                     target_deltas_mean[:, 0] + target_deltas_std[:, 0], label='Dx [mm]', color='m')
    plt.plot(x_axis, target_deltas_avg_imgs[:, 0], 'ob-', label='Dx <img> [mm]')
    plt.legend(loc='best')
    plt.xlabel('Screen')
    plt.ylabel('Target Offsets [mm]')
    plt.xticks(x_axis, screen_labels)
    plt.show(block=False)
    if save_dir:
        save_path = Path(save_dir) / f'target_offsets_dx_{analysis_label}.png'
        plt.savefig(save_path, dpi=300)

    plt.figure()
    plt.fill_between(x_axis,
                     target_deltas_mean[:, 1] - target_deltas_std[:, 1],
                     target_deltas_mean[:, 1] + target_deltas_std[:, 1], label='Dy [mm]', color='m')
    plt.plot(x_axis, target_deltas_avg_imgs[:, 1], 'ob-', label='Dy <img> [mm]')
    plt.legend(loc='best')
    plt.xlabel('Screen')
    plt.ylabel('Target Offsets [mm]')
    plt.xticks(x_axis, screen_labels)
    plt.show(block=False)
    if save_dir:
        save_path = Path(save_dir) / f'target_offsets_dy_{screen_labels[0]}-{screen_labels[-1]}.png'
        plt.savefig(save_path, dpi=300)

    return scans, target_deltas_mean, target_deltas_std, target_deltas_avg_imgs


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
    ax, _ = show_one(scan.avg_img, size_factor=1., x_lim=x_lim, y_lim=y_lim, colormap='hot',
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
    # GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

    base_path = Path(r'C:\Users\GuillaumePlateau\Documents\LBL\Data')
    # base_path = Path(r'Z:\data')

    # _tags = [(2023, 5, 9, 12, 'UC_ALineEbeam1', 'A1', 90),
    #          (2023, 5, 9, 14, 'UC_ALineEBeam2', 'A2', 90),
    #          (2023, 5, 9, 15, 'UC_ALineEBeam3', 'A3', 0)]

    _tags = [(2023, 5, 9, 22, 'UC_VisaEBeam1', 'U1', 0)]

    _no_scans = [(base_path/'Undulator'/f'Y{tag[0]}'/f'{tag[1]:02d}-{cal.month_name[tag[1]][:3]}' /
                  f'{str(tag[0])[-2:]}_{tag[1]:02d}{tag[2]:02d}'/'scans'/f'Scan{tag[3]:03d}', tag[4])
                 for tag in _tags]
    _labels = [tag[5] for tag in _tags]
    _rotations = [tag[6] for tag in _tags]
    screen_scan_analysis(_no_scans, _labels, rotate_deg=_rotations, ignore_experiment_name=(str(base_path)[0] == 'C'))
    print('done')
