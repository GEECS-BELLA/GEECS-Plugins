import os
import numpy as np
from geecs_api.api_defs import SysPath
from geecs_api.tools.scans.scan import Scan
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.tools.images.batch_analyses import average_images, analyze_images, summarize_image_analyses
from geecs_api.tools.images.spot_analysis import spot_analysis, fwhm
from geecs_api.tools.images.displays import show_one
import matplotlib.pyplot as plt


# no_scans = list of (scan path, scan number, camera name)
def screen_scan_analysis(no_scans: list[tuple[SysPath, str]], screen_labels: list[str], rotate_deg: list[int]):
    scans: list[Scan] = []
    for no_scan, rot_deg in zip(no_scans, rotate_deg):
        # create Scan object
        scan = Scan(folder=no_scan[0], match_exp=True)
        scan.screen_label = screen_labels[0]
        scan.camera = no_scan[1]

        # average/analyze camera images
        images_folder: SysPath = os.path.join(scan.get_folder(), scan.camera)
        scan.avg_img = np.rot90(average_images(images_folder), int(rot_deg / 90))
        scan.avg_img_analysis = spot_analysis(scan.avg_img.astype('float64'))
        scan.analyses = analyze_images(images_folder, rotate_deg=rot_deg, hp_median=2, hp_threshold=3.,
                                       denoise_cycles=0, gauss_filter=5., com_threshold=0.5)
        scan.analyses_summary = summarize_image_analyses(scan.analyses)

        # distance [pix] from target
        try:
            scan.target = (float(GeecsDevice.exp_info['devices'][scan.camera]['Target.X']['defaultvalue']),
                           float(GeecsDevice.exp_info['devices'][scan.camera]['Target.Y']['defaultvalue']))
        except Exception:
            scan.target = (0., 0.)
        scan.target_delta_pix = np.array(scan.avg_img_analysis['position_max']) - scan.target

        # plot
        plot_scan_image(scan, block_execution=False)

        # store scan
        scans.append(scan)


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
    print(f'y_lim: {y_lim}')

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

    # save
    image_path = scan.get_analysis_folder()/f'avg_img_{scan.camera}.png'
    plt.savefig(image_path, dpi=300)


if __name__ == '__main__':
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

    # base_path = r'C:\Users\GuillaumePlateau\Documents\LBL\Data'
    base_path = r'Z:\data'
    _no_scan_27 = (base_path + r'\Undulator\Y2023\04-Apr\23_0418\scans\Scan027', 'UC_ALineEbeam1')
    _no_scan_28 = (base_path + r'\Undulator\Y2023\04-Apr\23_0418\scans\Scan028', 'UC_ALineEBeam2')
    _no_scan_29 = (base_path + r'\Undulator\Y2023\04-Apr\23_0418\scans\Scan029', 'UC_ALineEBeam3')
    screen_scan_analysis([_no_scan_27, _no_scan_28, _no_scan_29], ['A1', 'A2', 'A3'], rotate_deg=[90, 90, 0])
    print('done')
