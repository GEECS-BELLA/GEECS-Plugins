import os
from geecs_api.api_defs import SysPath
from geecs_api.tools.scans.scan import Scan
from geecs_api.tools.images.batch_analyses import average_images
from geecs_api.tools.images.spot_analysis import spot_analysis, fwhm
from geecs_api.tools.images.displays import show_one
import matplotlib.pyplot as plt


# no_scans = list of (scan path, scan number, camera name)
def screen_scan_analysis(no_scans: list[tuple[SysPath, int, str]], screen_labels: list[str]):
    scans: list[Scan] = []
    for no_scan in no_scans:
        # create Scan object
        scan = Scan(folder=no_scan[0], match_exp=False)  # tmp: set to True
        scan.screen_label = screen_labels[0]
        scan.camera = no_scan[2]
        scans.append(scan)

        # average camera images
        images_folder: SysPath = os.path.join(scan.get_folder(), scan.camera)
        scan.avg_img = average_images(images_folder)
        scan.spot_analysis = spot_analysis(scan.avg_img.astype('float64'),
                                           mad_multiplier=0., filter_size=5, threshold=0.9)

        box_factor: float = 2.5
        x_lim = (round((scan.spot_analysis['opt_x'][2] - box_factor * fwhm(scan.spot_analysis['opt_x'][3])) / 10) * 10,
                 round((scan.spot_analysis['opt_x'][2] + box_factor * fwhm(scan.spot_analysis['opt_x'][3])) / 10) * 10)
        y_lim = (round((scan.spot_analysis['opt_y'][2] - box_factor * fwhm(scan.spot_analysis['opt_y'][3])) / 10) * 10,
                 round((scan.spot_analysis['opt_y'][2] + box_factor * fwhm(scan.spot_analysis['opt_y'][3])) / 10) * 10)

        show_one(scan.avg_img, block_execution=True, size_factor=1.5, x_lim=x_lim, y_lim=y_lim, colormap='hot',
                 centroid_ij=(scan.spot_analysis['position'][0], scan.spot_analysis['position'][1]),
                 centroid_color='m.', hide_ticks=False, show_colorbar=True,
                 show_contours=True, contours=6, contours_colors='k', contours_labels=False)

        # plt.figure()
        # plt.plot(scan.spot_analysis['axis_x'], scan.spot_analysis['data_x'], 'b-', label='data')
        # plt.plot(scan.spot_analysis['axis_x'], scan.spot_analysis['fit_x'], 'm-', label='fit')
        # plt.show(block=True)
        #
        # plt.figure()
        # plt.plot(scan.spot_analysis['axis_y'], scan.spot_analysis['data_y'], 'b-', label='data')
        # plt.plot(scan.spot_analysis['axis_y'], scan.spot_analysis['fit_y'], 'm-', label='fit')
        # plt.show(block=True)


if __name__ == '__main__':
    # _no_scan = (r'C:\Users\GuillaumePlateau\Documents\LBL\Data\Undulator\Y2023\04-Apr\23_0418\scans\Scan027',
    #             27, 'UC_ALineEbeam1')
    _no_scan = (r'C:\Users\GuillaumePlateau\Documents\LBL\Data\Undulator\Y2023\04-Apr\23_0418\scans\Scan028',
                28, 'UC_ALineEBeam2')
    # _no_scan = (r'C:\Users\GuillaumePlateau\Documents\LBL\Data\Undulator\Y2023\04-Apr\23_0418\scans\Scan029',
    #             29, 'UC_ALineEBeam3')
    screen_scan_analysis([_no_scan], ['A3'])
