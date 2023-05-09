import os
import numpy as np
from geecs_api.api_defs import SysPath
from geecs_api.tools.scans.scan import Scan
from geecs_api.tools.images.batch_analyses import average_images, analyze_images
from geecs_api.tools.images.spot_analysis import spot_analysis, fwhm
from geecs_api.tools.images.displays import show_one
import matplotlib.pyplot as plt


# no_scans = list of (scan path, scan number, camera name)
def screen_scan_analysis(no_scans: list[tuple[SysPath, int, str]], screen_labels: list[str], rotate_deg: int = 0):
    scans: list[Scan] = []
    for no_scan in no_scans:
        # create Scan object
        scan = Scan(folder=no_scan[0], match_exp=False)  # tmp: set to True
        scan.screen_label = screen_labels[0]
        scan.camera = no_scan[2]
        scans.append(scan)

        # average/analyze camera images
        images_folder: SysPath = os.path.join(scan.get_folder(), scan.camera)
        scan.avg_img = np.rot90(average_images(images_folder), int(rotate_deg / 90))
        avg_img_analysis = spot_analysis(scan.avg_img.astype('float64'))

        scan.analyses = analyze_images(images_folder, rotate_deg=rotate_deg)
        scan_pos_max = np.array([analysis['position_max'] for analysis in scan.analyses])

        box_factor: float = 2.5
        left = min(avg_img_analysis['opt_x_com'][2] - box_factor * fwhm(avg_img_analysis['opt_x_com'][3]),
                   avg_img_analysis['opt_x_max'][2] - box_factor * fwhm(avg_img_analysis['opt_x_max'][3]))
        right = max(avg_img_analysis['opt_x_com'][2] + box_factor * fwhm(avg_img_analysis['opt_x_com'][3]),
                    avg_img_analysis['opt_x_max'][2] + box_factor * fwhm(avg_img_analysis['opt_x_max'][3]))
        top = min(avg_img_analysis['opt_y_com'][2] - box_factor * fwhm(avg_img_analysis['opt_y_com'][3]),
                  avg_img_analysis['opt_y_max'][2] - box_factor * fwhm(avg_img_analysis['opt_y_max'][3]))
        bottom = max(avg_img_analysis['opt_y_com'][2] + box_factor * fwhm(avg_img_analysis['opt_y_com'][3]),
                     avg_img_analysis['opt_y_max'][2] + box_factor * fwhm(avg_img_analysis['opt_y_max'][3]))

        x_lim = (round(left / 10.) * 10, round(right / 10.) * 10)
        y_lim = (round(top / 10.) * 10, round(bottom / 10.) * 10)

        show_one(scan.avg_img, block_execution=False, size_factor=1., x_lim=x_lim, y_lim=y_lim, colormap='hot',
                 markers_ij=[avg_img_analysis['position_com'], avg_img_analysis['position_max']],
                 markers_color='c.', hide_ticks=False, show_colorbar=True,
                 show_contours=True, contours=5, contours_colors='k', contours_labels=False)

        show_one(scan.avg_img, block_execution=True, size_factor=1., x_lim=x_lim, y_lim=y_lim, colormap='hot',
                 markers_ij=[analysis['position_max'] for analysis in scan.analyses],
                 markers_color='c.', hide_ticks=False, show_colorbar=True,
                 show_contours=True, contours=5, contours_colors='k', contours_labels=False)

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
    _no_scan = (r'C:\Users\GuillaumePlateau\Documents\LBL\Data\Undulator\Y2023\04-Apr\23_0418\scans\Scan027',
                27, 'UC_ALineEbeam1')
    # _no_scan = (r'C:\Users\GuillaumePlateau\Documents\LBL\Data\Undulator\Y2023\04-Apr\23_0418\scans\Scan028',
    #             28, 'UC_ALineEBeam2')
    # _no_scan = (r'C:\Users\GuillaumePlateau\Documents\LBL\Data\Undulator\Y2023\04-Apr\23_0418\scans\Scan029',
    #             29, 'UC_ALineEBeam3')
    screen_scan_analysis([_no_scan], ['A3'], rotate_deg=90)
