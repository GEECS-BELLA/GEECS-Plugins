import os
import time
import numpy as np
from numpy.polynomial.polynomial import polyfit
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Any, Optional
from geecs_python_api.controls.api_defs import ScanTag
from geecs_python_api.controls.experiment.htu import HtuExp
from geecs_python_api.analysis.images.scans.scan_data import ScanData
from geecs_python_api.tools.interfaces.prompts import text_input
from geecs_python_api.analysis.images.scans.scan_images import ScanImages
from geecs_python_api.analysis.images.scans.scan_analysis import ScanAnalysis
from geecs_python_api.tools.images.filtering import FiltersParameters
from geecs_python_api.tools.interfaces.exports import save_py
from geecs_python_api.tools.images.displays import polyfit_label
# from geecs_python_api.controls.devices.HTU.transport.magnets import Quads
from geecs_python_api.tools.images.spot import fwhm_to_std


class QuadAnalysis(ScanAnalysis):
    def __init__(self, scan_data: ScanData, scan_images: ScanImages,
                 quad: int, fwhms_metric: str = 'median', quad_2_screen: float = 1.):
        super().__init__(scan_data, scan_images, 'U_EMQTripletBipolar')
        self.quad_number: int = quad
        self.quad_variable: str = f'Current_Limit.Ch{quad}'

        self.fwhms_metric: str = fwhms_metric
        self.quad_2_screen: float = quad_2_screen

    def analyze(self, variable: Optional[str] = None, initial_filtering=FiltersParameters(), ask_rerun: bool = True,
                blind_loads: bool = False, store_images: bool = True, store_scalars: bool = True,
                save_plots: bool = False, save: bool = False) -> Optional[Path]:
        if not variable:
            variable = self.quad_variable

        super().analyze(variable, initial_filtering, ask_rerun, blind_loads,
                        store_images, store_scalars, save_plots, save)

        figs = super().render(physical_units=True, x_label='Current [A]',
                              show_xy=True, show_fwhms=True, show_deltas=True,
                              xy_metric='median', fwhms_metric=self.fwhms_metric, deltas_metric='median',
                              xy_fit=1, fwhms_fit=2, deltas_fit=2,
                              show_figs=True, save_dir=self.scan_data.get_analysis_folder(), sync=False)

        setpoints: np.ndarray = self.data_dict['setpoints']
        range_x = np.array([np.min(setpoints), np.max(setpoints)])
        range_y = np.array([np.min(setpoints), np.max(setpoints)])

        while True:
            try:
                lim_str = text_input(f'Lower current limit to consider for FWHM-X, e.g. "none" or -1.5 : ')
                lim_low = np.min(setpoints) if lim_str.lower() == 'none' else float(lim_str)
                lim_str = text_input(f'Upper current limit to consider for FWHM-X, e.g. "none" or 3.5 : ')
                lim_high = np.max(setpoints) if lim_str.lower() == 'none' else float(lim_str)
                range_x = np.array([lim_low, lim_high])

                lim_str = text_input(f'Lower current limit to consider for FWHM-Y, e.g. "none" or -1.5 : ')
                lim_low = np.min(setpoints) if lim_str.lower() == 'none' else float(lim_str)
                lim_str = text_input(f'Upper current limit to consider for FWHM-Y, e.g. "none" or 3.5 : ')
                lim_high = np.max(setpoints) if lim_str.lower() == 'none' else float(lim_str)
                range_y = np.array([lim_low, lim_high])

                break
            except Exception:
                continue
            finally:
                try:
                    for fig_ax in figs:
                        plt.close(fig_ax[0])
                except Exception:
                    pass

        selected_x: np.ndarray = (setpoints >= np.min(range_x)) * (setpoints <= np.max(range_x))
        scan_x = setpoints[selected_x]

        selected_y: np.ndarray = (setpoints >= np.min(range_y)) * (setpoints <= np.max(range_y))
        scan_y = setpoints[selected_y]

        sample_analysis = self.data_dict['analyses'][0]
        um_per_pix: float = sample_analysis['summary']['um_per_pix']
        positions: {} = sample_analysis['image_analyses'][0]['positions']
        twiss_analysis: dict[str, Any] = {}

        for pos in positions['short_names']:
            twiss_analysis[pos] = {}
            data_val, data_err_low, data_err_high = ScanAnalysis.fetch_metrics(self.data_dict['analyses'],
                                                                               self.fwhms_metric, 'fwhms',
                                                                               pos, 'pix_ij', um_per_pix)
            twiss_analysis[pos]['fwhms'] = (data_val, data_err_low, data_err_high)

            # noinspection PyTypeChecker
            sig_x2 = fwhm_to_std(data_val[selected_x, 1] * 1e-6)**2
            fit_x_pars: np.ndarray = np.flip(polyfit(scan_x, sig_x2, 2))
            twiss_analysis[pos]['epsilon_x'], twiss_analysis[pos]['alpha_x'], twiss_analysis[pos]['beta_x'] = \
                QuadAnalysis.twiss_parameters(fit_x_pars, self.quad_2_screen)

            # noinspection PyTypeChecker
            sig_y2 = fwhm_to_std(data_val[selected_y, 0] * 1e-6)**2
            fit_y_pars: np.ndarray = np.flip(polyfit(scan_y, sig_y2, 2))
            twiss_analysis[pos]['epsilon_y'], twiss_analysis[pos]['alpha_y'], twiss_analysis[pos]['beta_y'] = \
                QuadAnalysis.twiss_parameters(fit_y_pars, self.quad_2_screen)

            twiss_analysis[pos]['sigma2_x'] = sig_x2
            twiss_analysis[pos]['sigma2_y'] = sig_y2
            twiss_analysis[pos]['fit_pars'] = np.stack([fit_y_pars, fit_x_pars]).transpose()

        twiss_analysis['quad_2_screen'] = self.quad_2_screen
        twiss_analysis['indexes_selected'] = np.stack([selected_y, selected_x]).transpose()
        twiss_analysis['setpoints_selected'] = np.stack([range_y, range_x]).transpose()
        self.data_dict['twiss'] = twiss_analysis

        if save:
            export_file_path = self.scan_data.get_analysis_folder() / f'quad_scan_analysis_{self.device_name}'
            save_py(file_path=export_file_path, data=self.data_dict)
            print(f'Data exported to:\n\t{export_file_path}.dat')
        else:
            export_file_path = None

        return export_file_path

    def render_twiss(self, physical_units: bool = True, save_dir: Optional[Path] = None):
        if 'twiss' not in self.data_dict:
            return []

        twiss = self.data_dict['twiss']

        units_factor: float
        units_label: str

        sample_analysis = self.data_dict['analyses'][0]
        um_per_pix: float = sample_analysis['summary']['um_per_pix']
        positions: {} = sample_analysis['image_analyses'][0]['positions']

        plot_labels = ['X', 'Y']

        for i_ax, (pos, title) in enumerate(zip(positions['short_names'], positions['long_names'])):
            try:
                fig, axs = plt.subplots(ncols=1, nrows=1, sharex='col',
                                        figsize=(ScanImages.fig_size[0], ScanImages.fig_size[1]))
                if physical_units:
                    if max(np.max(twiss[pos]['sigma2_x']), np.max(twiss[pos]['sigma2_y'])) < 1e-4:
                        units_factor = 1e6
                        units_label = r'mm$^2$'
                    else:
                        units_factor = 1
                        units_label = r'm$^2$'
                else:
                    units_factor = 1
                    units_label = r'pix$^2$'

                for i_xy, var, c_fill, c_val in zip([1, 0], plot_labels, ['m', 'y'], ['b', 'g']):
                    indexes = twiss['indexes_selected'][:, i_xy]

                    # all points, in gray
                    low = fwhm_to_std(twiss[pos]['fwhms'][0][:, i_xy] - twiss[pos]['fwhms'][1][:, i_xy]) * 1e-6  # [m]
                    high = fwhm_to_std(twiss[pos]['fwhms'][0][:, i_xy] + twiss[pos]['fwhms'][2][:, i_xy]) * 1e-6  # [m]
                    ctr = fwhm_to_std(twiss[pos]['fwhms'][0][:, i_xy]) * 1e-6  # [m]

                    axs.fill_between(self.data_dict['setpoints'], units_factor * low**2, units_factor * high**2,
                                     color='gray', alpha=0.166)
                    axs.plot(self.data_dict['setpoints'], units_factor * ctr**2, f'o-',
                             color='gray', linewidth=1, markersize=3)

                    # selected points, colored
                    low = fwhm_to_std(twiss[pos]['fwhms'][0][indexes, i_xy] - twiss[pos]['fwhms'][1][indexes, i_xy])
                    low *= 1e-6  # [m]
                    high = fwhm_to_std(twiss[pos]['fwhms'][0][indexes, i_xy] + twiss[pos]['fwhms'][2][indexes, i_xy])
                    high *= 1e-6  # [m]
                    ctr = fwhm_to_std(twiss[pos]['fwhms'][0][indexes, i_xy]) * 1e-6  # [m]
                    axs.fill_between(self.data_dict['setpoints'][indexes],
                                     units_factor * low**2, units_factor * high**2,
                                     color=c_fill, alpha=0.33)
                    axs.plot(self.data_dict['setpoints'][indexes], units_factor * ctr**2, f'o{c_val}-',
                             label=rf'{var} ({self.fwhms_metric}) [{units_label}]', linewidth=1, markersize=3)

                    # Twiss fit
                    fit_sets = np.linspace(self.data_dict['setpoints'][indexes][0],
                                           self.data_dict['setpoints'][indexes][-1], 1000)
                    fit_vals = np.polyval(twiss[pos]['fit_pars'][:, i_xy], fit_sets) * units_factor
                    axs.plot(fit_sets, fit_vals, 'k', linestyle='--', linewidth=0.66,
                             label=rf'$\epsilon_{var.lower()}$ = {twiss[pos][f"epsilon_{var.lower()}"]:.2e}, '
                                   + rf'$\alpha_{var.lower()}$ = {twiss[pos][f"alpha_{var.lower()}"]:.1f}, '
                                   + rf'$\beta_{var.lower()}$ = {twiss[pos][f"beta_{var.lower()}"]:.1f}' + '\nfit: '
                                   + polyfit_label(list(twiss[pos]['fit_pars'][:, i_xy] * units_factor),
                                                   res=2, latex=True))

                axs.legend(loc='best', prop={'size': 8})
                axs.set_ylabel(rf'$\sigma^2$ [{units_label}]')
                axs.set_title(rf'{title} ({um_per_pix:.2f} $\mu$m/pix)')
                axs.set_xlabel('Current [A]')

                if save_dir:
                    save_path = save_dir / f'quad_scan_analysis_{pos}_{self.fwhms_metric}.png'
                    if save_path.is_file():
                        os.remove(save_path)
                    plt.savefig(save_path, dpi=300)
                    while not save_path.is_file():
                        time.sleep(0.1)
                    plt.savefig(save_path, dpi=300)

            except Exception as ex:
                print(str(ex))
                pass

        plt.show(block=True)

    @staticmethod
    def twiss_parameters(poly_pars, quad_2_screen: float = 1.) -> tuple[float, float, float]:
        a1 = poly_pars[0]
        a2 = poly_pars[1] / (-2 * a1)
        a3 = poly_pars[2] - (a1 * a2**2)

        epsilon = np.sqrt(a1 * a3) / (quad_2_screen**2)
        beta = np.sqrt(a1 / a3)
        alpha = (a2 + 1/quad_2_screen) * beta

        return epsilon, alpha, beta


if __name__ == '__main__':
    # initialization
    # --------------------------------------------------------------------------
    _htu = HtuExp(get_info=True)
    _base_tag = ScanTag(2023, 8, 8, 22)

    # _device = Quads()
    # _camera = Camera('UC_TopView')
    _device = 'U_EMQTripletBipolar'
    _camera = 'A3'

    _quad = 3
    _quad_2_screen = 2.126  # [m]

    _folder = ScanData.build_folder_path(_base_tag, _htu.base_path)
    _scan_data = ScanData(_folder, ignore_experiment_name=_htu.is_offline)
    _scan_images = ScanImages(_scan_data, _camera)
    _quad_analysis = QuadAnalysis(_scan_data, _scan_images, _quad, fwhms_metric='median', quad_2_screen=_quad_2_screen)

    _filters = FiltersParameters(contrast=1.333, hp_median=2, hp_threshold=3., denoise_cycles=0, gauss_filter=5.,
                                 com_threshold=0.8, bkg_image=None, box=True, ellipse=False)

    # scan analysis
    # --------------------------------------------------------------------------
    _path = _quad_analysis.analyze(None, initial_filtering=_filters, ask_rerun=False, blind_loads=True,
                                   store_images=False, store_scalars=False, save_plots=False, save=True)

    _quad_analysis.render_twiss(physical_units=True, save_dir=_quad_analysis.scan_data.get_analysis_folder())

    # _device.close()
    # _camera.close()
    print('done')
