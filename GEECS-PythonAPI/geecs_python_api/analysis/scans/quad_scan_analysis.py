from __future__ import annotations

import os
import time
import numpy as np
from typing import Literal, NewType, TYPE_CHECKING
from numpy.polynomial.polynomial import polyfit
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Union, Any, Optional
from geecs_python_api.controls.devices.HTU.diagnostics.cameras import Camera
from geecs_python_api.controls.devices.HTU.transport.electromagnets.quad import EMQTriplet
from geecs_python_api.controls.api_defs import ScanTag
from geecs_python_api.tools.interfaces.prompts import text_input
from geecs_python_api.analysis.scans.scan_images import ScanImages
from geecs_python_api.analysis.scans.scan_analysis import ScanAnalysis
from geecs_python_api.tools.images.filtering import FiltersParameters
from geecs_python_api.tools.interfaces.exports import save_py
from geecs_python_api.tools.images.displays import polyfit_label
from geecs_python_api.tools.images.spot import fwhm_to_std

# This package's shared units registry
from ... import ureg
Q_ = ureg.Quantity
if TYPE_CHECKING:
    from pint import Quantity
    QuantityArray = NewType("QuantityArray", Quantity)

class QuadAnalysis(ScanAnalysis):
    """

    Parameters
    ----------
    scan_tag : ScanTag
    quad : int
        Number of the EMQ that was scanned
    camera : Union[int, Camera, str]
    fwhms_metric : str, optional
        default 'median'
    quad_2_screen : float, optional
        distance from center of EMQ to screen, in meters, by default 1.
    """

    def __init__(self, scan_tag: ScanTag, quad: int, camera: Union[int, Camera, str],
                 fwhms_metric: str = 'median', quad_2_screen: float = 1.):
        super().__init__(scan_tag, camera)
        self.quad_number: int = quad
        self.quad_variable: str = f'Current_Limit.Ch{quad}'

        self.fwhms_metric: str = fwhms_metric
        self.quad_2_screen: Quantity = quad_2_screen * ureg.meter

    def analyze(self, variable: Optional[str] = None, initial_filtering=FiltersParameters(), ask_rerun: bool = True,
                blind_loads: bool = False, store_images: bool = True, store_scalars: bool = True,
                save_plots: bool = False, save_data_dict: bool = False) -> Optional[Path]:
        if not variable:
            variable = self.quad_variable

        # run parent analysis (beam metrics vs scan parameter)
        super().analyze(variable, initial_filtering, ask_rerun, blind_loads,
                        store_images, store_scalars, save_plots, save_data_dict)

        # collect setpoints
        self.data_dict['emq1_A'] = np.median(self.scan_images.scan_scalar_data['U_EMQTripletBipolar']['Current_Limit.Ch1'])
        self.data_dict['emq2_A'] = np.median(self.scan_images.scan_scalar_data['U_EMQTripletBipolar']['Current_Limit.Ch2'])
        self.data_dict['emq3_A'] = np.median(self.scan_images.scan_scalar_data['U_EMQTripletBipolar']['Current_Limit.Ch3'])

        self.data_dict['jet_z'] = np.median(self.scan_images.scan_scalar_data['U_ESP_JetXYZ']['Position.Axis 3'])
        self.data_dict['hexapod_x'] = np.median(self.scan_images.scan_scalar_data['U_Hexapod']['xpos'])
        self.data_dict['source_pmq_mm'] = self.data_dict['jet_z'] + self.data_dict['hexapod_x'] + 35.5  # 2019 calibration

        # render parent analysis
        save_plots_dir = self.scan_data.get_analysis_folder() if save_plots else None

        figs = super().render(physical_units=True, x_label='Current [A]',
                              show_xy=True, show_fwhms=True, show_deltas=True,
                              xy_metric='median', fwhms_metric=self.fwhms_metric, deltas_metric='median',
                              xy_fit=1, fwhms_fit=2, deltas_fit=2,
                              show_figs=True, save_dir=save_plots_dir, sync=False)

        # request Twiss analysis ranges from user
        setpoints: QuantityArray = self.data_dict['setpoints'] * ureg.ampere
        range_x: QuantityArray = Q_.from_list([np.min(setpoints), np.max(setpoints)])
        range_y: QuantityArray = Q_.from_list([np.min(setpoints), np.max(setpoints)])

        while True:
            try:
                lim_str = text_input(f'Lower current limit to consider for FWHM-X, e.g. "none" or -1.5 : ')
                lim_low: Quantity = np.min(setpoints) if lim_str.lower() == 'none' else float(lim_str) * ureg.ampere
                lim_str = text_input(f'Upper current limit to consider for FWHM-X, e.g. "none" or 3.5 : ')
                lim_high: Quantity = np.max(setpoints) if lim_str.lower() == 'none' else float(lim_str) * ureg.ampere
                range_x = Q_.from_list([lim_low, lim_high])

                lim_str = text_input(f'Lower current limit to consider for FWHM-Y, e.g. "none" or -1.5 : ')
                lim_low = np.min(setpoints) if lim_str.lower() == 'none' else float(lim_str) * ureg.ampere
                lim_str = text_input(f'Upper current limit to consider for FWHM-Y, e.g. "none" or 3.5 : ')
                lim_high = np.max(setpoints) if lim_str.lower() == 'none' else float(lim_str) * ureg.ampere
                range_y = Q_.from_list([lim_low, lim_high])

                break
            except Exception:
                continue
            finally:
                try:
                    for fig_ax in figs:
                        plt.close(fig_ax[0])
                except Exception:
                    pass

        in_range_x: np.ndarray = (setpoints >= np.min(range_x)) & (setpoints <= np.max(range_x))
        in_range_y: np.ndarray = (setpoints >= np.min(range_y)) & (setpoints <= np.max(range_y))

        # run Twiss analysis
        sample_analysis = self.data_dict['analyses'][0]
        um_per_pix: float = sample_analysis['summary']['um_per_pix']
        positions: dict = sample_analysis['image_analyses'][0]['positions']
        twiss_analysis: dict[str, Any] = {}

        # central energy of electron beam, needed for calculating beam rigidity
        ebeam_energy: Quantity = 100.0 * ureg.MeV

        for pos in positions['short_names']:
            twiss_analysis[pos] = {}
            # data_val is an N x 2 array of FWHM values in micrometer, for y and x
            data_val, data_err_low, data_err_high = ScanAnalysis.fetch_metrics(self.data_dict['analyses'],
                                                                               self.fwhms_metric, 'fwhms',
                                                                               pos, 'pix_ij', um_per_pix)
            twiss_analysis[pos]['fwhms'] = (data_val, data_err_low, data_err_high)

            fwhm_x: QuantityArray = data_val[in_range_x, 1] * ureg.micrometer
            fwhm_y: QuantityArray = data_val[in_range_y, 0] * ureg.micrometer

            def obtain_twiss_parameters_through_quadratic_fit(fwhm: QuantityArray, emq_current_setpoints: QuantityArray, emq: EMQTriplet.EMQ):
                """ 
                Uses formulas from https://uspas.fnal.gov/materials/08UMD/Emittance%20Measurement%20-%20Quadrupole%20Scan.pdf

                Fits a parabola to sigma^2 vs inverse focal length of the quadrupole, 
                where 1/f = k1 * l, the focusing strength times the effective length
                
                Parameters
                ----------
                fwhm : QuantityArray
                    Full-width half-maximum values
                emq_current_setpoints : QuantityArray
                    EMQ currents
                emq : EMQTriplet.EMQ
                    Is needed to convert current to focal length
                    
                Returns
                -------
                epsilon : Quantity [length]*[angle]
                    unnormalized emittance
                alpha : Quantity [dimensionless]
                beta : Quantity [length]/[angle]
                sigma_squared : Quantity [length]**2
                fit_pars : tuple[Quantity [length]**2/[length]**-2, 
                                 Quantity [length]**2/[length]**-1, 
                                 Quantity [length]**2
                                ]
                    quadratic fit results in order quadratic, linear, constant
                """
                sigma_squared: QuantityArray = fwhm_to_std(fwhm)**2
                inverse_focal_length: QuantityArray = Q_(emq.k1(emq_current_setpoints.m_as('amp'), ebeam_energy_MeV=ebeam_energy.m_as('MeV')), 'm^-2') * Q_(emq.length, 'm')

                # unit-aware versions of np.polyfit and QuadAnalysis.min_constrained_quadratic_fit
                @ureg.wraps(('=A/B^2', '=A/B', '=A'), ('=B', '=A', None))
                def quadratic_fit(x, y, min_constrained: bool = False) -> tuple[Quantity, Quantity, Quantity]:
                    """ Returns quadratic fit parameters a*x^2 + b*x + c in order [a, b, c] """
                    if min_constrained:
                        return tuple(QuadAnalysis.min_constrained_quadratic_fit(x, y))
                    else:
                        return tuple(np.flip(polyfit(x, y, 2)))

                fit_pars: tuple[Quantity, Quantity, Quantity] 
                fit_pars = quadratic_fit(inverse_focal_length, sigma_squared)
                fit_min: Quantity = fit_pars[2] - (fit_pars[1]**2) / (4 * fit_pars[0])
                if fit_min.magnitude < 0:
                    fit_pars = quadratic_fit(inverse_focal_length, sigma_squared, min_constrained=True)

                epsilon, alpha, beta = QuadAnalysis.twiss_parameters(fit_pars, self.quad_2_screen)

                # Twiss calculates are done under the small angle approximation 
                # so somewhere an arctan was neglected, so the angle unit needs 
                # to be added
                epsilon *= ureg.radian
                beta /= ureg.radian                

                return epsilon, alpha, beta, sigma_squared, fit_pars

            def store_twiss_results(xy: Literal['x', 'y'], epsilon: Quantity, alpha: Quantity, beta: Quantity, sigma_squared: QuantityArray, fit_pars: tuple[Quantity, Quantity, Quantity]):
                """ Save magnitudes of Twiss and fit parameters in dict """
                twiss_analysis[pos][f'epsilon_{xy}'] = epsilon.m_as('meter*radian')
                twiss_analysis[pos][f'alpha_{xy}']   = alpha.m_as('dimensionless')
                twiss_analysis[pos][f'beta_{xy}'] = beta.m_as('meter/radian')
                twiss_analysis[pos][f'sigma2_{xy}'] = sigma_squared.m_as('meter^2')
                twiss_analysis[pos][f'fit_pars_{xy}'] = np.array((fit_pars[0].m_as('meter^2/meter^-2'), fit_pars[1].m_as('meter^2/meter^-1'), fit_pars[2].m_as('meter^2')))

            epsilon, alpha, beta, sigma_squared, fit_pars = obtain_twiss_parameters_through_quadratic_fit(fwhm_x, setpoints[in_range_x], EMQTriplet.emqs[self.quad_number - 1])
            store_twiss_results('x', epsilon, alpha, beta, sigma_squared, fit_pars)

            epsilon, alpha, beta, sigma_squared, fit_pars = obtain_twiss_parameters_through_quadratic_fit(fwhm_y, setpoints[in_range_y], EMQTriplet.emqs[self.quad_number - 1])
            store_twiss_results('y', epsilon, alpha, beta, sigma_squared, fit_pars)

            # 3 x 2 array of quadratic fit parameters in order of [a.x^2, b.x, c]
            # in order of y, x
            twiss_analysis[pos]['fit_pars'] = np.stack([twiss_analysis[pos][f'fit_pars_y'], 
                                                        twiss_analysis[pos][f'fit_pars_x']
                                                      ]).transpose()
            # [setpoint for sigma_y min, setpoint for sigma_x min]
            twiss_analysis[pos]['1/f_at_fit_min'] = -twiss_analysis[pos]['fit_pars'][1,:] / (2 * twiss_analysis[pos]['fit_pars'][0,:]) 
            
            def convert_inverse_focal_length_to_emq_current(inverse_focal_length: QuantityArray) -> QuantityArray:
                """
                Parameters
                ----------
                inverse_focal_length : QuantityArray
                    array of [y, x] inverse focal lengths

                Returns
                -------
                current : QuantityArray
                    array of [y, x] current setpoints
                """
                # get k1 from 1/f = k1 * l
                k1 = inverse_focal_length / Q_(EMQTriplet.emqs[self.quad_number - 1].length, 'm')
                # calculate k1 at unit current
                focal_strength_per_current = Q_(EMQTriplet.emqs[self.quad_number - 1].k1(1.0, ebeam_energy_MeV=ebeam_energy.m_as('MeV')), 'm^-2 / amp')
                return k1 / focal_strength_per_current
            
            twiss_analysis[pos]['setpoint_at_fit_min'] = convert_inverse_focal_length_to_emq_current(twiss_analysis[pos]['1/f_at_fit_min'] * ureg.meter**-1).m_as('amp')

        twiss_analysis['quad_2_screen'] = self.quad_2_screen.m_as('meter')
        twiss_analysis['indexes_selected'] = np.stack([in_range_y, in_range_x]).astype(int).transpose()
        twiss_analysis['setpoints_selected'] = np.stack([range_y, range_x]).m_as('amp').transpose()
        self.data_dict['twiss'] = twiss_analysis

        # export data
        if save_data_dict:
            data_dict_saved = {key: self.data_dict[key] for key in
                               ['setpoints', 'scan_folder', 'jet_z', 'hexapod_x', 'source_pmq_mm',
                                'emq1_A', 'emq2_A', 'emq3_A', 'twiss']}
            data_dict_saved['um_per_pix'] = um_per_pix

            export_file_path = self.scan_data.get_analysis_folder() / f'quad_scan_analysis_{self.device_name}'
            save_py(file_path=export_file_path, data=data_dict_saved)
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
        positions: dict = sample_analysis['image_analyses'][0]['positions']

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
        # Using formulas from https://uspas.fnal.gov/materials/08UMD/Emittance%20Measurement%20-%20Quadrupole%20Scan.pdf
        A = poly_pars[0]
        B = poly_pars[1] / (-2 * A)
        C = poly_pars[2] - (A * B**2)

        epsilon = np.sqrt(A * C) / (quad_2_screen**2)
        beta = np.sqrt(A / C)
        alpha = (B + 1/quad_2_screen) * beta

        return epsilon, alpha, beta

    @staticmethod
    def min_constrained_quadratic_fit(x_vals: np.ndarray, y_vals: np.ndarray) -> np.ndarray:
        k = np.argmin(y_vals)
        yk = y_vals[k]
        xk = x_vals[k]

        zi = y_vals - yk
        ti = (x_vals - xk)**2
        c = np.sum(ti * zi) / np.sum(ti**2)

        coefficients = [c, -2 * c * xk, c * (xk**2) + yk]

        return np.array(coefficients)
