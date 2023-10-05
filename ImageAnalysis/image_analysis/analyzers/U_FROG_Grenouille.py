from __future__ import annotations
from typing import Union, TYPE_CHECKING

import numpy as np
if TYPE_CHECKING:
    from ..types import Array2D
    from numpy.typing import NDArray

from ..base import ImageAnalyzer
from ..utils import ROI, read_imaq_image, NotAPath

from ..algorithms.grenouille import GrenouilleRetrieval
from .. import ureg, Q_, Quantity

from scipy.signal import peak_widths
from scipy.special import erf

class U_FROG_GrenouilleImageAnalyzer(ImageAnalyzer):
    """ ImageAnalyzer for Grenouille
    """
    def __init__(self, 
                 pulse_center_wavelength: Quantity = Q_(800, 'nm'),
                 grenouille_trace_center_wavelength: Quantity = Q_(400, 'nm'),
                 grenouille_trace_wavelength_step: Quantity = Q_(0.0798546, 'nm'),
                 grenouille_trace_time_delay_step: Quantity = Q_(0.893676, 'fs'),
                ):
        """
        Parameters
        ----------
        pulse_center_wavelength : [length] Quantity
            the center, or carrier, wavelength of the laser pulse.
        grenouille_trace_center_wavelength : [length] Quantity
            center wavelength of the spectrometer axis (0, vertical). For the 
            'second_harmonic_generation' nonlinear effect, this will be around 
            half the wavelength of the pulse
        grenouille_trace_wavelength_step : [length] Quantity
            resolution of spectrometer axis (0, vertical). 
        time_delay_step: [time] Quantity
            resolution of time_delay (tau) axis (1, horizontal). The center 
            of the axis should represent tau = 0
        """

        self.pulse_center_wavelength = pulse_center_wavelength
        self.grenouille_trace_center_wavelength = grenouille_trace_center_wavelength
        self.grenouille_trace_wavelength_step = grenouille_trace_wavelength_step
        self.grenouille_trace_time_delay_step = grenouille_trace_time_delay_step

        self.grenouille_retrieval = GrenouilleRetrieval(
            calculate_next_E_method='generalized_projection/along_gradient',
            nonlinear_effect='second_harmonic_generation',
            max_number_of_iterations=100,
            pulse_center_wavelength=self.pulse_center_wavelength,
        )

    def analyze_image(self, grenouille_trace: Array2D, auxiliary_data: dict | None = None) -> dict[str, Union[float, NDArray]]:
        pulse: NDArray[np.complex128] = self.grenouille_retrieval.calculate_pulse(grenouille_trace, 
                                            grenouille_trace_center_wavelength=self.grenouille_trace_center_wavelength,
                                            grenouille_trace_wavelength_step=self.grenouille_trace_wavelength_step,
                                            time_delay_step=self.grenouille_trace_time_delay_step,
                                        )
        
        # measurements on pulse
        # FWHM 
        power = np.square(np.abs(pulse))
        peak_i = np.argmax(power)
        fwhms_i, _, _, _ = peak_widths(power, [peak_i])
        fwhm = fwhms_i[0] * self.grenouille_retrieval.time_step

        # shortest interval containing 76% of power, which is equal to FWHM for a Gaussian pulse
        fwhm_area = erf(np.sqrt(np.log(2))) # approx 76%
        cumulative_power_normalized = np.cumsum(power) / np.sum(power)
        time_intervals = np.interp(cumulative_power_normalized + fwhm_area,  cumulative_power_normalized, self.grenouille_retrieval.E_t, left=Q_('nan fs'), right=Q_('nan fs')) - self.grenouille_retrieval.E_t
        _76_percent_interval = np.min(time_intervals)

        analysis_results = {'pulse_E_field_AU': np.rec.fromarrays([self.grenouille_retrieval.E_t, pulse], names=["time_fs", "pulse_E_field_AU"]),
                            'fwhm_fs': fwhm.m_as('fs'),
                            '76%_interval_fs': _76_percent_interval.m_as('fs'), 
                           }
        
        return analysis_results
