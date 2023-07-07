# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 10:44:15 2023

@author: Reinier van Mourik, TAU Systems
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NewType, Optional

from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity

if TYPE_CHECKING:
    from pint import Quantity
    QuantityArray = NewType("QuantityArray", Quantity)

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

def invert_wavelength_angular_frequency(wavelength_or_angular_frequency):
    return Q_(2*np.pi, 'radians') * ureg.speed_of_light / wavelength_or_angular_frequency

class GrenouilleRetrieval:
    """
    """

    def __init__(self, 
                 wavelength_center: Quantity = 2 * Q_(400, 'nm'),
                 wavelength_step: Quantity = 2 * Q_(0.0798546, 'nm'),
                 time_delay_step: Quantity = Q_(0.893676, 'fs'),
                 calculate_next_E_method: str = "integration",
                 nonlinear_effect: str = "second_harmonic_generation",
                 number_of_iterations: int = 100,
                ):

        self.wavelength_center = wavelength_center
        self.wavelength_step = wavelength_step
        self.time_delay_step = time_delay_step

        self.number_of_iterations = number_of_iterations

        if calculate_next_E_method not in {'integration', 
                                           'generalized_projection/along_gradient', 
                                           'generalized_projection/full_minimization'
                                          }:
            raise ValueError(f"Unknown calculate_next_E_method ({calculate_next_E_method}). "
                             "Should be one of 'integration', 'generalized_projection/along_gradient', "
                             "'generalized_projection/full_minimization'."
                            )
        self.calculate_next_E_method = calculate_next_E_method

        if nonlinear_effect not in {'self-diffraction',
                                    'second_harmonic_generation', 
                                   }:
            raise ValueError(f"Unknown nonlinear_effect ({nonlinear_effect}). "
                             "Should be one of 'self-diffraction', 'second_harmonic_generation'."
                            )
        self.nonlinear_effect = nonlinear_effect

    def calculate_pulse(self, grenouille_trace: np.ndarray, 
                        initial_E: Optional[np.ndarray] = None,
                       ):

        self.shape = grenouille_trace.shape
        self.grenouille_trace = grenouille_trace

        self.time_step = (# Nyquist should cover 350-450nm range, i.e. be 
                          # greater than angular frequency corresponding to 
                          # 350nm  
                          (invert_wavelength_angular_frequency(Q_(700, 'nm'))
                          # convert to sampling rate, in 1/[time]
                           * 2 / Q_(2*np.pi, 'radian')
                          # invert to get time step
                          ) ** -1
                         )

        self._calculate_I_FROG()
        if initial_E is None:
            self.E = self._default_initial_E()
        else:
            self.E = initial_E

        for iteration in range(self.number_of_iterations):
            self._calculate_E_sig()
            self._calculate_FT_E_sig()
            self._replace_FT_E_sig_magnitude()
            self._calculate_inverse_FT_E_sig()
            self._calculate_next_E()

        return self.E

    @property
    def grenouille_trace_λ(self) -> QuantityArray:
        return self.wavelength_center + (np.arange(self.shape[0]) - (self.shape[0] - 1) / 2) * self.wavelength_step

    @property
    def t(self) -> QuantityArray:
        return np.arange(self.shape[0]) * self.time_step
    
    @property
    def τ(self) -> QuantityArray:
        return (np.arange(self.shape[1]) - ((self.shape[1] - 1) / 2)) * self.time_delay_step

    @property
    def ω(self) -> QuantityArray:
        """ Returns the frequency axis of the FT of E_sig, as an array from
            -Ny..Ny 
        """
        return Q_(2*np.pi, 'radians') / self.time_step * np.fft.fftshift(np.fft.fftfreq(self.shape[0]))

    def _calculate_I_FROG(self):
        self.I_FROG = np.transpose(
            np.fromiter((np.interp((invert_wavelength_angular_frequency(self.ω)),  self.grenouille_trace_λ, grenouille_trace_column, left=0.0, right=0.0)
                        for grenouille_trace_column in self.grenouille_trace.T
                        ), (float, self.shape[0]))
        )

    def _default_initial_E(self):
        wavelength_FWHM = Q_(4, 'nm')
        pulse_FWHM = Q_(30, 'fs')
        omega_center = invert_wavelength_angular_frequency(self.wavelength_center)
        omega_FWHM = omega_center/self.wavelength_center * wavelength_FWHM
        E_FT = np.exp(-np.square(self.ω - omega_center) / (omega_FWHM**2 / (4 * np.log(2))))
        pulse = np.exp(-np.square(self.t - self.t[-1] / 2) / (pulse_FWHM**2 / (4 * np.log(2))))
        assert E_FT.check('')
        assert pulse.check('')
        return pulse.m * np.fft.ifft(np.fft.ifftshift(E_FT.m))

    def _calculate_E_sig(self):
        """ Calculate E_sig(t, τ) = E(t) gate(E(t - τ))
            
            gate(E(t)) depends on the non-linear effect.
        """

        E_interpolator_real = InterpolatedUnivariateSpline(self.t.m_as('sec'), self.E.real, ext='zeros')
        E_interpolator_imag = InterpolatedUnivariateSpline(self.t.m_as('sec'), self.E.imag, ext='zeros')

        def gate(t):

            def gate_self_diffraction(t):
                """ gate(t) = |E(t)|^2 """
                return (  np.square(E_interpolator_real(t.m_as('sec')))
                        + np.square(E_interpolator_imag(t.m_as('sec')))
                       )
            def gate_second_harmonic_generation(t):
                """ gate(t) = |E(t)| """
                return np.sqrt(  np.square(E_interpolator_real(t.m_as('sec')))
                               + np.square(E_interpolator_imag(t.m_as('sec')))
                              )

            return {'self-diffraction': gate_self_diffraction,
                    'second_harmonic_generation': gate_second_harmonic_generation,
                   }[self.nonlinear_effect](t)

        def E_sig_column(τ):
            """ Return E(t) * gate(E(t-τ)) for a given tau
                i.e. single column of E_sig(t, tau)
            """
            return self.E * gate(self.t - τ)

        self.E_sig_tτ = np.transpose(
            np.fromiter((E_sig_column(τ) for τ in self.τ), 
                        (np.complex128, len(self.t))
                       )
        )

    def _calculate_FT_E_sig(self):
        self.E_sig_ωτ = np.fft.fftshift(np.fft.fft(self.E_sig_tτ, axis=0), axes=(0,))

    def _replace_FT_E_sig_magnitude(self):
        self.E_sig_ωτ *= np.sqrt(self.I_FROG) / np.clip(np.abs(self.E_sig_ωτ), 1e-15, np.inf)

    def _calculate_inverse_FT_E_sig(self):
        self.E_sig_tτ = np.fft.ifft(np.fft.ifftshift(self.E_sig_ωτ, axes=(0,)), axis=0)

    def _calculate_next_E(self):
        
        def _calculate_next_E_by_integration():
            @ureg.wraps('=A*B', ('=A', '=B'))
            def trapz_ua(Y, x):
                return np.trapz(Y, x, axis=1)
            self.E = trapz_ua(self.E_sig_tτ, self.τ)

        def _calculate_next_E_by_generalized_projection_along_gradient():
            raise NotImplementedError()

        def _calculate_next_E_by_generalized_projection_full_minimization():
            raise NotImplementedError()

        return {'integration': _calculate_next_E_by_integration,
                'generalized_projection/along_gradient': _calculate_next_E_by_generalized_projection_along_gradient,
                'generalized_projection/full_minimization': _calculate_next_E_by_generalized_projection_full_minimization,
               }[self.calculate_next_E_method]()


    def simulate_grenouille_trace(self, E, grenouille_trace_shape):
        self.shape = grenouille_trace_shape
        self.E = E
        self._calculate_E_sig()
        
        # spectrogram(ω, τ), for ω = -Ny..Ny
        spectrogram = np.square(np.abs(np.fft.fftshift(np.fft.fft(self.E_sig_tτ, axis=0), axes=(0,))))

        # interpolate I_frog at lambda
        return np.transpose(
            np.fromiter(
                (np.interp(invert_wavelength_angular_frequency(self.grenouille_trace_λ), 
                           self.ω, spectrogram_column, left=0.0, right=0.0
                          )
                 for spectrogram_column in spectrogram.T
                ), (float, self.shape[0])
            )     
        )


if __name__ == "__main__":
    pass
