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
from scipy.optimize import least_squares

def invert_wavelength_angular_frequency(wavelength_or_angular_frequency):
    return Q_(2*np.pi, 'radians') * ureg.speed_of_light / wavelength_or_angular_frequency

class GrenouilleRetrieval:
    """
    [1] R. Trebino and D. J. Kane, “Using phase retrieval to measure the intensity 
       and phase of ultrashort pulses: frequency-resolved optical gating,” J. Opt. 
       Soc. Am. A, vol. 10, no. 5, p. 1101, May 1993, doi: 10.1364/JOSAA.10.001101.
    [2] K. W. DeLong, B. Kohler, K. Wilson, D. N. Fittinghoff, and R. Trebino, 
       “Pulse retrieval in frequency-resolved optical gating based on the method 
       of generalized projections,” Opt. Lett., vol. 19, no. 24, p. 2152, Dec. 
       1994, doi: 10.1364/OL.19.002152.

    """

    def __init__(self, 
                 calculate_next_E_method: str = "integration",
                 nonlinear_effect: str = "second_harmonic_generation",
                 number_of_iterations: int = 100,
                 pulse_center_wavelength: Quantity = Q_(800, 'nm'),
                 min_pulse_wavelength: Quantity = Q_(700, 'nm'), 
                ):
        """
        Parameters
        ----------
        calculate_next_E_method : str
            How to obtain the next iteration of E(t) from E_sig(t, τ)
                'integration': integrate E_sig(t, τ)dτ, as in [1]
                'generalized_projection/along_gradient': project E_sig(t, τ) onto
                    space of E_sig(t, τ)'s that can be generated from E(t), as in 
                    [2]. Search along the dZ/dE line
                'generalized_projection/full_minimization': generalized projection
                    as with previous, but optimize Z via all E points.
        nonlinear_effect : str
            The FROG setup's nonlinear effect, which determines the gate function
            as well as possible frequency doubling.
                'self_diffraction'
                'second_harmonic_generation'
        number_of_iterations: int 
            Maximum number of iterations
        pulse_center_wavelength : [length] Quantity
            the center, or carrier, wavelength of the laser pulse.
        min_pulse_wavelength: [length] Quantity
            The minimum wavelength of the pulse spectrum to resolve. This 
            determines the time step and the angular frequency of E_sig axis 0
        """


        if calculate_next_E_method not in {'integration', 
                                           'generalized_projection/along_gradient', 
                                           'generalized_projection/full_minimization'
                                          }:
            raise ValueError(f"Unknown calculate_next_E_method ({calculate_next_E_method}). "
                             "Should be one of 'integration', 'generalized_projection/along_gradient', "
                             "'generalized_projection/full_minimization'."
                            )
        self.calculate_next_E_method = calculate_next_E_method

        if nonlinear_effect not in {'self_diffraction',
                                    'second_harmonic_generation', 
                                   }:
            raise ValueError(f"Unknown nonlinear_effect ({nonlinear_effect}). "
                             "Should be one of 'self_diffraction', 'second_harmonic_generation'."
                            )
        self.nonlinear_effect = nonlinear_effect

        self.number_of_iterations = number_of_iterations
        self.pulse_center_wavelength = pulse_center_wavelength
        self.min_pulse_wavelength = min_pulse_wavelength

        # time step should be small enough so that Nyquist frequency is at least
        # equal to frequency corresponding to min_wavelength
        self.time_step = ((invert_wavelength_angular_frequency(self.min_pulse_wavelength)
                          # convert to sampling rate, in 1/[time]
                           * 2 / Q_(2*np.pi, 'radian')
                          # invert to get time step
                          ) ** -1
                         )

    def calculate_pulse(self, grenouille_trace: np.ndarray, 
                        grenouille_trace_center_wavelength: Quantity = Q_(400, 'nm'),
                        grenouille_trace_wavelength_step: Quantity = Q_(0.0798546, 'nm'),
                        time_delay_step: Quantity = Q_(0.893676, 'fs'),
                        pulse_duration: Quantity = Q_(800, 'fs'),
                        initial_E: Optional[np.ndarray] = None,
                       ):
        """
        Parameters
        ----------
        grenouille_trace : np.ndarray
            the Grenouille image to be analyzed, with wavelength on axis 0 
            (vertical) and time delay (tau) on axis 1 (horizontal)
        grenouille_trace_center_wavelength : [length] Quantity
            center wavelength of the spectrometer axis (0, vertical). For the 
            'second_harmonic_generation' nonlinear effect, this will be around 
            half the wavelength of the pulse
        grenouille_trace_wavelength_step : [length] Quantity
            resolution of spectrometer axis (0, vertical). 
        time_delay_step: [time] Quantity
            resolution of time_delay (tau) axis (1, horizontal). The center 
            of the axis should represent tau = 0
        pulse_duration: [time] Quantity
            length of the reconstructed pulse
        initial_E : complex np.ndarray of length grenouille_trace.shape[0]
            starting guess for E field pulse
        
        """

        self.shape = grenouille_trace.shape
        self.grenouille_trace = grenouille_trace

        self.grenouille_trace_center_wavelength = grenouille_trace_center_wavelength
        self.grenouille_trace_wavelength_step = grenouille_trace_wavelength_step
        self.time_delay_step = time_delay_step
        self.pulse_duration = pulse_duration

        # total time, i.e. length of time axis, should be such that it matches
        # (roughly) the resolution of grenouille trace wavelength resolution
        total_time = ((self.frequency_multiplier * self.grenouille_trace_center_wavelength)**2 
                      / ureg.speed_of_light 
                      / (self.frequency_multiplier * self.grenouille_trace_wavelength_step)
                     )

        self.time_axis_length = int(np.ceil((total_time / self.time_step) / 2)) * 2 + 1

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
            self._center_and_zero_phase_E()

        return self.E

    @property
    def grenouille_trace_λ(self) -> QuantityArray:
        return self.grenouille_trace_center_wavelength + (np.arange(self.shape[0]) - (self.shape[0] - 1) / 2) * self.grenouille_trace_wavelength_step

    @property
    def t(self) -> QuantityArray:
        """ Time axis for the Esig(t, τ) 2D array
        """
        return (np.arange(self.time_axis_length) - (self.time_axis_length - 1) / 2) * self.time_step

    @property
    def E_t(self) -> QuantityArray:
        """ Time axis for self.E. Will be padded to len(self.t) for calculations
        """
        # make it an odd length, so there is a t=0 point
        E_t_length = int(np.ceil((self.pulse_duration / self.time_step).m_as('') / 2)) * 2 + 1
        return (np.arange(E_t_length) - (E_t_length - 1) // 2) * self.time_step

    @property
    def τ(self) -> QuantityArray:
        return (np.arange(self.shape[1]) - ((self.shape[1] - 1) / 2)) * self.time_delay_step

    @property
    def ω(self) -> QuantityArray:
        """ Returns the frequency axis of the FT of E_sig, as an array from
            -Ny..Ny 
        """
        return Q_(2*np.pi, 'radians') / self.time_step * np.fft.fftshift(np.fft.fftfreq(self.time_axis_length))

    @property
    def frequency_multiplier(self):
        return {'self_diffraction': 1,
                'second_harmonic_generation': 2,
               }[self.nonlinear_effect]

    def _calculate_I_FROG(self):
        """
        Place the Grenouille trace on the ω-τ grid.
        
        This involves 
        - multiplying the Grenouille wavelength in case of harmonics
        - converting the wavelength to angular frequency
        - centering around the carrier frequency
        

        """
        λ = invert_wavelength_angular_frequency(self.frequency_multiplier * (self.ω + invert_wavelength_angular_frequency(self.pulse_center_wavelength)))
        self.I_FROG = np.transpose(
            np.fromiter((np.interp(λ,  self.grenouille_trace_λ, grenouille_trace_column, 
                                   left=0.0, right=0.0
                                  )
                        for grenouille_trace_column in self.grenouille_trace.T
                        ), (float, self.time_axis_length))
        )

    def _default_initial_E(self):
        pulse_FWHM = Q_(50, 'fs')
        pulse = np.exp(-np.square(self.E_t) / (pulse_FWHM**2 / (4 * np.log(2))))
        return pulse.m

    def _calculate_E_sig_from_E(self, E: np.ndarray, 
                                E_t: Optional[QuantityArray] = None, 
                                E_sig_t: Optional[QuantityArray] = None
                               ) -> np.ndarray:
        """ Calculate E_sig(t, τ) = E(t) gate(E(t - τ))
            
            gate(E(t)) depends on the non-linear effect.

        Parameters
        ----------
        E : complex E-field
        E_t : [time] Quantity array
            time axis corresponding to E. default self.E_t
        E_sig_t: [time] Quantity array
            time axis of resulting E_sig(t, τ) 2d array
        """
        if E_t is None:
            E_t = self.E_t

        if E_sig_t is None:
            E_sig_t = self.t

        E_interpolator_real = InterpolatedUnivariateSpline(E_t.m_as('sec'), E.real, ext='zeros')
        E_interpolator_imag = InterpolatedUnivariateSpline(E_t.m_as('sec'), E.imag, ext='zeros')

        def gate(t):

            def gate_self_diffraction(t):
                """ gate(t) = |E(t)|^2 """
                return (  np.square(E_interpolator_real(t.m_as('sec')))
                        + np.square(E_interpolator_imag(t.m_as('sec')))
                       )
            def gate_second_harmonic_generation(t):
                """ gate(t) = E(t) """
                return E_interpolator_real(t.m_as('sec')) + 1j * E_interpolator_imag(t.m_as('sec'))

            return {'self_diffraction': gate_self_diffraction,
                    'second_harmonic_generation': gate_second_harmonic_generation,
                   }[self.nonlinear_effect](t)

        E_at_E_sig_t = E_interpolator_real(E_sig_t.m_as('sec')) + 1j*E_interpolator_imag(E_sig_t.m_as('sec'))

        def E_sig_column(τ):
            """ Return E(t) * gate(E(t-τ)) for a given tau
                i.e. single column of E_sig(t, τ)
            """
            return E_at_E_sig_t * gate(E_sig_t - τ)

        return np.transpose(
            np.fromiter((E_sig_column(τ) for τ in self.τ), 
                        (np.complex128, len(E_sig_t))
                       )
        )

    def _calculate_E_sig(self):
        self.E_sig_tτ = self._calculate_E_sig_from_E(self.E)

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
            self.E = np.interp(self.E_t,   self.t, trapz_ua(self.E_sig_tτ, self.τ).m)

        def _calculate_next_E_by_generalized_projection_along_gradient():
            
            def resid(E_est):
                E_sig_est = self._calculate_E_sig_from_E(E_est)
                return np.abs(self.E_sig_tτ - E_sig_est).flatten()

            REAL = 1; IMAG = 2; 
            dEi = 1e-6
            Z0 = np.sum(np.square(resid(self.E)))

            def dZdEi(i, real_or_imag):
                dE = np.zeros(len(self.E)); 
                if real_or_imag == REAL: 
                    dE[i] += dEi
                elif real_or_imag == IMAG:
                    dE[i] += 1j * dEi
                Z = np.sum(np.square(resid(self.E + dE)))
                return (Z - Z0) / dEi
            
            dZdE = (  np.array([dZdEi(i, REAL) for i in range(len(self.E))]) 
                    + 1j * np.array([dZdEi(i, IMAG) for i in range(len(self.E))]) 
                   )

            def loss(beta):
                return resid(self.E + beta * dZdE)
            
            solution = least_squares(loss, 0.0)

            self.E += solution.x * dZdE

        def _calculate_next_E_by_generalized_projection_full_minimization():
            def loss(E_split_in_real_imag):
                E_real, E_imag = E_split_in_real_imag.reshape((2, -1))
                E_sig_est = self._calculate_E_sig_from_E(E_real + 1j * E_imag)

                return np.abs(self.E_sig_tτ - E_sig_est).flatten()

            solution = least_squares(loss, 
                                     np.concatenate([self.E.real, self.E.imag]),
                                     method='lm'
                                    )
            E_real, E_imag = solution.x.reshape((2, -1))

            self.E = E_real + 1j * E_imag

        return {'integration': _calculate_next_E_by_integration,
                'generalized_projection/along_gradient': _calculate_next_E_by_generalized_projection_along_gradient,
                'generalized_projection/full_minimization': _calculate_next_E_by_generalized_projection_full_minimization,
               }[self.calculate_next_E_method]()

    def _center_and_zero_phase_E(self):
        # center E around t = 0
        center_of_mass = (self.E_t * np.abs(self.E)).sum() / np.abs(self.E).sum()
        self.E = np.interp(self.E_t,  self.E_t - center_of_mass, self.E, left=0.0, right=0.0)
        
        # zero the phase at t = 0
        self.E *= np.exp(-1j * np.angle(self.E[(len(self.E) - 1) // 2]))

    def simulate_grenouille_trace(self, E, grenouille_trace_shape):
        self.shape = grenouille_trace_shape
        self.E = E
        self._calculate_E_sig()

        # spectrogram(ω, τ), for ω = -Ny..Ny
        spectrogram = np.square(np.abs(np.fft.fftshift(np.fft.fft(self.E_sig_tτ, axis=0), axes=(0,))))

        # interpolate I_frog at lambda
        return np.transpose(
            np.fromiter(
                (np.interp(invert_wavelength_angular_frequency(self.frequency_multiplier * self.grenouille_trace_λ), 
                           self.ω, spectrogram_column, left=0.0, right=0.0
                          )
                 for spectrogram_column in spectrogram.T
                ), (float, self.shape[0])
            )     
        )


if __name__ == "__main__":
    pass
