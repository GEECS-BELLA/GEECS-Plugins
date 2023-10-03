# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 10:44:15 2023

@author: Reinier van Mourik, TAU Systems
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NewType, Optional
from math import ceil

from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity

if TYPE_CHECKING:
    from pint import Quantity
    QuantityArray = NewType("QuantityArray", Quantity)

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import least_squares, minimize

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

        # min_pulse_wavelength: [length] Quantity
        #     The minimum wavelength of the pulse spectrum to resolve. This 
        #     determines the time step and the angular frequency of E_sig axis 0
        self.min_pulse_wavelength = 7/8 * self.pulse_center_wavelength


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


        # maximum time step of E_sig(t, tau) t axis
        # time step should be small enough so that Nyquist frequency is at least
        # equal to frequency corresponding to minimum resolvable wavelength
        maximum_time_step = ((invert_wavelength_angular_frequency(self.min_pulse_wavelength)
                             # convert to sampling rate, in 1/[time]
                             * 2 / Q_(2*np.pi, 'radian')
                             # invert to get time step
                             ) ** -1
                            )

        # if the calculated E_sig(t, tau) t step is smaller than a few times the 
        # tau step (time_delay_step), it's computationally advantageous to make it an 
        # integer part of time_delay_step
        if maximum_time_step < 5 * self.time_delay_step:
            self.time_step_time_delay_step_factor = int(ceil((self.time_delay_step / maximum_time_step)))
            self.time_step = self.time_delay_step / self.time_step_time_delay_step_factor

        # otherwise use it as is
        else:
            self.time_step_time_delay_step_factor = None
            self.time_step = maximum_time_step

        # total time, i.e. length of time axis, should be such that it matches
        # (roughly) the resolution of grenouille trace wavelength resolution
        total_time = ((self.frequency_multiplier * self.grenouille_trace_center_wavelength)**2 
                      / ureg.speed_of_light 
                      / (self.frequency_multiplier * self.grenouille_trace_wavelength_step)
                     )

        self.time_axis_length = int(ceil((total_time / self.time_step) / 2)) * 2 + 1

        self._calculate_I_FROG()
        if initial_E is None:
            self.E = self._default_initial_E()
        else:
            self.E = initial_E

        for iteration in range(self.number_of_iterations):
            self.E_sig_tτ = self._calculate_E_sig()
            self.E_sig_ωτ = self._calculate_FT_E_sig()
            self.E_sig_ωτ = self._replace_FT_E_sig_magnitude()
            self.E_sig_tτ = self._calculate_inverse_FT_E_sig()
            self.E = self._calculate_next_E()
            self.E = self._center_and_zero_phase_E()

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
        E_t_length = int(ceil((self.pulse_duration / self.time_step) / 2)) * 2 + 1
        return (np.arange(E_t_length) - (E_t_length - 1) // 2) * self.time_step

    @property
    def τ(self) -> QuantityArray:
        return (np.arange(self.shape[1]) - ((self.shape[1] - 1) // 2)) * self.time_delay_step

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

    def _calculate_E_sig_from_E_with_interpolation(self, E: np.ndarray, 
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

    def _calculate_E_sig_from_E_exact_time_steps(self, E: np.ndarray) -> np.ndarray:
        """ Calculate E_sig(t, τ) = E(t) gate(E(t - τ))

            gate(E(t)) depends on the non-linear effect.

            Assumes time_step is an integer multiple of time_delay_step

        Parameters
        ----------
        E : complex E-field
        """
        
        E_on_E_sig_t_axis = self._pad_to_t_axis(E)

        # calculate gate(E), which depends on the nonlinear effect
        if self.nonlinear_effect == 'self_diffraction':
            gate_E = np.square(np.abs(E_on_E_sig_t_axis))
        elif self.nonlinear_effect == 'second_harmonic_generation':
            gate_E = E_on_E_sig_t_axis.copy()

        # calculate E_sig(t, τ) = E(t) gate(E(t - τ))

        return np.transpose(
            np.fromiter(
                (E_on_E_sig_t_axis * shift_1d_array(gate_E, num_time_steps_shift)
                 for num_time_steps_shift 
                 in (np.arange(len(self.τ)) - (len(self.τ) - 1) // 2) * self.time_step_time_delay_step_factor
                ), (np.complex128, len(self.t))
            )
        )

    def _calculate_Z_from_E(self, E: np.ndarray):
        """ Calculate sum(|E_sig(t,τ) - E(t)E(t-τ))|^2)

        At the step where E(t) is calculated from E_sig(t,τ) (after it has been
        inverse-FT'd from the modified E_sig(ω,τ) ), for the generalized_projection
        methods Z is the objective to minimize.

        """
        return np.sum(np.square(np.abs(self.E_sig_tτ - self._calculate_E_sig(self._pad_to_t_axis(E)))))

    def _calculate_dZdE_from_E_exact_time_steps(self, E: np.ndarray):

        pad_len_on_either_side = (len(self.t) - len(E)) // 2
        E_on_t_axis = self._pad_to_t_axis(E)        
        assert len(self.t) - len(E) == 2 * pad_len_on_either_side

        # dZ/dRe(E[l])
        dZ_dReE = np.zeros(len(E_on_t_axis))
        # dZ/dIm(E[l])
        dZ_dImE = np.zeros(len(E_on_t_axis))

        tsf = self.time_step_time_delay_step_factor

        # for l in range(len(E)):
        #     for j in range(self.E_sig_tτ.shape[1]):
        #         # j is index to self.E_sig_tτ, while τj is self.τ[j] / self.time_delay_step
        #         τj = j - ((self.shape[1] - 1) // 2)
        #         assert abs(τj - self.τ[j] / self.time_delay_step) < 1e-6

        #         if 0 <= l - tsf * τj < len(E):
        #             diff_E_sig = self.E_sig_tτ[l, j] - E[l] * E[l - tsf * τj]
        #             dZ_dReE[l] += (  2 * np.real(diff_E_sig) * (-np.real(E[l - tsf * τj]))
        #                            + 2 * np.imag(diff_E_sig) * (-np.imag(E[l - tsf * τj]))
        #                           )
        #             dZ_dImE[l] += (  2 * np.real(diff_E_sig) * (+np.imag(E[l - tsf * τj]))
        #                            + 2 * np.imag(diff_E_sig) * (-np.real(E[l - tsf * τj]))
        #                           )

        #         if 0 <= l + tsf * τj < len(E):
        #             diff_E_sig = self.E_sig_tτ[l + tsf * τj, j] - E[l + tsf * τj] * E[l]
        #             dZ_dReE[l] += (  2 * np.real(diff_E_sig) * (-np.real(E[l + tsf * τj]))
        #                            + 2 * np.imag(diff_E_sig) * (-np.imag(E[l + tsf * τj]))
        #                           )
        #             dZ_dImE[l] += (  2 * np.real(diff_E_sig) * (+np.imag(E[l + tsf * τj]))
        #                            + 2 * np.imag(diff_E_sig) * (-np.real(E[l + tsf * τj]))
        #                           )

        for j in range(self.E_sig_tτ.shape[1]):
            # j is index to self.E_sig_tτ, while τj is self.τ[j] / self.time_delay_step
            τj = j - ((self.shape[1] - 1) // 2)
            assert abs(τj - self.τ[j] / self.time_delay_step) < 1e-6

            E_shifted = shift_1d_array(E_on_t_axis, tsf * τj)
            diff_E_sig = self.E_sig_tτ[:, j] - (E_on_t_axis * E_shifted)
            dZ_dReE += (  2 * np.real(diff_E_sig) * (-np.real(E_shifted))
                        + 2 * np.imag(diff_E_sig) * (-np.imag(E_shifted))
                       )
            dZ_dImE += (  2 * np.real(diff_E_sig) * (+np.imag(E_shifted))
                        + 2 * np.imag(diff_E_sig) * (-np.real(E_shifted))
                       )

            E_shifted = shift_1d_array(E_on_t_axis, -tsf * τj)
            diff_E_sig = shift_1d_array(self.E_sig_tτ[:, j], -tsf * τj) - E_shifted * E_on_t_axis
            dZ_dReE += (  2 * np.real(diff_E_sig) * (-np.real(E_shifted))
                        + 2 * np.imag(diff_E_sig) * (-np.imag(E_shifted))
                       )
            dZ_dImE += (  2 * np.real(diff_E_sig) * (+np.imag(E_shifted))
                        + 2 * np.imag(diff_E_sig) * (-np.real(E_shifted))
                       )

        # crop relevant section if E was padded at the start of this method
        if pad_len_on_either_side > 0:
            dZ_dReE = dZ_dReE[pad_len_on_either_side:-pad_len_on_either_side]
            dZ_dImE = dZ_dImE[pad_len_on_either_side:-pad_len_on_either_side]

        return dZ_dReE, dZ_dImE

    def _calculate_E_sig(self, E: np.ndarray = None):
        # select E_sig calculation method depending on whether time step is 
        # integer multiple of time_delay_step

        if E is None:
            E = self.E

        if self.time_step_time_delay_step_factor is None:
            return self._calculate_E_sig_from_E_with_interpolation(E)
        else:
            return self._calculate_E_sig_from_E_exact_time_steps(E)

    def _calculate_FT_E_sig(self, E_sig_tτ = None):
        if E_sig_tτ is None:
            E_sig_tτ = self.E_sig_tτ
        return np.fft.fftshift(np.fft.fft(E_sig_tτ, axis=0), axes=(0,))

    def _replace_FT_E_sig_magnitude(self):
        return self.E_sig_ωτ * np.sqrt(self.I_FROG) / np.clip(np.abs(self.E_sig_ωτ), 1e-15, np.inf)

    def _calculate_inverse_FT_E_sig(self, E_sig_ωτ = None):
        if E_sig_ωτ is None:
            E_sig_ωτ = self.E_sig_ωτ
        return np.fft.ifft(np.fft.ifftshift(E_sig_ωτ, axes=(0,)), axis=0)

    def _calculate_next_E(self, E_sig_tτ = None, E_current = None):

        if E_sig_tτ is None:
            E_sig_tτ = self.E_sig_tτ

        if E_current is None:
            E_current = self.E

        def _calculate_next_E_by_integration():
            @ureg.wraps('=A*B', ('=A', '=B', None))
            def trapz_ua(Y, x, axis=0):
                return np.trapz(Y, x, axis=axis)
            E = np.interp(self.E_t,   self.t, trapz_ua(E_sig_tτ, self.τ, axis=1))
            
            if self.nonlinear_effect == 'self_diffraction':
                raise NotImplementedError()
            elif self.nonlinear_effect == 'second_harmonic_generation':
                E /= np.sqrt(trapz_ua(E, self.E_t))

            return E.m_as('')

        def _calculate_next_E_by_generalized_projection_along_gradient():

            if self.time_step_time_delay_step_factor is None:

                E_sig_tτ_at_E_t = np.transpose(
                    np.fromiter((np.interp(self.E_t,  self.t, E_sig_tτ_col, left=0.0, right=0.0)
                                for E_sig_tτ_col in E_sig_tτ.T
                                ), (np.complex128, len(self.E_t))
                            )
                )

                def resid(E_est):
                    E_sig_est = self._calculate_E_sig_from_E(E_est, E_t=self.E_t, E_sig_t=self.E_t)
                    return np.abs(E_sig_tτ_at_E_t - E_sig_est).flatten()

                REAL = 1; IMAG = 2; 
                dEi = 1e-6
                Z0 = np.sum(np.square(resid(E_current)))

                def dZdEi(i, real_or_imag):
                    dE = np.zeros(len(E_current), np.complex128); 
                    if real_or_imag == REAL: 
                        dE[i] += dEi
                    elif real_or_imag == IMAG:
                        dE[i] += 1j * dEi
                    Z = np.sum(np.square(resid(E_current + dE)))
                    return (Z - Z0) / dEi

                dZdE = (  np.array([dZdEi(i, REAL) for i in range(len(E_current))]) 
                        + 1j * np.array([dZdEi(i, IMAG) for i in range(len(E_current))]) 
                    )

                def loss(beta):
                    return resid(E_current + beta * dZdE)

                solution = least_squares(loss, 0.0)
                beta = solution.x

                return E_current + beta * dZdE

            else:  # self.time_step_time_delay_step_factor is not None

                dZdReE, dZdImE = self._calculate_dZdE_from_E_exact_time_steps(E_current)

                def fun(β):
                    return self._calculate_Z_from_E(E_current + β * (dZdReE + 1j * dZdImE))
                
                def jac(β):
                    # dZ/dβ = dZ/dE|E=E' . dZ/dE|E=E0, where E' = E0 + β * (dZdE|E=E0)
                    dZ_dE_Re_Ep, dZ_dE_Im_Ep = self._calculate_dZdE_from_E_exact_time_steps(E_current + β * (dZdReE + 1j * dZdImE))
                    return np.dot(dZ_dE_Re_Ep, dZdReE) + np.dot(dZ_dE_Im_Ep, dZdImE)

                solution = minimize(fun, 0.0, jac=jac)
                beta = solution.x

                return E_current + beta * (dZdReE + 1j * dZdImE)

        def _calculate_next_E_by_generalized_projection_full_minimization():
            # def loss(E_split_in_real_imag):
            #     E_real, E_imag = E_split_in_real_imag.reshape((2, -1))
            #     E_sig_est = self._calculate_E_sig(E_real + 1j * E_imag)

            #     return np.abs(E_sig_tτ - E_sig_est).flatten()

            # solution = least_squares(loss, 
            #                          np.concatenate([E_current.real, E_current.imag]),
            #                          method='lm'
            #                         )

            def fun(E_split_in_real_imag):
                E_real, E_imag = E_split_in_real_imag.reshape((2, -1))
                return self._calculate_Z_from_E(E_real + 1j * E_imag)

            def jac(E_split_in_real_imag):
                E_real, E_imag = E_split_in_real_imag.reshape((2, -1))
                return np.concatenate(self._calculate_dZdE_from_E_exact_time_steps(E_real + 1j * E_imag))

            solution = minimize(fun, np.concatenate([self.E.real, self.E.imag]), jac=jac, method='Newton-CG')

            E_real, E_imag = solution.x.reshape((2, -1))

            return E_real + 1j * E_imag

        return {'integration': _calculate_next_E_by_integration,
                'generalized_projection/along_gradient': _calculate_next_E_by_generalized_projection_along_gradient,
                'generalized_projection/full_minimization': _calculate_next_E_by_generalized_projection_full_minimization,
               }[self.calculate_next_E_method]()

    def _center_and_zero_phase_E(self, E = None):

        if E is None:
            E = self.E

        # center E around t = 0
        center_of_mass = (self.E_t * np.abs(E)).sum() / np.abs(E).sum()
        E = np.interp(self.E_t,  self.E_t - center_of_mass, E, left=0.0, right=0.0)
        
        # zero the phase at t = 0
        E *= np.exp(-1j * np.angle(E[(len(E) - 1) // 2]))

        return E

    def _pad_to_t_axis(self, E: np.ndarray):
        """
        pad E(t) to be on self.t time axis. 
        Verifies that the E(t) axis lies on the self.t axis as expected.
        """
        pad_len_on_either_side = (len(self.t) - len(E)) // 2
        E_on_t_axis = np.pad(E, ((pad_len_on_either_side, pad_len_on_either_side)))
        assert len(E_on_t_axis) == len(self.t)

        return E_on_t_axis

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


def shift_1d_array(arr: np.ndarray, num_steps: int) -> np.ndarray:
    """ Roll items in a 1D array num_steps to the right, padding with zeros
    
    A version of the np.roll function (for 1D only) where instead of wrapping
    values around to the front (or back), zeros are inserted.

    Parameters
    ----------
    arr : 1d np.ndarray
    num_steps : int
        positive for right shift, negative for left shift

    """
    if arr.ndim != 1:
        raise ValueError("expected 1d array")
    if num_steps > 0:
        return np.roll(np.pad(arr, (0, num_steps)), num_steps)[:-num_steps]
    else:
        return np.roll(np.pad(arr, (-num_steps, 0)), num_steps)[-num_steps:]


if __name__ == "__main__":
    pass
