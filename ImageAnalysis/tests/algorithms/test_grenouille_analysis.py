from __future__ import annotations

import unittest

import numpy as np

from image_analysis.algorithms.grenouille import GrenouilleRetrieval
from image_analysis import ureg, Q_

class GrenouilleAnalysisTestCase(unittest.TestCase):
   
    def test_full_calculate_pulse(self):

        gr = GrenouilleRetrieval(calculate_next_E_method='integration')

        # ## Calculate sample grenouille trace from example E field pulse
        E_time_step = Q_(5, 'fs')
        E_t = (np.arange(127) - 64) * E_time_step
        E = np.exp(-1/2 * np.square(E_t) / Q_(30, 'fs')**2) * np.exp(1j * E_t * 2*np.pi / Q_(200, 'fs'))
        E = E.m_as('')

        grenouille_trace_center_wavelength = Q_(400, 'nm')
        grenouille_trace_wavelength_step = Q_(0.7, 'nm')
        grenouille_trace_time_delay_step = Q_(2.1, 'fs')
        grenouille_trace_shape = (80, 81)

        grenouille_trace = gr.simulate_grenouille_trace(E, E_time_step, grenouille_trace_shape, 
                               grenouille_trace_center_wavelength=grenouille_trace_center_wavelength, 
                               grenouille_trace_wavelength_step=grenouille_trace_wavelength_step, 
                               grenouille_trace_time_delay_step=grenouille_trace_time_delay_step,
                           )


        # ## Recover original pulse with integration method
        gr.calculate_next_E_method = 'integration'
        E_recovered = gr.calculate_pulse(grenouille_trace, 
                                         grenouille_trace_center_wavelength=grenouille_trace_center_wavelength, 
                                         grenouille_trace_wavelength_step=grenouille_trace_wavelength_step, 
                                         time_delay_step=grenouille_trace_time_delay_step,
                                        )

        E_recovered_at_E_t = np.interp(E_t,  gr.E_t, E_recovered, left=0.0, right=0.0)
        self.assertLess(np.max(np.abs(E_recovered_at_E_t - E)), 0.05)

        # ## check generalized_projection/along_gradient method
        gr.calculate_next_E_method = 'generalized_projection/along_gradient'
        E_recovered = gr.calculate_pulse(grenouille_trace, 
                                        grenouille_trace_center_wavelength=grenouille_trace_center_wavelength, 
                                        grenouille_trace_wavelength_step=grenouille_trace_wavelength_step, 
                                        time_delay_step=grenouille_trace_time_delay_step,
                                        )
        E_recovered_at_E_t = np.interp(E_t,  gr.E_t, E_recovered, left=0.0, right=0.0)
        self.assertLess(np.max(np.abs(E_recovered_at_E_t - E)), 0.05)


class UtilsTestCase(unittest.TestCase):
    def test_pad_to_t_axis(self):
        gr = GrenouilleRetrieval()
        gr.time_axis_length = 7
        padded_array = gr._pad_to_t_axis(np.array([1,2,3]))
        self.assertEqual(padded_array[2], 1)
        self.assertEqual(padded_array[3], 2)

if __name__ == "__main__":
    unittest.main()
