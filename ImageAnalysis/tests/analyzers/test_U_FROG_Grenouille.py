from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Type

import numpy as np

from image_analysis.analyzers.U_FROG_Grenouille import U_FROG_GrenouilleImageAnalyzer, U_FROG_GrenouilleAWSLambdaImageAnalyzer
from image_analysis.algorithms.grenouille import create_test_pulse_and_trace
from image_analysis import ureg, Q_
if TYPE_CHECKING:
    from image_analysis.algorithms.grenouille import GrenouillePulseAndTrace

class TestU_FROG_GrenouilleImageAnalyzer(unittest.TestCase):
    def _test_analyze_image(self, image_analyzer_class: Type[U_FROG_GrenouilleImageAnalyzer]):

        test_pulse_and_trace: GrenouillePulseAndTrace = create_test_pulse_and_trace()
        image_analyzer = image_analyzer_class(
            pulse_center_wavelength = test_pulse_and_trace.pulse_center_wavelength,
            grenouille_trace_center_wavelength = test_pulse_and_trace.grenouille_trace_center_wavelength,
            grenouille_trace_wavelength_step = test_pulse_and_trace.grenouille_trace_wavelength_step,
            grenouille_trace_time_delay_step = test_pulse_and_trace.grenouille_trace_time_delay_step,
        )

        test_pulse_t = test_pulse_and_trace.pulse_time_step * (np.arange(len(test_pulse_and_trace.pulse_E)) - len(test_pulse_and_trace.pulse_E) / 2)

        analyzer_results = image_analyzer.analyze_image(test_pulse_and_trace.grenouille_trace)
        E_recovered_t = analyzer_results['pulse_E_field_AU']['time_fs'] * ureg.femtosecond
        E_recovered = analyzer_results['pulse_E_field_AU']['E_real_AU'] + 1j * analyzer_results['pulse_E_field_AU']['E_imag_AU']

        E_recovered_at_test_pulse_t = np.interp(test_pulse_t,  E_recovered_t, E_recovered, left=0.0, right=0.0)
        self.assertLess(np.max(np.abs(E_recovered_at_test_pulse_t - test_pulse_and_trace.pulse_E)), 0.05)

    def test_analyze_image_local(self):
        """Test U_FROG_GrenouilleImageAnalyzer.analyze_image()
        """
        self._test_analyze_image(U_FROG_GrenouilleImageAnalyzer)

    def test_analyze_image_aws_lambda(self):
        """Test U_FROG_GrenouilleAWSLambdaImageAnalyzer.analyze_image()
        """
        self._test_analyze_image(U_FROG_GrenouilleAWSLambdaImageAnalyzer)

if __name__ == "__main__":
    unittest.main()
