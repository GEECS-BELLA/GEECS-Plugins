from __future__ import annotations
from typing import Optional, Union, TYPE_CHECKING
from io import BytesIO
from time import time_ns
import builtins

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

import boto3
import json

class U_FROG_GrenouilleImageAnalyzer(ImageAnalyzer):
    """ ImageAnalyzer for Grenouille
    """
    def __init__(self, 
                 pulse_center_wavelength: Quantity = Q_(800, 'nm'),
                 grenouille_trace_center_wavelength: Quantity = Q_(410.4, 'nm'),
                 grenouille_trace_wavelength_step: Quantity = Q_(0.157, 'nm'),
                 grenouille_trace_time_delay_step: Quantity = Q_(0.931, 'fs'),
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
        grenouille_trace_time_delay_step: [time] Quantity
            resolution of time_delay (tau) axis (1, horizontal). The center 
            of the axis should represent tau = 0
        """

        super().__init__()

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

    def _compile_analysis_results(self, pulse: np.ndarray, E_t: Quantity) -> dict:
        # measurements on pulse
        # FWHM 
        power = np.square(np.abs(pulse))
        peak_i = np.argmax(power)
        fwhms_i, _, _, _ = peak_widths(power, [peak_i])
        fwhm = fwhms_i[0] * self.grenouille_retrieval.time_step

        # shortest interval containing 76% of power, which is equal to FWHM for a Gaussian pulse
        fwhm_area = erf(np.sqrt(np.log(2))) # approx 76%
        cumulative_power_normalized = np.cumsum(power) / np.sum(power)
        time_intervals = np.interp(cumulative_power_normalized + fwhm_area,  cumulative_power_normalized, E_t, left=Q_(np.nan, 'fs'), right=Q_(np.nan, 'fs')) - E_t
        _76_percent_interval = np.nanmin(time_intervals)

        return {'pulse_E_field_AU': np.rec.fromarrays([E_t.m_as('fs'), pulse.real, pulse.imag], names=["time_fs", "E_real_AU", "E_imag_AU"]),
                'peak_power_AU': np.max(np.square(np.abs(pulse))),
                'fwhm_fs': fwhm.m_as('fs'),
                '76%_interval_fs': _76_percent_interval.m_as('fs'), 
               }


    def analyze_image(self, grenouille_trace: Array2D, auxiliary_data: dict | None = None) -> dict[str, Union[float, NDArray]]:
        pulse: NDArray[np.complex128] = self.grenouille_retrieval.calculate_pulse(grenouille_trace, 
                                            grenouille_trace_center_wavelength=self.grenouille_trace_center_wavelength,
                                            grenouille_trace_wavelength_step=self.grenouille_trace_wavelength_step,
                                            time_delay_step=self.grenouille_trace_time_delay_step,
                                        )
        
        return self._compile_analysis_results(pulse, self.grenouille_retrieval.E_t)
    


class U_FROG_GrenouilleAWSLambdaImageAnalyzer(U_FROG_GrenouilleImageAnalyzer):
    """ ImageAnalyzer for U_FROG_Grenouille that uses AWS Lambda function
    """

    # this ImageAnalyzer's analyze_image sends a request to AWS Lambda and waits
    # for its response, so it should be run asynchronously
    run_analyze_image_asynchronously = True

    def analyze_image(self, grenouille_trace: Array2D, auxiliary_data: Optional[dict] = None) -> dict[str, Union[float, NDArray]]:
        boto3_session = boto3.Session()
        
        s3 = boto3_session.resource('s3')
        lambda_client = boto3_session.client('lambda')

        s3_key = f"temp/calculate_pulse_from_grenouille_trace/{time_ns():d}.npy"
        with BytesIO() as f:
            np.save(f, grenouille_trace, allow_pickle=False)
            f.seek(0)
            s3.Bucket("tausystems-taumeasurement-image").upload_fileobj(f, s3_key)

        body = { # just test image and parameters for now
            's3_key': s3_key,  
            'grenouille_trace_center_wavelength_nm': self.grenouille_trace_center_wavelength.m_as('nanometer'),
            'grenouille_trace_wavelength_step_nm': self.grenouille_trace_wavelength_step.m_as('nanometer'),
            'time_delay_step_fs': self.grenouille_trace_time_delay_step.m_as('femtosecond'),
            'pulse_duration_fs': 800,
            'pulse_center_wavelength_nm': 800,
            'max_computation_time_sec': 30
        }

        payload = {
            "body": json.dumps(body),
            "version": "2.0",
        }

        response = lambda_client.invoke(
            FunctionName="arn:aws:lambda:us-west-1:460578213037:function:calculate_pulse_from_grenouille_trace",
            Payload=json.dumps(payload).encode('utf-8'),
            InvocationType='RequestResponse',  # It's not the Lambda invocation that's asynchronous, just analyze_image()
        )

        result = json.loads(response['Payload'].read())

        if 'errorType' in result:
            raise getattr(builtins, result['errorType'])(result['errorMessage'])

        pulse = np.array([e_real + 1j * e_imag for e_real, e_imag in result['pulse_E_field_AU_real_imag']])
        E_t = Q_(np.arange(result['time_fs']['len']) * result['time_fs']['step'] + result['time_fs']['start'], 'femtosecond')

        return self._compile_analysis_results(pulse, E_t)
