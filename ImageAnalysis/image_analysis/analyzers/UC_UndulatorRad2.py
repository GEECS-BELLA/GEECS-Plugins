from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ..types import Array2D, QuantityArray, Quantity
from collections import namedtuple

from ..base import ImageAnalyzer
from ..utils import ROI

from . import ureg
Q_ = ureg.Quantity

import numpy as np

from skimage.filters import butterworth

class UC_UndulatorRad2ImageAnalyzer(ImageAnalyzer):

    # transmission curve has columns 
    #   wavelength [nm], transmission T [%]
    transmission_curve = np.loadtxt(Path(__file__).parent/'data'/'UC_UndulatorRad2_transmission.dat', delimiter=',')
    # make transmission a number instead of percentage
    transmission_curve[:, 1] /= 100

    # spectrometer wavelength resolution per pixel in the horizontal direction
    wavelength_resolution = Q_(0.400, 'nm')

    # pixel on the full image that corresponds to 0 wavelength. Nominal location
    # of 0th order spot.
    zero_wavelength_ix = 775

    @property
    def wavelength(self) -> QuantityArray:
        return (np.arange(2592) - self.zero_wavelength_ix) * self.wavelength_resolution

    # filter parameters
    # low_cutoff_freq_ratio = 0.0001
    # high_cutoff_freq_ratio = 0.022

    def __init__(self, 
                 roi: ROI = ROI(top=567, bottom=1048, left=1397, right=2245),
                 high_cutoff_freq_ratio: float = 0.022,
                ):
        self.background: Optional[float] = None
        self.roi: ROI = roi
        self.high_cutoff_freq_ratio = high_cutoff_freq_ratio

    def set_background(self, image: Array2D):
        """ 
        """
        self.background = np.vstack([image[:self.roi.top, :], image[self.roi.bottom:, :]]).mean()

    def analyze_image(self, image: Array2D) -> dict[str, float|NDArray]:
        """ Calculate metrics. 
        """

        assert image.shape == (2048, 2592)
        
        # crop image to region where we have a good transmission curve
        s_x = (Q_(353.9, 'nm') < self.wavelength) & (self.wavelength < Q_(487.6, 'nm'))
        wavelength_cropped = self.wavelength[s_x]
        image = image[self.roi.top:self.roi.bottom, s_x]

        # subtract background, and clip negative values to 0
        if self.background is not None:
            image = np.clip(image - self.background, 0)

        # correct by transmission function
        image /= np.interp(wavelength_cropped.m_as('nm'),  self.transmission_curve[:, 0], self.transmission_curve[:, 1])[None, :]

        # filter cropped image by lowpass filters
        image = butterworth(image, self.high_cutoff_freq_ratio, high_pass=False)       
        
        # ## now assess metrics
        # the total counts at each wavelength, in counts/[length]
        radiation_density: QuantityArray = image.sum(axis=0) / self.wavelength_resolution  

        # peak density
        RadiationPeak = namedtuple('RadiationPeak', ['wavelength', 'peak_radiation'])
        def find_peak_between(lower_wavelength: Quantity, upper_wavelength: Quantity) -> tuple[Quantity, Quantity]:
            s = (lower_wavelength <= wavelength_cropped) & (wavelength_cropped <= upper_wavelength)
            peak_radiation_density_i = np.argmax(radiation_density[s])
            return RadiationPeak(wavelength_cropped[s][peak_radiation_density_i], 
                                 radiation_density[s][peak_radiation_density_i]
                                ) 

        radiation_peaks: dict[str, RadiationPeak] = {
            'violet': find_peak_between(Q_(368, 'nm'), Q_(452, 'nm')),
            'blue': find_peak_between(Q_(452, 'nm'), Q_(506, 'nm')),
        }

        return {'radiation_sum': image.sum(),
                'radiation_peak_blue_wavelength_nm': radiation_peaks['blue'].wavelength.m_as('nm'),
                'radiation_peak_blue_density_counts_nm^-1': radiation_peaks['blue'].peak_radiation.m_as('nm'),
                'radiation_peak_violet_wavelength_nm': radiation_peaks['violet'].wavelength.m_as('nm'),
                'radiation_peak_violet_density_counts_nm^-1': radiation_peaks['violet'].peak_radiation.m_as('nm'),
               }
