# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 10:44:15 2023

@author: Reinier van Mourik, TAU Systems
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NewType

from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity

if TYPE_CHECKING:
    from pint import Quantity
    QuantityArray = NewType("QuantityArray", Quantity)

import numpy as np
from scipy.linalg import toeplitz

class GrenouilleRetrieval:
    """
    """

    def __init__(self, 
                 wavelength_center: Quantity = Q_(400, 'nm'),
                 wavelength_step: Quantity = Q_(-0.0798546, 'nm'),
                 time_step: Quantity = Q_(0.893676, 'fs'),
                 next_E_sig_method: str = "inverse_FT",
                ):
        
        self.wavelength_center = wavelength_center
        self.wavelength_step = wavelength_step
        self.time_step = time_step

    def calculate_pulse(self, grenouille_trace: np.ndarray):
        self.shape = grenouille_trace.shape

    @property
    def t(self) -> QuantityArray:
        return np.arange(self.shape[0]) * self.time_step
    
    @property
    def tau(self) -> QuantityArray:
        return (np.arange(self.shape[0]) - ((self.shape[0] - 1) / 2)) * self.time_step

    @property
    def omega(self) -> QuantityArray:
        return np.fft.fftfreq(self.shape[1])

    def _calculate_E(self):
        self.E = np.trapz(self.E_sig, self.tau, axis=1)
    
    def _calculate_E_sig(self):
        pass

    def _calculate_FT_E_sig(self):
        pass

    def _replace_FT_E_sig_magnitude(self):
        pass

    def _calculate_next_E_sig(self):
        pass

    
    
