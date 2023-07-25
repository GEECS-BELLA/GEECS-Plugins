"""
Tues, Jul 25, 2023

This is a function that can be called to turn an array of pixels on HiResMagSpec into an approximate array of energies.

This allows us to use the energy scale included in 'interpSpec' while also using the raw image for post-analysis.

@Chris
"""

import numpy as np

const_HiResMagSpec_resolution = 0.043 # mm/pixel

def GetEstimatedEnergyAxis(pixel_axis, interpSpec_energy_array):
    mm_axis = np.linspace(0, len(pixel_axis), len(pixel_axis)) * const_HiResMagSpec_resolution
    xbeam = mm_axis[int(len(mm_axis) / 2)]  # 36 #10.5 #24
    E0 = interpSpec_energy_array[0]
    Em = interpSpec_energy_array[-1]
    px = mm_axis[-1]
    Ebend = (E0 * Em * px) / (E0 * xbeam - xbeam * Em + Em * px)
    dnom = xbeam * E0 / (E0 - Ebend)
    energy_axis = dnom * Ebend / (mm_axis + (dnom - xbeam))

    cor0 = 1.17
    cor1 = 0.03
    cor2 = -0.007
    energy_axis = energy_axis + (cor1) * np.power(energy_axis - Ebend, 1) + (cor2) * np.power(
        energy_axis - Ebend, 2) + cor0

    return energy_axis
