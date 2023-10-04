"""
Mon 8-7-2023

Module for containing functions to both calculate the energy axis calibration and retrieve the previously found fit
parameters.  Works for both the HiResMagSpec and the BCaveMagSpec

@Chris
"""

import numpy as np


def return_energy_axis(pixel_axis, mag_spec_name):
    if mag_spec_name == 'hires':
        energy_axis = return_hires_energy_axis(pixel_axis)
    elif mag_spec_name == 'acave3':
        energy_axis = return_acave3_energy_axis(pixel_axis)
    else:
        print("Invalid Mag Spec Name for Energy Axis!")
        energy_axis = pixel_axis
    return energy_axis


def return_hires_energy_axis(pixel_axis):
    """
    This is calibrated using the 825 mT settings to get a range of 89.368 to 114.861 MeV across 1287 pixels
    """
    mag_field = '825mT'
    if mag_field == '800mT':
        p0 = 8.66599527e+01
        p1 = 1.78007126e-02
        p2 = 1.10546749e-06
    elif mag_field == '825mT':
        p0 = 8.93681013e+01
        p1 = 1.83568540e-02
        p2 = 1.14012869e-06
    else:
        p0 = 0
        p1 = 0
        p2 = 0
    energy_axis = p0 + p1 * pixel_axis + p2 * np.power(pixel_axis, 2)
    return energy_axis


def return_acave3_energy_axis(pixel_axis):
    """
    Calibrated using the energy axis at 251 mT for the range of 82.167 to 605.241 MeV across 1049 pixels
    """
    mag_field = '251mT'
    if mag_field == '251mT':
        coefs_array = [1.06977145e-34, -5.88860941e-31, 1.40247546e-27, -1.88266185e-24, 1.55532229e-21,
                       -8.08854030e-19, 2.58007035e-16, -4.47996520e-14, 2.00337342e-12, 7.02713877e-10,
                       -7.32308369e-08, 7.28657608e-05, 8.09162396e-02, 8.21672776e+01]  # 0th order last
    else:
        coefs_array = np.zeros(14)
    energy_axis = np.poly1d(coefs_array)(pixel_axis)
    return energy_axis
