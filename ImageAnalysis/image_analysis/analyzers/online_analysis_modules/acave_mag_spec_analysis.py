# -*- coding: utf-8 -*-
"""
Created on Mon Aug 8 2023

Module for performing various analyses on MagSpec Images.

This updated version is cleaner and streamlined to work on LabView. Most notably by not opening TDMS file or
the interpSpec to parse any additional information. Just make sure some of the constants are correct.

Author: Chris
"""

import numpy as np
import time
from scipy import optimize

from . import hi_res_mag_spec_energy_axis as energy_axis_lookup


def print_time(label, start_time, do_print=False):
    if do_print:
        print(label, time.perf_counter() - start_time)
    return time.perf_counter()


def analyze_image(input_image, input_params, do_print=False):
    return_image = None
    return_dictionary = None
    return_lineouts = None
    return return_image, return_dictionary, return_lineouts
