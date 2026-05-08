"""This is the class for control of the
thorlabs piezo motor controller."""

# region Imports
#Imports for the class
import os
from collections import deque
from os.path import defpath

import yaml
import time
import numpy as np
import pandas as pd
import shelve
import sys
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import fsolve
sys.path.append('../../')
from geecs_python_api.controls.interface import GeecsDatabase
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
import asyncio
import collections
import statistics
# endregion

#region Main Code
class TLMDT69_Controller:

    def __init__(self, device):
        self.device = device

    def  move_axis(self, voltage, axis):
        """This will start monitoing the centroids async"""
        axis = axis.upper()
        command = axis+"_Voltage"
        self.device.set(command, voltage)
        time.sleep(3)

    def get_current_voltage(self, axis):
        axis = axis.upper()
        command = axis+"_Voltage"
        value = self.device.get(command)
        return value

#endregion
