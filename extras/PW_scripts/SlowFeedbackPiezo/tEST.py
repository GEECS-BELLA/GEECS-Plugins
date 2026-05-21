""""This is the class for the asynchronous acquisition
of the camera"""

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

GeecsDevice.exp_info = GeecsDatabase.collect_exp_info("kHzLPA")
camera = GeecsDevice('kHz_Cam2-2_AstrellaNF_Online')

#camera.subscribe_var_values(['meancounts', 'FWHMx', 'FWHMy','centroidx', 'centroidy'])

for i in range(10):
    camera.subscribe_var_values(['centroidx', 'centroidy'])
    time.sleep(1)
    a = camera.state
    print(a)

    camera.subscribe_var_values(['meancounts', 'FWHMx', 'FWHMy'])
    time.sleep(1)
    b = camera.state
    print(b)

    if len(b)>4:
        print(b[""])

    time.sleep(1)