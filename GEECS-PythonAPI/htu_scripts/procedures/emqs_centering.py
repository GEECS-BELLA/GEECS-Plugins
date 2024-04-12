import numpy as np
from pathlib import Path
from geecs_python_api.controls.interface import api_error
from geecs_python_api.controls.api_defs import ScanTag
from geecs_python_api.controls.experiment.htu import HtuExp
from geecs_python_api.analysis.scans import ScanData
from geecs_python_api.analysis.scans import ScanImages
from geecs_python_api.analysis.scans import ScanAnalysis


# Parameters - User
# ------------------------------------------------------------------------------
emq_currents = 7
q1_current_min, q1_current_max = -1.5, 1.5
q3_current_min, q3_current_max = -1.5, 1.5

steer_tolerance = 0.01
steer_currents = 7
s1x_current_min, s1x_current_max = -2., 2.
s1y_current_min, s1y_current_max = -2., 2.
s2x_current_min, s2x_current_max = -2., 2.
s2y_current_min, s2y_current_max = -2., 2.


# Parameters - Calculated
# ------------------------------------------------------------------------------
q1_A = np.linspace(q1_current_min, q1_current_max, emq_currents)
q3_A = np.linspace(q3_current_min, q3_current_max, emq_currents)

s1x_A = np.linspace(s1x_current_min, s1x_current_max, steer_currents)
s1y_A = np.linspace(s1y_current_min, s1y_current_max, steer_currents)
s1y_A[1::2] *= -1
s1_A = np.array([s1x_A, s1y_A])

s2x_A = np.linspace(s2x_current_min, s2x_current_max, steer_currents)
s2y_A = np.linspace(s2y_current_min, s2y_current_max, steer_currents)
s2y_A[1::2] *= -1
s2_A = np.array([s2x_A, s2y_A])


# Connect
# ------------------------------------------------------------------------------
htu = HtuExp(get_info=True)
htu.connect(laser=False, jet=False, diagnostics=False, transport=True)


# Disconnect
# ------------------------------------------------------------------------------
print(api_error)
htu.close()
print('done')
