import time
import numpy as np
import pandas as pd
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.diagnostics.ebeam_diagnostics import EBeamDiagnostics
from geecs_api.devices.HTU.diagnostics import UndulatorStage


# create experiment object
# htu = HtuExp()
GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

# e_beam = EBeamDiagnostics()
velmex = UndulatorStage()
time.sleep(.1)

# do something
# print(f'Velmex position: {velmex.get_position()}')
velmex.set_position(station=3, diagnostic='energy')

# cleanup connections
# e_beam.cleanup()
# for controller in e_beam.controllers:
#     controller.cleanup()
velmex.cleanup()
