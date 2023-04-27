import time
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.diagnostics.ebeam_diagnostics import EBeamDiagnostics


# create experiment object
# htu = HtuExp()
GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

e_beam = EBeamDiagnostics()
time.sleep(.1)

# do something

# cleanup connections
e_beam.cleanup()
for controller in e_beam.controllers:
    controller.cleanup()
