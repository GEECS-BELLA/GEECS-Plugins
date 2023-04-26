import time
from geecs_api.experiment import HtuExp
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.HTU.diagnostics.phosphors import *
from geecs_api.devices.HTU.diagnostics.beam import BeamDiagnostics


# create experiment object
# htu = HtuExp()
exp_info = GeecsDatabase.collect_exp_info('Undulator')

# a1_phosphor = PhosphorA1(exp_info)
# u9_phosphor = PhosphorU9(exp_info)
e_beam = BeamDiagnostics(exp_info)
time.sleep(.1)

# do something
e_beam.u9_phosphor.screen.is_phosphor_inserted()

# cleanup connections
e_beam.cleanup()
for controller in e_beam.controllers:
    controller.cleanup()
