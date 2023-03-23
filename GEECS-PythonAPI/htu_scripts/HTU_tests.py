import time
from geecs_api.experiment import HtuExp


# create experiment object
htu = HtuExp()

# do something/nothing
# htu.laser.pump_laser.shutters.insert(1, 'North', sync=True)
# htu.e_transport.hexapod.set_position('Y', 15., exec_timeout=120, sync=True)

htu.shutdown()

# display some states
time.sleep(1)
print(f'Shutter state:\n\t{htu.laser.pump_laser.shutters.state}')
print(f'Stage state:\n\t{htu.jet.stage.state}')
# print(f'Pressure state:\n\t{htu.jet.pressure.state}')
# print(f'Seed shutter (Amp4) state:\n\t{htu.laser.seed_laser.amp4_shutter.state}')
# print(f'Pump shutters state:\n\t{htu.laser.pump_laser.shutters.state}')

htu.cleanup()
