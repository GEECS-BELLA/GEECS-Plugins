import time
from geecs_api.experiment import HtuExp


# create experiment object
htu = HtuExp()

# do something/nothing
time.sleep(1)
htu.shutdown()

# display some states
print(f'Stage state:\n\t{htu.jet.stage.state}')
print(f'Pressure state:\n\t{htu.jet.pressure.state}')
print(f'Seed shutter (Amp4) state:\n\t{htu.laser.seed_laser.amp4_shutter.state}')
print(f'Pump shutters state:\n\t{htu.laser.pump_laser.shutters.state}')
