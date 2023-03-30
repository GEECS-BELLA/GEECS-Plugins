import time
from geecs_api.experiment import HtuExp


# create experiment object
htu = HtuExp()

# do something/nothing
# htu.jet.stage.set_position('Y', -5.5)
# htu.laser.pump_laser.shutters.insert(1, 'North', sync=True)
# htu.transport.hexapod.set_position('Y', 15., exec_timeout=120, sync=True)
# htu.diagnostics.e_beam.screens.remove_TC_phosphor()
# htu.diagnostics.e_beam.screens.insert_TC_phosphor()


time.sleep(1)
# f'HTU shutdown: {htu.shutdown()}'

# display some states
print(f'Stage state:\n\t{htu.jet.stage.state}')

# cleanup connections
htu.cleanup()
