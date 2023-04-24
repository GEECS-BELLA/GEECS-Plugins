import time
from geecs_api.experiment import HtuExp


# create experiment object
htu = HtuExp()

time.sleep(1)

# display some states
print(f'Stage state:\n\t{htu.jet.stage.state}')

# cleanup connections
htu.cleanup()
