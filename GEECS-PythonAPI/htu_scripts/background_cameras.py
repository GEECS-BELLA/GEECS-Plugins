import time
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.HTU.diagnostics.cameras import Camera


# create cameras
exp_info = GeecsDatabase.collect_exp_info('Undulator')

cameras = \
    [Camera(device_name=geecs_name, exp_info=exp_info)
     for geecs_name
     in ['UC_ALineEbeam1', 'UC_ALineEBeam2', 'UC_ALineEBeam3',
         'UC_TC_Phosphor', 'UC_DiagnosticsPhosphor', 'UC_Phosphor1',
         'UC_VisaEBeam1', 'UC_VisaEBeam2', 'UC_VisaEBeam3',
         'UC_VisaEBeam4', 'UC_VisaEBeam5', 'UC_VisaEBeam6',
         'UC_VisaEBeam7', 'UC_VisaEBeam8', 'UC_VisaEBeam9']]

for cam in cameras:
    cam.subscribe_var_values()

# collect backgrounds
time.sleep(0.5)
cameras[0].save_multiple_backgrounds(cameras, 30.)

# cleanup connections
for cam in cameras:
    cam.cleanup()
