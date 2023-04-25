import time
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.HTU.diagnostics.cameras import CameraTC, CameraDC, CameraP1, CameraA1, CameraA2, CameraA3


# create cameras
exp_info = GeecsDatabase.collect_exp_info('Undulator')

cameras = [CameraDC(exp_info), CameraTC(exp_info), CameraP1(exp_info),
           CameraA1(exp_info), CameraA2(exp_info), CameraA3(exp_info)]

for cam in cameras:
    cam.subscribe_var_values()

# collect backgrounds
time.sleep(0.5)
cameras[0].save_multiple_backgrounds(cameras, 30.)

# cleanup connections
for cam in cameras:
    cam.cleanup()
