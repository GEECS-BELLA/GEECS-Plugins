from geecs_python_api.controls.interface import GeecsDatabase
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.controls.devices.HTU.diagnostics.cameras import Camera


# parameters
local: bool = False
average: int = 30
exp_name: str = 'Undulator'

c_names = ['UC_TC_Phosphor', 'UC_DiagnosticsPhosphor', 'UC_Phosphor1',
           'UC_ALineEbeam1', 'UC_ALineEBeam2', 'UC_ALineEBeam3',
           'UC_VisaEBeam1', 'UC_VisaEBeam2', 'UC_VisaEBeam3',
           'UC_VisaEBeam4', 'UC_VisaEBeam5', 'UC_VisaEBeam6',
           'UC_VisaEBeam7', 'UC_VisaEBeam8', 'UC_VisaEBeam9']


# create cameras
GeecsDevice.exp_info = GeecsDatabase.collect_exp_info(exp_name)
cameras = [Camera(device_name=geecs_name) for geecs_name in c_names]
[cam.subscribe_var_values() for cam in cameras]


# collect backgrounds
if local:
    [cam.save_local_background(n_images=average) for cam in cameras]
else:
    Camera.save_multiple_backgrounds(cameras, exec_timeout=30.)


# close
[cam.close() for cam in cameras]
