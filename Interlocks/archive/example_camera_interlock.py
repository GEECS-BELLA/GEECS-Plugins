"""
Example 1: Using InterlockServer with cameras (original use case)
"""
import sys
import time
sys.path.append('../../')

from geecs_python_api.controls.interface import GeecsDatabase
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from interlock_server import InterlockServer, create_camera_check

# Setup
GeecsDevice.exp_info = GeecsDatabase.collect_exp_info("Bella")
cam1 = GeecsDevice('CAM-PL1-TapeDrivePointing')
cam1.subscribe_var_values(['MaxCounts'])

# Create server
server = InterlockServer(host="127.0.0.1", port=5001)

# Register camera monitor
camera_check = create_camera_check(cam1, 'MaxCounts', threshold=140, above=True)
server.register_monitor("CAM-PL1-TapeDrivePointing", camera_check, interval=0.5)

# Start server
server.start()

print("Camera interlock server running. Press Ctrl+C to stop.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    server.stop()
