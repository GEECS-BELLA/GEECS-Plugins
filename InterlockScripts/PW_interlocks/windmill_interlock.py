"""software interlock for the windmill PM."""

import logging
import os
import time

from dotenv import load_dotenv

# TODO:
# - Add which camera/device is failing to the diagnostic error message
# - Automatically try to reconnect if stale/offline
# - check clock difference between the udp trigger computer and computer their using
from geecs_python_api.controls.interlocks import (  # type: ignore
    InterlockConstructor,
    MultiCheck,
    ThresholdCheck,
)

from geecs_python_api.controls.interlocks.monitor_conditions import (
    ConfigurationChangeCheck,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",  # remove milliseconds for cleaner output
)

# Load environment variables for server config
load_dotenv()
SERVER_IP = os.getenv("SERVER_IP")
SERVER_PORT = int(os.getenv("SERVER_PORT"))

# Create interlock constructor
# If device doesn't send data for 5 seconds, all checks return unsafe
interlock = InterlockConstructor(
    "BELLA",
    host=SERVER_IP,
    port=SERVER_PORT,
    staleness_timeout_ms=5000,
)

# Register camera device and subscribe to required variables
tapePointing = interlock.add_device(
    "tapePointing",
    "CAM-PL1-TapeDrivePointing",
    [
        "MaxCounts",
        "MeanCounts",
        "MeanCountsRaw",
        "acq_timestamp",
        "Target.X",
        "Target.Y",
        "centroidx",
        "centroidy",
        "exposure",
        "gain",
    ],
)

AegaeonCam = interlock.add_device(
    "AegaeonCam",
    "CAM-2BL-Aegeaon",
    [
        "MaxCounts",
        "MeanCounts",
        "MeanValue",
        "MeanCountsRaw",
        "acq_timestamp",
        "Target.X",
        "Target.Y",
        "centroidx",
        "centroidy",
        "exposure",
        "gain",
    ],
)

# Optional: Configure device-specific settings
tapePointing.use_alias_in_TCP_subscription = False
AegaeonCam.use_alias_in_TCP_subscription = False

# # Create condition builders for each check
# max_counts_check = ThresholdCheck(
#     interlock.device_group, "cam2", "MaxCounts", 300, operator="<"
# )

max_counts_check = ThresholdCheck(
    interlock.device_group,
    "tapePointing",
    "MeanCounts",
    4,
    operator="<",  # check if meancounts goes below value, meaning no film
)


wiper_check = ThresholdCheck(
    interlock.device_group,
    "AegaeonCam",
    "MeanCounts",
    5,
    operator=">",  # check if meancounts goes above value, means wiper in beam path
)


tapePointing_config = ConfigurationChangeCheck(
    interlock.device_group, "tapePointing", ["exposure", "gain"]
)

AegaeonCam_config = ConfigurationChangeCheck(
    interlock.device_group, "AegaeonCam", ["exposure", "gain"]
)


# Combine conditions into single multi-check
camera_multi_check = MultiCheck(
    [
        max_counts_check,
        # mean_counts_check,
        # centroid_x_check,
        # centroid_y_check,
        wiper_check,
        tapePointing_config,
        AegaeonCam_config,
    ]
)

# max_counts_check = ThresholdCheck(
#     interlock.device_group, "cam1", "MaxCounts", 170, operator=">"
# )

# Register monitor with interlock server
interlock.add_monitor("Camera Multi Check", camera_multi_check, interval=0.1)

# interlock.add_monitor("Camera MaxCounts Check", max_counts_check, interval=0.1)

# Start server (subscribes devices, starts monitor threads)
interlock.start()
time.sleep(1)  # wait for devices to connect and subscribe
tapePointing_config.arm_interlock()
AegaeonCam_config.arm_interlock()

# Main loop - server runs in background threads
try:
    while True:
        # Can inspect monitor state and get diagnostic info
        state = interlock.get_monitor_state("Camera Multi Check")
        if state:  # unsafe
            diagnostic = interlock.get_diagnostic_info("Camera Multi Check")
            print(f"WARNING: Camera Multi Check - {diagnostic}")
        time.sleep(0.1)

except KeyboardInterrupt:
    # Stop server (unsubscribes devices, stops threads)
    interlock.stop()
