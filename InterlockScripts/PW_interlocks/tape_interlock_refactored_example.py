import time
import logging
from dotenv import load_dotenv
import os

from geecs_python_api.controls.interlocks import (
    InterlockConstructor,
    ThresholdCheck,
    AlignmentCheck,
    MultiCheck,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
cam1 = interlock.add_device(
    "cam1",
    "CAM-PL1-TapeDrivePointing",
    [
        "MaxCounts",
        # "MeanCounts",
        # "acq_timestamp",
        # "Target.X",
        # "Target.Y",
        # "centroidx",
        # "centroidy",
    ],
)

# Optional: Configure device-specific settings
cam1.use_alias_in_TCP_subscription = False

# # Create condition builders for each check
# max_counts_check = ThresholdCheck(
#     interlock.device_group, "cam1", "MaxCounts", 4000, operator=">"
# )

# mean_counts_check = ThresholdCheck(
#     interlock.device_group, "cam1", "MeanCounts", 0, operator="<"
# )

# centroid_x_check = AlignmentCheck(
#     interlock.device_group, "cam1", "centroidx", "Target.X", tolerance=5
# )

# centroid_y_check = AlignmentCheck(
#     interlock.device_group, "cam1", "centroidy", "Target.Y", tolerance=5
# )

# # Combine conditions into single multi-check
# camera_multi_check = MultiCheck(
#     [
#         max_counts_check,
#         mean_counts_check,
#         centroid_x_check,
#         centroid_y_check,
#     ]
# )

max_counts_check = ThresholdCheck(interlock.device_group, "cam1", "MaxCounts", 170, operator=">")

# Register monitor with interlock server
# interlock.add_monitor("Camera Multi Check", camera_multi_check, interval=0.1)

interlock.add_monitor("Camera MaxCounts Check", max_counts_check, interval=0.1)

# Start server (subscribes devices, starts monitor threads)
interlock.start()

# Main loop - server runs in background threads
try:
    while True:
        # Can inspect monitor state if needed
        # state = interlock.get_monitor_state('Camera Multi Check')
        # print(f"Monitor state: {state}")
        time.sleep(0.02)

except KeyboardInterrupt:
    # Stop server (unsubscribes devices, stops threads)
    interlock.stop()
