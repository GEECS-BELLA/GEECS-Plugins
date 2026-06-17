"""
Example monitoring script to print diagnostic info from interlocks.

This script connects to the InterlockConstructor (running in another process/script)
and prints diagnostic info about what's causing unsafe conditions.

Usage:
    1. Run tape_interlock_refactored_example.py in one terminal
    2. Run this script in another terminal to see real-time diagnostics
"""

import time
import logging

from geecs_python_api.controls.interlocks import (
    InterlockConstructor,
    ThresholdCheck,
    AlignmentCheck,
    MultiCheck,
)
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables
load_dotenv()
SERVER_IP = os.getenv("SERVER_IP")
SERVER_PORT = int(os.getenv("SERVER_PORT"))

# Create interlock constructor and set up the same monitors as the main script
interlock = InterlockConstructor(
    "BELLA",
    host=SERVER_IP,
    port=SERVER_PORT,
    staleness_timeout_ms=5000,
)

# Register camera device
cam1 = interlock.add_device(
    "cam1",
    "CAM-PL1-TapeDrivePointing",
    [
        "MaxCounts",
        "MeanCounts",
        "acq_timestamp",
        "Target.X",
        "Target.Y",
        "centroidx",
        "centroidy",
    ],
)

cam1.use_alias_in_TCP_subscription = False

# Create condition builders
max_counts_check = ThresholdCheck(
    interlock.device_group, "cam1", "MaxCounts", 1000, operator="<"
)

mean_counts_check = ThresholdCheck(
    interlock.device_group, "cam1", "MeanCounts", 0, operator="<"
)

centroid_x_check = AlignmentCheck(
    interlock.device_group, "cam1", "centroidx", "Target.X", tolerance=5
)

centroid_y_check = AlignmentCheck(
    interlock.device_group, "cam1", "centroidy", "Target.Y", tolerance=5
)

# Combine conditions
camera_multi_check = MultiCheck(
    [
        max_counts_check,
        mean_counts_check,
        centroid_x_check,
        centroid_y_check,
    ]
)

# Register monitor
interlock.add_monitor("Camera Multi Check", camera_multi_check, interval=0.1)

# Start server
interlock.start()

# Monitor loop - print diagnostic info when unsafe
print("=" * 80)
print("Monitoring interlock diagnostics...")
print("Press Ctrl+C to stop")
print("=" * 80)

try:
    last_state = None
    last_diagnostic = None

    while True:
        state = interlock.get_monitor_state("Camera Multi Check")
        diagnostic = interlock.get_diagnostic_info("Camera Multi Check")

        # Print only when state changes or diagnostic changes
        if state != last_state or diagnostic != last_diagnostic:
            timestamp = time.strftime("%H:%M:%S")
            
            if state:
                print(f"[{timestamp}] UNSAFE - {diagnostic}")
            else:
                print(f"[{timestamp}] SAFE")
            
            last_state = state
            last_diagnostic = diagnostic

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopping monitoring...")
    interlock.stop()
    print("Done!")
