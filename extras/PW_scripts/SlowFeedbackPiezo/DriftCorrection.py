"""This script is for the correction of the
drift over a period of time"""

# region Imports
#Imports for the class
import os
from collections import deque
from os.path import defpath

import yaml
import time
import numpy as np
import pandas as pd
import shelve
import sys
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import fsolve
from AsyncCamera import CameraCentroid
sys.path.append('../')
from geecs_python_api.controls.interface import GeecsDatabase
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
import asyncio
import signal
import functools

import collections
import statistics
# endregion

"""async def main(targetx, targety):

    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info("Bella")
    camera = GeecsDevice('CAM-HPD-CCD')
    monitor = CameraCentroid(camera)

    await monitor.start_monitoring(interval=0.2)

    print("Press enter to stop the loop")

    while True:
        mean = await monitor.get_running_mean()
        print(f"Current Mean: {mean}")

        if abs(mean[0] - targetx) > 5:
            #Add moving logic here
            print("Moving the x axis of mirror to a certain position")

        if abs(mean[1] - targetx) > 5:
            #Add moving logic here
            print("Moving the y axis of mirror to a certain position")

        await asyncio.sleep(1)

    await monitor.stop_monitoring()

    camera.close

if __name__ == "__main__":
    asyncio.run(main(500, 500))"""


async def main():
    # Create an event to signal when to stop
    stop_event = asyncio.Event()

    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info("Bella")
    camera = GeecsDevice('CAM-HPD-CCD')
    monitor = CameraCentroid(camera)

    # Define target values
    targetx = 100  # Replace with your actual target value
    targety = 100  # Replace with your actual target value

    # Start listening for Enter key in a separate executor
    loop = asyncio.get_running_loop()
    input_future = loop.run_in_executor(
        None,
        functools.partial(input, "Press Enter to stop monitoring...\n")
    )

    # Create a task that will set the event when Enter is pressed
    async def wait_for_input():
        await input_future
        print("\nEnter key pressed. Stopping monitoring...")
        stop_event.set()

    input_task = asyncio.create_task(wait_for_input())

    await monitor.start_monitoring(interval=0.2)

    print("Monitoring started. Press Enter at any time to stop.")

    try:
        while not stop_event.is_set():
            mean = await monitor.get_running_mean()
            print(f"Current Mean: {mean}")

            if abs(mean[0] - targetx) > 5:
                # Add moving logic here
                print("Moving the x axis of mirror to a certain position")

            if abs(mean[1] - targety) > 5:
                # Add moving logic here
                print("Moving the y axis of mirror to a certain position")

            # Wait for a short time or until the stop event is set
            try:
                await asyncio.wait_for(stop_event.wait(), 0.5)
            except asyncio.TimeoutError:
                # This is expected - just continue the loop
                pass

        print("Loop exited. Cleaning up...")
    finally:
        # Always ensure we stop monitoring and cancel the input task
        input_task.cancel()
        await monitor.stop_monitoring()
        camera.close
        print("Monitoring stopped and camera closed.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
        print("Script completed successfully")
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)