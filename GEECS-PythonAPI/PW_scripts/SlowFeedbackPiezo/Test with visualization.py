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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading
import collections
import statistics
import matplotlib
matplotlib.use('TkAgg')
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


# Shared data for communication between async loop and plotting
class SharedData:
    def __init__(self, max_points=100):
        self.current_mean = [0, 0]
        self.target = [0, 0]
        # Store history for plotting trails
        self.x_history = deque(maxlen=max_points)
        self.y_history = deque(maxlen=max_points)
        self.lock = threading.Lock()  # For thread-safe updates


# Global variables
stop_flag = False
shared_data = SharedData()


def input_thread():
    """Thread that waits for Enter key press"""
    global stop_flag
    input("Press Enter to stop monitoring...\n")
    stop_flag = True
    print("\nEnter key pressed. Stopping monitoring...")


def update_plot(frame, ax, scatter_current, scatter_target, line, shared_data):
    """Update function for the animation"""
    with shared_data.lock:
        current_x, current_y = shared_data.current_mean
        target_x, target_y = shared_data.target

        # Update history
        shared_data.x_history.append(current_x)
        shared_data.y_history.append(current_y)

        # Update scatter plots
        scatter_current.set_offsets([[current_x, current_y]])
        scatter_target.set_offsets([[target_x, target_y]])

        # Update line (trail)
        line.set_data(list(shared_data.x_history), list(shared_data.y_history))

        # Calculate distance from current to target
        distance = np.sqrt((current_x - target_x) ** 2 + (current_y - target_y) ** 2)

        # Set axis limits with target at center
        # Use either 50 units or the distance, whichever is greater
        axis_range = max(50, distance * 1.2)  # Add 20% margin to distance

        ax.set_xlim(target_x - axis_range, target_x + axis_range)
        ax.set_ylim(target_y - axis_range, target_y + axis_range)

        # Update title with position information
        ax.set_title(f'Centroid Tracking - Current: ({current_x:.1f}, {current_y:.1f}) '
                     f'Target: ({target_x:.1f}, {target_y:.1f}) Distance: {distance:.1f}')

    return scatter_current, scatter_target, line


def start_plotting_thread(shared_data):
    """Start the plotting in a separate thread"""
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))  # Square figure for equal scaling
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True)

    # Initial scatter plots
    scatter_current = ax.scatter([], [], color='blue', s=100, label='Current Position')
    scatter_target = ax.scatter([], [], color='red', marker='x', s=300, label='Target Position')

    # Line for the trail
    line, = ax.plot([], [], 'b-', alpha=0.3)

    # Add legend
    ax.legend()

    # Set initial view limits (will be updated in update_plot)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)

    # Set aspect ratio to be equal
    ax.set_aspect('equal')

    # Add grid lines at the target position
    def on_xlims_change(axes):
        target_x, target_y = shared_data.target
        ax.axhline(y=target_y, color='r', linestyle='--', alpha=0.1)
        ax.axvline(x=target_x, color='r', linestyle='--', alpha=0.1)

    ax.callbacks.connect('xlim_changed', on_xlims_change)

    # Create animation
    ani = FuncAnimation(
        fig,
        update_plot,
        fargs=(ax, scatter_current, scatter_target, line, shared_data),
        interval=100,
        blit=False
    )

    plt.tight_layout()
    plt.show()


async def main():
    global shared_data, stop_flag

    # Start the input thread to listen for Enter key
    threading.Thread(target=input_thread, daemon=True).start()

    # Start the plotting thread
    plotting_thread = threading.Thread(
        target=start_plotting_thread,
        args=(shared_data,),
        daemon=True
    )
    plotting_thread.start()

    # Initialize your hardware
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info("Bella")
    camera = GeecsDevice('CAM-HPD-CCD')
    monitor = CameraCentroid(camera)

    # Define target values and update shared data
    targetx = 820  # Replace with your actual target value
    targety = 615  # Replace with your actual target value

    with shared_data.lock:
        shared_data.target = [targetx, targety]

    await monitor.start_monitoring(interval=0.3)

    print("Monitoring started. Press Enter at any time to stop.")

    try:
        while not stop_flag:
            mean = await monitor.get_running_mean()

            # Update shared data for plotting
            with shared_data.lock:
                shared_data.current_mean = mean

            print(f"Current Mean: {mean}")

            if abs(mean[0] - targetx) > 5:
                # Add moving logic here
                print("Moving the x axis of mirror to a certain position")

            if abs(mean[1] - targety) > 5:
                # Add moving logic here
                print("Moving the y axis of mirror to a certain position")

            # Short sleep to allow checking the stop_flag frequently
            await asyncio.sleep(1)

        print("Loop exited. Cleaning up...")
    finally:
        # Always ensure we stop monitoring
        await monitor.stop_monitoring()
        camera.close
        print("Monitoring stopped and camera closed.")


if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        global stop_flag
        print('\nCtrl+C detected. Stopping...')
        stop_flag = True


    signal.signal(signal.SIGINT, signal_handler)

    try:
        asyncio.run(main())
        print("Script completed successfully")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)