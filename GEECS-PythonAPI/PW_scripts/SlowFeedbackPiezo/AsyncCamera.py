"""This is the class for the asynchronous acquisition
of the camera"""

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
sys.path.append('../../')
from geecs_python_api.controls.interface import GeecsDatabase
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
import asyncio
import collections
import statistics
# endregion

#region Main Code
class CameraCentroid:

    def __init__(self, camera_name, num_ave = 3):
        #self.GeecsDevice.exp_info = GeecsDatabase.collect_exp_info("Bella")
        #self.camera = GeecsDevice(str(camera_name))
        self.camera = camera_name
        self.camera.subscribe_var_values(['centroidx', 'centroidy'])
        self.num_ave = num_ave
        self.centroids = deque(maxlen= num_ave)
        self.running_mean = np.array([0.0, 0.0])
        self.is_running = False
        self._lock = asyncio.Lock()

    async def  start_monitoring(self, interval = 0.2):
        """This will start monitoing the centroids async"""
        self.is_running = True
        asyncio.create_task(self._acquisition_loop(interval))

    async def stop_monitoring(self):
        """This will stop the monioting loop"""
        self.is_running = False

    async def get_running_mean(self):
        """gets the current running mean"""
        async with self._lock:
            return self.running_mean.copy()

    async def get_centtroids(self):
        """gets all the stored centroid values"""
        async with self._lock:
            return list(self.centroids)

    async def _acquisition_loop(self, interval):
        while self.is_running:
            try:
                #We get the camera state
                a = self.camera.state


                #Check if data is valid
                if len(a) > 3:
                    b = list(a)
                    x_name = (b[-2])
                    y_name = (b[-1])
                    x = float(a[x_name])
                    y = float(a[y_name])
                    #update the centroids if the data is valid
                    await self._update_centroids([x, y])

            except Exception as e:
                print(f"Error acquiring centroids: {e}")

            #sleep for some duration
            await asyncio.sleep(interval)


    async def _update_centroids(self, centroid):
        """This function is responsible for updating the
        centroid array and calculating the running mean"""
        async with self._lock:
            self.centroids.append(centroid)
            if len(self.centroids) > 0:
                self.running_mean = np.mean(self.centroids, axis = 0)


#endregion

#region Test Code for HPD-CCD
async def main():
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info("kHzLPA")
    camera = GeecsDevice('kHz_Cam2-3_AstrellaMode')
    monitor = CameraCentroid(camera)

    await monitor.start_monitoring(interval=0.1)

    for i in range(100):

        mean = await monitor.get_running_mean()
        print(f"Current Mean: {mean}")

        await asyncio.sleep(0.1)

    await monitor.stop_monitoring()
    camera.close

    # Fix 3: Cancel any remaining tasks
    for task in asyncio.all_tasks():
        if task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

if __name__ == "__main__":
    asyncio.run(main())


#endregion
