#!/usr/bin/env python3
"""Initial rough testing script while figuring out all the types"""

import time
from src.khzwave import wfs
import logging
from matplotlib import pyplot as plt
logging.basicConfig()
wfs.LOG.setLevel(logging.INFO)
xx = wfs.WfsInterface()

# with open(xx.HEADER_FILE, 'r') as rf:
#     for line in rf:
#         print(line)

# Initialize Instrument up to select MLA step
yy = wfs.Wfs20Instrument(xx)

# Configure camera
yy.configureCamera()
print("spots x,y: ", yy.info.spots_x, yy.info.spots_y)

# Setup reference plane and pupils
yy.setReferencePupils()

# Take an image and read it back
# Take several to get auto modes working
yy.setExposureGain(exposureTime=1.0)
obj = None
for i in range(10):
    yy.takeSpotfieldImage()
    yy.printStatus()
    yy.getSpotfieldImage()
    yy.calcBeamInformation()
    yy.buildSpotfieldImageArray()
    obj = plt.imshow(yy.img.SpotfieldImage)
    # plt.show()
    time.sleep(1)

# #yy.calcWavefront()
# yy.getSpotIntensities()

# # Try high-speed mode
yy.setHighSpeedMode(adaptCentroids=False, allowAutoExposure=False)
yy.getHighSpeedWindows()
for i in range(100):
    ntime = time.time_ns()
    yy.takeSpotfieldImage()
    yy.calcSpotToReferenceDeviations()
    yy.calcZernikeLSF(order=4)
    print(time.time_ns() - ntime, yy.img.zernikeOrders, yy.img.roCMm, yy.img.arrayZernikeUm[:10], yy.img.arrayZernikeOrdersUm[:10])
    time.sleep(1)


yy.close()