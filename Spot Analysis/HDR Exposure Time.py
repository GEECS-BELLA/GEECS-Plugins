import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Loading exposure images into a list
# Load your images here
img1 = "/Users/anthonylblmac/Desktop/updated s6 and s7/Final Program/Best images/Scan002_UC_ModeImager_001.png"
img2 = "/Users/anthonylblmac/Desktop/updated s6 and s7/Final Program/Best images/Scan003_UC_ModeImager_001.png"
img3 = "/Users/anthonylblmac/Desktop/updated s6 and s7/Final Program/Best images/Scan004_UC_ModeImager_015.png"

img_fn = [img1, img2, img3]
img_list = [cv.imread(fn) for fn in img_fn]
#list exposure times for each (in seconds)
exposure_times = np.array([0.005, 0.005, 0.005], dtype=np.float32)

# Merge exposures to HDR image using Debevec method
merge_debevec = cv.createMergeDebevec()
hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy())

# Merge exposures to HDR image using Robertson method
merge_robertson = cv.createMergeRobertson()
hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())

# Tonemap HDR images using Debevec and Robertson methods
tonemap1 = cv.createTonemap(gamma=2.2)

# Tonemap Debevec HDR to 16-bit
res_debevec_16bit = tonemap1.process(hdr_debevec.copy()) * 65535
res_debevec_16bit = np.clip(res_debevec_16bit, 0, 65535).astype('uint16')

# Tonemap Robertson HDR to 16-bit
res_robertson_16bit = tonemap1.process(hdr_robertson.copy()) * 65535
res_robertson_16bit = np.clip(res_robertson_16bit, 0, 65535).astype('uint16')

# Exposure fusion using Mertens
merge_mertens = cv.createMergeMertens()
res_mertens = merge_mertens.process(img_list)

# Convert datatype to 16-bit and save Debevec, Robertson, and Mertens results
res_mertens_16bit = np.clip(res_mertens * 65535, 0, 65535).astype('uint16')

# Save 16-bit images
cv.imwrite('/Users/anthonylblmac/Desktop/updated s6 and s7/Final Program/Best images/res_debevec_16bit.png', res_debevec_16bit)
cv.imwrite('/Users/anthonylblmac/Desktop/updated s6 and s7/Final Program/Best images/res_robertson_16bit.png', res_robertson_16bit)
cv.imwrite('/Users/anthonylblmac/Desktop/updated s6 and s7/Final Program/Best images/res_mertens_16bit.png', res_mertens_16bit)


