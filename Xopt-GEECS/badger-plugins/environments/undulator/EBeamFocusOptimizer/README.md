# Using EMQs to Optimize for a Bright, Round ebeam

## Prerequisites

-Ebeams are visible on an ebeam camera (UC_AlineEBeam3)

-EMQs are turned on

## Usage

Could be possible to simply switch the `camera_name` variable and use this optimizer with a different camera.  Need to test.
Right now I simply take the max counts on the camera and divide by 1 plus the absolute difference between FWHM x and y.