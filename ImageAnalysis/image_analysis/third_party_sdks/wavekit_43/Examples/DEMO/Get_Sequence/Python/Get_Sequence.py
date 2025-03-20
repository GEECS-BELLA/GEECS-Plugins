# ImagineOpticProblemDemo.cpp : This file contains the 'main' function. Program execution begins and ends there.
#

# February 11, 2020.
# Oleg Obyedkov,  obyedkovo@corning.com
# This program tests the HASO4 capture image sequence mode using external HASO4 camera mode.
# See the main() function.
# Note: special hardware used for generation of precise external triggers (not controlled in this example).
import numpy as np
import os,sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..//..//..//..//" ))
import wavekit_py as wkpy

numImage = 128

class ImagineOpticProblemDemo:
    def __init__( self ):
        self.config = "D:\Trunk2\Python_interface\Sources\Demo\WFS_HASO4_first_7542.dat"
        self.camera = wkpy.Camera(config_file_path = self.config)
        self.camera.connect()
        print ("Imop_Camera_NewFromConfigFile() success\n");
        
        self.size, self.depth = self.camera.get_size();
        print ("Imop_Camera_GetSize() -> image:(" + str(self.size.X) + "," + str(self.size.Y) + "), depth:" + str(self.depth) + "\n");
        
        self.images = np.array(numImage, dtype = wkpy.Image)
        
        for i in range(numImage-1):
            image = wkpy.Image(size = self.size, bit_depth = self.depth)
            self.images = np.append(self.images, image)
            
        print ("Imop_Image_New() success, allocated " + str(numImage) + " images\n");
        
        self.slopes = wkpy.HasoSlopes(config_file_path = self.config)
        print ("Imop_HasoSlopes_NewFromConfigFile() success\n")
        
        self.sn, self.pupil = self.slopes.get_info()
        print ("Imop_HasoSlopes_GetInfo() -> pupil:(" + str(self.pupil.size.X) + "," +  str(self.pupil.size.Y) + "), sub-pupil:(" + str(self.pupil.steps.X) + "," + str(self.pupil.steps.Y) + ")\n");
        
        self.engine = wkpy.HasoEngine(config_file_path = self.config)
        print ("Imop_HasoEngine_NewFromConfigFile() success\n");
        
        dWavelength = 980.0
        self.slopes.set_wavelength(dWavelength)
        print ("Imop_HasoSlopes_SetWaveLength() success, wavelength:" + str(dWavelength) + "\n");

    def set_camera_preferences(self, nDurUs, nTimeoutMsec, cTriggerMode, nStartPupilX, nStartPupilY ): 
        startSubPupil = wkpy.uint2D(nStartPupilX, nStartPupilY)
        denoising_strength = 1.0
        
        self.engine.set_preferences(startSubPupil, denoising_strength, True)
        print ("Imop_HasoEngine_SetPreferences() success, start sub-pupil:(" + str(startSubPupil.X) + "," + str(startSubPupil.Y) + "), de-noise:" + str(denoising_strength))
        
        self.camera.stop()
        print ("Imop_Camera_Stop() success")

        self.camera.set_parameter_value("exposure_duration_us", nDurUs)
        print ("Imop_Camera_SetParameterInt() success, exposure:" + str(nDurUs) + "\n")
        
        self.camera.set_timeout(nTimeoutMsec)
        print ("Imop_Camera_SetTimeOut() success, timeout:" + str(nTimeoutMsec) + "\n")
        
        parm = "trigger_mode"
        self.camera.set_parameter_value(parm, cTriggerMode)
        print ("Imop_Camera_SetParameterString() success, '" + str(parm) + "':'" + str(cTriggerMode) + "'\n")

        self.camera.start(wkpy.E_CAMERA_ACQUISITION_MODE.NEW, wkpy.E_CAMERA_SYNCHRONIZATION_MODE.SYNCHRONOUS)

    def clean_up(self):
        self.camera.stop()
        self.camera.disconnect()
        
    def get_sequence(self):     
        self.images = self.camera.get_raw_sequence(numImage)
     
if __name__ == '__main__':

    imagine = ImagineOpticProblemDemo()
    
    #imagine.set_camera_preferences(250, 20000, "rising_edge", imagine.pupil.size.X / 2, imagine.pupil.size.Y / 2)
    #imagine.set_camera_preferences(250, 20000, "cam_internal", imagine.pupil.size.X / 2, imagine.pupil.size.Y / 2)

	# 1.	Here we are telling external device to trigger HASO4_first camera 'm_nNumImages' times with 10.0 msec period.
	#		i.e. 128 external triggers

	# 2.	Timeout happens here:
	#		Imop_Camera_GetSequence -> Error type camera_timeout_error.
	#		get_last_raw_image16() : Error in GetImagePointer - type = Camera timed out - message = Grab timed out.
    
    
    imagine.get_sequence();
    
    imagine.clean_up()
