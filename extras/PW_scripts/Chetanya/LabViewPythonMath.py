import sys
import time
import os
from geecs_python_api.controls.devices.scan_device import ScanDevice
from geecs_python_api.controls.devices.scan_device import GeecsDatabase

"""Todo List:
1. When using command line, send the name of the main device to the script
2. Canhe the stopfile name to be same as the device name so that we have a unique identifier

"""




# Getting the name of the experiment and Device Name 
#device_name = sys.argv[1]
device_name = 'CAM-HPD-CCD'
# region Exit Logic with command line
#Writing the process id to a file so we can close it later
stop_file = 'stop'+'_'+device_name+'.txt'
print(stop_file)

#type nul > stop.txt
#endregion



# Define the experiment to collect data from
ScanDevice.exp_info = GeecsDatabase.collect_exp_info("Bella")

"""
#Defining the return device
return_device = ScanDevice(device_name)
return_device.use_alias_in_TCP_subscription = False
"""

#Format
# Initialize the device
camera = ScanDevice('CAM-HPD-CCD')
camera.use_alias_in_TCP_subscription = False #This removes the use of aliases making it easier to code

#Subscibe to the variables you need
camera.subscribe_var_values(['meancounts', 'systimestamp'])

"""
#Get these values
for i in range(100): #Change this with while True
    a = camera.state # This asks for the current values of the chosen variables
    if 'meancounts' in a:  #Checking if the variables are present in the reply

        ##Replace this with do the math
        meancountval = 2*a['meancounts'] #If they are, then you can get the value by using the varibale name as dict key

        ###Replace this with setting the variable in the device
        print(i, meancountval, pid)
    time.sleep(0.3)
"""


#
try:
    while True:
        # Check for stop file
        if os.path.exists(stop_file):
            print("Stop file detected, exiting loop gracefully...")
            break

        a = camera.state # This asks for the current values of the chosen variables
        if 'meancounts' in a:  #Checking if the variables are present in the reply
            ##Replace this with do the math
            meancountval = 2*a['meancounts'] #If they are, then you can get the value by using the varibale name as dict key
            timestamp = a['systimestamp']
            ###Replace this with setting the variable in the device
            print(timestamp, meancountval, device_name)
        time.sleep(0.3)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    camera.close()
    #os.remove('script.pid')
    if os.path.exists(stop_file):
        os.remove(stop_file)


#Close the subscription at the end
#camera.close()
#os.remove('script.pid')

