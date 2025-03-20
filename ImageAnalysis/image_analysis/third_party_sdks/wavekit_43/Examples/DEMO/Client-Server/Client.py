"""
Imports 
Required modules :
-os
-PyQt4
-sys
-time
"""
import sys, os, time
from PySide2 import QtCore

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..//..//..//" ))
import wavekit_py as wkpy

class Client(QtCore.QObject):
    """Client Class
    Required Class to run this app.
    It contains Client init and connection.
    """
    #Qt Signal to send and receive datas from main
    signal_start_acq = QtCore.Signal()
    image_ready = QtCore.Signal(wkpy.Image)
    
    
    def __init__(self):
        """Init Function
        Used to instanciate all required objects/attributes
        """
        #Init mother class
        super(Client, self).__init__()
        #Init attributes
        self.getData = False
        self.onStopThread = False
        
        
    def start_acq(self):
        """Start Acquisition Function
        Connect Client to Server and Receive datas from it
        """
        #Connect Client to Server
        self.client = wkpy.Client(ip = "localhost", port = 8081, timeout = 15000)
        #Loop until app is finished
        while(True):
            #If Client getData is False, empty loop, 
            #else get datas from server
            if self.getData:
                time.sleep(1)
                #Check that server has data to send
                print(self.client.get_data_number())
                if(self.client.get_data_number() > 0):
                    try:
                    #Get datas from server
                        image, data_id = self.client.get_data_value(0)
                    except Exception as e:
                        print(str(e))
                #Return datas to GUI
                self.image_ready.emit(image)
            if self.onStopThread:
                break
