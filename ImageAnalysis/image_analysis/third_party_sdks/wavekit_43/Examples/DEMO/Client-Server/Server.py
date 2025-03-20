"""
Imports 
Required modules :
-os
-sys
-time
"""
import os, sys, time
from PySide2 import QtCore

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..//..//..//" ))
import wavekit_py as wkpy

class Server(QtCore.QObject):
    """Client Class
    Required Class to run this app.
    It contains Client init and connection.
    """
    
    
    def __init__(self):
        """Init Function
        Used to instanciate all required objects/attributes
        """
        self.server = wkpy.Server(config_file_path = os.path.join(os.path.dirname(__file__), "WFS_HASO3_32_4229.dat"), port = 8081)
    
    
    def send_data(self):
        """Start Acquisition Function
        Connect Client to Server and Receive datas from it
        """
        self.server.start()
        #Define image to send
        self.image = wkpy.Image(image_file_path = str(os.path.join(os.path.dirname(__file__), "imageTest.himg")))
        while(True):
            time.sleep(0.5)
            #Add image to server data
            self.server.add_data('image', wkpy.E_TYPES.IMAGE, self.image)
            print ('Image sent')
 
 
if __name__=='__main__':
    server = Server()
    server.send_data()
