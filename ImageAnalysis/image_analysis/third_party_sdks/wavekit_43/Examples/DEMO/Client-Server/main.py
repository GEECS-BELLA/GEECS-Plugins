"""
Imports 
Required modules :
-os
-matplotlib
-PySide2
-subproccess
-sys
-time
"""
import os, sys, time
from PySide2.QtWidgets import QDialog, QApplication
from PySide2 import QtGui  
from PySide2 import QtCore, QtWidgets, QtUiTools
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE, CREATE_NEW_CONSOLE

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..//..//..//" ))
import wavekit_py as wkpy

from Client import Client                
    
class Interface(QtWidgets.QMainWindow):
    """Interface Class
    Main class to run this app.
    It contains inits, GUI loop.
    """
    
    def __init__(
        self,
        parent = None
    ):
        """Init Function
        Used to instanciate all required objects/attributes
        """
        #Define Client and Server status
        self.is_client_started = False
        self.is_server_started = False
        
        #Init mother class
        super(Interface, self).__init__(parent) 
        
        #Load interface
        loader = QtUiTools.QUiLoader()
        file = QtCore.QFile(os.path.abspath("client.ui"))
        file.open(QtCore.QFile.ReadOnly)
        self.window = loader.load(file, parent) 
        file.close()        
        self.window.show()
        
        #Define a system kill on window close
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        
        #Init GUI
        self.init_ui()
        #Init Client
        self.init_client()
        #Define Display
        self.init_display()
        #Init Connections
        self.init_connections()
        #Init default values
        self.init_default_values()  
        #elf.show()
        
    def start_server(
        self
    ):
        """Start Server Function
        Used to launch server and update GUI
        """
        #Check that there is no server currently running
        assert(not self.is_server_started)
        #Launch command prompt with server
        self.cmd = Popen('python.exe Server.py', creationflags = CREATE_NEW_CONSOLE)
        #Update GUI objects
        self.window.ClientStartButton.setEnabled(True)
        self.window.ServerStopButton.setEnabled(True)
        self.window.ServerStartButton.setEnabled(False)
        self.window.serverStatus.setEnabled(True)
        self.window.is_server_started = True
        
        
    def stop_server(
        self
    ):        
        """Stop Server Function
        Used to stop server and update GUI
        """
        if self.is_server_started:
            #Update GUI objects
            self.window.ServerStartButton.setEnabled(True)
            self.window.ServerStopButton.setEnabled(False)
            self.stop_client()
            time.sleep(0.2)
            self.window.ClientStartButton.setEnabled(False)
            self.window.serverStatus.setEnabled(False)
            self.window.clientStatus.setEnabled(False)
            #Kill command prompt containing server
            self.cmd.kill()
        self.is_server_started = False        
          
    def __del__(
        self    
    ):
        """Delete Function
        Used to stop server and client and then wait for thread to kill itself
        """
        self.stop_server()
        self.client.image_ready.disconnect()
        self.client.onStopThread = True
        if self.thread is not None:
            self.thread.terminate()
            self.thread.wait()


    def init_ui(
        self
    ):
        """Init GUI Function
        Used to build up the GUI to be used in the app
        """
        #self.setupUi(self.window)


    def init_display(
        self
    ):
        """Init Display Function
        Used to define display informations
        """
        self.image_color_bar = None
        self.image_canvas = None
        #Define "blank" image to print while there is no Image to print
        self.image_figure, (self.image_axes, self.image_color_bar_axes) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [20, 1]})
        self.image_figure.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
        self.image_color_bar_axes.yaxis.tick_right()
        self.image_color_bar_axes.xaxis.set_visible(False)
        self.image_canvas  = FigureCanvas(self.image_figure)
        #Add Image Holder to ImageDisplay object
        self.window.ImageDisplay.layout().addWidget(self.image_canvas)


    def init_client(
        self
    ):
        """Init Client Function
        Used to instanciate Client and move it to a thread
        """
        #Instanciate Client object
        self.client = Client()
        #Instanciate Qt Thread
        self.thread = QtCore.QThread()
        #Move Client object to Thread
        self.client.moveToThread(self.thread)
        self.thread.start()


    def init_connections(
        self
    ):
        """Init Connections Function
        Used to connect events with GUI Objects and bind Qt Signals and Slots
        """
        self.window.ServerStartButton.clicked.connect(self.start_server)
        self.window.ServerStopButton.clicked.connect(self.stop_server)
        self.window.ClientStartButton.clicked.connect(self.start_client)
        self.window.ClientStopButton.clicked.connect(self.stop_client)
        self.client.image_ready.connect(self.display_image)
        self.client.signal_start_acq.connect(self.client.start_acq)


    def init_default_values(
        self
    ):
        """Init Default values Function
        Used to set Gui objects status
        """
        self.window.ClientStartButton.setEnabled(False)
        self.window.ClientStopButton.setEnabled(False)
        self.window.ServerStopButton.setEnabled(False)
        self.window.serverStatus.setEnabled(False)
        self.window.clientStatus.setEnabled(False)
        
        
    def start_client(
        self
    ):
        """Start Client Function
        Used to update Client thread and GUI
        """
        #Update Client thread to get image from server
        self.client.getData = True
        if not self.is_client_started :
            self.client.signal_start_acq.emit()
            self.is_client_started = True
        #Update GUI objects
        self.window.ClientStopButton.setEnabled(True)
        self.window.ClientStartButton.setEnabled(False)
        self.window.clientStatus.setEnabled(True)


    def stop_client(
        self
    ):
        """Stop Client Function
        Used to update Client thread and GUI
        """
        #Update Client object
        self.client.getData = False
        #Update GUI objects
        self.window.ClientStopButton.setEnabled(False)
        self.window.ClientStartButton.setEnabled(True)
        self.window.clientStatus.setEnabled(False)
        
        
    def display_image(
        self, 
        image
    ):
        """Display Image Function
        Used to get datas from Image and print buffer
        """
        data = None
        try:
            data = image.get_data()
        except:
            pass
        if data is not None:
            plot = self.image_axes.imshow(data, interpolation = 'none')
            self.image_axes.invert_yaxis()
            self.image_axes.tick_params(axis='x', colors='white')
            self.image_axes.tick_params(axis='y', colors='white')
            self.image_color_bar = self.image_figure.colorbar(plot, cax = self.image_color_bar_axes)
            self.image_color_bar_axes.tick_params(axis='y', colors='white')                
            self.image_canvas.draw() 
    
if __name__=='__main__':
    """Main Function
    Used to launch app
    """
    app = QApplication(sys.argv)
    interface = Interface()
    app.exec_()
