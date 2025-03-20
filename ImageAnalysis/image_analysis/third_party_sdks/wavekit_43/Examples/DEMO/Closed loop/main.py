"""
Imports 
Required modules :
-os
-matplotlib
-PySide2
-sys
-time
"""
import os, sys, time

from PySide2 import QtCore, QtGui, QtWidgets, QtUiTools
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..//..//..//" ))
import wavekit_py as wkpy

class Interface(QtWidgets.QMainWindow):

    def __init__(self, parent = None):
        """Init Function
        Used to instanciate all required objects/attributes
        """
        #Init mother class
        super(Interface, self).__init__(parent)
        
        #Load interface
        loader = QtUiTools.QUiLoader()
        file = QtCore.QFile(os.path.abspath("closed_loop.ui"))
        file.open(QtCore.QFile.ReadOnly)
        self.window = loader.load(file, parent) 
        file.close()        
        self.window.show()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        
        #Init variable
        self.backup_path = None
        self.haso_path = None
        self.wfc_path = None
        self.image = None
        self.init_ui()
        self.init_connections()
    
    def init_ui(self):
    
        fg_color = 'white'
        
        # Init variable
        self.image_cb = None
        self.image_canvas = None
        self.command_canvas = None
        self.command_plot = None

        # Create figure containing image
        self.image_figure, (self.image_ax, self.image_cb_ax) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [20, 1]})
        self.image_figure.suptitle(r'$\Delta$ Wavefront', fontsize=16, color = fg_color)
        self.image_figure.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
        self.image_ax.axes.tick_params(color = fg_color, labelcolor = fg_color)
        self.image_cb_ax.yaxis.tick_right()
        self.image_cb_ax.xaxis.set_visible(False)
        self.image_cb_ax.axes.tick_params(color = fg_color, labelcolor = fg_color)

        # Create canvas containing image figure
        self.image_canvas  = FigureCanvas(self.image_figure)
        # Add image canvas to ImageDisplay widget
        self.window.ImageDisplay.layout().addWidget(self.image_canvas)
        self.image_ax.set_visible(False)
        self.image_cb_ax.set_visible(False)
        
        # Create figure containing plot
        self.command_figure = plt.figure("Command")
        self.command_ax = self.command_figure.add_subplot(1,1,1)
        self.command_ax.axes.tick_params(color = fg_color, labelcolor = fg_color)
        self.command_figure.suptitle(r'$\Delta$ Positions of mirror actuators', fontsize=16, color = fg_color)
        self.command_figure.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
        self.command_ax.set_visible(False)
        
        self.command_max = 1
        self.command_min = -1

        # Create canvas containing plot figure
        self.command_canvas  = FigureCanvas(self.command_figure)
        # Add plot canvas to PlotDisplay widget
        self.window.PlotDisplay.layout().addWidget(self.command_canvas)
        
        self.window.AcquisitionStatus.setEnabled(False)
        self.window.HasoStatus.setEnabled(False)
        self.window.WFCStatus.setEnabled(False)
        
        self.window.StartButton.setEnabled(False)
        
        self.window.WFSconnectButton.setEnabled(False)
        self.window.WFSdisconnectButton.setEnabled(False)
        self.window.WFCconnectButton.setEnabled(False)
        self.window.WFCdisconnectButton.setEnabled(False)
    
    def init_connections(self) :
        self.window.getHasoFileButton.clicked.connect(self.get_haso_path)
        self.window.getWFCFileButton.clicked.connect(self.get_wfc_path)
        self.window.getBackupFileButton.clicked.connect(self.get_backup_path)

        #haso connection
        self.window.WFSconnectButton.clicked.connect(self.connect_haso)
        self.window.WFSdisconnectButton.clicked.connect(self.disconnect_haso)
        
        #corrector connection
        self.window.WFCconnectButton.clicked.connect(self.connect_wfc)
        self.window.WFCdisconnectButton.clicked.connect(self.disconnect_wfc)
        
        self.window.StartButton.clicked.connect(self.start_loop)
    
    def get_backup_path(self) :
        research = QtWidgets.QFileDialog()
        research.setFileMode(QtWidgets.QFileDialog.DirectoryOnly);
        file = research.getOpenFileName(self, 'Choose file', __file__)
        self.backup_path = str(file[0])
        self.window.BackupFilePath.setText(self.backup_path)
        self.check_start_ready()
        
    def get_haso_path(self) :
        research = QtWidgets.QFileDialog()
        research.setFileMode(QtWidgets.QFileDialog.DirectoryOnly);
        file = research.getOpenFileName(self, 'Choose file', __file__)
        self.haso_path = str(file[0])
        self.window.HasoFilePath.setText(self.haso_path)
        self.window.WFSconnectButton.setEnabled(True)
        
    def get_wfc_path(self) :
        research = QtWidgets.QFileDialog()
        research.setFileMode(QtWidgets.QFileDialog.DirectoryOnly);
        file = research.getOpenFileName(self, 'Choose file', __file__)
        self.wfc_path = str(file[0])
        self.window.WFCFilePath.setText(self.wfc_path)
        self.window.WFCconnectButton.setEnabled(True)
    
    def error(self, message):
        self.statusBar().showMessage(message)
        time.sleep(1.5)
        self.statusBar().clearMessage()
    
    def connect_haso(self):
        try :
            self.camera = wkpy.Camera(config_file_path = self.haso_path)
            self.camera.connect()
            self.camera.start(
                wkpy.E_CAMERA_ACQUISITION_MODE.NEW,
                wkpy.E_CAMERA_SYNCHRONIZATION_MODE.SYNCHRONOUS
                )
            self.camera.set_parameter_value('exposure_duration_us', 1000)
            self.image = self.camera.get_raw_image()
            self.on_haso_connection(True)
        except Exception as e :
            print(str(e))
            self.error(str(e))
    
    def disconnect_haso(self):
        try :
            self.camera.disconnect()
            self.on_haso_connection(False)
        except Exception as e :
            self.error(str(e))
    
    def on_haso_connection(self, connected):
        self.window.WFSconnectButton.setEnabled(not connected)
        self.window.WFSdisconnectButton.setEnabled(connected)
        self.window.HasoStatus.setEnabled(connected)
        self.window.getHasoFileButton.setEnabled(not connected)
        self.check_start_ready()
    
    def connect_wfc(self):
        try :
            self.wfc = wkpy.WavefrontCorrector(config_file_path = self.wfc_path)
            self.wfc.connect(True)
            self.on_wfc_connection(True)
        except Exception as e :
            print(str(e))
            self.error(str(e))
    
    def disconnect_wfc(self):
        try :
            self.wfc.disconnect()
            self.on_wfc_connection(False)
        except Exception as e :
            self.error(str(e))
    
    def on_wfc_connection(self, connected):
        self.window.WFCconnectButton.setEnabled(not connected)
        self.window.WFCdisconnectButton.setEnabled(connected)
        self.window.WFCStatus.setEnabled(connected)
        self.window.getWFCFileButton.setEnabled(not connected)
        self.check_start_ready()
    
    def check_start_ready(self):
        self.window.StartButton.setEnabled(
            self.window.HasoStatus.isEnabled()
            and self.window.WFCStatus.isEnabled()
            and self.backup_path is not None)
    
    def start_loop(self):
        self.on_loop_running(True)
        self.closed_loop()
    
    def on_loop_running(self, running):
        self.window.StartButton.setEnabled(not running)
        self.window.WFSdisconnectButton.setEnabled(not running)
        self.window.WFCdisconnectButton.setEnabled(not running)
        self.window.AcquisitionStatus.setEnabled(running)
    
    def closed_loop(self):
        try :
            nb_iter = self.window.nbLoop.value()
            corr_data_manager = wkpy.CorrDataManager(
                haso_config_file_path = self.haso_path,
                interaction_matrix_file_path = self.backup_path
                )
            
            """Get command dynamic range
            """
            prefs = corr_data_manager.get_actuator_prefs(0)
            self.command_min = prefs.min_value*0.01
            self.command_max = prefs.max_value*0.01
            
            """Compute reference slopes target tilt and focus
            """
            hasoengine = wkpy.HasoEngine(config_file_path = self.haso_path)
            """Get computed slopes
            """
            slopes = hasoengine.compute_slopes(self.image, False)[1]
            
            """Create ref_hasoslopeserence slopes
            """
            ref_hasoslopes = wkpy.HasoSlopes(hasoslopes = slopes)
            
            processor_list = wkpy.SlopesPostProcessorList()
            processor_list.insert_filter(0, False, False, False, True, True, True)

            hasodata = wkpy.HasoData(hasoslopes = ref_hasoslopes)
            hasodata.apply_slopes_post_processor_list(processor_list)
            ref_hasoslopes = hasodata.get_hasoslopes()[0] #Get only computed slopes

            processor_list.delete_processor(0)
            processor_list.insert_substractor(0, ref_hasoslopes, "")

            hasodata.set_hasoslopes(slopes)
            hasodata.apply_slopes_post_processor_list(processor_list)    
            delta_hasoslopes = hasodata.get_hasoslopes()[0]

            """Set wavefrontcorrector pref_hasoslopeserences
            """
            self.wfc.set_temporization(20)

            """Compute command matrix
            """
            corr_data_manager.set_command_matrix_prefs(32, False)
            corr_data_manager.compute_command_matrix()

            """Loop with RMS printing
            """
            compute_phase_set = wkpy.ComputePhaseSet(type_phase = wkpy.E_COMPUTEPHASESET.ZONAL)
            loop_smoothing = wkpy.LoopSmoothing(level = "MEDIUM")
            gain = 0.2
            
            self.image_ax.set_visible(True)
            self.image_cb_ax.set_visible(True)
            self.command_ax.set_visible(True)
            
            for x in range(nb_iter):
                self.image = self.camera.get_raw_image()
                
                slopes = hasoengine.compute_slopes(self.image, False)[1]
                hasodata.set_hasoslopes(slopes)
                hasodata.apply_slopes_post_processor_list(processor_list)
                delta_hasoslopes = hasodata.get_hasoslopes()[0]
                phase = wkpy.Compute.phase_zonal(compute_phase_set, hasodata)
                delta_commands, applied_gain = corr_data_manager.compute_closed_loop_iteration(
                    delta_hasoslopes,
                    False,
                    loop_smoothing,
                    gain
                    )
                self.wfc.move_to_relative_positions(delta_commands)
                
                rms, pv, max_, min_ = phase.get_statistics()
                
                self.display(phase.get_data()[0].copy(), delta_commands, rms, pv)
                
                """
                1sec wait between iterations
                You can improve calculations performance by removing this waiting instruction
                (but the display refresh of this demo is not optimized for the full performance)
                If you do so, you can improve the display refresh by either using threads or by flushing events
                """
                time.sleep(1)
            
            self.on_loop_running(False)
            
        except Exception as e :
            print(str(e))
            self.error(str(e))
            self.on_loop_running(False)
        
    
    def display(self, data, command, rms, pv):
        image_plot = self.image_ax.imshow(data, interpolation = 'none')
        self.image_ax.invert_yaxis()
        self.image_cb = self.image_figure.colorbar(image_plot, cax = self.image_cb_ax)
        self.image_canvas.draw()
 
        self.command_ax.clear()
        self.command_ax.bar(range(len(command)), command)
        ymin = self.command_min
        ymax = self.command_max
        for x in command:
            if x > ymax :
                ymax = x
            if x < ymin:
                ymin = x
                
        self.command_ax.set_ylim([ymin, ymax])
        self.command_canvas.draw()
        
        self.window.RMSValue.setText(str(rms))
        self.window.PVValue.setText(str(pv))
        
        self.repaint()

    def main(self):
        '''Display main Window'''
        #self.show()
    
    
    
    
if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    interface = Interface()
    interface.main()
    app.exec_()