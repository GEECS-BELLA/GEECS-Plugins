"""
Imports 
Required modules :
-os
-matplotlib
-PySide2
-sys
-time
"""

import os, sys, glob, time
from PySide2 import QtCore, QtGui, QtWidgets, QtUiTools

from datetime import datetime, date, time, timedelta

import ctypes
import numpy as np
import scipy
import scipy.ndimage as ndimage
import subprocess
import threading, copy
import time

from PIL import Image

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# sys.exec(r'C:\Python27\Lib\site-packages\PyQt4\pyrcc4.exe -o Haso_rc.py Haso.qrc')
# sys.exec(r'C:\Python27\Lib\site-packages\PyQt4\pyuic4.bat -o Haso_ui.py Haso.ui')


import internal.pydic as pydic

import cv2 as cv 

#sys.path.append(os.path.dirname(__file__)  + '/../../../')
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..//..//..//" ))
import wavekit_py as wkpy

def extract_info_from_configfile(haso_config_file_path):
    try :
        hasoconfig, hasospec, wavelenght  = wkpy.HasoConfig.get_config(haso_config_file_path[0]);
    except Exception as e :
        print(str(e))
        print("test 2")
        errorCamera()
        
    return (hasospec.nb_subapertures.X, hasospec.nb_subapertures.Y), (hasospec.ulens_step.X, hasospec.ulens_step.Y), hasospec.micro_lens_focal

def errorCamera():
    widget = QtWidgets.QWidget()
    choice = QtWidgets.QMessageBox.warning(widget, 'Error!',
                                            "Camera can't be initialized",
                                            QtWidgets.QMessageBox.Ok )
    sys.exit()

class Worker(QtCore.QObject):

    image_ready = QtCore.Signal(wkpy.Image, float)
    error = QtCore.Signal(str)

    def __init__(self, configfile):
        """Init Function
        Used to instanciate all required objects/attributes
        """
        #Init mother class
        super(Worker, self).__init__()
        
        #Pre init Camera
        self.configfile = configfile
        self.camera = None
        
        #Init Camera
        try :
            self.camera = wkpy.Camera(config_file_path = self.configfile[0])
            self.camera.connect()
            self.camera.start(0, 1)
        except Exception as e :
            print(str(e))
            errorCamera()

        #Init image
        self.image = wkpy.Image(size = wkpy.uint2D(1,1), bit_depth = 16)
        self.firstImage = np.zeros((1,1))
        
        
        #Init variable
        self.expotime = 4000
        self.expotime_changed = True

        self.nbImages = 1
        self.nbImages_changed = True

        self.isFirstAcquisition = True
        self.acquire = False

    def __del__(self):
        if self.camera is not None:
            self.camera.stop()
            self.camera.disconnect()
        
    def set_exposure_time(self, expotime) :
        self.expotime = expotime
        self.expotime_changed = True

    def set_nb_of_images(self, n) :
        self.nbImages = n
        self.nbImages_changed = True
    
    def start_acq(self):
        self.isFirstAcquisition = True
        while(self.acquire) :

            try :
                if self.camera is not None:
                    if self.expotime_changed:
                    
                        self.camera.set_parameter_value("exposure_duration_us", self.expotime)
                        self.expotime_changed = False
                    
                    if self.nbImages_changed:
                        self.camera.set_nb_images_to_sum(self.nbImages)
                        self.nbImages_changed = False
                    
                    try :
                        self.image = self.camera.snap_raw_image()
                        time.sleep(0.1)
                    except Exception as e:
                        time.sleep(0.1)
                        continue

                image_array = self.image.get_data()
                if self.isFirstAcquisition:
                    self.firstImage = image_array.copy()
                    self.isFirstAcquisition = False

            except Exception as e :
                self.error.emit('ERROR !! ' + str(e))
                continue

            try :
                min, max, sat = self.image.get_statistics(10)
            except Exception as e :
                self.error.emit('ERROR !! ' + str(e))

            self.image_ready.emit(self.image, sat)
            self.error.emit(str(''))

class HasoDemo(QtWidgets.QMainWindow):

    start_acq = QtCore.Signal()
    
    def __init__(
        self,
        configfile = None,
        parent = None,
        figures = True
    ):
        ''' Create class instance'''
        super(HasoDemo, self).__init__(parent)
        loader = QtUiTools.QUiLoader()
        file = QtCore.QFile(os.path.abspath("internal//baseApp.ui"))
        file.open(QtCore.QFile.ReadOnly)
        self.window = loader.load(file, parent) 
        file.close()        
        self.window.show()
        
        self.thread = None # initialize to ensure proper deletion

        # Select configuration file
        # configfile = r'__IO_SOFT__!' + os.path.join(os.path.dirname(os.path.realpath(__file__)), '22479584\\rev 0\\Fichier de Config V4.x\\WFS_HASOFake_0000.dat')
        if configfile is None:
            file_dialog = QtWidgets.QFileDialog()
            file = file_dialog.getOpenFileNames(self, 'Select configuration file', __file__,str("Config file (*.dat)" ))
            if len(file) < 1: exit(0)
            configfile = file[0]
            
        # Build UI
        #self.setupUi(self)
           
         
        self.configfile = configfile
        self.filter = [False, False, False, False, False]
        try:
            print(str(self.configfile))
            self.slopes = wkpy.HasoSlopes(config_file_path = self.configfile[0])
        except Exception as e:
            print(str(e))
            print("test")
            errorCamera()

        self.size, self.steps, ulens_focal = extract_info_from_configfile(self.configfile)
        nb_ticks = 5
        self.xticks = np.linspace(0,self.size[0],nb_ticks)
        self.xticklabels = [   '{:.1f}'.format(v/1000.)
            for v in np.linspace(
                -self.size[0]*self.steps[0] / 2.,
                self.size[0]*self.steps[0] / 2.,
                nb_ticks
            )
        ]

        self.yticks = np.linspace(0,self.size[1],nb_ticks)
        self.yticklabels = [   '{:.1f}'.format(v/1000.)
            for v in np.linspace(
                -self.size[1]*self.steps[1] / 2.,
                self.size[1]*self.steps[1] / 2.,
                nb_ticks
            )
        ]

        if figures:
            self.wavefront_figure, (self.wavefront_ax, self.wavefront_cb_ax) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [20, 1]})
            self.wavefront_figure.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
            self.wavefront_cb_ax.yaxis.tick_right()
            self.wavefront_cb_ax.xaxis.set_visible(False)

            self.image_figure, (self.image_ax, self.image_cb_ax) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [20, 1]})
            self.image_figure.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
            self.image_cb_ax.yaxis.tick_right()
            self.image_cb_ax.xaxis.set_visible(False)

        self.wavefront_canvas = None
        self.wavefront_cb = None

        self.image_canvas = None
        self.image_cb = None

        self.worker = Worker(self.configfile)
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        self.thread.start()
        self.connectActions()
        
        self.set_exposure_time()
        self.set_nb_of_images()
        
        self.save_image_dir = os.getcwd()
        self.save_image_dir_wavefront = os.getcwd()
        self.window.saveDirectoryPath.setText(os.getcwd())
        self.window.saveDirectoryPathWavefront.setText(os.getcwd())

    def __del__(self):
        ''' Delete class instance'''
        if self.thread is not None:
            self.thread.terminate()

    def connectActions(self):
        ''' Connect class methods to user interactions (buttons, checkboxes...)'''
        # ui
        self.window.actionPlay.triggered.connect(self.handleAcquisition)
        
        self.window.expotime.valueChanged.connect(self.set_exposure_time)
        self.window.nbImages.valueChanged.connect(self.set_nb_of_images)
        
        self.window.filter_tx.clicked.connect(self.set_filter)
        self.window.filter_ty.clicked.connect(self.set_filter)
        self.window.filter_f.clicked.connect(self.set_filter)
        self.window.filter_a0.clicked.connect(self.set_filter)
        self.window.filter_a45.clicked.connect(self.set_filter)
        
        self.window.saveCurrentFile.clicked.connect(self.save_current_image)
        self.window.saveDirectory_2.clicked.connect(self.get_path_save_directory)

        self.window.saveDirectoryWavefront.clicked.connect(self.get_path_save_directory_wavefront)
        self.window.saveWavefrontButton.clicked.connect(self.save_current_image_wavefront)

        # to worker
        self.start_acq.connect(self.worker.start_acq)
        # from worker
        self.worker.image_ready.connect(self.display_image)
        self.worker.error.connect(self.display_error)

    def onChangeIndex(self, index):
        '''Disables play button when user switches to Wavefront tab'''
        if self.window.tabWidget.currentIndex() == 0:
            self.window.actionPlay.setEnabled(True)
        else: self.window.actionPlay.setDisabled(True) 
        self.window.actionPlay.setChecked(False)
        self.thread.acquire = False

    def handleAcquisition(self):
        '''Starts (or stops) parallel acquisition'''
        if self.window.actionPlay.isChecked() :
            self.worker.acquire = True
            self.start_acq.emit()
        else:
            self.worker.acquire = False
            self.thread.acquire = False

    def set_exposure_time(self):
        self.worker.set_exposure_time(self.window.expotime.value())
        
    def set_nb_of_images(self):
        self.worker.set_nb_of_images(self.window.nbImages.value())

    def set_filter(self, wavefront) :
        self.filter = [
            bool(self.window.filter_tx.isChecked()),
            bool(self.window.filter_ty.isChecked()),
            bool(self.window.filter_f.isChecked()),
            bool(self.window.filter_a0.isChecked()),
            bool(self.window.filter_a45.isChecked())
        ]
        if not self.window.actionPlay.isChecked() :
            self.compute_wf()

    def compute_wf(self):
        filter_array = (ctypes.c_byte * 5)()
        for i,f in enumerate(self.filter):
            filter_array[i] = ctypes.c_byte(0) if f else ctypes.c_byte(1)
        
        phase = wkpy.Phase(hasoslopes = self.slopes, type_ = wkpy.E_COMPUTEPHASESET.ZONAL, filter_ = filter_array)
        WF = phase.get_data()
        rms, pv, max, min = phase.get_statistics()
        self.display_wavefront(WF[0], pv, rms)
        self.display_geo()
        del phase

    # a modifier par :
    # slopes -> new_from_image
    def process(self) :
        if self.window.tabWidget.currentIndex() == 0:
            # compute slopes
            self.slopes = wkpy.HasoSlopes(image = self.image, config_file_path = str(self.configfile))
            try :
                dx, dy = pydic.run(
                    imarray_ref, imarray,
                    # roi = [(172, 107), (500, 424)],
                    map = maskarray
                )
                # Missing something like a pixel to mrad factor.
                # stupid value there, but allow visualization.
                dx = 1000. * dx
                dy = 1000. * dy
                
            except Exception as e: 
                self.display_error('Displacement computation failed : ' + str(e))
                return

            self.slopes.set_slopes(dx, dy)

            pupil = cv.resize(maskarray, dsize = (dx.shape[1], dx.shape[0]), interpolation=cv.INTER_NEAREST)
            self.slopes.apply_pupil(pupil)



    def get_path_save_directory_wavefront(self) :
        research = QtGui.QFileDialog()
        research.setFileMode(QtGui.QFileDialog.DirectoryOnly);
        file = QtCore.QStringList(research.getExistingDirectory(self, 'Choose save directory', __file__))
        self.save_image_dir_wavefront = file[0]
        self.window.saveDirectoryPathWavefront.setText(self.save_image_dir_wavefront)

    def save_current_image_wavefront(self) :
        timestamp = datetime.strftime(datetime.now(), 'wavefront_%Y-%m-%d-%H-%M-%S')
        self.slopes.save_to_file(os.path.join(str(self.save_image_dir_wavefront), str(timestamp) + '.has') , '', '')

    def get_path_save_directory(self) :
        research = QtGui.QFileDialog()
        research.setFileMode(QtGui.QFileDialog.DirectoryOnly);
        file = QtCore.QStringList(research.getExistingDirectory(self, 'Choose save directory', __file__))
        self.save_image_dir = file[0]
        self.window.saveDirectoryPath.setText(save_image_dir)

    def save_current_image(self) :
        timestamp = datetime.strftime(datetime.now(), 'image_%Y-%m-%d-%H-%M-%S')
        try:
            self.worker.image.save(os.path.join(str(self.save_image_dir), str(timestamp) + '.himg'))
        except Exception as e:
            self.display_error(str(e))

    def display_wavefront(self, wavefront, pv, rms) :
        if self.wavefront_canvas is None:
            self.wavefront_canvas  = FigureCanvas(self.wavefront_figure)
            self.window.wavefront_display.layout().addWidget(self.wavefront_canvas)

        self.wavefront_ax.clear()
        if self.wavefront_cb is not None:
            self.wavefront_cb_ax.cla()
            self.wavefront_cb = None
        plot = self.wavefront_ax.imshow(wavefront, interpolation = 'none')

        self.wavefront_ax.invert_yaxis()

        self.wavefront_ax.tick_params(axis='x', colors='white')
        self.wavefront_ax.tick_params(axis='y', colors='white')
        cmin, cmax = plot.get_clim()
        if (cmin > -0.025 and cmax < 0.025) :
            plot.set_clim(-0.025, 0.025)

        self.wavefront_cb = self.wavefront_figure.colorbar(plot, cax = self.wavefront_cb_ax)
        self.wavefront_cb_ax.tick_params(axis='y', colors='white')
        
        self.wavefront_canvas.draw()

        self.window.wf_rms_value.setText('{:.3f}'.format(rms))
        self.window.wf_pv_value.setText('{:.3f}'.format(pv))
        self.display_geo()

    def display_geo(self) :
        geo = self.slopes.get_geometric_properties()
        self.window.tilt_x_value.setText('{:.3f}'.format(geo[0]))
        self.window.tilt_y_value.setText('{:.3f}'.format(geo[1]))
        self.window.curv_value.setText('{:.3f}'.format(geo[2]))

    def display_image(self, image, sat) :
        if self.image_canvas is None:
            self.image_canvas  = FigureCanvas(self.image_figure)
            self.window.image_display.layout().addWidget(self.image_canvas)
        
        self.image_ax.clear()
        if self.image_cb is not None:
            self.image_cb_ax.cla()
            self.image_cb = None
            
        if self.window.tabWidget.currentIndex() == 0:
            plot = self.image_ax.imshow(image.get_data(), interpolation = 'none')

            self.image_ax.invert_yaxis()

            self.image_ax.tick_params(axis='x', colors='white')
            self.image_ax.tick_params(axis='y', colors='white')

            self.image_cb = self.wavefront_figure.colorbar(plot, cax = self.image_cb_ax)
            self.image_cb_ax.tick_params(axis='y', colors='white')
            
            self.image_canvas.draw()
        
            self.window.sat_value.setText('{:.1f}'.format(sat))
        else: 
            self.engine = wkpy.HasoEngine(config_file_path = self.configfile[0])
            try:
                trimmer_quality, self.slopes = self.engine.compute_slopes(image, 0, False)
            except Exception as e:
                self.display_error('Signal too weak to compute slopes')
                    
            #compute phase
            filter_array = (ctypes.c_byte * 5)()
            for i,f in enumerate(self.filter):
                filter_array[i] = ctypes.c_byte(0) if f else ctypes.c_byte(1)
            phase = wkpy.Phase(hasoslopes = self.slopes, type_ = wkpy.E_COMPUTEPHASESET.ZONAL, filter_ = filter_array)
            WF = phase.get_data()
            self.wavefront = WF[0]
            rms, pv, max, min = phase.get_statistics()
            self.display_wavefront(self.wavefront, pv, rms)
            self.display_geo()
            del phase
            
    
    def display_error(self, message):
        self.window.statusbar.showMessage(message)

    def main(self):
        '''Display main Window'''
        self.show()

if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    HasoDemo = HasoDemo(figures = True)
#    image = ndimage.imread('internal\\22.tiff')
#    mask = HasoDemo.compute_auto_mask(image)
#    plt.figure()
#    plt.imshow(mask, interpolation='nearest')
#    imagemask = mask*image
#    plt.figure()
#    plt.imshow(imagemask, interpolation='nearest')
#    plt.show()
   
    app.exec_()
