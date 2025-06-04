#!/usr/bin/python

from PySide2 import QtCore, QtGui, QtWidgets

import os, sys, glob
import ctypes
import numpy as np

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtWidgets.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig)

import os,sys
sys.path.append(os.path.dirname(__file__)  + '/../../../')
import wavekit_py as wkpy


#if os.path.exists('C:\\Program Files\\Imagine Optic\\WaveView\\dlls'):
#    os.environ['PATH'] = 'C:\\Program Files\\Imagine Optic\\WaveView\\dlls' + os.pathsep + os.environ['PATH']
#elif os.path.exists('C:\\Program Files (x86)\\Imagine Optic\\WaveView\\dlls'):
#    os.environ['PATH'] = 'C:\\Program Files (x86)\\Imagine Optic\\WaveView\\dlls' + os.pathsep + os.environ['PATH']
#else: 
#    a = QtGui.QApplication(sys.argv)
#    QtGui.QMessageBox.critical(QtGui.QWidget(), 'Alert', "Not find WaveView dlls", QtGui.QMessageBox.Ok)
#    sys.exit(0)
#dllHandle = ctypes.windll.kernel32.LoadLibraryA("c_interface_vc100.dll")
#hLib = ctypes.CDLL(None, handle = dllHandle)

class float2D(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]
class uint2D(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint), ("y", ctypes.c_uint)]

def computeWF(hasodata_file_path, filter):
    
    E_COMPUTEPHASESET_ZONAL = 3  # ///< incremental phase reconstruction algorithm

    filter_array = (ctypes.c_char * 5)()
    for i,f in enumerate(filter):
        filter_array[i] = ctypes.c_char('0') if f else ctypes.c_char('1')
        
    hasoslopes = wkpy.HasoSlopes(has_file_path = hasodata_file_path)
    phase = wkpy.Phase(hasoslopes = hasoslopes,
                type_ = wkpy.E_COMPUTEPHASESET.ZONAL, 
                filter_ = filter_array  )
    #print message.value
    
    
    sn, dimenssions = hasoslopes.get_info()
    buf = ctypes.c_float*(dimenssions.size.X*dimenssions.size.Y)

    buf, pupil = phase.get_data()

    return buf
    
class ImageViewer(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        """Init Function
        Used to instanciate all required objects/attributes
        """
        #Init mother class
        super(ImageViewer, self).__init__(parent)
        
        #Load interface
        loader = QtUiTools.QUiLoader()
        file = QtCore.QFile(os.path.abspath("ImageViewer.ui"))
        file.open(QtCore.QFile.ReadOnly)
        self.window = loader.load(file, parent) 
        file.close()        
        self.window.show()
        
        #Init variable
        self.hasList = []
        self.views   = []
        self.currentIndex = -1
        self.filter = [False,False,False,False,False]

        #connect Actions
        self.connectActions()
        
        #Init display
        self.figure  = Figure()
        self.canvas  = FigureCanvas(self.figure)
        self.displayareawidget.layout().addWidget(self.canvas)

        #Init tool bar
        self.addToolBar(QtCore.Qt.BottomToolBarArea, NavigationToolbar(self.canvas, self))

       

        
    def main(self):
        self.show()
        
    def openFile(self):
        fileName = QtGui.QFileDialog.getOpenFileName(
                        self,
                        "Select file ...",
                        QtCore.QDir.homePath(),
                        "has files (*.has)"
                    )
        if fileName:
            self.hasList = [str(fileName)]
            self.views = []
            self.currentIndex = 0
            self.processFiles()
        
    def nextFile(self):
        self.currentIndex = (self.currentIndex + 1) % len(self.hasList)
        self.display()
        
    def connectActions(self):
        self.actionQuit.triggered.connect(QtGui.qApp.quit)
        self.actionOpenFile.triggered.connect(self.openFile) 
        self.nextbutton.clicked.connect(self.nextFile)
        self.previousbutton.clicked.connect(self.prevFile)
        
        self.TiltXcheckBox.clicked.connect(self.updateFilter)
        self.TiltYcheckBox.clicked.connect(self.updateFilter)
        self.FocuscheckBox.clicked.connect(self.updateFilter)
        self.Astig0checkBox.clicked.connect(self.updateFilter)
        self.Astig45checkBox.clicked.connect(self.updateFilter)
        
    def prevFile(self):
        self.currentIndex = (self.currentIndex - 1) % len(self.hasList)
        self.display()
	
    def processFiles(self):
        self.views = [computeWF(f, self.filter) for f in self.hasList]
        self.display()
        
    def display(self) :
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.imshow(self.views[self.currentIndex], interpolation = 'none')
        self.canvas.draw()
        self.imagenamelabel.setText(self.hasList[self.currentIndex])
        
    def updateFilter(self):
        self.filter[0] = bool(self.TiltXcheckBox.isChecked()  )
        self.filter[1] = bool(self.TiltYcheckBox.isChecked()  )
        self.filter[2] = bool(self.FocuscheckBox.isChecked()  )
        self.filter[3] = bool(self.Astig0checkBox.isChecked() )
        self.filter[4] = bool(self.Astig45checkBox.isChecked())
        if self.currentIndex >= 0:
            self.processFiles()
        
if __name__=='__main__':
    app = QtGui.QApplication(sys.argv)
    ImageViewer = ImageViewer()
    ImageViewer.main()
    app.exec_()
