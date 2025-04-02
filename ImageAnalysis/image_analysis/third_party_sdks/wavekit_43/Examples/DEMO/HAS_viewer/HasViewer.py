#!/usr/bin/python

import subprocess
import os, sys, glob
import ctypes
import re

dependencies = ['matplotlib', 'numpy']
subprocess.call([sys.executable, '-m', 'pip', 'install'] + dependencies)

import numpy as np
from PySide2 import QtGui, QtCore, QtWidgets, QtUiTools

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

import imageViewer
import os,sys
sys.path.append(os.path.dirname(__file__)  + '/../../../')
import wavekit_py as wkpy

class float2D(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]
class uint2D(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint), ("y", ctypes.c_uint)]

def getParent(f):
    if(f.parent().data() != None):
        return getParent(f.parent()) + "\\" + f.data()
    else:
        m = re.search(r'\((?P<DriveName>[A-Z]\:)\)', f.data())
        if m is not None:
            return m.group('DriveName') + "\\"
        else:
            return f.data().toString()

def computeWF(hasodata_file_path):
    
    E_COMPUTEPHASESET_ZONAL = 3  # ///< incremental phase reconstruction algorithm

    filter = (ctypes.c_char * 5)()
    filter[0] = ctypes.c_char(1)
    filter[1] = ctypes.c_char(1)
    filter[2] = ctypes.c_char(1)
    filter[3] = ctypes.c_char(1)
    filter[4] = ctypes.c_char(1)
       
    hasoslopes = wkpy.HasoSlopes(has_file_path = hasodata_file_path)
    phase = wkpy.Phase(hasoslopes = hasoslopes,
                type_ = wkpy.E_COMPUTEPHASESET.ZONAL, 
                filter_ = filter  )
    
    sn, dimensions = hasoslopes.get_info()
    buf = ctypes.c_float*(dimensions.size.X*dimensions.size.Y)
    
    buf, pupil = phase.get_data()
    

    return buf

class HasViewer(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        """Init Function
        Used to instanciate all required objects/attributes
        """
        #Init mother class
        super(HasViewer, self).__init__(parent)
        
        #Load interface
        loader = QtUiTools.QUiLoader()
        file = QtCore.QFile(os.path.abspath("HasViewer.ui"))
        file.open(QtCore.QFile.ReadOnly)
        self.window = loader.load(file, parent) 
        file.close()        
        self.window.show()
        
        #Init variable
        self.hasList = []
        self.views   = []
        self.currentIndex = -1
        self.doubleClick = False
        
        #connect actions
        self.connectActions()

        #Init treeView
        self.model = QtWidgets.QFileSystemModel()
        filters = ["*.has"]
        self.DirName = QtCore.QDir.homePath()
        self.model.setRootPath('')
        self.model.setFilter(QtCore.QDir.AllDirs | QtCore.QDir.NoDotAndDotDot | QtCore.QDir.AllEntries)
        self.model.setNameFilters(filters) 
        self.model.setNameFilterDisables(False) 
        self.window.treeView.setModel(self.model)
        
        #Add splitter
        self.window.splitter.setStretchFactor(1,0)
        self.window.treeView.setColumnWidth(0, 160)
        self.window.treeView.resizeColumnToContents(True)
        
        #Define policy
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.window.treeView.setSizePolicy(sizePolicy)
        
        #Hide column of treeView
        self.window.treeView.hideColumn(1)
        self.window.treeView.hideColumn(2)
        self.window.treeView.hideColumn(3)

        #Init list of has file
        self.processFiles()
        
        
        
        
    def main(self):
        self.show()

    def clickedEvent(self, index):
        ImageViewer = imageViewer.ImageViewer()
        ImageViewer.hasList = self.hasList
        ImageViewer.currentIndex = index #self.currentIndex
        ImageViewer.processFiles()
        ImageViewer.show()

    def connectActions(self):
        self.window.actionQuit.triggered.connect(QtWidgets.QApplication.quit)
        self.window.actionOpenDir.triggered.connect(self.openDirFromMenu)
        self.window.treeView.doubleClicked.connect(self.setDoubleClick)
        self.window.treeView.clicked.connect(self.showSomething)

    def setDoubleClick(self):
        self.doubleClick = True
    
    def showSomething(self):
        if self.doubleClick :
            self.openDirFromTree()
            self.doubleClick = False
        else :
            self.showSlopesNumber()

    def showSlopesNumber(self):
        ok = self.clearLayout(self.window.gridLayout);
        item = self.window.treeView.selectedIndexes()[0]
        DirName = getParent(item)
        self.hasList = []
        self.hasList = [os.path.abspath(f) for f in glob.glob(os.path.join(str(DirName), "*.has"))]
        self.window.slopesNumberBar.clearMessage()
        self.window.slopesNumberBar.showMessage("Slopes number : " + str(len(self.hasList)))
        self.addWidgetNumber()
        
    def addWidgetNumber(self):
        slopesNumber = QtWidgets.QLabel()
        slopesNumber.setAlignment(QtCore.Qt.AlignCenter)
        slopesNumber.setText("Slopes number : " + str(len(self.hasList)) + ".\nDouble click to view thumbnails and expand directory.")
        self.window.gridLayout.addWidget(slopesNumber)
        
    def openDirFromTree(self):
        ok = self.clearLayout((self.window.gridLayout))

        item = self.window.treeView.selectedIndexes()[0]
        DirName = getParent(item)

        # update header
        model1 = QtGui.QStandardItemModel(0,1,self)
        model1.setHeaderData(0, QtCore.Qt.Horizontal, str(DirName))
        self.DirName = DirName
        self.window.treeView.header().setModel(model1)
        self.hasList = []
        self.hasList = [os.path.abspath(f) for f in glob.glob(os.path.join(str(DirName), "*.has"))]    
        
        self.processFiles()

    def openDirFromMenu(self):
        if self.DirName != None:
            dirFile = self.DirName 
        else: 
            dirFile = QtCore.QDir.homePath()
        DirName = QtGui.QFileDialog.getExistingDirectory(
                        self,
                        "Select directory ...",
                        dirFile,
                        QtGui.QFileDialog.ShowDirsOnly | QtGui.QFileDialog.DontResolveSymlinks
                    )
        if DirName:
            self.index = self.model.index(DirName)
            self.treeView.setCurrentIndex(self.index)
            self.openDirFromTree()

    def processFiles(self):
        self.views = [computeWF(f) for f in self.hasList]
        self.display()

    def display(self):
        minSize = QtCore.QSize(150,150)
        spacing = self.window.scrollAreaWidgetContents.layout().spacing()
        # remplacer le spacing soustrait par margin left + right
        contentsMargins = self.window.gridLayout.contentsMargins()
        margin = contentsMargins.left() + contentsMargins.right()
        ncols = ((self.window.scrollArea.viewport().width() - margin) / (minSize.width() + spacing))
        col  = 0 
        row  = 0
        for (index,f) in enumerate(self.views):
            slopesGroup = QtWidgets(self.imageListBox)
            slopesGroup.setMinimumSize(minSize)
            slopesLayout = QtWidgets.QGridLayout()

            figure = Figure()
            canvas = FigureCanvas(figure)
            slopesLayout.addWidget(canvas,0,0)
            
            fileName, fileExtension = os.path.splitext(os.path.basename(self.hasList[index]))
            slopesLabel = QtWidgets.QLabel(slopesGroup)
            slopesLabel.setText(fileName)
            slopesLayout.addWidget(slopesLabel,1,0)
            ax = figure.add_subplot(111)
            ax.clear()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.imshow(f, interpolation = 'none')

            canvas.mouseReleaseEvent = lambda event, arg = index : self.clickedEvent(arg)
            canvas.draw()
            slopesGroup.setLayout(slopesLayout)
            self.gridLayout.addWidget(slopesGroup, row, col)

            col = col+1
            if col >= ncols:
                col = 0
                row = row + 1

    def clearLayout(self, layout):
        if layout != None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget() is not None:
                    child.widget().deleteLater()
                elif child.layout() is not None:
                    clearLayout(child.layout())
        self.currentIndex = -1
        self.hasList = []
        return True
    
if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    HasViewer = HasViewer()
    app.exec_()
