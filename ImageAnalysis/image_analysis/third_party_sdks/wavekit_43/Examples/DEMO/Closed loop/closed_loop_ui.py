# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'closed_loop.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(956, 731)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(956, 731))
        MainWindow.setStyleSheet(_fromUtf8("QMainWindow#MainWindow\n"
"{\n"
"    background-color: rgb(36, 36, 36);\n"
"    border:2px solid rgb(56, 56, 56);\n"
"    border-left:2px solid rgb(56, 56, 56);\n"
"}\n"
"\n"
"QWidget\n"
"{\n"
"    font: 10pt \"Open Sans\";\n"
"    color: white; \n"
"    background-color: rgb(66, 66, 66);\n"
"    selection-background-color :rgb(64, 167, 195);\n"
"}\n"
"\n"
"QFrame#AcquisitionParameters, QFrame#FilteringParameters, QFrame#Diagnosis, QFrame#Diagnosis_2, QFrame#EntryFile_2, QFrame#SaveFile, QFrame#SaveFileWavefront {\n"
"    background-color: rgb(36, 36, 36);\n"
"}\n"
"\n"
"QWidget#currentFile, QWidget#refFile, QWidget#Mask, QWidget#maskLoad {\n"
"    background-color: rgb(36, 36, 36);\n"
"}\n"
"\n"
"QMainWindow#mainWindow\n"
"{\n"
"    margin:5px 0px 5px 0px;\n"
"}\n"
"\n"
"/*toolbars*/\n"
"\n"
"\n"
"QToolBar {\n"
"    border: none;\n"
"    margin:2px;\n"
"}\n"
"\n"
"QToolBar {\n"
"    background-color: rgb(64, 167, 195);\n"
"    padding: 0px 0px 0px 5px;\n"
"    margin-bottom:5px;\n"
"}\n"
"\n"
"QToolBar QWidget {\n"
"    background-color: rgb(64, 167, 195);\n"
"}\n"
"\n"
"QMessageBox {\n"
"    background-color: rgb(56, 56, 56);\n"
"    border-bottom: 2px solid rgb(66, 66, 66);\n"
"    border-left: 2px solid rgb(66, 66, 66);\n"
"    border-right: 2px solid rgb(66, 66, 66);\n"
"}\n"
"\n"
"/* Push button */\n"
"\n"
"QPushButton\n"
"{\n"
"    background-color: rgb(56, 56, 56);\n"
"}\n"
"\n"
"QPushButton:disabled\n"
"{\n"
"    background-color: rgb(96, 96, 96);\n"
"}\n"
"\n"
"QPushButton:checked\n"
"{\n"
"    background-color: rgb(64, 167, 195);\n"
"}\n"
"\n"
"QFrame QPushButton\n"
"{\n"
"    background-color: rgb(96,96, 96);\n"
"    border: 0px solid white;\n"
"}\n"
"\n"
"/* Label */\n"
"\n"
"QLabel\n"
"{\n"
"    background-color: transparent;\n"
"}\n"
"\n"
"/* Tool Tip */\n"
"\n"
"QToolTip \n"
"{ \n"
"    border: 1px solid rgb(64, 167, 195);\n"
"}\n"
"\n"
"/* Checkbox */\n"
"\n"
"QCheckBox\n"
"{\n"
"    background-color:transparent;\n"
"    color: rgb(255, 255, 255);\n"
"}\n"
"\n"
"QCheckBox::indicator\n"
"{\n"
"    border: 1px solid white;\n"
"    background-color: rgb(56, 56, 56);\n"
"}\n"
"\n"
"QCheckBox::indicator:checked \n"
"{\n"
"    background-color:rgb(64, 167, 195);\n"
"}\n"
"\n"
"/* Combobox */\n"
"\n"
"QComboBox\n"
"{\n"
"    background-color: rgb(56, 56, 56);\n"
"    border: 1px solid rgb(66,66,66);\n"
"}\n"
"\n"
"/* Line edit */\n"
"\n"
"QLineEdit\n"
"{\n"
"    background-color: rgb(56, 56, 56);\n"
"    border: 1px solid rgb(96, 96, 96);\n"
"}\n"
"\n"
"/* Spinbox */\n"
"\n"
"QDoubleSpinBox, QSpinBox\n"
"{\n"
"    font: 10pt \"Open Sans\";\n"
"    background-color: rgb(56, 56, 56);\n"
"    border: 1px solid rgb(66,66,66);\n"
"    padding-right: 15px;\n"
"}\n"
"\n"
"QDoubleSpinBox:disabled, QSpinBox:disabled\n"
"{\n"
"    border: 0px solid rgb(66,66,66);\n"
"    padding-right: 5px;\n"
"}\n"
"\n"
"QDoubleSpinBox::up-button:disabled, QSpinBox::up-button:disabled, \n"
"QDoubleSpinBox::down-button:disabled, QSpinBox::down-button:disabled\n"
"{\n"
"    width: 0px;\n"
"    height: 0px;\n"
"}\n"
"\n"
"QTabWidget::pane { \n"
"    border: none;\n"
"    background-color: rgb(56, 56, 56);\n"
"}\n"
"\n"
"QTabBar::tab {\n"
"    background-color: rgb(66, 66, 66);\n"
"    border: none;\n"
"    padding:5px;\n"
"    padding-right: 25px;\n"
"    min-width: 40ex;\n"
"    margin-right:2px;\n"
"    margin-left:2px;\n"
"}\n"
"\n"
"QPushButton:enabled {\n"
"    background-color:#ffffff;\n"
"    color: black;\n"
"}\n"
"\n"
"QPushButton#AcquisitionStatus:enabled, QPushButton#HasoStatus:enabled, QPushButton#WFCStatus:enabled {\n"
"    background-color: green;\n"
"}\n"
"\n"
"QPushButton#AcquisitionStatus:disabled, QPushButton#HasoStatus:disabled, QPushButton#WFCStatus:disabled {\n"
"    background-color: red;\n"
"}\n"
"\n"
"QTabBar::tab:selected {\n"
"    color: white;\n"
"    font: bold 12pt \"Open Sans\";\n"
"    border : 1px solid white;\n"
"}\n"
"\n"
"QTabBar::tab:!selected {\n"
"    color: rgb(64, 167, 195);\n"
"    font: 12pt \"Open Sans\";\n"
"    border : 1px solid rgb(64, 167, 195);\n"
"}"))
        self.centralwidget = QtGui.QWidget(MainWindow)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.commandFrame = QtGui.QFrame(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.commandFrame.sizePolicy().hasHeightForWidth())
        self.commandFrame.setSizePolicy(sizePolicy)
        self.commandFrame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.commandFrame.setFrameShadow(QtGui.QFrame.Raised)
        self.commandFrame.setObjectName(_fromUtf8("commandFrame"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.commandFrame)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.widget_6 = QtGui.QWidget(self.commandFrame)
        self.widget_6.setObjectName(_fromUtf8("widget_6"))
        self.horizontalLayout_8 = QtGui.QHBoxLayout(self.widget_6)
        self.horizontalLayout_8.setObjectName(_fromUtf8("horizontalLayout_8"))
        self.verticalLayout_2.addWidget(self.widget_6)
        self.widget_4 = QtGui.QWidget(self.commandFrame)
        self.widget_4.setObjectName(_fromUtf8("widget_4"))
        self.horizontalLayout_6 = QtGui.QHBoxLayout(self.widget_4)
        self.horizontalLayout_6.setObjectName(_fromUtf8("horizontalLayout_6"))
        self.label_2 = QtGui.QLabel(self.widget_4)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout_6.addWidget(self.label_2)
        self.HasoStatus = QtGui.QPushButton(self.widget_4)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(10)
        sizePolicy.setVerticalStretch(10)
        sizePolicy.setHeightForWidth(self.HasoStatus.sizePolicy().hasHeightForWidth())
        self.HasoStatus.setSizePolicy(sizePolicy)
        self.HasoStatus.setMinimumSize(QtCore.QSize(10, 10))
        self.HasoStatus.setMaximumSize(QtCore.QSize(10, 10))
        self.HasoStatus.setText(_fromUtf8(""))
        self.HasoStatus.setObjectName(_fromUtf8("HasoStatus"))
        self.horizontalLayout_6.addWidget(self.HasoStatus)
        self.verticalLayout_2.addWidget(self.widget_4)
        self.widget = QtGui.QWidget(self.commandFrame)
        self.widget.setObjectName(_fromUtf8("widget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.widget)
        self.horizontalLayout.setMargin(0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.HasoFilePath = QtGui.QLineEdit(self.widget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(24)
        sizePolicy.setHeightForWidth(self.HasoFilePath.sizePolicy().hasHeightForWidth())
        self.HasoFilePath.setSizePolicy(sizePolicy)
        self.HasoFilePath.setMinimumSize(QtCore.QSize(0, 24))
        self.HasoFilePath.setObjectName(_fromUtf8("HasoFilePath"))
        self.horizontalLayout.addWidget(self.HasoFilePath)
        self.getHasoFileButton = QtGui.QPushButton(self.widget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(24)
        sizePolicy.setVerticalStretch(24)
        sizePolicy.setHeightForWidth(self.getHasoFileButton.sizePolicy().hasHeightForWidth())
        self.getHasoFileButton.setSizePolicy(sizePolicy)
        self.getHasoFileButton.setMinimumSize(QtCore.QSize(24, 24))
        self.getHasoFileButton.setMaximumSize(QtCore.QSize(24, 24))
        self.getHasoFileButton.setObjectName(_fromUtf8("getHasoFileButton"))
        self.horizontalLayout.addWidget(self.getHasoFileButton)
        self.verticalLayout_2.addWidget(self.widget)
        self.WFSconnectButton = QtGui.QPushButton(self.commandFrame)
        self.WFSconnectButton.setMinimumSize(QtCore.QSize(0, 24))
        self.WFSconnectButton.setObjectName(_fromUtf8("WFSconnectButton"))
        self.verticalLayout_2.addWidget(self.WFSconnectButton)
        self.WFSdisconnectButton = QtGui.QPushButton(self.commandFrame)
        self.WFSdisconnectButton.setObjectName(_fromUtf8("WFSdisconnectButton"))
        self.verticalLayout_2.addWidget(self.WFSdisconnectButton)
        self.widget_5 = QtGui.QWidget(self.commandFrame)
        self.widget_5.setObjectName(_fromUtf8("widget_5"))
        self.horizontalLayout_7 = QtGui.QHBoxLayout(self.widget_5)
        self.horizontalLayout_7.setObjectName(_fromUtf8("horizontalLayout_7"))
        self.label_3 = QtGui.QLabel(self.widget_5)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.horizontalLayout_7.addWidget(self.label_3)
        self.WFCStatus = QtGui.QPushButton(self.widget_5)
        self.WFCStatus.setMinimumSize(QtCore.QSize(10, 10))
        self.WFCStatus.setMaximumSize(QtCore.QSize(10, 10))
        self.WFCStatus.setText(_fromUtf8(""))
        self.WFCStatus.setObjectName(_fromUtf8("WFCStatus"))
        self.horizontalLayout_7.addWidget(self.WFCStatus)
        self.verticalLayout_2.addWidget(self.widget_5)
        self.widget_3 = QtGui.QWidget(self.commandFrame)
        self.widget_3.setObjectName(_fromUtf8("widget_3"))
        self.horizontalLayout_5 = QtGui.QHBoxLayout(self.widget_3)
        self.horizontalLayout_5.setMargin(0)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.WFCFilePath = QtGui.QLineEdit(self.widget_3)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(24)
        sizePolicy.setHeightForWidth(self.WFCFilePath.sizePolicy().hasHeightForWidth())
        self.WFCFilePath.setSizePolicy(sizePolicy)
        self.WFCFilePath.setMinimumSize(QtCore.QSize(0, 24))
        self.WFCFilePath.setObjectName(_fromUtf8("WFCFilePath"))
        self.horizontalLayout_5.addWidget(self.WFCFilePath)
        self.getWFCFileButton = QtGui.QPushButton(self.widget_3)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(24)
        sizePolicy.setVerticalStretch(24)
        sizePolicy.setHeightForWidth(self.getWFCFileButton.sizePolicy().hasHeightForWidth())
        self.getWFCFileButton.setSizePolicy(sizePolicy)
        self.getWFCFileButton.setMinimumSize(QtCore.QSize(24, 24))
        self.getWFCFileButton.setMaximumSize(QtCore.QSize(24, 24))
        self.getWFCFileButton.setObjectName(_fromUtf8("getWFCFileButton"))
        self.horizontalLayout_5.addWidget(self.getWFCFileButton)
        self.verticalLayout_2.addWidget(self.widget_3)
        self.WFCconnectButton = QtGui.QPushButton(self.commandFrame)
        self.WFCconnectButton.setMinimumSize(QtCore.QSize(0, 24))
        self.WFCconnectButton.setObjectName(_fromUtf8("WFCconnectButton"))
        self.verticalLayout_2.addWidget(self.WFCconnectButton)
        self.WFCdisconnectButton = QtGui.QPushButton(self.commandFrame)
        self.WFCdisconnectButton.setObjectName(_fromUtf8("WFCdisconnectButton"))
        self.verticalLayout_2.addWidget(self.WFCdisconnectButton)
        self.label_4 = QtGui.QLabel(self.commandFrame)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.verticalLayout_2.addWidget(self.label_4)
        self.widget_7 = QtGui.QWidget(self.commandFrame)
        self.widget_7.setObjectName(_fromUtf8("widget_7"))
        self.horizontalLayout_9 = QtGui.QHBoxLayout(self.widget_7)
        self.horizontalLayout_9.setMargin(0)
        self.horizontalLayout_9.setSpacing(0)
        self.horizontalLayout_9.setObjectName(_fromUtf8("horizontalLayout_9"))
        self.BackupFilePath = QtGui.QLineEdit(self.widget_7)
        self.BackupFilePath.setMinimumSize(QtCore.QSize(0, 24))
        self.BackupFilePath.setObjectName(_fromUtf8("BackupFilePath"))
        self.horizontalLayout_9.addWidget(self.BackupFilePath)
        self.getBackupFileButton = QtGui.QPushButton(self.widget_7)
        self.getBackupFileButton.setMinimumSize(QtCore.QSize(24, 24))
        self.getBackupFileButton.setMaximumSize(QtCore.QSize(24, 24))
        self.getBackupFileButton.setObjectName(_fromUtf8("getBackupFileButton"))
        self.horizontalLayout_9.addWidget(self.getBackupFileButton)
        self.verticalLayout_2.addWidget(self.widget_7)
        self.widget_2 = QtGui.QWidget(self.commandFrame)
        self.widget_2.setObjectName(_fromUtf8("widget_2"))
        self.horizontalLayout_4 = QtGui.QHBoxLayout(self.widget_2)
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.label = QtGui.QLabel(self.widget_2)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout_4.addWidget(self.label)
        self.AcquisitionStatus = QtGui.QPushButton(self.widget_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.AcquisitionStatus.sizePolicy().hasHeightForWidth())
        self.AcquisitionStatus.setSizePolicy(sizePolicy)
        self.AcquisitionStatus.setMinimumSize(QtCore.QSize(10, 10))
        self.AcquisitionStatus.setMaximumSize(QtCore.QSize(10, 10))
        self.AcquisitionStatus.setText(_fromUtf8(""))
        self.AcquisitionStatus.setObjectName(_fromUtf8("AcquisitionStatus"))
        self.horizontalLayout_4.addWidget(self.AcquisitionStatus)
        self.verticalLayout_2.addWidget(self.widget_2)
        self.nbLoop = QtGui.QSpinBox(self.commandFrame)
        self.nbLoop.setMinimumSize(QtCore.QSize(0, 24))
        self.nbLoop.setMinimum(1)
        self.nbLoop.setMaximum(100)
        self.nbLoop.setProperty("value", 10)
        self.nbLoop.setObjectName(_fromUtf8("nbLoop"))
        self.verticalLayout_2.addWidget(self.nbLoop)
        self.StartButton = QtGui.QPushButton(self.commandFrame)
        self.StartButton.setMinimumSize(QtCore.QSize(64, 30))
        self.StartButton.setObjectName(_fromUtf8("StartButton"))
        self.verticalLayout_2.addWidget(self.StartButton)
        self.widget_8 = QtGui.QWidget(self.commandFrame)
        self.widget_8.setObjectName(_fromUtf8("widget_8"))
        self.verticalLayout_5 = QtGui.QVBoxLayout(self.widget_8)
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.label_5 = QtGui.QLabel(self.widget_8)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.verticalLayout_5.addWidget(self.label_5)
        self.RMSValue = QtGui.QLabel(self.widget_8)
        self.RMSValue.setText(_fromUtf8(""))
        self.RMSValue.setObjectName(_fromUtf8("RMSValue"))
        self.verticalLayout_5.addWidget(self.RMSValue)
        self.label_6 = QtGui.QLabel(self.widget_8)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.verticalLayout_5.addWidget(self.label_6)
        self.PVValue = QtGui.QLabel(self.widget_8)
        self.PVValue.setText(_fromUtf8(""))
        self.PVValue.setObjectName(_fromUtf8("PVValue"))
        self.verticalLayout_5.addWidget(self.PVValue)
        self.verticalLayout_2.addWidget(self.widget_8)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.horizontalLayout_3.addWidget(self.commandFrame)
        self.displayFrame = QtGui.QFrame(self.centralwidget)
        self.displayFrame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.displayFrame.setFrameShadow(QtGui.QFrame.Raised)
        self.displayFrame.setObjectName(_fromUtf8("displayFrame"))
        self.verticalLayout = QtGui.QVBoxLayout(self.displayFrame)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.ImageDisplay = QtGui.QWidget(self.displayFrame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ImageDisplay.sizePolicy().hasHeightForWidth())
        self.ImageDisplay.setSizePolicy(sizePolicy)
        self.ImageDisplay.setMinimumSize(QtCore.QSize(0, 400))
        self.ImageDisplay.setObjectName(_fromUtf8("ImageDisplay"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.ImageDisplay)
        self.verticalLayout_3.setMargin(0)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.verticalLayout.addWidget(self.ImageDisplay)
        self.PlotDisplay = QtGui.QWidget(self.displayFrame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PlotDisplay.sizePolicy().hasHeightForWidth())
        self.PlotDisplay.setSizePolicy(sizePolicy)
        self.PlotDisplay.setMinimumSize(QtCore.QSize(0, 200))
        self.PlotDisplay.setMaximumSize(QtCore.QSize(16777215, 200))
        self.PlotDisplay.setObjectName(_fromUtf8("PlotDisplay"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.PlotDisplay)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.verticalLayout.addWidget(self.PlotDisplay)
        self.horizontalLayout_3.addWidget(self.displayFrame)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionPlay = QtGui.QAction(MainWindow)
        self.actionPlay.setCheckable(True)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/WaveView/icon/Play")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/WaveView/icon/Pause")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionPlay.setIcon(icon)
        self.actionPlay.setObjectName(_fromUtf8("actionPlay"))

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "Closed loop", None))
        self.label_2.setText(_translate("MainWindow", "HASO Sensor file", None))
        self.getHasoFileButton.setText(_translate("MainWindow", "....", None))
        self.WFSconnectButton.setText(_translate("MainWindow", "Connect HASO Sensor", None))
        self.WFSdisconnectButton.setText(_translate("MainWindow", "Disconnect HASO", None))
        self.label_3.setText(_translate("MainWindow", "Corrector file", None))
        self.getWFCFileButton.setText(_translate("MainWindow", "....", None))
        self.WFCconnectButton.setText(_translate("MainWindow", "Connect Corrector", None))
        self.WFCdisconnectButton.setText(_translate("MainWindow", "Disconnect Corrector", None))
        self.label_4.setText(_translate("MainWindow", "Correction backup file", None))
        self.getBackupFileButton.setText(_translate("MainWindow", ".....", None))
        self.label.setText(_translate("MainWindow", "Closed loop", None))
        self.StartButton.setText(_translate("MainWindow", "Start", None))
        self.label_5.setText(_translate("MainWindow", "RMS (µm)", None))
        self.label_6.setText(_translate("MainWindow", "PV (µm)", None))
        self.actionPlay.setText(_translate("MainWindow", "Play", None))

