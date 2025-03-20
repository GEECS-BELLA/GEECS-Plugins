# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'client.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

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
"QPushButton#clientStatus:enabled, QPushButton#serverStatus:enabled {\n"
"    background-color: green;\n"
"}\n"
"\n"
"QPushButton#clientStatus:disabled, QPushButton#serverStatus:disabled {\n"
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
        self.serverFrame = QtGui.QFrame(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.serverFrame.sizePolicy().hasHeightForWidth())
        self.serverFrame.setSizePolicy(sizePolicy)
        self.serverFrame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.serverFrame.setFrameShadow(QtGui.QFrame.Raised)
        self.serverFrame.setObjectName(_fromUtf8("serverFrame"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.serverFrame)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.widget = QtGui.QWidget(self.serverFrame)
        self.widget.setObjectName(_fromUtf8("widget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.widget)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label_2 = QtGui.QLabel(self.widget)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout.addWidget(self.label_2)
        self.clientStatus = QtGui.QPushButton(self.widget)
        self.clientStatus.setMinimumSize(QtCore.QSize(10, 10))
        self.clientStatus.setMaximumSize(QtCore.QSize(10, 10))
        self.clientStatus.setText(_fromUtf8(""))
        self.clientStatus.setObjectName(_fromUtf8("clientStatus"))
        self.horizontalLayout.addWidget(self.clientStatus)
        self.verticalLayout_2.addWidget(self.widget)
        self.ClientStartButton = QtGui.QPushButton(self.serverFrame)
        self.ClientStartButton.setMinimumSize(QtCore.QSize(64, 30))
        self.ClientStartButton.setObjectName(_fromUtf8("ClientStartButton"))
        self.verticalLayout_2.addWidget(self.ClientStartButton)
        self.ClientStopButton = QtGui.QPushButton(self.serverFrame)
        self.ClientStopButton.setMinimumSize(QtCore.QSize(64, 30))
        self.ClientStopButton.setObjectName(_fromUtf8("ClientStopButton"))
        self.verticalLayout_2.addWidget(self.ClientStopButton)
        self.widget_2 = QtGui.QWidget(self.serverFrame)
        self.widget_2.setObjectName(_fromUtf8("widget_2"))
        self.horizontalLayout_4 = QtGui.QHBoxLayout(self.widget_2)
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.label = QtGui.QLabel(self.widget_2)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout_4.addWidget(self.label)
        self.serverStatus = QtGui.QPushButton(self.widget_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.serverStatus.sizePolicy().hasHeightForWidth())
        self.serverStatus.setSizePolicy(sizePolicy)
        self.serverStatus.setMinimumSize(QtCore.QSize(10, 10))
        self.serverStatus.setMaximumSize(QtCore.QSize(10, 10))
        self.serverStatus.setText(_fromUtf8(""))
        self.serverStatus.setObjectName(_fromUtf8("serverStatus"))
        self.horizontalLayout_4.addWidget(self.serverStatus)
        self.verticalLayout_2.addWidget(self.widget_2)
        self.ServerStartButton = QtGui.QPushButton(self.serverFrame)
        self.ServerStartButton.setMinimumSize(QtCore.QSize(64, 30))
        self.ServerStartButton.setObjectName(_fromUtf8("ServerStartButton"))
        self.verticalLayout_2.addWidget(self.ServerStartButton)
        self.ServerStopButton = QtGui.QPushButton(self.serverFrame)
        self.ServerStopButton.setMinimumSize(QtCore.QSize(64, 30))
        self.ServerStopButton.setObjectName(_fromUtf8("ServerStopButton"))
        self.verticalLayout_2.addWidget(self.ServerStopButton)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.horizontalLayout_3.addWidget(self.serverFrame)
        self.clientFrame = QtGui.QFrame(self.centralwidget)
        self.clientFrame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.clientFrame.setFrameShadow(QtGui.QFrame.Raised)
        self.clientFrame.setObjectName(_fromUtf8("clientFrame"))
        self.verticalLayout = QtGui.QVBoxLayout(self.clientFrame)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.ImageDisplay = QtGui.QWidget(self.clientFrame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ImageDisplay.sizePolicy().hasHeightForWidth())
        self.ImageDisplay.setSizePolicy(sizePolicy)
        self.ImageDisplay.setObjectName(_fromUtf8("ImageDisplay"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.ImageDisplay)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.verticalLayout.addWidget(self.ImageDisplay)
        self.horizontalLayout_3.addWidget(self.clientFrame)
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
        MainWindow.setWindowTitle(_translate("MainWindow", "Client viewer", None))
        self.label_2.setText(_translate("MainWindow", "Client", None))
        self.ClientStartButton.setText(_translate("MainWindow", "Start", None))
        self.ClientStopButton.setText(_translate("MainWindow", "Stop", None))
        self.label.setText(_translate("MainWindow", "Server", None))
        self.ServerStartButton.setText(_translate("MainWindow", "Start", None))
        self.ServerStopButton.setText(_translate("MainWindow", "Stop", None))
        self.actionPlay.setText(_translate("MainWindow", "Play", None))

