# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\v4.2\Tools\Tools_python\has_viewer\ImageViewer.ui'
#
# Created: Tue Mar 26 14:39:39 2019
#      by: PyQt4 UI code generator 4.9.6
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
        MainWindow.resize(620, 420)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.widget_2 = QtGui.QWidget(self.centralwidget)
        self.widget_2.setObjectName(_fromUtf8("widget_2"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.widget_2)
        self.verticalLayout_3.setMargin(0)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.label = QtGui.QLabel(self.widget_2)
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout_3.addWidget(self.label)
        self.TiltXcheckBox = QtGui.QCheckBox(self.widget_2)
        self.TiltXcheckBox.setObjectName(_fromUtf8("TiltXcheckBox"))
        self.verticalLayout_3.addWidget(self.TiltXcheckBox)
        self.TiltYcheckBox = QtGui.QCheckBox(self.widget_2)
        self.TiltYcheckBox.setObjectName(_fromUtf8("TiltYcheckBox"))
        self.verticalLayout_3.addWidget(self.TiltYcheckBox)
        self.FocuscheckBox = QtGui.QCheckBox(self.widget_2)
        self.FocuscheckBox.setObjectName(_fromUtf8("FocuscheckBox"))
        self.verticalLayout_3.addWidget(self.FocuscheckBox)
        self.Astig0checkBox = QtGui.QCheckBox(self.widget_2)
        self.Astig0checkBox.setObjectName(_fromUtf8("Astig0checkBox"))
        self.verticalLayout_3.addWidget(self.Astig0checkBox)
        self.Astig45checkBox = QtGui.QCheckBox(self.widget_2)
        self.Astig45checkBox.setObjectName(_fromUtf8("Astig45checkBox"))
        self.verticalLayout_3.addWidget(self.Astig45checkBox)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem)
        self.verticalLayout_2.addWidget(self.widget_2)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtGui.QLayout.SetDefaultConstraint)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.imageviewerwidget = QtGui.QWidget(self.centralwidget)
        self.imageviewerwidget.setObjectName(_fromUtf8("imageviewerwidget"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.imageviewerwidget)
        self.verticalLayout_4.setMargin(0)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.navigationlayout = QtGui.QHBoxLayout()
        self.navigationlayout.setSizeConstraint(QtGui.QLayout.SetDefaultConstraint)
        self.navigationlayout.setObjectName(_fromUtf8("navigationlayout"))
        self.imagenamelabel = QtGui.QLabel(self.imageviewerwidget)
        self.imagenamelabel.setObjectName(_fromUtf8("imagenamelabel"))
        self.navigationlayout.addWidget(self.imagenamelabel)
        self.previousbutton = QtGui.QPushButton(self.imageviewerwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.previousbutton.sizePolicy().hasHeightForWidth())
        self.previousbutton.setSizePolicy(sizePolicy)
        self.previousbutton.setObjectName(_fromUtf8("previousbutton"))
        self.navigationlayout.addWidget(self.previousbutton)
        self.nextbutton = QtGui.QPushButton(self.imageviewerwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.nextbutton.sizePolicy().hasHeightForWidth())
        self.nextbutton.setSizePolicy(sizePolicy)
        self.nextbutton.setObjectName(_fromUtf8("nextbutton"))
        self.navigationlayout.addWidget(self.nextbutton)
        self.verticalLayout_4.addLayout(self.navigationlayout)
        self.displayareawidget = QtGui.QWidget(self.imageviewerwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.displayareawidget.sizePolicy().hasHeightForWidth())
        self.displayareawidget.setSizePolicy(sizePolicy)
        self.displayareawidget.setObjectName(_fromUtf8("displayareawidget"))
        self.verticalLayout_5 = QtGui.QVBoxLayout(self.displayareawidget)
        self.verticalLayout_5.setMargin(0)
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.verticalLayout_4.addWidget(self.displayareawidget)
        self.verticalLayout_4.setStretch(1, 1)
        self.verticalLayout.addWidget(self.imageviewerwidget)
        self.horizontalLayout.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 620, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        MainWindow.setMenuBar(self.menubar)
        self.actionOpenFile = QtGui.QAction(MainWindow)
        self.actionOpenFile.setObjectName(_fromUtf8("actionOpenFile"))
        self.actionQuit = QtGui.QAction(MainWindow)
        self.actionQuit.setObjectName(_fromUtf8("actionQuit"))
        self.menuFile.addAction(self.actionOpenFile)
        self.menuFile.addAction(self.actionQuit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.label.setText(_translate("MainWindow", "Filter", None))
        self.TiltXcheckBox.setText(_translate("MainWindow", "Tilt X", None))
        self.TiltYcheckBox.setText(_translate("MainWindow", "Tilt Y", None))
        self.FocuscheckBox.setText(_translate("MainWindow", "Focus", None))
        self.Astig0checkBox.setText(_translate("MainWindow", "Astig 0", None))
        self.Astig45checkBox.setText(_translate("MainWindow", "Astig 45", None))
        self.imagenamelabel.setText(_translate("MainWindow", "File name", None))
        self.previousbutton.setText(_translate("MainWindow", "Previous", None))
        self.nextbutton.setText(_translate("MainWindow", "Next", None))
        self.menuFile.setTitle(_translate("MainWindow", "File", None))
        self.actionOpenFile.setText(_translate("MainWindow", "Open File ...", None))
        self.actionOpenFile.setIconText(_translate("MainWindow", "Open File", None))
        self.actionOpenFile.setToolTip(_translate("MainWindow", "Open File", None))
        self.actionQuit.setText(_translate("MainWindow", "Quit", None))

