# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\ScAnalyzer.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(906, 421)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.buttonStart = QtWidgets.QPushButton(self.centralwidget)
        self.buttonStart.setGeometry(QtCore.QRect(20, 350, 93, 28))
        self.buttonStart.setObjectName("buttonStart")
        self.buttonStop = QtWidgets.QPushButton(self.centralwidget)
        self.buttonStop.setGeometry(QtCore.QRect(130, 350, 93, 28))
        self.buttonStop.setObjectName("buttonStop")
        self.inputYear = QtWidgets.QLineEdit(self.centralwidget)
        self.inputYear.setGeometry(QtCore.QRect(100, 50, 113, 22))
        self.inputYear.setObjectName("inputYear")
        self.inputMonth = QtWidgets.QLineEdit(self.centralwidget)
        self.inputMonth.setGeometry(QtCore.QRect(100, 80, 113, 22))
        self.inputMonth.setObjectName("inputMonth")
        self.inputDay = QtWidgets.QLineEdit(self.centralwidget)
        self.inputDay.setGeometry(QtCore.QRect(100, 110, 113, 22))
        self.inputDay.setObjectName("inputDay")
        self.labelYear = QtWidgets.QLabel(self.centralwidget)
        self.labelYear.setGeometry(QtCore.QRect(50, 50, 31, 21))
        self.labelYear.setObjectName("labelYear")
        self.labelMonth = QtWidgets.QLabel(self.centralwidget)
        self.labelMonth.setGeometry(QtCore.QRect(50, 80, 41, 21))
        self.labelMonth.setObjectName("labelMonth")
        self.labelDay = QtWidgets.QLabel(self.centralwidget)
        self.labelDay.setGeometry(QtCore.QRect(50, 110, 41, 21))
        self.labelDay.setObjectName("labelDay")
        self.logDisplay = QtWidgets.QTextBrowser(self.centralwidget)
        self.logDisplay.setGeometry(QtCore.QRect(240, 50, 631, 311))
        self.logDisplay.setObjectName("logDisplay")
        self.inputStartScan = QtWidgets.QLineEdit(self.centralwidget)
        self.inputStartScan.setGeometry(QtCore.QRect(100, 180, 113, 22))
        self.inputStartScan.setObjectName("inputStartScan")
        self.labelStartScan = QtWidgets.QLabel(self.centralwidget)
        self.labelStartScan.setGeometry(QtCore.QRect(30, 180, 61, 21))
        self.labelStartScan.setObjectName("labelStartScan")
        self.labelIgnore = QtWidgets.QLabel(self.centralwidget)
        self.labelIgnore.setGeometry(QtCore.QRect(20, 210, 81, 21))
        self.labelIgnore.setObjectName("labelIgnore")
        self.inputIgnore = QtWidgets.QLineEdit(self.centralwidget)
        self.inputIgnore.setGeometry(QtCore.QRect(100, 210, 113, 22))
        self.inputIgnore.setObjectName("inputIgnore")
        self.labelScanSettings = QtWidgets.QLabel(self.centralwidget)
        self.labelScanSettings.setGeometry(QtCore.QRect(70, 150, 81, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.labelScanSettings.setFont(font)
        self.labelScanSettings.setObjectName("labelScanSettings")
        self.buttonAnalysisActivator = QtWidgets.QPushButton(self.centralwidget)
        self.buttonAnalysisActivator.setGeometry(QtCore.QRect(740, 10, 121, 31))
        self.buttonAnalysisActivator.setAutoDefault(False)
        self.buttonAnalysisActivator.setDefault(False)
        self.buttonAnalysisActivator.setFlat(False)
        self.buttonAnalysisActivator.setObjectName("buttonAnalysisActivator")
        self.checkBoxOverwrite = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBoxOverwrite.setGeometry(QtCore.QRect(20, 250, 191, 20))
        self.checkBoxOverwrite.setObjectName("checkBoxOverwrite")
        self.lineDocumentID = QtWidgets.QLineEdit(self.centralwidget)
        self.lineDocumentID.setGeometry(QtCore.QRect(100, 310, 113, 22))
        self.lineDocumentID.setObjectName("lineDocumentID")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 310, 81, 21))
        self.label.setObjectName("label")
        self.checkBoxScanlog = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBoxScanlog.setGeometry(QtCore.QRect(20, 280, 181, 20))
        self.checkBoxScanlog.setObjectName("checkBoxScanlog")
        self.lineExperimentName = QtWidgets.QLineEdit(self.centralwidget)
        self.lineExperimentName.setGeometry(QtCore.QRect(360, 20, 113, 22))
        self.lineExperimentName.setReadOnly(True)
        self.lineExperimentName.setObjectName("lineExperimentName")
        self.labelExperimentName = QtWidgets.QLabel(self.centralwidget)
        self.labelExperimentName.setGeometry(QtCore.QRect(250, 20, 111, 21))
        self.labelExperimentName.setObjectName("labelExperimentName")
        self.buttonReset = QtWidgets.QPushButton(self.centralwidget)
        self.buttonReset.setGeometry(QtCore.QRect(480, 20, 51, 21))
        self.buttonReset.setObjectName("buttonReset")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.buttonStart.setText(_translate("MainWindow", "Start"))
        self.buttonStop.setText(_translate("MainWindow", "Stop"))
        self.labelYear.setText(_translate("MainWindow", "Year"))
        self.labelMonth.setText(_translate("MainWindow", "Month"))
        self.labelDay.setText(_translate("MainWindow", "Day"))
        self.labelStartScan.setText(_translate("MainWindow", "Start Scan"))
        self.labelIgnore.setText(_translate("MainWindow", "Ignore Scans"))
        self.labelScanSettings.setText(_translate("MainWindow", "Scan Settings"))
        self.buttonAnalysisActivator.setText(_translate("MainWindow", "Analysis Activator"))
        self.checkBoxOverwrite.setText(_translate("MainWindow", "Overwrite Processed Scans"))
        self.label.setText(_translate("MainWindow", "DocumentID"))
        self.checkBoxScanlog.setText(_translate("MainWindow", "Write to Scan Log"))
        self.labelExperimentName.setText(_translate("MainWindow", "Experiment Name:"))
        self.buttonReset.setText(_translate("MainWindow", "Reset"))
