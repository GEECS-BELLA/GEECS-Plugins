# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\ShotControlEditor.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 430)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(20, 20, 121, 21))
        self.label.setObjectName("label")
        self.lineConfigurationSelect = QtWidgets.QLineEdit(Dialog)
        self.lineConfigurationSelect.setGeometry(QtCore.QRect(150, 20, 231, 22))
        self.lineConfigurationSelect.setObjectName("lineConfigurationSelect")
        self.buttonNewConfiguration = QtWidgets.QPushButton(Dialog)
        self.buttonNewConfiguration.setGeometry(QtCore.QRect(50, 50, 81, 28))
        self.buttonNewConfiguration.setObjectName("buttonNewConfiguration")
        self.buttonDeleteConfiguration = QtWidgets.QPushButton(Dialog)
        self.buttonDeleteConfiguration.setGeometry(QtCore.QRect(270, 50, 81, 28))
        self.buttonDeleteConfiguration.setObjectName("buttonDeleteConfiguration")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(20, 100, 131, 21))
        self.label_2.setObjectName("label_2")
        self.lineDeviceName = QtWidgets.QLineEdit(Dialog)
        self.lineDeviceName.setGeometry(QtCore.QRect(150, 100, 231, 22))
        self.lineDeviceName.setObjectName("lineDeviceName")
        self.buttonCopyConfiguration = QtWidgets.QPushButton(Dialog)
        self.buttonCopyConfiguration.setGeometry(QtCore.QRect(160, 50, 81, 28))
        self.buttonCopyConfiguration.setObjectName("buttonCopyConfiguration")
        self.line = QtWidgets.QFrame(Dialog)
        self.line.setGeometry(QtCore.QRect(7, 80, 381, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.listShotControlVariables = QtWidgets.QListWidget(Dialog)
        self.listShotControlVariables.setGeometry(QtCore.QRect(20, 160, 181, 211))
        self.listShotControlVariables.setObjectName("listShotControlVariables")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(20, 130, 61, 21))
        self.label_3.setObjectName("label_3")
        self.lineVariableName = QtWidgets.QLineEdit(Dialog)
        self.lineVariableName.setGeometry(QtCore.QRect(80, 130, 161, 22))
        self.lineVariableName.setObjectName("lineVariableName")
        self.buttonAddVariable = QtWidgets.QPushButton(Dialog)
        self.buttonAddVariable.setGeometry(QtCore.QRect(250, 130, 61, 22))
        self.buttonAddVariable.setObjectName("buttonAddVariable")
        self.buttonRemoveVariable = QtWidgets.QPushButton(Dialog)
        self.buttonRemoveVariable.setGeometry(QtCore.QRect(320, 130, 61, 22))
        self.buttonRemoveVariable.setObjectName("buttonRemoveVariable")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(270, 160, 55, 16))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(220, 170, 31, 16))
        self.label_5.setObjectName("label_5")
        self.lineOffState = QtWidgets.QLineEdit(Dialog)
        self.lineOffState.setGeometry(QtCore.QRect(220, 190, 161, 22))
        self.lineOffState.setObjectName("lineOffState")
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setGeometry(QtCore.QRect(220, 220, 71, 16))
        self.label_6.setObjectName("label_6")
        self.lineScanState = QtWidgets.QLineEdit(Dialog)
        self.lineScanState.setGeometry(QtCore.QRect(220, 240, 161, 22))
        self.lineScanState.setObjectName("lineScanState")
        self.label_7 = QtWidgets.QLabel(Dialog)
        self.label_7.setGeometry(QtCore.QRect(220, 270, 101, 16))
        self.label_7.setObjectName("label_7")
        self.lineStandbyState = QtWidgets.QLineEdit(Dialog)
        self.lineStandbyState.setGeometry(QtCore.QRect(220, 290, 161, 22))
        self.lineStandbyState.setObjectName("lineStandbyState")
        self.buttonSaveConfiguration = QtWidgets.QPushButton(Dialog)
        self.buttonSaveConfiguration.setGeometry(QtCore.QRect(100, 390, 91, 28))
        self.buttonSaveConfiguration.setObjectName("buttonSaveConfiguration")
        self.buttonCloseWindow = QtWidgets.QPushButton(Dialog)
        self.buttonCloseWindow.setGeometry(QtCore.QRect(210, 390, 91, 28))
        self.buttonCloseWindow.setObjectName("buttonCloseWindow")
        self.line_2 = QtWidgets.QFrame(Dialog)
        self.line_2.setGeometry(QtCore.QRect(10, 380, 381, 3))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.label_8 = QtWidgets.QLabel(Dialog)
        self.label_8.setGeometry(QtCore.QRect(220, 320, 161, 16))
        self.label_8.setObjectName("label_8")
        self.lineSingleShotState = QtWidgets.QLineEdit(Dialog)
        self.lineSingleShotState.setGeometry(QtCore.QRect(220, 340, 161, 22))
        self.lineSingleShotState.setObjectName("lineSingleShotState")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        Dialog.setTabOrder(self.lineConfigurationSelect, self.buttonNewConfiguration)
        Dialog.setTabOrder(self.buttonNewConfiguration, self.buttonCopyConfiguration)
        Dialog.setTabOrder(self.buttonCopyConfiguration, self.buttonDeleteConfiguration)
        Dialog.setTabOrder(self.buttonDeleteConfiguration, self.lineDeviceName)
        Dialog.setTabOrder(self.lineDeviceName, self.lineVariableName)
        Dialog.setTabOrder(self.lineVariableName, self.buttonAddVariable)
        Dialog.setTabOrder(self.buttonAddVariable, self.buttonRemoveVariable)
        Dialog.setTabOrder(self.buttonRemoveVariable, self.listShotControlVariables)
        Dialog.setTabOrder(self.listShotControlVariables, self.lineOffState)
        Dialog.setTabOrder(self.lineOffState, self.lineScanState)
        Dialog.setTabOrder(self.lineScanState, self.lineStandbyState)
        Dialog.setTabOrder(self.lineStandbyState, self.buttonSaveConfiguration)
        Dialog.setTabOrder(self.buttonSaveConfiguration, self.buttonCloseWindow)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Shot Control Editor"))
        self.label.setText(_translate("Dialog", "Configuration Select:"))
        self.buttonNewConfiguration.setText(_translate("Dialog", "New"))
        self.buttonDeleteConfiguration.setText(_translate("Dialog", "Delete"))
        self.label_2.setText(_translate("Dialog", "Shot Control Device:"))
        self.buttonCopyConfiguration.setText(_translate("Dialog", "Copy"))
        self.label_3.setText(_translate("Dialog", "Variables:"))
        self.buttonAddVariable.setText(_translate("Dialog", "Add"))
        self.buttonRemoveVariable.setText(_translate("Dialog", "Remove"))
        self.label_4.setText(_translate("Dialog", "States"))
        self.label_5.setText(_translate("Dialog", "OFF:"))
        self.label_6.setText(_translate("Dialog", "ON (Scan):"))
        self.label_7.setText(_translate("Dialog", "ON (Standby):"))
        self.buttonSaveConfiguration.setText(_translate("Dialog", "Save"))
        self.buttonCloseWindow.setText(_translate("Dialog", "Close"))
        self.label_8.setText(_translate("Dialog", "Single Shot (Synchronize):"))
