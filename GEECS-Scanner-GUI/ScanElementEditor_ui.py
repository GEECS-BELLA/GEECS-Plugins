# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\ScanElementEditor.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(391, 556)
        self.listDevices = QtWidgets.QListWidget(Dialog)
        self.listDevices.setGeometry(QtCore.QRect(20, 110, 171, 141))
        self.listDevices.setObjectName("listDevices")
        self.listVariables = QtWidgets.QListWidget(Dialog)
        self.listVariables.setGeometry(QtCore.QRect(200, 110, 171, 131))
        self.listVariables.setObjectName("listVariables")
        self.checkboxSaveNonscalar = QtWidgets.QCheckBox(Dialog)
        self.checkboxSaveNonscalar.setGeometry(QtCore.QRect(200, 250, 151, 20))
        self.checkboxSaveNonscalar.setObjectName("checkboxSaveNonscalar")
        self.checkboxSynchronous = QtWidgets.QCheckBox(Dialog)
        self.checkboxSynchronous.setGeometry(QtCore.QRect(200, 280, 131, 20))
        self.checkboxSynchronous.setObjectName("checkboxSynchronous")
        self.lineDeviceName = QtWidgets.QLineEdit(Dialog)
        self.lineDeviceName.setGeometry(QtCore.QRect(20, 60, 171, 22))
        self.lineDeviceName.setObjectName("lineDeviceName")
        self.lineVariableName = QtWidgets.QLineEdit(Dialog)
        self.lineVariableName.setGeometry(QtCore.QRect(200, 60, 171, 22))
        self.lineVariableName.setObjectName("lineVariableName")
        self.buttonAddDevice = QtWidgets.QPushButton(Dialog)
        self.buttonAddDevice.setGeometry(QtCore.QRect(20, 84, 81, 25))
        self.buttonAddDevice.setObjectName("buttonAddDevice")
        self.buttonRemoveDevice = QtWidgets.QPushButton(Dialog)
        self.buttonRemoveDevice.setGeometry(QtCore.QRect(110, 84, 81, 25))
        self.buttonRemoveDevice.setObjectName("buttonRemoveDevice")
        self.buttonAddVariable = QtWidgets.QPushButton(Dialog)
        self.buttonAddVariable.setGeometry(QtCore.QRect(200, 84, 81, 25))
        self.buttonAddVariable.setObjectName("buttonAddVariable")
        self.buttonRemoveVariable = QtWidgets.QPushButton(Dialog)
        self.buttonRemoveVariable.setGeometry(QtCore.QRect(290, 84, 81, 25))
        self.buttonRemoveVariable.setObjectName("buttonRemoveVariable")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(80, 40, 55, 21))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(240, 40, 101, 20))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(60, 310, 41, 21))
        self.label_3.setObjectName("label_3")
        self.lineActionName = QtWidgets.QLineEdit(Dialog)
        self.lineActionName.setGeometry(QtCore.QRect(20, 330, 121, 22))
        self.lineActionName.setObjectName("lineActionName")
        self.buttonAddAction = QtWidgets.QPushButton(Dialog)
        self.buttonAddAction.setGeometry(QtCore.QRect(20, 354, 61, 25))
        self.buttonAddAction.setObjectName("buttonAddAction")
        self.buttonRemoveAction = QtWidgets.QPushButton(Dialog)
        self.buttonRemoveAction.setGeometry(QtCore.QRect(80, 354, 61, 25))
        self.buttonRemoveAction.setObjectName("buttonRemoveAction")
        self.listActions = QtWidgets.QListWidget(Dialog)
        self.listActions.setGeometry(QtCore.QRect(20, 380, 121, 131))
        self.listActions.setObjectName("listActions")
        self.labelActionOption1 = QtWidgets.QLabel(Dialog)
        self.labelActionOption1.setGeometry(QtCore.QRect(180, 340, 191, 21))
        self.labelActionOption1.setObjectName("labelActionOption1")
        self.lineActionOption1 = QtWidgets.QLineEdit(Dialog)
        self.lineActionOption1.setGeometry(QtCore.QRect(180, 360, 191, 22))
        self.lineActionOption1.setObjectName("lineActionOption1")
        self.buttonMoveSooner = QtWidgets.QPushButton(Dialog)
        self.buttonMoveSooner.setGeometry(QtCore.QRect(150, 400, 21, 28))
        self.buttonMoveSooner.setObjectName("buttonMoveSooner")
        self.buttonMoveLater = QtWidgets.QPushButton(Dialog)
        self.buttonMoveLater.setGeometry(QtCore.QRect(150, 440, 21, 28))
        self.buttonMoveLater.setObjectName("buttonMoveLater")
        self.radioIsSetup = QtWidgets.QRadioButton(Dialog)
        self.radioIsSetup.setGeometry(QtCore.QRect(150, 490, 101, 20))
        self.radioIsSetup.setChecked(True)
        self.radioIsSetup.setObjectName("radioIsSetup")
        self.radioIsPost = QtWidgets.QRadioButton(Dialog)
        self.radioIsPost.setGeometry(QtCore.QRect(260, 490, 121, 20))
        self.radioIsPost.setObjectName("radioIsPost")
        self.lineActionOption2 = QtWidgets.QLineEdit(Dialog)
        self.lineActionOption2.setGeometry(QtCore.QRect(180, 410, 191, 22))
        self.lineActionOption2.setObjectName("lineActionOption2")
        self.lineActionOption3 = QtWidgets.QLineEdit(Dialog)
        self.lineActionOption3.setGeometry(QtCore.QRect(180, 460, 191, 22))
        self.lineActionOption3.setObjectName("lineActionOption3")
        self.labelActionOption2 = QtWidgets.QLabel(Dialog)
        self.labelActionOption2.setGeometry(QtCore.QRect(180, 390, 191, 21))
        self.labelActionOption2.setObjectName("labelActionOption2")
        self.labelActionOption3 = QtWidgets.QLabel(Dialog)
        self.labelActionOption3.setGeometry(QtCore.QRect(180, 440, 191, 21))
        self.labelActionOption3.setObjectName("labelActionOption3")
        self.label_7 = QtWidgets.QLabel(Dialog)
        self.label_7.setGeometry(QtCore.QRect(20, 10, 91, 21))
        self.label_7.setObjectName("label_7")
        self.lineElementName = QtWidgets.QLineEdit(Dialog)
        self.lineElementName.setGeometry(QtCore.QRect(120, 10, 251, 22))
        self.lineElementName.setObjectName("lineElementName")
        self.separator2 = QtWidgets.QFrame(Dialog)
        self.separator2.setGeometry(QtCore.QRect(10, 300, 371, 16))
        self.separator2.setFrameShape(QtWidgets.QFrame.HLine)
        self.separator2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.separator2.setObjectName("separator2")
        self.separator1 = QtWidgets.QFrame(Dialog)
        self.separator1.setGeometry(QtCore.QRect(10, 30, 371, 16))
        self.separator1.setFrameShape(QtWidgets.QFrame.HLine)
        self.separator1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.separator1.setObjectName("separator1")
        self.separator3 = QtWidgets.QFrame(Dialog)
        self.separator3.setGeometry(QtCore.QRect(10, 510, 371, 16))
        self.separator3.setFrameShape(QtWidgets.QFrame.HLine)
        self.separator3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.separator3.setObjectName("separator3")
        self.buttonWindowSave = QtWidgets.QPushButton(Dialog)
        self.buttonWindowSave.setGeometry(QtCore.QRect(50, 522, 93, 28))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.buttonWindowSave.setFont(font)
        self.buttonWindowSave.setObjectName("buttonWindowSave")
        self.buttonWindowLoad = QtWidgets.QPushButton(Dialog)
        self.buttonWindowLoad.setGeometry(QtCore.QRect(150, 522, 93, 28))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.buttonWindowLoad.setFont(font)
        self.buttonWindowLoad.setObjectName("buttonWindowLoad")
        self.buttonWindowCancel = QtWidgets.QPushButton(Dialog)
        self.buttonWindowCancel.setGeometry(QtCore.QRect(250, 522, 93, 28))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.buttonWindowCancel.setFont(font)
        self.buttonWindowCancel.setObjectName("buttonWindowCancel")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(20, 260, 121, 16))
        self.label_4.setObjectName("label_4")
        self.linePostAnalysis = QtWidgets.QLineEdit(Dialog)
        self.linePostAnalysis.setGeometry(QtCore.QRect(20, 280, 171, 22))
        self.linePostAnalysis.setObjectName("linePostAnalysis")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Scan Element Editor"))
        self.checkboxSaveNonscalar.setText(_translate("Dialog", "Save Nonscalar Data?"))
        self.checkboxSynchronous.setText(_translate("Dialog", "Synchronous?"))
        self.buttonAddDevice.setText(_translate("Dialog", "Add"))
        self.buttonRemoveDevice.setText(_translate("Dialog", "Remove"))
        self.buttonAddVariable.setText(_translate("Dialog", "Add"))
        self.buttonRemoveVariable.setText(_translate("Dialog", "Remove"))
        self.label.setText(_translate("Dialog", "Devices"))
        self.label_2.setText(_translate("Dialog", "Device Variables"))
        self.label_3.setText(_translate("Dialog", "Actions"))
        self.buttonAddAction.setText(_translate("Dialog", "Add"))
        self.buttonRemoveAction.setText(_translate("Dialog", "Remove"))
        self.labelActionOption1.setText(_translate("Dialog", "Option 1:"))
        self.buttonMoveSooner.setText(_translate("Dialog", "↑"))
        self.buttonMoveLater.setText(_translate("Dialog", "↓"))
        self.radioIsSetup.setText(_translate("Dialog", "Setup Action"))
        self.radioIsPost.setText(_translate("Dialog", "Postscan Action"))
        self.labelActionOption2.setText(_translate("Dialog", "Option 2:"))
        self.labelActionOption3.setText(_translate("Dialog", "Option 3:"))
        self.label_7.setText(_translate("Dialog", "Element Name:"))
        self.buttonWindowSave.setText(_translate("Dialog", "Save"))
        self.buttonWindowLoad.setText(_translate("Dialog", "Open"))
        self.buttonWindowCancel.setText(_translate("Dialog", "Cancel"))
        self.label_4.setText(_translate("Dialog", "Post-Analysis Class:"))