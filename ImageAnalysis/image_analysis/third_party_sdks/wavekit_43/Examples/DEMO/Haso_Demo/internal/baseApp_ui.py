# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'internal/baseApp.ui'
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
        MainWindow.setStyleSheet(_fromUtf8("QMainWindow#MainWindow\n"
"{\n"
"    background-color: rgb(66, 66, 66);\n"
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
"   background-color: rgb(66, 66, 66);\n"
"    border: none;\n"
"    padding:5px;\n"
"    padding-right: 25px;\n"
"    min-width: 40ex;\n"
"    margin-right:2px;\n"
"    margin-left:2px;\n"
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
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setMargin(0)
        self.horizontalLayout.setSpacing(10)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.tabWidget = QtGui.QTabWidget(self.centralwidget)
        self.tabWidget.setTabShape(QtGui.QTabWidget.Rounded)
        self.tabWidget.setIconSize(QtCore.QSize(24, 24))
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.tab = QtGui.QWidget()
        self.tab.setObjectName(_fromUtf8("tab"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout(self.tab)
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.image_display = QtGui.QWidget(self.tab)
        self.image_display.setObjectName(_fromUtf8("image_display"))
        self.horizontalLayout_4 = QtGui.QHBoxLayout(self.image_display)
        self.horizontalLayout_4.setMargin(0)
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.horizontalLayout_3.addWidget(self.image_display)
        self.widget_2 = QtGui.QWidget(self.tab)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_2.sizePolicy().hasHeightForWidth())
        self.widget_2.setSizePolicy(sizePolicy)
        self.widget_2.setMinimumSize(QtCore.QSize(280, 0))
        self.widget_2.setObjectName(_fromUtf8("widget_2"))
        self.verticalLayout = QtGui.QVBoxLayout(self.widget_2)
        self.verticalLayout.setContentsMargins(-1, -1, -1, 0)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.label_8 = QtGui.QLabel(self.widget_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        self.label_8.setStyleSheet(_fromUtf8("font: bold 12pt \"Open Sans\";"))
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.verticalLayout.addWidget(self.label_8)
        self.AcquisitionParameters = QtGui.QFrame(self.widget_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.AcquisitionParameters.sizePolicy().hasHeightForWidth())
        self.AcquisitionParameters.setSizePolicy(sizePolicy)
        self.AcquisitionParameters.setFrameShape(QtGui.QFrame.Box)
        self.AcquisitionParameters.setFrameShadow(QtGui.QFrame.Plain)
        self.AcquisitionParameters.setObjectName(_fromUtf8("AcquisitionParameters"))
        self.formLayout = QtGui.QFormLayout(self.AcquisitionParameters)
        self.formLayout.setHorizontalSpacing(20)
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.label = QtGui.QLabel(self.AcquisitionParameters)
        self.label.setObjectName(_fromUtf8("label"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.label)
        self.expotime = QtGui.QSpinBox(self.AcquisitionParameters)
        self.expotime.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.expotime.setMinimum(50)
        self.expotime.setMaximum(1000000)
        self.expotime.setProperty("value", 100)
        self.expotime.setObjectName(_fromUtf8("expotime"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.expotime)
        self.label_2 = QtGui.QLabel(self.AcquisitionParameters)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.label_2)
        self.nbImages = QtGui.QSpinBox(self.AcquisitionParameters)
        self.nbImages.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.nbImages.setProperty("value", 1)
        self.nbImages.setObjectName(_fromUtf8("nbImages"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.nbImages)
        self.verticalLayout.addWidget(self.AcquisitionParameters)
        spacerItem = QtGui.QSpacerItem(5, 5, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem)
        self.label_17 = QtGui.QLabel(self.widget_2)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Open Sans"))
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_17.setFont(font)
        self.label_17.setStyleSheet(_fromUtf8("font: bold 12pt \"Open Sans\";"))
        self.label_17.setObjectName(_fromUtf8("label_17"))
        self.verticalLayout.addWidget(self.label_17)
        self.Diagnosis_2 = QtGui.QFrame(self.widget_2)
        self.Diagnosis_2.setFrameShape(QtGui.QFrame.Box)
        self.Diagnosis_2.setFrameShadow(QtGui.QFrame.Plain)
        self.Diagnosis_2.setObjectName(_fromUtf8("Diagnosis_2"))
        self.formLayout_4 = QtGui.QFormLayout(self.Diagnosis_2)
        self.formLayout_4.setHorizontalSpacing(20)
        self.formLayout_4.setObjectName(_fromUtf8("formLayout_4"))
        self.label_5 = QtGui.QLabel(self.Diagnosis_2)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.formLayout_4.setWidget(0, QtGui.QFormLayout.LabelRole, self.label_5)
        self.sat_value = QtGui.QLabel(self.Diagnosis_2)
        self.sat_value.setMinimumSize(QtCore.QSize(80, 0))
        self.sat_value.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.sat_value.setObjectName(_fromUtf8("sat_value"))
        self.formLayout_4.setWidget(0, QtGui.QFormLayout.FieldRole, self.sat_value)
        self.verticalLayout.addWidget(self.Diagnosis_2)
        self.label_11 = QtGui.QLabel(self.widget_2)
        self.label_11.setStyleSheet(_fromUtf8("font: bold 12pt \"Open Sans\";"))
        self.label_11.setObjectName(_fromUtf8("label_11"))
        self.verticalLayout.addWidget(self.label_11)
        self.SaveFile = QtGui.QFrame(self.widget_2)
        self.SaveFile.setFrameShape(QtGui.QFrame.Box)
        self.SaveFile.setFrameShadow(QtGui.QFrame.Plain)
        self.SaveFile.setObjectName(_fromUtf8("SaveFile"))
        self.formLayout_3 = QtGui.QFormLayout(self.SaveFile)
        self.formLayout_3.setContentsMargins(9, -1, -1, -1)
        self.formLayout_3.setHorizontalSpacing(9)
        self.formLayout_3.setObjectName(_fromUtf8("formLayout_3"))
        self.saveDirectory = QtGui.QLabel(self.SaveFile)
        self.saveDirectory.setObjectName(_fromUtf8("saveDirectory"))
        self.formLayout_3.setWidget(0, QtGui.QFormLayout.LabelRole, self.saveDirectory)
        self.widget = QtGui.QWidget(self.SaveFile)
        self.widget.setObjectName(_fromUtf8("widget"))
        self.horizontalLayout_10 = QtGui.QHBoxLayout(self.widget)
        self.horizontalLayout_10.setMargin(0)
        self.horizontalLayout_10.setSpacing(0)
        self.horizontalLayout_10.setObjectName(_fromUtf8("horizontalLayout_10"))
        self.saveDirectoryPath = QtGui.QLineEdit(self.widget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.saveDirectoryPath.sizePolicy().hasHeightForWidth())
        self.saveDirectoryPath.setSizePolicy(sizePolicy)
        self.saveDirectoryPath.setObjectName(_fromUtf8("saveDirectoryPath"))
        self.horizontalLayout_10.addWidget(self.saveDirectoryPath)
        self.saveDirectory_2 = QtGui.QPushButton(self.widget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(20)
        sizePolicy.setVerticalStretch(18)
        sizePolicy.setHeightForWidth(self.saveDirectory_2.sizePolicy().hasHeightForWidth())
        self.saveDirectory_2.setSizePolicy(sizePolicy)
        self.saveDirectory_2.setMinimumSize(QtCore.QSize(24, 0))
        self.saveDirectory_2.setObjectName(_fromUtf8("saveDirectory_2"))
        self.horizontalLayout_10.addWidget(self.saveDirectory_2)
        self.formLayout_3.setWidget(0, QtGui.QFormLayout.FieldRole, self.widget)
        self.saveCurrentFile = QtGui.QPushButton(self.SaveFile)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.saveCurrentFile.sizePolicy().hasHeightForWidth())
        self.saveCurrentFile.setSizePolicy(sizePolicy)
        self.saveCurrentFile.setMinimumSize(QtCore.QSize(0, 24))
        self.saveCurrentFile.setObjectName(_fromUtf8("saveCurrentFile"))
        self.formLayout_3.setWidget(1, QtGui.QFormLayout.SpanningRole, self.saveCurrentFile)
        self.verticalLayout.addWidget(self.SaveFile)
        self.widget_3 = QtGui.QWidget(self.widget_2)
        self.widget_3.setObjectName(_fromUtf8("widget_3"))
        self.horizontalLayout_5 = QtGui.QHBoxLayout(self.widget_3)
        self.horizontalLayout_5.setMargin(0)
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        spacerItem1 = QtGui.QSpacerItem(20, 500, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.horizontalLayout_5.addItem(spacerItem1)
        self.verticalLayout.addWidget(self.widget_3)
        self.horizontalLayout_3.addWidget(self.widget_2)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/WaveView/icon/cinema_blue.svg")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/WaveView/icon/Camera")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.tabWidget.addTab(self.tab, icon, _fromUtf8(""))
        self.tab_2 = QtGui.QWidget()
        self.tab_2.setObjectName(_fromUtf8("tab_2"))
        self.horizontalLayout_6 = QtGui.QHBoxLayout(self.tab_2)
        self.horizontalLayout_6.setObjectName(_fromUtf8("horizontalLayout_6"))
        self.wavefront_display = QtGui.QWidget(self.tab_2)
        self.wavefront_display.setMinimumSize(QtCore.QSize(0, 578))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Open Sans"))
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.wavefront_display.setFont(font)
        self.wavefront_display.setObjectName(_fromUtf8("wavefront_display"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.wavefront_display)
        self.horizontalLayout_2.setMargin(0)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.horizontalLayout_6.addWidget(self.wavefront_display)
        self.widget_4 = QtGui.QWidget(self.tab_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_4.sizePolicy().hasHeightForWidth())
        self.widget_4.setSizePolicy(sizePolicy)
        self.widget_4.setMinimumSize(QtCore.QSize(5, 5))
        self.widget_4.setMaximumSize(QtCore.QSize(16777215, 578))
        self.widget_4.setObjectName(_fromUtf8("widget_4"))
        self.verticalLayout_11 = QtGui.QVBoxLayout(self.widget_4)
        self.verticalLayout_11.setContentsMargins(-1, 0, -1, 0)
        self.verticalLayout_11.setObjectName(_fromUtf8("verticalLayout_11"))
        self.label_7 = QtGui.QLabel(self.widget_4)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        self.label_7.setStyleSheet(_fromUtf8("font: bold 12pt \"Open Sans\";"))
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.verticalLayout_11.addWidget(self.label_7)
        self.FilteringParameters = QtGui.QFrame(self.widget_4)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.FilteringParameters.sizePolicy().hasHeightForWidth())
        self.FilteringParameters.setSizePolicy(sizePolicy)
        self.FilteringParameters.setFrameShape(QtGui.QFrame.Box)
        self.FilteringParameters.setFrameShadow(QtGui.QFrame.Plain)
        self.FilteringParameters.setObjectName(_fromUtf8("FilteringParameters"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.FilteringParameters)
        self.verticalLayout_3.setMargin(3)
        self.verticalLayout_3.setSpacing(6)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.filter_tx = QtGui.QCheckBox(self.FilteringParameters)
        self.filter_tx.setObjectName(_fromUtf8("filter_tx"))
        self.verticalLayout_3.addWidget(self.filter_tx)
        self.filter_ty = QtGui.QCheckBox(self.FilteringParameters)
        self.filter_ty.setObjectName(_fromUtf8("filter_ty"))
        self.verticalLayout_3.addWidget(self.filter_ty)
        self.filter_f = QtGui.QCheckBox(self.FilteringParameters)
        self.filter_f.setObjectName(_fromUtf8("filter_f"))
        self.verticalLayout_3.addWidget(self.filter_f)
        self.filter_a0 = QtGui.QCheckBox(self.FilteringParameters)
        self.filter_a0.setObjectName(_fromUtf8("filter_a0"))
        self.verticalLayout_3.addWidget(self.filter_a0)
        self.filter_a45 = QtGui.QCheckBox(self.FilteringParameters)
        self.filter_a45.setObjectName(_fromUtf8("filter_a45"))
        self.verticalLayout_3.addWidget(self.filter_a45)
        self.verticalLayout_11.addWidget(self.FilteringParameters)
        self.label_9 = QtGui.QLabel(self.widget_4)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        self.label_9.setStyleSheet(_fromUtf8("font: bold 11pt \"Open Sans\";"))
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.verticalLayout_11.addWidget(self.label_9)
        self.Diagnosis = QtGui.QFrame(self.widget_4)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Diagnosis.sizePolicy().hasHeightForWidth())
        self.Diagnosis.setSizePolicy(sizePolicy)
        self.Diagnosis.setFrameShape(QtGui.QFrame.Box)
        self.Diagnosis.setFrameShadow(QtGui.QFrame.Plain)
        self.Diagnosis.setObjectName(_fromUtf8("Diagnosis"))
        self.formLayout_2 = QtGui.QFormLayout(self.Diagnosis)
        self.formLayout_2.setMargin(3)
        self.formLayout_2.setHorizontalSpacing(20)
        self.formLayout_2.setObjectName(_fromUtf8("formLayout_2"))
        self.label_15 = QtGui.QLabel(self.Diagnosis)
        self.label_15.setObjectName(_fromUtf8("label_15"))
        self.formLayout_2.setWidget(0, QtGui.QFormLayout.LabelRole, self.label_15)
        self.label_6 = QtGui.QLabel(self.Diagnosis)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.formLayout_2.setWidget(1, QtGui.QFormLayout.LabelRole, self.label_6)
        self.wf_pv_value = QtGui.QLabel(self.Diagnosis)
        self.wf_pv_value.setMinimumSize(QtCore.QSize(80, 0))
        self.wf_pv_value.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.wf_pv_value.setObjectName(_fromUtf8("wf_pv_value"))
        self.formLayout_2.setWidget(1, QtGui.QFormLayout.FieldRole, self.wf_pv_value)
        self.label_16 = QtGui.QLabel(self.Diagnosis)
        self.label_16.setObjectName(_fromUtf8("label_16"))
        self.formLayout_2.setWidget(2, QtGui.QFormLayout.LabelRole, self.label_16)
        self.wf_rms_value = QtGui.QLabel(self.Diagnosis)
        self.wf_rms_value.setMinimumSize(QtCore.QSize(80, 0))
        self.wf_rms_value.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.wf_rms_value.setObjectName(_fromUtf8("wf_rms_value"))
        self.formLayout_2.setWidget(2, QtGui.QFormLayout.FieldRole, self.wf_rms_value)
        self.label_14 = QtGui.QLabel(self.Diagnosis)
        self.label_14.setObjectName(_fromUtf8("label_14"))
        self.formLayout_2.setWidget(3, QtGui.QFormLayout.LabelRole, self.label_14)
        self.label_3 = QtGui.QLabel(self.Diagnosis)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.formLayout_2.setWidget(4, QtGui.QFormLayout.LabelRole, self.label_3)
        self.tilt_x_value = QtGui.QLabel(self.Diagnosis)
        self.tilt_x_value.setMinimumSize(QtCore.QSize(80, 0))
        self.tilt_x_value.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.tilt_x_value.setObjectName(_fromUtf8("tilt_x_value"))
        self.formLayout_2.setWidget(4, QtGui.QFormLayout.FieldRole, self.tilt_x_value)
        self.label_4 = QtGui.QLabel(self.Diagnosis)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.formLayout_2.setWidget(5, QtGui.QFormLayout.LabelRole, self.label_4)
        self.tilt_y_value = QtGui.QLabel(self.Diagnosis)
        self.tilt_y_value.setMinimumSize(QtCore.QSize(80, 0))
        self.tilt_y_value.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.tilt_y_value.setObjectName(_fromUtf8("tilt_y_value"))
        self.formLayout_2.setWidget(5, QtGui.QFormLayout.FieldRole, self.tilt_y_value)
        self.label_13 = QtGui.QLabel(self.Diagnosis)
        self.label_13.setObjectName(_fromUtf8("label_13"))
        self.formLayout_2.setWidget(6, QtGui.QFormLayout.LabelRole, self.label_13)
        self.curv_value = QtGui.QLabel(self.Diagnosis)
        self.curv_value.setMinimumSize(QtCore.QSize(80, 0))
        self.curv_value.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.curv_value.setObjectName(_fromUtf8("curv_value"))
        self.formLayout_2.setWidget(6, QtGui.QFormLayout.FieldRole, self.curv_value)
        self.verticalLayout_11.addWidget(self.Diagnosis)
        self.label_10 = QtGui.QLabel(self.widget_4)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy)
        self.label_10.setStyleSheet(_fromUtf8("font: bold 11pt \"Open Sans\";"))
        self.label_10.setObjectName(_fromUtf8("label_10"))
        self.verticalLayout_11.addWidget(self.label_10)
        self.SaveFileWavefront = QtGui.QFrame(self.widget_4)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.SaveFileWavefront.sizePolicy().hasHeightForWidth())
        self.SaveFileWavefront.setSizePolicy(sizePolicy)
        self.SaveFileWavefront.setFrameShape(QtGui.QFrame.Box)
        self.SaveFileWavefront.setFrameShadow(QtGui.QFrame.Plain)
        self.SaveFileWavefront.setObjectName(_fromUtf8("SaveFileWavefront"))
        self.formLayout_5 = QtGui.QFormLayout(self.SaveFileWavefront)
        self.formLayout_5.setObjectName(_fromUtf8("formLayout_5"))
        self.label_12 = QtGui.QLabel(self.SaveFileWavefront)
        self.label_12.setObjectName(_fromUtf8("label_12"))
        self.formLayout_5.setWidget(0, QtGui.QFormLayout.LabelRole, self.label_12)
        self.widget_6 = QtGui.QWidget(self.SaveFileWavefront)
        self.widget_6.setObjectName(_fromUtf8("widget_6"))
        self.horizontalLayout_7 = QtGui.QHBoxLayout(self.widget_6)
        self.horizontalLayout_7.setMargin(0)
        self.horizontalLayout_7.setSpacing(0)
        self.horizontalLayout_7.setObjectName(_fromUtf8("horizontalLayout_7"))
        self.saveDirectoryPathWavefront = QtGui.QLineEdit(self.widget_6)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.saveDirectoryPathWavefront.sizePolicy().hasHeightForWidth())
        self.saveDirectoryPathWavefront.setSizePolicy(sizePolicy)
        self.saveDirectoryPathWavefront.setMinimumSize(QtCore.QSize(0, 24))
        self.saveDirectoryPathWavefront.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.saveDirectoryPathWavefront.setFocusPolicy(QtCore.Qt.TabFocus)
        self.saveDirectoryPathWavefront.setObjectName(_fromUtf8("saveDirectoryPathWavefront"))
        self.horizontalLayout_7.addWidget(self.saveDirectoryPathWavefront)
        self.saveDirectoryWavefront = QtGui.QPushButton(self.widget_6)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(20)
        sizePolicy.setVerticalStretch(18)
        sizePolicy.setHeightForWidth(self.saveDirectoryWavefront.sizePolicy().hasHeightForWidth())
        self.saveDirectoryWavefront.setSizePolicy(sizePolicy)
        self.saveDirectoryWavefront.setMinimumSize(QtCore.QSize(24, 0))
        self.saveDirectoryWavefront.setSizeIncrement(QtCore.QSize(0, 24))
        self.saveDirectoryWavefront.setObjectName(_fromUtf8("saveDirectoryWavefront"))
        self.horizontalLayout_7.addWidget(self.saveDirectoryWavefront)
        self.formLayout_5.setWidget(0, QtGui.QFormLayout.FieldRole, self.widget_6)
        self.saveWavefrontButton = QtGui.QPushButton(self.SaveFileWavefront)
        self.saveWavefrontButton.setMinimumSize(QtCore.QSize(0, 24))
        self.saveWavefrontButton.setObjectName(_fromUtf8("saveWavefrontButton"))
        self.formLayout_5.setWidget(1, QtGui.QFormLayout.SpanningRole, self.saveWavefrontButton)
        self.verticalLayout_11.addWidget(self.SaveFileWavefront)
        spacerItem2 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout_11.addItem(spacerItem2)
        self.horizontalLayout_6.addWidget(self.widget_4)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(_fromUtf8(":/WaveView/icon/phi_blue.svg")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon1.addPixmap(QtGui.QPixmap(_fromUtf8(":/WaveView/icon/Wavefront")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.tabWidget.addTab(self.tab_2, icon1, _fromUtf8(""))
        self.horizontalLayout.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtGui.QToolBar(MainWindow)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.toolBar.sizePolicy().hasHeightForWidth())
        self.toolBar.setSizePolicy(sizePolicy)
        self.toolBar.setMovable(False)
        self.toolBar.setObjectName(_fromUtf8("toolBar"))
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionPlay = QtGui.QAction(MainWindow)
        self.actionPlay.setCheckable(True)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(_fromUtf8(":/WaveView/icon/Play")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon2.addPixmap(QtGui.QPixmap(_fromUtf8(":/WaveView/icon/Pause")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionPlay.setIcon(icon2)
        self.actionPlay.setObjectName(_fromUtf8("actionPlay"))
        self.toolBar.addAction(self.actionPlay)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "Haso Demo", None))
        self.label_8.setText(_translate("MainWindow", "Acquisition parameters", None))
        self.label.setText(_translate("MainWindow", "Exposure time (µs)", None))
        self.label_2.setText(_translate("MainWindow", "Image number", None))
        self.label_17.setText(_translate("MainWindow", "Diagnosis", None))
        self.label_5.setText(_translate("MainWindow", "Saturation (%)", None))
        self.sat_value.setText(_translate("MainWindow", "---", None))
        self.label_11.setText(_translate("MainWindow", "Save", None))
        self.saveDirectory.setText(_translate("MainWindow", "Save directory", None))
        self.saveDirectory_2.setText(_translate("MainWindow", "....", None))
        self.saveCurrentFile.setText(_translate("MainWindow", "Save file", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Hartmanngram", None))
        self.label_7.setText(_translate("MainWindow", "Filtering parameters", None))
        self.filter_tx.setText(_translate("MainWindow", "Tilt X", None))
        self.filter_ty.setText(_translate("MainWindow", "Tilt Y", None))
        self.filter_f.setText(_translate("MainWindow", "Focus", None))
        self.filter_a0.setText(_translate("MainWindow", "Astig 0°", None))
        self.filter_a45.setText(_translate("MainWindow", "Astig 45°", None))
        self.label_9.setText(_translate("MainWindow", "Diagnosis", None))
        self.label_15.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Wavefront statistics</span></p></body></html>", None))
        self.label_6.setText(_translate("MainWindow", "PV (µm)", None))
        self.wf_pv_value.setText(_translate("MainWindow", "---", None))
        self.label_16.setText(_translate("MainWindow", "RMS (µm)", None))
        self.wf_rms_value.setText(_translate("MainWindow", "---", None))
        self.label_14.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">Chief ray</span></p></body></html>", None))
        self.label_3.setText(_translate("MainWindow", "Tilt X (mrad)", None))
        self.tilt_x_value.setText(_translate("MainWindow", "---", None))
        self.label_4.setText(_translate("MainWindow", "Tilt Y (mrad)", None))
        self.tilt_y_value.setText(_translate("MainWindow", "---", None))
        self.label_13.setText(_translate("MainWindow", "Curvature R (mm)", None))
        self.curv_value.setText(_translate("MainWindow", "---", None))
        self.label_10.setText(_translate("MainWindow", "Save", None))
        self.label_12.setText(_translate("MainWindow", "Save directory", None))
        self.saveDirectoryWavefront.setText(_translate("MainWindow", ".....", None))
        self.saveWavefrontButton.setText(_translate("MainWindow", "Save wavefront", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Wavefront            ", None))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar", None))
        self.actionPlay.setText(_translate("MainWindow", "Play", None))

import baseApp_rc
