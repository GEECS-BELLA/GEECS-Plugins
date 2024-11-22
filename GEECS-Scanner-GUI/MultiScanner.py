"""
GUI that can organize a list of presets into a multi-scan script.  This list can be separated into two lists to
independently set presets for the save device elements

-Chris
"""

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from GEECSScanner import GEECSScannerWindow
    from PyQt5.QtWidgets import QListWidget, QListWidgetItem

import os
import yaml
import time
from PyQt5.QtWidgets import QWidget, QInputDialog, QFileDialog, QMessageBox
from PyQt5.QtCore import QTimer, QObject, QThread, pyqtSignal, pyqtSlot
from MultiScanner_ui import Ui_Form
from multiscan_sound_player import play_finish_jingle


class MultiScanner(QWidget):
    def __init__(self, main_window: 'GEECSScannerWindow', multiscan_configurations_location):
        super().__init__()

        self.main_window = main_window

        # Initializes the gui elements
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle("Multi-Scanner")

        # Exit button closes the window and tells the main gui that multiscan mode is disabled
        self.ui.buttonExit.clicked.connect(self.close_window)

        # Populate the preset list and allow double-clicking to apply preset to main window gui
        self.ui.buttonRefreshPresetList.clicked.connect(self.populate_preset_list)
        self.populate_preset_list()
        self.ui.listAvailablePresets.itemDoubleClicked.connect(self.apply_preset_to_main_window)

        # Initializes the two lists to empty
        self.element_preset_list: list[str] = []
        self.scan_preset_list: list[str] = []

        # Buttons to add and remove presets to each of the two lists
        self.ui.buttonAddElement.clicked.connect(self.add_element_preset)
        self.ui.buttonRemoveElement.clicked.connect(self.remove_element_preset)
        self.ui.buttonAddScan.clicked.connect(self.add_scan_preset)
        self.ui.buttonRemoveScan.clicked.connect(self.remove_scan_preset)

        # Checkbox to enable the "split list" mode, otherwise the multiscan is a simple single list of full presets
        self.ui.checkBoxEnableScanList.clicked.connect(self.toggle_split_preset_mode)
        self.toggle_split_preset_mode()

        # Buttons to move presets up and down in their respective lists
        self.ui.buttonElementSooner.clicked.connect(self.move_element_sooner)
        self.ui.buttonElementLater.clicked.connect(self.move_element_later)
        self.ui.buttonScanSooner.clicked.connect(self.move_scan_sooner)
        self.ui.buttonScanLater.clicked.connect(self.move_scan_later)

        # Buttons to perform some basic item-copying functions to the two lists
        self.ui.buttonCopyElementToScan.clicked.connect(self.copy_list_element_to_scan)
        self.ui.buttonCopyScanToElement.clicked.connect(self.copy_list_scan_to_element)
        self.ui.buttonCopyRowElement.clicked.connect(self.copy_row_to_list_element)
        self.ui.buttonCopyRowScan.clicked.connect(self.copy_row_to_list_scan)

        # Buttons to save and load multiscan configurations
        self.config_folder = multiscan_configurations_location
        self.ui.buttonSaveMultiscan.clicked.connect(self.save_multiscan_configuration)
        self.ui.buttonLoadMultiscan.clicked.connect(self.load_multiscan_configuration)

        # Initial range of the spin box by the start button.  The maximum is later set to be the current list size
        self.ui.spinBoxStartPosition.setMinimum(1)
        self.ui.spinBoxStartPosition.setMaximum(1)

        # Initial state of the worker thread that checks submits scans when scan_manager is accepting scan requests
        self.worker_thread = None
        self.worker = None
        self.is_stopping = False

        # Buttons to start and stop the multiscan, which interact with the worker thread that waits and submits scans
        self.ui.buttonStartMultiscan.clicked.connect(self.start_multiscan)
        self.ui.buttonStopMultiscan.clicked.connect(self.stop_multiscan)

        # Timer that updates the info on the gui according to the current progress of the worker thread
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        self.ui.lineProgress.setText("-/-")
        self.ui.lineProgress.setReadOnly(True)
        self.timer.start(500)
        self.update_progress()

    def populate_preset_list(self):
        """Search for scan presets and adds them to the list of available presets"""
        self.ui.listAvailablePresets.clear()
        for preset in self.main_window.load_preset_list():
            self.ui.listAvailablePresets.addItem(preset)

    def apply_preset_to_main_window(self):
        """Applies the selected scan preset to the main window gui"""
        selected_element = self.ui.listAvailablePresets.selectedItems()
        for preset in selected_element:
            preset_name = f"{preset.text()}"

        self.main_window.apply_preset_from_name(preset_name)

    def refresh_multiscan_lists(self, list_widget: 'QListWidget' = None, index: int = None):
        """Updates the gui lists with the current state of the class lists, while preserving the selection position"""
        self.ui.listElementPresets.clear()
        self.ui.listScanPresets.clear()

        if self.ui.checkBoxEnableScanList:
            number_scans = max(len(self.element_preset_list), len(self.scan_preset_list))
        else:
            number_scans = len(self.element_preset_list)

        for i in range(number_scans):
            try: self.ui.listElementPresets.addItem(f"{i+1}:  {self.element_preset_list[i]}")
            except IndexError: self.ui.listElementPresets.addItem(f"{i+1}.")
            if self.ui.checkBoxEnableScanList.isChecked():
                try: self.ui.listScanPresets.addItem(f"{i+1}:  {self.scan_preset_list[i]}")
                except IndexError:  self.ui.listScanPresets.addItem(f"{i+1}.")

        if index is not None and list_widget is not None:
            list_widget.setCurrentRow(index)
            list_widget.setCurrentRow(index)

        if number_scans == 0:
            number_scans = 1
        if self.ui.spinBoxStartPosition.value() > number_scans:
            self.ui.spinBoxStartPosition.setValue(number_scans)
        self.ui.spinBoxStartPosition.setMaximum(number_scans)

    def add_element_preset(self):
        """Adds preset to the list of element presets"""
        self.add_to_list(self.ui.listAvailablePresets.selectedItems(), self.element_preset_list)

    def remove_element_preset(self):
        """Removes selected element preset from the list"""
        self.remove_from_list(self.ui.listElementPresets, self.element_preset_list)

    def add_scan_preset(self):
        """Adds preset to the list of scan presets"""
        self.add_to_list(self.ui.listAvailablePresets.selectedItems(), self.scan_preset_list)

    def remove_scan_preset(self):
        """Removes selected scan preset from the list"""
        self.remove_from_list(self.ui.listScanPresets, self.scan_preset_list)

    def add_to_list(self, selected_items: list['QListWidgetItem'], target_list: list[str]):
        """Generic function to add item to a specified list"""
        if not selected_items:
            return
        for selection in selected_items:
            target_list.append(selection.text())
        self.refresh_multiscan_lists()

    def remove_from_list(self, list_widget: 'QListWidget', target_list: list[str]):
        """Generic function to remove selection from specified list"""
        selected_items = list_widget.selectedItems()
        if not selected_items:
            return
        for selection in selected_items:
            index = list_widget.row(selection)
            if index < len(target_list):
                del target_list[list_widget.row(selection)]
        self.refresh_multiscan_lists()

    def toggle_split_preset_mode(self):
        """Enables/disables gui buttons based on if the user has the 'split preset' mode enabled through the checkbox"""
        is_enabled = self.ui.checkBoxEnableScanList.isChecked()
        self.ui.listScanPresets.setEnabled(is_enabled)
        self.ui.buttonAddScan.setEnabled(is_enabled)
        self.ui.buttonRemoveScan.setEnabled(is_enabled)
        self.ui.buttonScanSooner.setEnabled(is_enabled)
        self.ui.buttonScanLater.setEnabled(is_enabled)
        self.ui.buttonCopyElementToScan.setEnabled(is_enabled)
        self.ui.buttonCopyScanToElement.setEnabled(is_enabled)
        self.ui.buttonCopyRowScan.setEnabled(is_enabled)
        self.ui.buttonCopyRowElement.setEnabled(is_enabled)

        self.refresh_multiscan_lists()

    def move_element_sooner(self):
        """Move selected element preset sooner in the list"""
        self.move_ordering(list_widget=self.ui.listElementPresets, target_list=self.element_preset_list, sooner=True)

    def move_element_later(self):
        """Move selected element preset later in the list"""
        self.move_ordering(list_widget=self.ui.listElementPresets, target_list=self.element_preset_list, later=True)

    def move_scan_sooner(self):
        """Move scan element preset sooner in the list"""
        self.move_ordering(list_widget=self.ui.listScanPresets, target_list=self.scan_preset_list, sooner=True)

    def move_scan_later(self):
        """Move scan element preset later in the list"""
        self.move_ordering(list_widget=self.ui.listScanPresets, target_list=self.scan_preset_list, later=True)

    def move_ordering(self, list_widget: 'QListWidget', target_list: list[str],
                      sooner: bool = False, later: bool = False):
        """Generic function to move the selected action to an earlier or later position in the same list"""
        selected_items = list_widget.selectedItems()
        if not selected_items:
            return
        for selection in selected_items:
            i = list_widget.row(selection)
        if sooner and 0 < i < len(target_list):
            target_list[i], target_list[i - 1] = target_list[i - 1], target_list[i]
            i = i - 1
        if later and 0 <= i < len(target_list) - 1:
            target_list[i], target_list[i + 1] = target_list[i + 1], target_list[i]
            i = i + 1
        self.refresh_multiscan_lists(list_widget=list_widget, index=i)

    def copy_list_element_to_scan(self):
        """Copies the full element preset list into the scan preset list"""
        self.scan_preset_list = self.element_preset_list.copy()
        self.refresh_multiscan_lists()

    def copy_list_scan_to_element(self):
        """Copies the full scan preset list into the element preset list"""
        self.element_preset_list = self.scan_preset_list.copy()
        self.refresh_multiscan_lists()

    def copy_row_to_list_element(self):
        """Copies the selected row in the element list to all available slots in the element list"""
        self.copy_row_to_list(list_widget=self.ui.listElementPresets, target_list=self.element_preset_list)

    def copy_row_to_list_scan(self):
        """Copies the selected row in the scan list to all available slots in the scan list"""
        self.copy_row_to_list(list_widget=self.ui.listScanPresets, target_list=self.scan_preset_list)

    def copy_row_to_list(self, list_widget: 'QListWidget', target_list: list[str]):
        """Logic of copying a row element to the rest of the list.  Would be easy except that I also want to add the
        feature of allowing to copy a blank line to clear the list, and copy a single line to fill that list to the
        same length as the other list.  TODO there are two for loops here and I think they could be written better..."""
        selected_items = list_widget.selectedItems()
        if not selected_items:
            return
        for selection in selected_items:
            i = list_widget.row(selection)

        if i >= len(target_list):
            replacement = ''
        else:
            replacement = target_list[i]

        for j in range(max(len(self.element_preset_list), len(self.scan_preset_list))):
            if j >= len(target_list):
                target_list.append(replacement)
            else:
                target_list[j] = replacement

        for j in list(reversed(range(max(len(self.element_preset_list), len(self.scan_preset_list))))):
            if target_list[j] == '':
                del target_list[j]

        self.refresh_multiscan_lists(list_widget=list_widget, index=i)

    def save_multiscan_configuration(self):
        """Saves the current multiscan configuration to a yaml file specified by the user"""
        text, ok = QInputDialog.getText(self, 'Save Configuration', 'Enter filename:')
        if ok and text:
            settings = {
                'Element Presets': self.element_preset_list,
                'Split Preset Mode': self.ui.checkBoxEnableScanList.isChecked()
            }
            if len(self.scan_preset_list) > 0:
                settings['Scan Presets'] = self.scan_preset_list

            os.makedirs(self.config_folder, exist_ok=True)

            with open(f"{self.config_folder}/{text}.yaml", 'w') as file:
                yaml.dump(settings, file, default_flow_style=False)

    def load_multiscan_configuration(self):
        """Prompts the user to specify a yaml file from which to load a multiscan configuration"""
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Select a YAML File", self.config_folder, "YAML Files (*yaml)",
                                                   options=options)
        if file_name:
            self.load_settings_from_file(file_name)
            self.toggle_split_preset_mode()
            self.refresh_multiscan_lists()

    def load_settings_from_file(self, file_name):
        """Loads multiscan configuration from the given file name"""
        with open(file_name, 'r') as file:
            settings = yaml.safe_load(file)

        if 'Element Presets' in settings:
            self.element_preset_list = settings['Element Presets']
        else:
            self.element_preset_list = []

        if 'Split Preset Mode' in settings and settings['Split Preset Mode']:
            self.ui.checkBoxEnableScanList.setChecked(True)
            if 'Scan Presets' in settings:
                self.scan_preset_list = settings['Scan Presets']
            else:
                self.scan_preset_list = []
        else:
            self.ui.checkBoxEnableScanList.setChecked(False)
            self.scan_preset_list = []

    def start_multiscan(self):
        """Initializes a thread to periodically send presets and start scan commands to GEECS Scanner."""
        self.ui.buttonStartMultiscan.setEnabled(False)

        if len(self.element_preset_list) == 0:
            self.display_message("Number of element presets must be greater than zero.")
        elif self.ui.checkBoxEnableScanList.isChecked() and len(self.element_preset_list) != len(self.scan_preset_list):
            self.display_message("Need equivalent number of presents in both element and scan lists.")
        else:
            # Make a list of presets to execute
            element_list = self.element_preset_list
            scan_list = []
            if self.ui.checkBoxEnableScanList.isChecked():
                scan_list = self.scan_preset_list

            # Start a thread to push the next scan when the main gui window can accept a new scan
            self.worker = Worker(main_window=self.main_window, start_number=self.ui.spinBoxStartPosition.value()-1,
                                 element_presets=element_list, scan_presets=scan_list)
            self.worker_thread = QThread()
            self.worker.moveToThread(self.worker_thread)

            self.worker_thread.started.connect(self.worker.start_work)
            self.worker.submit_next.connect(self.push_next_preset_scan)
            self.worker.finished.connect(self.worker_thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.worker_thread.finished.connect(self.cleanup_worker)
            self.worker_thread.finished.connect(self.worker_thread.deleteLater)

            self.worker_thread.start()

            self.ui.buttonStopMultiscan.setEnabled(True)

    @staticmethod
    def display_message(message: str):
        """Displays an error message associated with starting a multiscan with improper preset lists"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Error starting multiscan:")
        msg.setInformativeText(message)
        msg.setWindowTitle("Multiscan Error")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    @pyqtSlot(int, list, list)
    def push_next_preset_scan(self, current_position: int, element_presets: list[str], scan_presets: list[str]):
        """When signal is received, gives the main gui window the specified preset commands and starts the next scan"""
        info = f"Multi-Scan #{current_position + 1}: {element_presets[current_position]}"
        if not scan_presets:
            self.main_window.apply_preset_from_name(element_presets[current_position])
        else:
            self.main_window.apply_preset_from_name(element_presets[current_position], load_scan_params=False)
            self.main_window.apply_preset_from_name(scan_presets[current_position], load_save_elements=False)
            info = info + f" --> {scan_presets[current_position]}"

        self.main_window.ui.textEditScanInfo.setText(info)
        self.main_window.initialize_scan()

        print("Initialized")

    def stop_multiscan(self):
        """Stop the ongoing multiscan and any current scan on the main window, as well as clean up the worker"""
        self.is_stopping = True
        self.ui.buttonStopMultiscan.setEnabled(False)

        if self.worker:
            # Kill the multiscan push thread
            self.worker.stop_work()
            self.cleanup_worker()

            # Send stop command to main window
            self.main_window.stop_scan()

    def cleanup_worker(self):
        """Sets the worker and its thread back to None"""
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
        self.worker = None

    def update_progress(self):
        """Checks for the current status of multiscan and updates the display of the GUI accordingly"""

        # If the multiscan is currently running, check the position in the thread list & set is_running to False if done
        if self.worker:
            self.ui.lineProgress.setText(self.worker.get_status())

        # If is_stopping is true, check if the main window has reset and set is_stopping back to False
        if self.is_stopping:
            if not self.worker:
                self.is_stopping = False

        start_button_logic = not self.worker and not self.is_stopping
        self.ui.buttonStartMultiscan.setEnabled(start_button_logic)
        stop_button_logic = (self.worker is not None) and not self.is_stopping
        self.ui.buttonStopMultiscan.setEnabled(stop_button_logic)

        return

    def close_window(self):
        """Upon exiting the window, stop any concurrent multiscan and tell the main window to exit multiscan mode"""
        self.stop_multiscan()
        self.main_window.exit_multiscan_mode()
        self.close()

    def closeEvent(self, event):
        """Upon exiting the window, stop any concurrent multiscan and tell the main window to exit multiscan mode"""
        self.stop_multiscan()
        self.main_window.exit_multiscan_mode()
        event.accept()


class Worker(QObject):
    """Thread that executes the submitted multiscan.  Continuously checks to see if a scan could be submitted, and if so
    then sends the signal to submit the next scan with the given presets"""
    finished = pyqtSignal()
    submit_next = pyqtSignal(int, list, list)

    def __init__(self, main_window: 'GEECSScannerWindow', start_number: int,
                 element_presets: list[str], scan_presets: list[str] = None):
        super().__init__()
        self._running = False
        self.main_window = main_window  # Required to check current scan status
        self.element_presets = element_presets
        self.scan_presets = scan_presets
        self.current_position = start_number
        self.is_waiting = False

    def start_work(self):
        """Begin loop to check status of GEECS Scanner every second"""
        self._running = True
        while self._running:
            if self.check_condition():
                self.send_command()
                self.is_waiting = True
            time.sleep(1)
        play_finish_jingle()
        self.finished.emit()

    def stop_work(self):
        """Sets a flag for the thread to exit the multiscan loop"""
        self._running = False

    def check_condition(self) -> bool:
        """
        :return:  True if the main window can accept a new scan request, False otherwise
        """
        if self.main_window.is_ready_for_scan():
            return not self.is_waiting
        else:
            self.is_waiting = False
            return False

    def send_command(self):
        """Emits signal to start next scan, sending the two preset lists and current index in the multiscan"""
        if self.current_position >= len(self.element_presets):
            self.stop_work()
        else:
            self.submit_next.emit(self.current_position, self.element_presets, self.scan_presets)
            self.current_position += 1

    def get_status(self) -> str:
        """
        :return: Position of current multiscan position and the length of the full multiscan script
        """
        return f"{self.current_position}/{len(self.element_presets)}"
