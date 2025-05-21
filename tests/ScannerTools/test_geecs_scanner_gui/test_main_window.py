from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot
    from PyQt5.QtWidgets import QRadioButton

import pytest
import copy

from pathlib import Path

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QTimer

from geecs_scanner.app import (GEECSScannerWindow, SaveElementEditor, ScanVariableEditor,
                               ShotControlEditor, MultiScanner, ActionLibrary)
from geecs_scanner.app.geecs_scanner import MAXIMUM_SCAN_SIZE

from geecs_scanner.utils import ApplicationPaths as AppPaths
from geecs_scanner.app.lib.gui_utilities import read_yaml_file_to_dict
from geecs_scanner.utils.exceptions import ConflictingScanElements


@pytest.fixture
def app(qtbot: QtBot):
    """ Initializes the GEECS Scanner window in debug mode, manually setting the Exp Name and App Paths """
    window = GEECSScannerWindow()
    qtbot.addWidget(window)
    window.show()
    window.activateWindow()
    window.setFocus()

    # Hard-code application paths to test experiment in this package
    AppPaths.BASE_PATH = Path(__file__).parents[1] / "test_configs"
    window.app_paths = AppPaths(experiment="Test", create_new=False)

    # Set experiment name to "Test" and update the GUI accordingly
    window.ui.experimentDisplay.setText("Test")

    return window


def test_element_list(app, qtbot: QtBot):
    """ Tests moving elements from one list to another, as well as the refresh lists button """

    # First, test the ability to discover and move elements between the `found` and `selected` lists
    list1 = app.ui.foundDevices
    list2 = app.ui.selectedDevices
    assert list1.count() == 2
    assert list2.count() == 0

    list1.setCurrentRow(1)
    qtbot.mouseClick(app.ui.addDeviceButton, Qt.LeftButton)
    assert list1.count() == 1
    assert list2.count() == 1

    list1.setCurrentRow(0)
    qtbot.mouseClick(app.ui.addDeviceButton, Qt.LeftButton)
    assert list1.count() == 0
    assert list2.count() == 2

    # Next, if we create a new element does it correctly refresh the list
    temp_file = app.app_paths.save_devices() / "temp.yaml"
    with open(temp_file, "w"):
        pass
    assert list1.count() == 0
    assert list2.count() == 2

    qtbot.mouseClick(app.ui.buttonRefreshLists, Qt.LeftButton)
    assert list1.count() == 1
    assert list2.count() == 2

    temp_file.unlink()
    qtbot.mouseClick(app.ui.buttonRefreshLists, Qt.LeftButton)
    assert list1.count() == 0
    assert list2.count() == 2

    # Lastly, move the two selected elements back to found
    list2.setCurrentRow(0)
    qtbot.mouseClick(app.ui.removeDeviceButton, Qt.LeftButton)
    list2.setCurrentRow(0)
    qtbot.mouseClick(app.ui.removeDeviceButton, Qt.LeftButton)
    assert list1.count() == 2
    assert list2.count() == 0

    list1.clearSelection()
    list2.clearSelection()


def test_open_element_editor(app, qtbot):
    """ Tests opening the element editor using 'New' and 'Edit' with elements selected/not selected """

    def check_and_close_element_editor(check_value: str = ''):
        active_modal = QApplication.activeModalWidget()
        if active_modal:
            if isinstance(active_modal, SaveElementEditor):
                dialog: SaveElementEditor = active_modal
                assert dialog.ui.lineElementName.text().strip() == check_value
            else:
                raise AssertionError("Unexpected active modal to check value of")
            active_modal.close()
        else:
            raise AssertionError("No active modal to check value of")

    QTimer.singleShot(200, lambda: check_and_close_element_editor(check_value=''))
    qtbot.mouseClick(app.ui.newDeviceButton, Qt.LeftButton)

    assert app.element_editor is not None
    app.element_editor = None

    app.ui.foundDevices.clearSelection()
    QTimer.singleShot(200, lambda: check_and_close_element_editor(check_value=''))
    qtbot.mouseClick(app.ui.editDeviceButton, Qt.LeftButton)

    assert app.element_editor is not None
    app.element_editor = None

    test_ind = 0
    app.ui.foundDevices.setCurrentRow(test_ind)
    test_text = app.ui.foundDevices.item(test_ind).text()
    QTimer.singleShot(200, lambda: check_and_close_element_editor(check_value=test_text))
    qtbot.mouseClick(app.ui.editDeviceButton, Qt.LeftButton)

    assert app.element_editor is not None
    app.element_editor = None
    app.ui.foundDevices.clearSelection()


def test_menu_options(app, qtbot: QtBot):
    from geecs_scanner.app.geecs_scanner import BOOLEAN_OPTIONS, STRING_OPTIONS
    assert len(app.all_options) == len(BOOLEAN_OPTIONS) + len(STRING_OPTIONS)

    if len(BOOLEAN_OPTIONS) > 0:
        bool_opt = app.all_options[0]
        assert bool_opt.get_name() == BOOLEAN_OPTIONS[0]
        initial_value = bool_opt.isChecked()

        bool_opt.trigger()
        assert initial_value is not bool_opt.isChecked()

        bool_opt.trigger()
        assert initial_value is bool_opt.isChecked()

    if len(STRING_OPTIONS) > 0:
        str_opt = app.all_options[len(BOOLEAN_OPTIONS)]
        assert str_opt.get_name() == STRING_OPTIONS[0]
        assert str_opt.get_value() == ''
        assert str_opt.isChecked() is False

        str_opt.trigger()
        assert str_opt.get_value() == 'test'
        assert str_opt.isChecked() is True

        str_opt.trigger()
        assert str_opt.get_value() == ''
        assert str_opt.isChecked() is False


def test_opening_side_guis(app, qtbot: QtBot):
    """ Tests opening the other 4 main gui windows and closing them """

    def close_window(gui_reference, gui_class):
        active_modal = QApplication.activeModalWidget()
        if active_modal:
            assert isinstance(active_modal, gui_class)
            active_modal.close()
        elif gui_reference:
            assert isinstance(gui_reference, gui_class)
            if hasattr(gui_reference, 'close'):
                gui_reference.close()
            else:
                raise AssertionError(f"No function 'close' for '{gui_class}'.")
        else:
            raise AssertionError(f"No active gui found for '{gui_class}'.")

    # 1D Scan Variable Editor
    QTimer.singleShot(200, lambda: close_window(app.variable_editor, ScanVariableEditor))
    qtbot.mouseClick(app.ui.buttonScanVariables, Qt.LeftButton)
    assert app.variable_editor is not None
    app.variable_editor = None

    # Shot Control/Timing Device Editor
    QTimer.singleShot(200, lambda: close_window(app.timing_editor, ShotControlEditor))
    qtbot.mouseClick(app.ui.buttonOpenTimingSetup, Qt.LeftButton)
    assert app.timing_editor is not None
    app.timing_editor = None

    # Action Library
    QTimer.singleShot(200, lambda: close_window(app.action_library_window, ActionLibrary))
    qtbot.mouseClick(app.ui.buttonActionLibrary, Qt.LeftButton)
    assert app.action_library_window is not None
    assert app.is_in_action_library is True
    qtbot.wait(300)
    assert app.action_library_window is None
    assert app.is_in_action_library is False

    # Multiscanner
    QTimer.singleShot(200, lambda: close_window(app.multiscanner_window, MultiScanner))
    qtbot.mouseClick(app.ui.buttonLaunchMultiScan, Qt.LeftButton)
    assert app.multiscanner_window is not None
    assert app.is_in_multiscan is True
    qtbot.wait(300)
    assert app.multiscanner_window is None
    assert app.is_in_multiscan is False


def test_adjusting_scan_parameters(app, qtbot: QtBot):
    """ Tests the basic functionality of changing the scan settings.  This ensures that (1) the correct line edits
     are enabled when their respective radio button is selected and (2) """
    scan_line_elements = [app.ui.lineStartValue, app.ui.lineStopValue, app.ui.lineStepSize, app.ui.lineShotStep]

    def check_scan_line_edits(selected_radio: QRadioButton, check_value: bool):
        """ First checks that only the specified radio button is the one currently checked.  Next, checks that
         the line edits are enabled/disabled correctly based on the given expected boolean """
        assert app.ui.noscanRadioButton.isChecked() is (selected_radio is app.ui.noscanRadioButton)
        assert app.ui.scanRadioButton.isChecked() is (selected_radio is app.ui.scanRadioButton)
        assert app.ui.backgroundRadioButton.isChecked() is (selected_radio is app.ui.backgroundRadioButton)

        assert app.ui.lineScanVariable.isEnabled() is check_value
        for element in scan_line_elements:
            assert element.isEnabled() is check_value

        assert app.ui.lineNumShots.isEnabled() is not check_value

    # Click on 'NoScan', check line edit status
    qtbot.mouseClick(app.ui.noscanRadioButton, Qt.LeftButton)
    check_scan_line_edits(app.ui.noscanRadioButton, False)

    # Click on 'Background', check line edit status.  Set number of noscan shots to 50
    qtbot.mouseClick(app.ui.backgroundRadioButton, Qt.LeftButton)
    check_scan_line_edits(app.ui.backgroundRadioButton, False)

    app.ui.lineNumShots.setText('50')
    app.update_noscan_num_shots()
    assert app.noscan_num == 50

    # Click on 'Scan', check line edit status.  Check number of scan variables and set test values to all line edits
    qtbot.mouseClick(app.ui.scanRadioButton, Qt.LeftButton)
    check_scan_line_edits(app.ui.scanRadioButton, True)

    assert len(app.scan_variable_list) == 2
    assert len(app.scan_composite_list) == 2
    app.ui.lineScanVariable.setText(app.scan_variable_list[0])
    for i in range(len(scan_line_elements)):
        scan_line_elements[i].setText(str(i))
    app.calculate_num_shots()

    # Click on 'NoScan', check line edit status.  Check that "50" was preserved in number of noscan shots and reset it
    qtbot.mouseClick(app.ui.noscanRadioButton, Qt.LeftButton)
    check_scan_line_edits(app.ui.noscanRadioButton, False)

    assert app.noscan_num == 50
    assert int(app.ui.lineNumShots.text()) == 50
    app.ui.lineNumShots.setText('100')
    app.update_noscan_num_shots()

    # Check that scan line edits are empty
    assert app.ui.lineScanVariable.text() == ''
    for line in scan_line_elements:
        assert line.text() == ''

    # Click on 'Scan', check that previously entered values were preserved.  Reset it and go back to 'NoScan'
    qtbot.mouseClick(app.ui.scanRadioButton, Qt.LeftButton)
    assert app.ui.lineScanVariable.text() == app.scan_variable_list[0]
    for i in range(len(scan_line_elements)):
        assert int(float(scan_line_elements[i].text())) == i
        scan_line_elements[i].setText("")
    app.calculate_num_shots()

    qtbot.mouseClick(app.ui.noscanRadioButton, Qt.LeftButton)


def test_scan_preset(app, qtbot: QtBot):
    """ Tests saving, loading, and deleting presets """
    assert app.ui.listScanPresets.count() == 0

    app.save_current_preset(filename="blank")
    assert app.ui.listScanPresets.count() == 1

    app.ui.foundDevices.setCurrentRow(0)
    qtbot.mouseClick(app.ui.addDeviceButton, Qt.LeftButton)
    qtbot.mouseClick(app.ui.scanRadioButton, Qt.LeftButton)
    app.ui.lineScanVariable.setText(app.scan_variable_list[0])

    app.save_current_preset(filename="test")
    assert app.ui.listScanPresets.count() == 2
    assert app.ui.noscanRadioButton.isChecked() is False

    app.ui.listScanPresets.setCurrentRow(0)
    app.apply_preset()  # Double-clicking is not possible, just calling the function
    assert app.ui.selectedDevices.count() == 0
    assert app.ui.noscanRadioButton.isChecked() is True
    assert app.ui.lineScanVariable.text() == ""
    assert int(float(app.ui.lineNumShots.text())) == 100

    app.ui.listScanPresets.setCurrentRow(1)
    app.apply_preset()
    assert app.ui.selectedDevices.count() == 1
    assert app.ui.scanRadioButton.isChecked() is True
    assert app.ui.lineScanVariable.text() == app.scan_variable_list[0]

    app.ui.listScanPresets.setCurrentRow(0)
    app.apply_preset()
    app.ui.listScanPresets.setCurrentRow(1)
    app.delete_selected_preset()
    assert app.ui.listScanPresets.count() == 1
    app.ui.listScanPresets.setCurrentRow(0)
    app.delete_selected_preset()
    assert app.ui.listScanPresets.count() == 0
    assert app.ui.selectedDevices.count() == 0
    assert int(float(app.ui.lineNumShots.text())) == 100


def test_shot_calculation(app, qtbot: QtBot):
    """ Tests calculating the number of shots in various configurations """
    app.save_current_preset(filename="blank")

    def calculate_and_assert(expected_string: str = "N/A"):
        app.calculate_num_shots()
        assert app.ui.lineNumShots.text() == expected_string

    qtbot.mouseClick(app.ui.scanRadioButton, Qt.LeftButton)
    assert app.ui.lineNumShots.text() == "N/A"

    app.ui.lineStartValue.setText("1.0")
    app.ui.lineStopValue.setText("2.0")
    app.ui.lineStepSize.setText("0.5")
    app.ui.lineShotStep.setText("10")
    calculate_and_assert("30")

    app.ui.lineStepSize.setText("-0.2")
    calculate_and_assert("60")

    app.ui.lineStopValue.setText("-1.0")
    calculate_and_assert("110")

    app.ui.lineStepSize.setText("0.5")
    calculate_and_assert("50")

    app.ui.lineShotStep.setText("10.5")
    calculate_and_assert()
    app.ui.lineShotStep.setText("0")
    calculate_and_assert()
    app.ui.lineShotStep.setText("10")

    app.ui.lineStepSize.setText("0")
    calculate_and_assert()
    app.ui.lineStepSize.setText("50")
    calculate_and_assert("10")
    app.ui.lineStepSize.setText("0.5")

    app.ui.lineStopValue.setText("1.0")
    calculate_and_assert("10")

    app.ui.lineShotStep.setText(str(MAXIMUM_SCAN_SIZE + 1))
    calculate_and_assert()

    app.ui.listScanPresets.setCurrentRow(0)
    app.apply_preset()
    app.ui.listScanPresets.setCurrentRow(0)
    app.delete_selected_preset()


def test_list_of_steps(app, qtbot: QtBot):
    app.save_current_preset(filename="blank")

    # Tooltip button is visible when entering Scan mode
    assert app.ui.toolbuttonStepList.isVisible() is False
    qtbot.mouseClick(app.ui.scanRadioButton, Qt.LeftButton)
    assert app.ui.toolbuttonStepList.isVisible() is True
    assert app.ui.toolbuttonStepList.toolTip().strip() == ""

    # Test that it returns the correct list of steps when choosing a basic 1D scan
    app.ui.lineScanVariable.setText(app.scan_variable_list[0])
    app.ui.lineStartValue.setText("1")
    app.ui.lineStopValue.setText("2")
    app.ui.lineStepSize.setText("0.5")
    app.ui.lineShotStep.setText("20")
    app.calculate_num_shots()
    tooltip_string = app.ui.toolbuttonStepList.toolTip().strip()
    assert tooltip_string != ""

    # If we change the number of shots, then the tool tip string should not change
    app.ui.lineShotStep.setText("10")
    app.calculate_num_shots()
    assert app.ui.toolbuttonStepList.toolTip().strip() == tooltip_string

    # Test 'relative' composite variables, though we are just seeing that the code executes without error
    app.ui.lineScanVariable.setText("test_comp_1")
    app.calculate_num_shots()
    assert tooltip_string != ""

    # TODO could consider making something to test 'absolute', but this requires a `get` command to a GEECS Device

    app.ui.listScanPresets.setCurrentRow(0)
    app.apply_preset()
    app.ui.listScanPresets.setCurrentRow(0)
    app.delete_selected_preset()


def test_dictionary_combining(app, qtbot: QtBot):
    """ Tests combining element dictionaries while (1) throwing errors for conflicting flags and (2) no duplicates """
    test_dict_location = app.app_paths.experiment() / 'aux_configs'
    dict_a = read_yaml_file_to_dict(test_dict_location / 'test_a.yaml')
    dict_b = read_yaml_file_to_dict(test_dict_location / 'test_b.yaml')
    dict_c = read_yaml_file_to_dict(test_dict_location / 'test_c.yaml')

    test_ab = copy.deepcopy(dict_a['Devices'])
    app.combine_elements(test_ab, dict_b['Devices'])
    assert len(test_ab['device_1']['variable_list']) == 3  # Should not append two copies of a particular variable
    assert test_ab['device_1']['synchronous'] is True
    assert len(test_ab['device_2']['variable_list']) == 1

    try:
        app.combine_elements(dict_a['Devices'], dict_c['Devices'])
        raise AssertionError("Mismatched 'synchronous' flags, should raise an error")
    except ConflictingScanElements:
        assert True

    test_bc = copy.deepcopy(dict_b['Devices'])
    app.combine_elements(test_bc, dict_c['Devices'])
    assert len(test_bc['device_1']['variable_list']) == 2
    assert test_ab['device_1']['synchronous'] is True
    assert test_bc['device_2']['synchronous'] is False
    assert len(test_bc['device_2']['variable_list']) == 1


# TODO Test Updating config

# TODO Test starting/stopping scan and logic for which buttons are enabled


if __name__ == '__main__':
    pytest.main()
