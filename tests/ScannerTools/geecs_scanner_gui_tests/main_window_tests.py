from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

import pytest

from pathlib import Path

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QTimer

from geecs_scanner.app.geecs_scanner import GEECSScannerWindow
from geecs_scanner.app.save_element_editor import SaveElementEditor
from geecs_scanner.utils import ApplicationPaths as AppPaths


@pytest.fixture
def app(qtbot: QtBot):
    """ Initializes the GEECS Scanner window in debug mode, manually setting the Exp Name and App Paths """
    window = GEECSScannerWindow(unit_test_mode=True)
    qtbot.addWidget(window)
    window.show()
    window.activateWindow()
    window.setFocus()

    # Hard-code application paths to test experiment in this package
    AppPaths.BASE_PATH = Path(__file__).parent / "test_configs"
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


if __name__ == '__main__':
    pytest.main()
