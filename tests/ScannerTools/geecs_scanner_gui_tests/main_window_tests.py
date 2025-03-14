import pytest
from pytestqt.qtbot import QtBot

from pathlib import Path

from PyQt5.QtCore import Qt

from geecs_scanner.app.geecs_scanner import GEECSScannerWindow
from geecs_scanner.utils import ApplicationPaths as AppPaths


@pytest.fixture
def app(qtbot: QtBot):
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
    list1 = app.ui.foundDevices
    list2 = app.ui.selectedDevices
    assert list1.count() == 2

    list1.setCurrentRow(0)
    qtbot.mouseClick(app.ui.addDeviceButton, Qt.LeftButton)
    assert list1.count() == 1
    assert list2.count() == 1

    list2.setCurrentRow(0)
    qtbot.mouseClick(app.ui.removeDeviceButton, Qt.LeftButton)
    assert list1.count() == 2
    assert list2.count() == 0

    list1.setCurrentRow(1)
    qtbot.mouseClick(app.ui.addDeviceButton, Qt.LeftButton)
    list1.setCurrentRow(0)
    qtbot.mouseClick(app.ui.addDeviceButton, Qt.LeftButton)
    assert list1.count() == 0
    assert list2.count() == 2


def test_button_click(app, qtbot):
    qtbot.mouseClick(app.ui.newDeviceButton, Qt.LeftButton)
    assert app.element_editor is not None


if __name__ == '__main__':
    pytest.main()
