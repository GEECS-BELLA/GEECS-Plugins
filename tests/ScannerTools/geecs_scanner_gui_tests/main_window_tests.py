import pytest

from pathlib import Path

from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

from geecs_scanner.app.geecs_scanner import GEECSScannerWindow
from geecs_scanner.utils import ApplicationPaths as AppPaths


@pytest.fixture
def app(qtbot):
    window = GEECSScannerWindow(unit_test_mode=True)
    qtbot.addWidget(window)
    window.show()

    # Hard-code application paths to test experiment in this package
    AppPaths.BASE_PATH = Path(__file__).parent / "test_configs"
    window.app_paths = AppPaths(experiment="Test", create_new=False)

    # Set experiment name to "Test" and update the GUI accordingly
    window.ui.experimentDisplay.setText("Test")

    return window


def test_button_click(app, qtbot):
    qtbot.mouseClick(app.ui.newDeviceButton, Qt.LeftButton)
    assert app.element_editor is not None


if __name__ == '__main__':
    pytest.main()
