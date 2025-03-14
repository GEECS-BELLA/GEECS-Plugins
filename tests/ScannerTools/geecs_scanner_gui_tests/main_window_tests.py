import sys
import pytest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

from geecs_scanner.app.geecs_scanner import GEECSScannerWindow


@pytest.fixture
def app(qtbot):
    window = GEECSScannerWindow()
    qtbot.addWidget(window)
    window.show()
    return window


def test_button_click(app, qtbot):
    qtbot.mouseClick(app.ui.newDeviceButton, Qt.LeftButton)
    assert app.element_editor is not None


if __name__ == '__main__':
    pytest.main()
