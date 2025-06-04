"""
This is a test that the tests will work, and it gives a base example of how to test PyQT5 GUI features

-Chris
"""

import pytest
from PyQt5.QtWidgets import QMainWindow, QPushButton
from PyQt5.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Test Window')
        self.button = QPushButton('Click Me', self)
        self.button.clicked.connect(self.on_button_click)
        self.setCentralWidget(self.button)
        self.button_clicked = False

    def on_button_click(self):
        self.button_clicked = True


@pytest.fixture
def app(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    return window


def test_button_click(app, qtbot):
    qtbot.mouseClick(app.button, Qt.LeftButton)
    assert app.button_clicked is True


if __name__ == '__main__':
    pytest.main()
