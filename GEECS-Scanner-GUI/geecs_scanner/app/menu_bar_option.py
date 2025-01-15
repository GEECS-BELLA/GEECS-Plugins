"""
A general class to host menu options used by GEECSScanner.  In the init of the main application, just need to specify
the name of the option and these classes handle the rest.

The base "MenuBarOption" is mostly designed as the "-Bool" variant, but the config loading was different to load bools
rather than strings so a separate subclass was made specifically to load a default boolean.

TODO Add a description when initializes so that users have more to go off of than just a cryptic name.  Could be shown
TODO  when hovering over the option in the menu bar, or during the update dialog for the string option.

-Chris
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union
if TYPE_CHECKING:
    from .GEECSScanner import GEECSScannerWindow

import configparser
from pathlib import Path
from PyQt5.QtWidgets import QAction, QInputDialog
from PyQt5.QtCore import QCoreApplication

# TODO this should live somewhere else, more than GEECS Scanner needs this now...
CONFIG_PATH = Path('~/.config/geecs_python_api/config.ini').expanduser()
CONFIG_SECTION = 'Options'


class MenuBarOption(QAction):
    def __init__(self, main_window: GEECSScannerWindow, name: str):
        super().__init__(main_window)

        self.setCheckable(True)
        self.setChecked(False)
        self.setObjectName(name)

        _translate = QCoreApplication.translate
        self.setText(_translate("MainWindow", name))

        self.triggered.connect(self.update_default)

    def update_default(self):
        config = configparser.ConfigParser()
        config.read(CONFIG_PATH)

        if not config.has_section(CONFIG_SECTION):
            config.add_section(CONFIG_SECTION)

        config.set(CONFIG_SECTION, self.get_name(), str(self.get_value()))

        with open(CONFIG_PATH, 'w') as file:
            config.write(file)

    def get_name(self) -> str:
        return self.objectName()

    def get_value(self) -> Union[bool, str]:
        return self.isChecked()


class MenuBarOptionBool(MenuBarOption):
    def __init__(self, main_window: GEECSScannerWindow, name: str):
        super().__init__(main_window, name)
        self.value: bool = False
        self.load_default()
        self.setChecked(self.value)

    def load_default(self):
        config = configparser.ConfigParser()
        config.read(CONFIG_PATH)
        if config.has_section(CONFIG_SECTION) and config.has_option(CONFIG_SECTION, self.get_name()):
            self.value = config.getboolean(CONFIG_SECTION, self.get_name())


class MenuBarOptionStr(MenuBarOption):
    def __init__(self, main_window: GEECSScannerWindow, name: str):
        super().__init__(main_window, name)
        self.main_window = main_window

        self.value: str = ""
        self.load_default()
        self.setChecked(not self.value == "")

    def load_default(self):
        config = configparser.ConfigParser()
        config.read(CONFIG_PATH)
        if config.has_section(CONFIG_SECTION) and config.has_option(CONFIG_SECTION, self.get_name()):
            self.value = config[CONFIG_SECTION][self.get_name()]

    def get_value(self) -> Union[bool, str]:
        return self.value

    def update_default(self):
        self.prompt_new_value()
        super().update_default()

    def prompt_new_value(self):
        text, ok = QInputDialog.getText(self.main_window, 'Update Optional Parameter', self.get_name(), text=self.value)
        if ok:
            self.value = text.strip()

        self.setChecked(not self.value == "")
