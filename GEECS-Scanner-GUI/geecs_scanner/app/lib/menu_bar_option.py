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
    from .. import GEECSScannerWindow

import configparser
from ...utils import ApplicationPaths as AppPaths
from PyQt5.QtWidgets import QAction, QInputDialog
from PyQt5.QtCore import QCoreApplication


CONFIG_SECTION = 'Options'


class MenuBarOption(QAction):
    def __init__(self, main_window: GEECSScannerWindow, name: str):
        """
        Base class that contains functions relevant no matter how associated values to the given option are saved.  Here
        I assume a boolean value that is stored only in the "isChecked()" function of the action button.

        :param main_window: Reference to the main window gui for initialization and prompting dialogs
        :param name: Name of the option.  Shown in the menu bar, config file, and passed as the key of a dict
        """
        super().__init__(main_window)

        self.unit_test_mode = main_window.unit_test_mode

        self.setCheckable(True)
        self.setChecked(False)
        self.setObjectName(name)

        _translate = QCoreApplication.translate
        self.setText(_translate("MainWindow", name))

        self.triggered.connect(self.update_default)

    def update_default(self):
        """ Writes the current result of "get_value()" to the config file under `[CONFIG_SECTION][self.get_name()]` """
        if self.unit_test_mode:
            return

        config = configparser.ConfigParser()
        config.read(AppPaths.config_file())

        if not config.has_section(CONFIG_SECTION):
            config.add_section(CONFIG_SECTION)

        config.set(CONFIG_SECTION, self.get_name(), str(self.get_value()))

        with open(AppPaths.config_file(), 'w') as file:
            config.write(file)

    def get_name(self) -> str:
        """
        :return: the name of the option, currently stored within its 'objectName()'
        """
        return self.objectName()

    def get_value(self) -> Union[bool, str]:
        """
        :return: Returns whether the action button is currently checked as the default value.  Can be overridden to
        instead return a str or a different bool
        """
        return self.isChecked()


class MenuBarOptionBool(MenuBarOption):
    def __init__(self, main_window: GEECSScannerWindow, name: str):
        """
        A more specific subclass that works with a bool value associated with an action option.  The only difference
        here is that this subclass loads the starting value of the bool option from the config file

        :param main_window: Reference to the main window gui for initialization and prompting dialogs
        :param name: Name of the option.  Shown in the menu bar, config file, and passed as the key of a dict
        """
        super().__init__(main_window, name)
        self.value: bool = False
        self.load_default()
        self.setChecked(self.value)

    def load_default(self):
        """ Loads the boolean value of the given option from the user config .ini file """
        if self.unit_test_mode:
            return

        config = configparser.ConfigParser()
        config.read(AppPaths.config_file())
        if config.has_section(CONFIG_SECTION) and config.has_option(CONFIG_SECTION, self.get_name()):
            self.value = config.getboolean(CONFIG_SECTION, self.get_name())


class MenuBarOptionStr(MenuBarOption):
    def __init__(self, main_window: GEECSScannerWindow, name: str):
        """
        Subclass that works with a str value associated with an action option.  Since a str cannot be represented by a
        simple `isChecked()` function, the value is instead set through a dialog box that appears when the action is
        clicked.  This value is written to the config .ini file, and the action button itself appears checked when the
        str value is not None.

        :param main_window: Reference to the main window gui for initialization and prompting dialogs
        :param name: Name of the option.  Shown in the menu bar, config file, and passed as the key of a dict
        """
        super().__init__(main_window, name)
        self.main_window = main_window

        self.value: str = ""
        self.load_default()
        self.setChecked(not self.value == "")

    def load_default(self):
        """ Loads the str value of the given option from the user config .ini file """
        if self.unit_test_mode:
            return

        config = configparser.ConfigParser()
        config.read(AppPaths.config_file())
        if config.has_section(CONFIG_SECTION) and config.has_option(CONFIG_SECTION, self.get_name()):
            self.value = config[CONFIG_SECTION][self.get_name()]

    def get_value(self) -> Union[bool, str]:
        """
        :return: the str value associated with the action option
        """
        return self.value

    def update_default(self):
        """ First prompts for a new str value before continuing on to the default `update_default()` function """
        self.prompt_new_value()
        super().update_default()

    def prompt_new_value(self):
        """ Opens a dialog box to prompt the user for a new value """
        if self.unit_test_mode:
            self.value = 'test' if not self.value else ''

        else:
            text, ok = QInputDialog.getText(self.main_window, 'Update Optional Parameter', self.get_name(), text=self.value)
            if ok:
                self.value = text.strip()

        self.setChecked(not self.value == "")
