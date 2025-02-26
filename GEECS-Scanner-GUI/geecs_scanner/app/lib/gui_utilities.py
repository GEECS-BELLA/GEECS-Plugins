from __future__ import annotations
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt5.QtCore import QObject
from pathlib import Path
import yaml

from PyQt5.QtWidgets import QLineEdit, QCompleter
from PyQt5.QtCore import Qt, QObject


def display_completer_list(window: QObject, location: QLineEdit, completer_list: list[str], max_visible_lines: int = 6):
    """ Displays a completer list at a given location

    :param window: GUI window that calls this (ie; use 'self')
    :param location: GUI element at which to show the completer list
    :param completer_list: strings to show in the completer pop-up
    :param max_visible_lines: maximum number of completer entries to show
    """
    if location.isEnabled():
        location.selectAll()
        completer = QCompleter(completer_list, window)
        completer.setMaxVisibleItems(max_visible_lines)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        completer.setCaseSensitivity(Qt.CaseSensitive)

        location.setCompleter(completer)
        location.setFocus()
        completer.complete()


def display_completer_variable_list(window: QObject, database_dict: dict,
                                    list_location: QLineEdit, device_location: QLineEdit):
    """ Displays list of variables at one location using the device name at another location

    :param window: GUI window that calls this (ie; use 'self')
    :param database_dict: database containing all devices and associated variables
    :param list_location: GUI element at which to show the completer list
    :param device_location: GUI element where the device name is given
    """
    device_name = device_location.text().strip()
    if device_name in database_dict:
        variable_list = sorted(database_dict[device_name].keys())
        display_completer_list(window=window, location=list_location, completer_list=variable_list)


def parse_variable_text(text) -> Union[int, float, str]:
    """ Attempts to convert a string first to an int, then a float, and finally just returns the string if unsuccessful

    :param text: string
    :return: either an int, float, or string of the input, in that order
    """
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def write_dict_to_yaml_file(filename: Path, dictionary: dict):
    """ Write the given dictionary to the given yaml file, lives here in case writing ever changes

    :param filename: yaml filename
    :param dictionary: complete dictionary to be written
    """
    with open(filename, 'w') as f:
        yaml.dump(dictionary, f, default_flow_style=False)
