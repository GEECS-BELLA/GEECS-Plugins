import sys
import traceback


def exception_hook(exctype, value, tb):
    """
    Global wrapper to print out tracebacks of python errors during the execution of a PyQT window.

    :param exctype: Exception Type
    :param value: Value of the exception
    :param tb: Traceback
    """
    print("An error occurred:")
    traceback.print_exception(exctype, value, tb)
    sys.__excepthook__(exctype, value, tb)
    sys.exit(1)


class ActionError(Exception):
    """ Exception for action-related errors:  such as wrong action name or failed get command """
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class ConflictingScanElements(Exception):
    """ Exception for when a scan is submitted but the scan elements have conflicting flags """
    def __init__(self, message):
        super().__init__(message)
        self.message = message
