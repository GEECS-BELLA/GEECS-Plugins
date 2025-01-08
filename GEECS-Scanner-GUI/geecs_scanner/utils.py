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
