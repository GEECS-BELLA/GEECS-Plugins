from typing import Optional


class ConfigurationError(Exception):
    """ Exception raised for errors in the configuration file. """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class GeecsDeviceInstantiationError(Exception):
    """Exception raised when a GEecs device fails to instantiate."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ErrorAPI(Exception):

    def __init__(self, message='', source='', warning=False):
        message = message.strip()
        source = source.strip()
        self.is_error = not warning and bool(message)
        self.is_warning = warning and bool(message)

        if self.is_error or self.is_warning:
            self.error_msg = message
            self.error_src = source
            super().__init__(self.error_msg)
        else:
            self.error_msg = self.error_src = ''

    def merge(self, message='', source='', warning=False):
        is_error = not warning and bool(message)
        is_warning = warning and bool(message)

        if not self.is_error and (is_error or not self.is_warning):
            self.is_error, self.is_warning, self.error_msg, self.error_src = [is_error, is_warning, message, source]

        if is_error or is_warning:
            self.error_handler((message, source, is_warning, is_error))
        else:
            self.error_handler(None)

    def error(self, message='', source=''):
        self.merge(message=message, source=source, warning=False)

    def warning(self, message='', source=''):
        self.merge(message=message, source=source, warning=True)

    def clear(self):
        self.error_msg = self.error_src = ''
        self.is_error = self.is_warning = False

    def error_handler(self, new_error: Optional[tuple[str, str, bool, bool]] = None):
        if new_error:
            print(ErrorAPI._print_str(new_error[0], new_error[1], new_error[2], new_error[3]))
        elif self.is_error or self.is_warning:
            print(ErrorAPI._print_str(self.error_msg, self.error_src, self.is_warning, self.is_error))
        self.clear()

    @staticmethod
    def _print_str(message='', source='', warning=False, error=False):
        if error:
            err_str = f'Error:\n\tMessage: {message}\n\tSource:  {source}'

        elif warning:
            err_str = f'Warning:\n\tMessage: {message}\n\tSource:  {source}'

        else:
            err_str = 'No error'

        return err_str

    def __str__(self):
        return ErrorAPI._print_str(self.error_msg, self.error_src, self.is_warning, self.is_error)


# define global
api_error = ErrorAPI()


if __name__ == '__main__':
    print(api_error)

    api_error.warning('initial', 'main')
    api_error.error('my error', 'sub')

    api_error.clear()

    api_error.warning('initial', 'main')
    api_error.error('my further error', 'sub')
