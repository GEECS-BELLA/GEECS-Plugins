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

    def merge_with_previous(self, prev_err=None):
        if prev_err and isinstance(prev_err, ErrorAPI):
            if prev_err.is_error or (not prev_err.is_error and not self.is_error and prev_err.is_warning):
                self.is_error, self.is_warning, self.error_msg, self.error_src = \
                    [prev_err.is_error, prev_err.is_warning, prev_err.error_msg, prev_err.error_src]

    def __str__(self):
        return f'Error Object:\n\tError: {self.is_error}\n\tWarning: {self.is_warning}\n\tMessage: {self.error_msg}' \
               f'\n\tSource: {self.error_src}'


if __name__ == '__main__':
    ini_err = ErrorAPI('initial', 'main', True)
    new_err = ErrorAPI('', 'sub')
    new_err.merge_with_previous(ini_err)
    print(new_err)
