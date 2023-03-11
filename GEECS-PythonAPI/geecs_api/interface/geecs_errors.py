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

        if not self.is_error and (self.is_error or is_error or not self.is_warning):
            self.is_error, self.is_warning, self.error_msg, self.error_src = [is_error, is_warning, message, source]

    def error(self, message='', source=''):
        self.merge(message=message, source=source, warning=False)

    def warning(self, message='', source=''):
        self.merge(message=message, source=source, warning=True)

    def clear(self):
        self.error_msg = self.error_src = ''
        self.is_error = self.is_warning = False

    def __str__(self):
        if self.is_error:
            err_str = f'Error:\n\tMessage: {self.error_msg}\n\tSource:  {self.error_src}'

        elif self.is_warning:
            err_str = f'Warning:\n\tMessage: {self.error_msg}\n\tSource:  {self.error_src}'

        else:
            err_str = 'No error'

        return err_str


# define global
api_error = ErrorAPI()


if __name__ == '__main__':
    print(api_error)

    api_error.warning('initial', 'main')
    print(api_error)

    api_error.error('my error', 'sub')
    print(api_error)

    # api_error.clear()

    api_error.error('my further error', 'sub')
    print(api_error)
