class ErrorAPI(Exception):
    def __init__(self, error_msg=''):
        super().__init__('\n\nError: ' + error_msg)
