import logging
from typing import Optional

# Set up a module-level logger
logger = logging.getLogger(__name__)

class GeecsDeviceInstantiationError(Exception):
    """Exception raised when a GEecs device fails to instantiate."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ErrorAPI(Exception):
    def __init__(self, message: str = '', source: str = '', warning: bool = False):
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

    def merge(self, message: str = '', source: str = '', warning: bool = False):
        is_error = not warning and bool(message)
        is_warning = warning and bool(message)

        # Update error state if not already flagged
        if not self.is_error and (is_error or not self.is_warning):
            self.is_error, self.is_warning, self.error_msg, self.error_src = (
                is_error, is_warning, message, source
            )

        if is_error or is_warning:
            self.error_handler((message, source, is_warning, is_error))
        else:
            self.error_handler(None)

    def error(self, message: str = '', source: str = ''):
        self.merge(message=message, source=source, warning=False)

    def warning(self, message: str = '', source: str = ''):
        self.merge(message=message, source=source, warning=True)

    def clear(self):
        self.error_msg = self.error_src = ''
        self.is_error = self.is_warning = False

    def error_handler(self, new_error: Optional[tuple[str, str, bool, bool]] = None):
        if new_error:
            msg, src, warning_flag, error_flag = new_error
        elif self.is_error or self.is_warning:
            msg, src, warning_flag, error_flag = self.error_msg, self.error_src, self.is_warning, self.is_error
        else:
            msg, src, warning_flag, error_flag = "No error", "", False, False

        formatted_msg = self._format_message(msg, src)

        # Log with the appropriate logger method; the logger's own format will be applied.
        if error_flag:
            logger.error(formatted_msg)
        elif warning_flag:
            logger.warning(formatted_msg)
        else:
            logger.info(formatted_msg)

        self.clear()

    @staticmethod
    def _format_message(message: str = '', source: str = '') -> str:
        # If a source is provided, prefix the message with it.
        return f"[{source}] {message}" if source else message

    def __str__(self) -> str:
        return self._format_message(self.error_msg, self.error_src)

# Global instance (maintaining existing usage)
api_error = ErrorAPI()

if __name__ == '__main__':
    # Configure logging (for demo purposes)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    logger.info("Testing ErrorAPI:")
    print(api_error)

    # Test a warning
    api_error.warning('initial warning', 'main')
    # Test an error
    api_error.error('my error', 'sub')

    api_error.clear()

    # Another round of tests
    api_error.warning('another warning', 'main')
    api_error.error('a further error', 'sub')
