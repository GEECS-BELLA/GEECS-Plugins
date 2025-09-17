"""Error definitions for geecs api.

Note: ErrorAPI is being Deprecated in favor of calls to standard logging.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class GeecsDeviceInstantiationError(Exception):
    """Raised when a GEECS device fails to instantiate."""


class ErrorAPI:
    """Deprecated logging wrapper kept for backward compatibility."""

    def __init__(
        self, message: str = "", source: str = "", warning: bool = False
    ) -> None:
        """Initialize with optional message/state and emit a deprecation warning."""
        warnings.warn(
            "ErrorAPI is deprecated; use the standard logging module directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        message = message.strip()
        source = source.strip()
        self.is_error: bool = (not warning) and bool(message)
        self.is_warning: bool = warning and bool(message)
        self.error_msg: str = message if (self.is_error or self.is_warning) else ""
        self.error_src: str = source if (self.is_error or self.is_warning) else ""

    def merge(self, message: str = "", source: str = "", warning: bool = False) -> None:
        """Update internal state and log via standard logging."""
        warnings.warn(
            "ErrorAPI.merge is deprecated; use logger.error/warning/info instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        message = message.strip()
        source = source.strip()
        is_error = (not warning) and bool(message)
        is_warning = warning and bool(message)

        if not self.is_error and (is_error or not self.is_warning):
            self.is_error, self.is_warning, self.error_msg, self.error_src = (
                is_error,
                is_warning,
                message,
                source,
            )

        if is_error or is_warning:
            self.error_handler((message, source, is_warning, is_error))
        else:
            self.error_handler(None)

    def error(self, message: str = "", source: str = "") -> None:
        """Log an error message."""
        self.merge(message=message, source=source, warning=False)

    def warning(self, message: str = "", source: str = "") -> None:
        """Log a warning message."""
        self.merge(message=message, source=source, warning=True)

    def clear(self) -> None:
        """Clear internal error state."""
        self.error_msg = self.error_src = ""
        self.is_error = self.is_warning = False

    def error_handler(
        self, new_error: Optional[Tuple[str, str, bool, bool]] = None
    ) -> None:
        """Format and route the message to logging, then clear state."""
        if new_error:
            msg, src, warning_flag, error_flag = new_error
        elif self.is_error or self.is_warning:
            msg, src, warning_flag, error_flag = (
                self.error_msg,
                self.error_src,
                self.is_warning,
                self.is_error,
            )
        else:
            msg, src, warning_flag, error_flag = "No error", "", False, False

        if error_flag:
            logger.error("[%s] %s", src, msg) if src else logger.error("%s", msg)
        elif warning_flag:
            logger.warning("[%s] %s", src, msg) if src else logger.warning("%s", msg)
        else:
            logger.info("[%s] %s", src, msg) if src else logger.info("%s", msg)

        self.clear()

    @staticmethod
    def _format_message(message: str = "", source: str = "") -> str:
        """Return a source-prefixed message string (deprecated helper)."""
        return f"[{source}] {message}" if source else message

    def __str__(self) -> str:  # pragma: no cover
        """Return the formatted message for legacy prints."""
        return self._format_message(self.error_msg, self.error_src)


# Backward-compatible singleton (prefer using logging directly elsewhere)
api_error = ErrorAPI()
