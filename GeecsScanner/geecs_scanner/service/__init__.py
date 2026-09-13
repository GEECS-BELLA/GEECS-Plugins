"""The scanner's service layer: pure Python over the queue client.

Nothing in this package imports FastAPI.  :class:`ScannerService` takes the
client, the resolver and the stream cache it is given, and every method
returns a Pydantic model or raises :class:`ScannerError`.  The web layer
is three lines per route over it; tests drive it directly.
"""

from geecs_scanner.service.errors import ScannerError
from geecs_scanner.service.scanner import ScannerService
from geecs_scanner.service.streams import ProgressCache

__all__ = ["ProgressCache", "ScannerError", "ScannerService"]
