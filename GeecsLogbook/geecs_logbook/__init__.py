"""The GEECS scan logbook: a day-document view over scan folders.

A day is a query, not a document — see :mod:`geecs_logbook.scan_reader`.
The package is a *consumer* of scan folders and never creates one.
"""

from geecs_logbook.models import DaySummary, ScanStatus, ScanSummary
from geecs_logbook.app import __version__, create_app
from geecs_logbook.scan_reader import read_day, read_scan, scan_status

__all__ = [
    "DaySummary",
    "ScanStatus",
    "ScanSummary",
    "__version__",
    "create_app",
    "read_day",
    "read_scan",
    "scan_status",
]
