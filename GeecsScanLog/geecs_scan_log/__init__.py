"""The GEECS scan logbook: a day-document view over scan folders.

A day is a query, not a document — see :mod:`geecs_scan_log.scan_reader`.
The package is a *consumer* of scan folders and never creates one.
"""

from geecs_scan_log.models import DaySummary, ScanStatus, ScanSummary
from geecs_scan_log.router import create_log_router
from geecs_scan_log.scan_reader import read_day, read_scan, scan_status

__all__ = [
    "DaySummary",
    "ScanStatus",
    "ScanSummary",
    "create_log_router",
    "read_day",
    "read_scan",
    "scan_status",
]
