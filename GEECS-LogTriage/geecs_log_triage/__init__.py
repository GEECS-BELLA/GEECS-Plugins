"""GEECS-LogTriage: harvest, classify, and aggregate scan-log errors.

Public API
----------
LogEntry, Severity
    Re-exported from :mod:`geecs_data_utils.scan_log_loader` for convenience.
ErrorFingerprint, ErrorOccurrence, TriageReport, Classification
    Triage-specific Pydantic v2 models.
parse_scan_log
    Parse a `scan.log` file into a list of `LogEntry` (re-exported).
load_scan_log
    Load `scan.log` from a scan folder (re-exported).
fingerprint_for
    Compute a stable :class:`ErrorFingerprint` for a `LogEntry`.
classify
    Map an exception type / message into a :class:`Classification`.
harvest_date, harvest_date_range, harvest_scan_folder
    High-level helpers that walk scans and produce a :class:`TriageReport`.
"""

from geecs_data_utils import LogEntry, Severity, load_scan_log, parse_scan_log
from geecs_log_triage.schemas import (
    Classification,
    ErrorFingerprint,
    ErrorOccurrence,
    TriageReport,
)
from geecs_log_triage.fingerprint import fingerprint_for, normalize_message
from geecs_log_triage.classifier import classify
from geecs_log_triage.harvester import (
    day_folder_for,
    harvest_date,
    harvest_date_range,
    harvest_scan,
    harvest_scan_folder,
)
from geecs_log_triage.render import render_markdown

__all__ = [
    # re-exports from data-utils
    "LogEntry",
    "Severity",
    "load_scan_log",
    "parse_scan_log",
    # triage models
    "Classification",
    "ErrorFingerprint",
    "ErrorOccurrence",
    "TriageReport",
    # functions
    "fingerprint_for",
    "normalize_message",
    "classify",
    "day_folder_for",
    "harvest_date",
    "harvest_date_range",
    "harvest_scan",
    "harvest_scan_folder",
    "render_markdown",
]
