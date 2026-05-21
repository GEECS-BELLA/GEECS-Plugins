"""Pydantic v2 data models specific to log triage.

The lower-level :class:`LogEntry` and :class:`Severity` types are owned by
:mod:`geecs_data_utils.scan_log_loader` and re-exported here for convenience
(so callers can ``from geecs_log_triage import LogEntry`` without needing a
second import).

Triage-specific types defined here:

- :class:`Classification` - triage category for an error fingerprint.
- :class:`ErrorFingerprint` - stable hash + signature for dedup.
- :class:`ErrorOccurrence` - one occurrence of an error in one scan.
- :class:`TriageReport` - top-level aggregate emitted by the harvester / CLI.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

# Re-export the shared loader types so triage consumers only need one import.
from geecs_data_utils import LogEntry, Severity  # noqa: F401  (re-export)


class Classification(str, Enum):
    """Triage category for an error fingerprint.

    Drives downstream behavior (Stage 2 issue filing, Stage 3 fix
    attempts). See `classifier.CLASSIFICATION_MAP` for the mapping rules.
    """

    BUG_CANDIDATE = "bug_candidate"
    CONFIG_ISSUE = "config_issue"
    HARDWARE_ISSUE = "hardware_issue"
    OPERATOR_ERROR = "operator_error"
    UNKNOWN = "unknown"


class ErrorFingerprint(BaseModel):
    """A stable identifier for an error, used to dedup across scans.

    Two errors with cosmetic differences (different scan numbers,
    timestamps, file paths in the message) should share a fingerprint.

    Attributes
    ----------
    fingerprint_hash : str
        12-character sha1 hex digest of the signature string.
    signature : str
        Human-readable signature used to compute the hash.
    exception_type : str, optional
        Extracted exception class name (e.g., "KeyError"), if present.
    normalized_message : str
        Message body with numerics, IDs, and paths normalized.
    sample_traceback : str, optional
        Representative traceback from the first occurrence (truncated).
    classification : Classification
        Triage category for routing in Stage 2.
    """

    fingerprint_hash: str = Field(min_length=12, max_length=12)
    signature: str
    exception_type: Optional[str] = None
    normalized_message: str
    sample_traceback: Optional[str] = None
    classification: Classification = Classification.UNKNOWN


class ErrorOccurrence(BaseModel):
    """A single occurrence of an error in a particular scan.

    Attributes
    ----------
    fingerprint : ErrorFingerprint
        The error's fingerprint (shared by similar occurrences).
    entry : LogEntry
        The originating log entry (preserves traceback, timestamp, etc.).
    scan_id : str
        Scan identifier in `Scan{NNN}` form.
    scan_folder : Path
        Absolute path to the scan folder.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fingerprint: ErrorFingerprint
    entry: LogEntry
    scan_id: str
    scan_folder: Path


class TriageReport(BaseModel):
    """Top-level aggregate produced by the harvester / CLI.

    Attributes
    ----------
    date_range : tuple[date, date]
        Inclusive (start, end) of dates examined.
    experiment : str, optional
        Experiment name passed to the harvester (None if scanning a
        directly-given folder).
    min_level : Severity
        Lowest severity level kept in `grouped` occurrences.
    total_scans_examined : int
        Number of scan folders that were walked.
    total_log_entries : int
        Total parsed log entries across all scans, before level filtering.
    total_errors : int
        Total `ErrorOccurrence` records grouped (after level filtering).
    grouped : dict[str, list[ErrorOccurrence]]
        Map from `fingerprint_hash` to all occurrences sharing it.
    generated_at : datetime
        UTC timestamp when the report was produced.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    date_range: tuple[date, date]
    experiment: Optional[str] = None
    min_level: Severity = Severity.ERROR
    total_scans_examined: int = 0
    total_log_entries: int = 0
    total_errors: int = 0
    grouped: dict[str, list[ErrorOccurrence]] = Field(default_factory=dict)
    generated_at: datetime
