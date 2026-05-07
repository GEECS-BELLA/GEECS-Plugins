"""Classify a log entry / error into one of the triage `Classification` buckets.

Deterministic mapping from exception type (preferred when available from
a traceback) and message content. The map encodes the GEECS exception
taxonomy declared in ``GEECS-Scanner-GUI/geecs_scanner/utils/exceptions.py``
plus common Python builtins.

Classifications drive Stage 2 routing:

- ``BUG_CANDIDATE``  - file as a bug, eligible for Stage 3 fix attempts.
- ``CONFIG_ISSUE``   - file as needs-human-review, do **not** auto-fix code.
- ``HARDWARE_ISSUE`` - file as flaky-hardware tracking, do not auto-fix.
- ``OPERATOR_ERROR`` - typically batch into a daily digest, not per-fingerprint.
- ``UNKNOWN``        - fall through; Stage 2 will likely defer.
"""

from __future__ import annotations

from typing import Optional

from geecs_data_utils import LogEntry  # re-exported by data-utils
from geecs_log_triage.schemas import Classification


# Exception type -> Classification.
#
# Keep this list short and intentional. When new GEECS exception types are
# added, edit this map rather than scattering classification logic through
# the codebase.
CLASSIFICATION_MAP: dict[str, Classification] = {
    # GEECS-specific (from geecs_scanner.utils.exceptions and friends).
    "ActionError": Classification.CONFIG_ISSUE,
    "ConflictingScanElements": Classification.CONFIG_ISSUE,
    "GeecsDeviceInstantiationError": Classification.HARDWARE_ISSUE,
    # Python builtins commonly indicating a code bug.
    "KeyError": Classification.BUG_CANDIDATE,
    "AttributeError": Classification.BUG_CANDIDATE,
    "TypeError": Classification.BUG_CANDIDATE,
    "ValueError": Classification.BUG_CANDIDATE,
    "IndexError": Classification.BUG_CANDIDATE,
    "AssertionError": Classification.BUG_CANDIDATE,
    "ZeroDivisionError": Classification.BUG_CANDIDATE,
    "NameError": Classification.BUG_CANDIDATE,
    "ImportError": Classification.BUG_CANDIDATE,
    "ModuleNotFoundError": Classification.BUG_CANDIDATE,
    # Network / hardware-ish.
    "ConnectionRefusedError": Classification.HARDWARE_ISSUE,
    "ConnectionResetError": Classification.HARDWARE_ISSUE,
    "ConnectionAbortedError": Classification.HARDWARE_ISSUE,
    "TimeoutError": Classification.HARDWARE_ISSUE,
    "BrokenPipeError": Classification.HARDWARE_ISSUE,
    # Filesystem / data.
    "FileNotFoundError": Classification.OPERATOR_ERROR,
    "PermissionError": Classification.OPERATOR_ERROR,
    "IsADirectoryError": Classification.OPERATOR_ERROR,
    "NotADirectoryError": Classification.OPERATOR_ERROR,
    # Pydantic v2 validation: usually config/operator-side.
    "ValidationError": Classification.CONFIG_ISSUE,
}


# Substrings in the message body that, in the absence of a parsed exception
# type, hint at a particular classification. Lower-priority than the
# exception map.
_MESSAGE_HINTS: list[tuple[str, Classification]] = [
    ("Uncaught exception", Classification.BUG_CANDIDATE),
    ("subscription failed", Classification.HARDWARE_ISSUE),
    ("device not responding", Classification.HARDWARE_ISSUE),
    ("scan aborted", Classification.OPERATOR_ERROR),
    ("config", Classification.CONFIG_ISSUE),
]


def classify(
    entry: LogEntry,
    exception_type: Optional[str] = None,
) -> Classification:
    """Return the `Classification` for a `LogEntry`.

    Parameters
    ----------
    entry : LogEntry
        Parsed log record.
    exception_type : str, optional
        Exception class name extracted from the traceback. When provided and
        present in :data:`CLASSIFICATION_MAP`, it determines the classification.

    Returns
    -------
    Classification
        The triage category.
    """
    if exception_type and exception_type in CLASSIFICATION_MAP:
        return CLASSIFICATION_MAP[exception_type]

    # Try to peel an exception type out of the message itself
    # (e.g., "Failed to ...: ConnectionRefusedError").
    for known in CLASSIFICATION_MAP:
        if known in entry.message:
            return CLASSIFICATION_MAP[known]

    msg_lower = entry.message.lower()
    for hint, klass in _MESSAGE_HINTS:
        if hint.lower() in msg_lower:
            return klass

    return Classification.UNKNOWN
