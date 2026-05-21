"""Fingerprinting: turn a `LogEntry` into a stable, dedup-friendly hash.

The goal is that two errors with cosmetic-only differences (different scan
numbers, timestamps, file paths in the message body, hex IDs, etc.) hash
to the same value, so Stage 2 issue filing can dedup against existing
GitHub issues.

Algorithm
---------
1. If the record carries a traceback:
   - Extract the first traceback line that names a `geecs_*` source file
     (the "top user frame"). If none found, fall back to the last frame.
   - Extract the exception line (last non-empty line; format:
     ``ExceptionType: message``).
   - Signature: ``"{ExceptionType}@{filename}:{function}"``.
2. Else: signature is ``"{logger_name}|{level}|{normalized_message}"``.
3. Hash: ``sha1(signature.encode()).hexdigest()[:12]``.

Normalization replaces:
- digit runs    -> ``N``
- absolute paths -> their basename
- 8+ char hex / uuid-like tokens -> ``X``
- trailing whitespace and punctuation are stripped.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional

from geecs_data_utils import LogEntry  # re-exported by data-utils
from geecs_log_triage.schemas import (
    Classification,
    ErrorFingerprint,
)

# --------------------------------------------------------------------------
# Normalization helpers
# --------------------------------------------------------------------------

_PATH_RE = re.compile(r"(?:[A-Za-z]:)?[\\/](?:[^\s'\"<>]+[\\/])+([^\s'\"<>]+)")
_HEX_RE = re.compile(r"\b(?:0x)?[0-9a-fA-F]{8,}\b")
_DIGIT_RE = re.compile(r"\d+")
_TRAILING_PUNCT_RE = re.compile(r"[\s\.\,\:\;\!\?\)\]\}]+$")


def normalize_message(msg: str) -> str:
    """Normalize a log message so cosmetic variation does not change hash.

    Parameters
    ----------
    msg : str
        Raw log message.

    Returns
    -------
    str
        Normalized message suitable for use in a fingerprint signature.
    """
    s = _PATH_RE.sub(lambda m: m.group(1), msg)
    s = _HEX_RE.sub("X", s)
    s = _DIGIT_RE.sub("N", s)
    s = _TRAILING_PUNCT_RE.sub("", s)
    return s.strip()


# --------------------------------------------------------------------------
# Traceback parsing
# --------------------------------------------------------------------------

_TB_FRAME_RE = re.compile(
    r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)\s*$'
)
_TB_EXCEPTION_RE = re.compile(
    r"^(?P<type>[A-Za-z_][A-Za-z0-9_\.]*(?:Error|Exception|Warning)):\s*(?P<msg>.*)$"
)


def _basename(path: str) -> str:
    """Return the file basename (POSIX or Windows separator agnostic)."""
    for sep in ("/", "\\"):
        if sep in path:
            path = path.rsplit(sep, 1)[1]
    return path


def _extract_traceback_signature(
    tb: str,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (exception_type, top_user_file, top_user_function) from a traceback.

    "Top user frame" prefers frames whose file path contains ``geecs``.
    Falls back to the last frame in the traceback.
    """
    frames: list[tuple[str, str]] = []
    exception_type: Optional[str] = None
    for line in tb.splitlines():
        m = _TB_FRAME_RE.match(line)
        if m:
            frames.append((m.group("file"), m.group("func")))
            continue
        m = _TB_EXCEPTION_RE.match(line)
        if m:
            exception_type = m.group("type")

    if not frames:
        return exception_type, None, None

    user_frame = None
    for f in frames:
        if "geecs" in f[0].lower():
            user_frame = f
    if user_frame is None:
        user_frame = frames[-1]

    return exception_type, _basename(user_frame[0]), user_frame[1]


# --------------------------------------------------------------------------
# Public fingerprint API
# --------------------------------------------------------------------------


def _hash_signature(signature: str) -> str:
    """Return a 12-char sha1 hex digest of the signature string."""
    return hashlib.sha1(signature.encode("utf-8"), usedforsecurity=False).hexdigest()[
        :12
    ]


def fingerprint_for(
    entry: LogEntry,
    classification: Classification = Classification.UNKNOWN,
) -> ErrorFingerprint:
    """Compute an `ErrorFingerprint` for a parsed `LogEntry`.

    Parameters
    ----------
    entry : LogEntry
        The parsed log record.
    classification : Classification, optional
        Pre-computed classification. Defaults to `UNKNOWN`; callers
        normally pass the value returned by :func:`geecs_log_triage.classify`.

    Returns
    -------
    ErrorFingerprint
        Stable fingerprint with hash, signature, normalized message, and
        (when available) extracted exception type and sample traceback.
    """
    normalized = normalize_message(entry.message)
    exc_type: Optional[str] = None
    sample_tb: Optional[str] = None

    if entry.traceback:
        exc_type, top_file, top_func = _extract_traceback_signature(entry.traceback)
        if top_file is not None and exc_type is not None:
            signature = f"{exc_type}@{top_file}:{top_func}"
        elif exc_type is not None:
            signature = f"{exc_type}|{normalized}"
        else:
            signature = f"{entry.logger_name}|{entry.level.value}|{normalized}"
        sample_tb = (
            entry.traceback
            if len(entry.traceback) <= 4000
            else entry.traceback[:4000] + "\n...<truncated>..."
        )
    else:
        signature = f"{entry.logger_name}|{entry.level.value}|{normalized}"

    return ErrorFingerprint(
        fingerprint_hash=_hash_signature(signature),
        signature=signature,
        exception_type=exc_type,
        normalized_message=normalized,
        sample_traceback=sample_tb,
        classification=classification,
    )
