"""Thin re-export of the scan-log parser.

The actual parsing implementation lives in
:mod:`geecs_data_utils.scan_log_loader` so it's reusable by any consumer
(notebooks, plotting helpers, etc.) without depending on the triage layer.

This module is kept as a stable import path for log-triage code; callers
can still write ``from geecs_log_triage import parse_scan_log``.
"""

from __future__ import annotations

from geecs_data_utils.scan_log_loader import (
    HEADER_RE,
    parse_lines,
    parse_scan_log,
)

__all__ = ["HEADER_RE", "parse_lines", "parse_scan_log"]
