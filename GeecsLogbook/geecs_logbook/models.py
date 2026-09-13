"""Derived views of a scan folder.

Nothing in this module is stored by the logbook. Every field is read from
files the scanner already wrote — ``ScanInfoScanNNN.ini`` and the folder
listing — so a ``ScanSummary`` can always be rebuilt from the share and can
never drift from the data it describes.

Human commentary is a separate concern and does not appear here.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

ScanStatus = Literal["success", "failed", "aborted", "incomplete", "unknown"]

#: How a scan status maps onto the kit's shared status vocabulary
#: (:data:`geecs_web_theme.STATES`), which is what drives the chip colour.
#:
#: The mapping is deliberately NOT the identity on two names, and the
#: reason is in "Status is reported, not inferred" in this package's
#: CLAUDE.md. ``incomplete`` — an empty ``ScanEndInfo`` — is the most
#: common state on the real share, so painting it amber painted most of a
#: day amber; it is an absence of information, which is what the kit's
#: ``unknown`` means. Our ``unknown`` is the opposite case: something WAS
#: written and we cannot read it, which earns the amber.
#:
#: Nothing is lost where two statuses do share a colour, because the chip
#: keeps its own *word* — the reader still sees "aborted" or "not
#: finalised" written on it. Colour carries severity; text carries which.
#:
#: ``tests/test_models.py`` pins every value to a real kit state and every
#: :data:`ScanStatus` to an entry here, so adding a status without deciding
#: its severity fails rather than rendering an uncoloured chip.
KIT_STATE: dict[str, str] = {
    "success": "ok",
    "failed": "failed",
    "aborted": "degraded",
    # NOT degraded. `incomplete` means ScanEndInfo is still empty — the most
    # common state on the real share (37 of 49 across four sampled days), and
    # this file's "Status is reported, not inferred" section records that
    # painting it amber "painted most of a day amber". It is an absence of
    # information, which is what the kit's `unknown` means.
    "incomplete": "unknown",
    # ...and this one IS the suspicious case: a non-empty ScanEndInfo we do
    # not recognise. Something was written and we cannot read it, which is
    # worth a colour. This pair is deliberately not the identity mapping.
    "unknown": "degraded",
}


class ScanSummary(BaseModel):
    """One scan, as its folder describes it.

    Attributes
    ----------
    number : int
        The scan number, i.e. ``NNN`` in ``ScanNNN``.
    started : datetime or None
        When the scan ran, from the first record in its ``scan.log``.
        Falls back to the ``ScanInfo`` file's own modification time for
        archive scans that predate ``scan.log`` — approximate, and flagged
        as such by ``started_approximate``. ``None`` when neither exists.
        Never the scan *folder's* mtime: any later pass that writes into
        the folder moves it, measured over an hour off the real start.
    started_approximate : bool
        Whether ``started`` is the fallback rather than the log's own
        timestamp, so the view can mark it rather than implying precision
        it does not have.
    parameter : str or None
        ``Scan Parameter`` — the scanned variable, or ``"Shotnumber"`` for
        a no-scan acquisition.
    start, end, step_size : float or None
        The scan range, as written by the scanner.
    shots_per_step : int or None
        ``Shots per step``.
    mode : str or None
        ``ScanMode`` — typically ``"standard"`` or ``"noscan"``.
    plan, scanner : str or None
        ``Plan`` and ``Scanner``, e.g. ``"scan"`` / ``"bluesky"``.
    trigger_profile : str or None
        ``Trigger profile``.
    background : bool or None
        ``Background``.
    purpose : str or None
        ``ScanStartInfo`` — the operator's stated reason for the scan,
        captured at submission. This is the one human-authored field the
        scanner already records, so the logbook displays it rather than
        asking for it again.
    status : ScanStatus
        Derived from ``ScanEndInfo``; see :func:`~.scan_reader.scan_status`.
    failure_reason : str or None
        The text of ``ScanEndInfo`` with its ``fail:`` prefix stripped,
        when the scan failed. ``None`` otherwise.
    devices : list of str
        Per-device subdirectory names found in the scan folder.
    has_scan_info : bool
        Whether ``ScanInfoScanNNN.ini`` exists. False is a normal state for
        a scan that is still running or that ended before writing it.
    """

    number: int
    started: Optional[datetime] = None
    started_approximate: bool = False
    parameter: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None
    step_size: Optional[float] = None
    shots_per_step: Optional[int] = None
    mode: Optional[str] = None
    plan: Optional[str] = None
    scanner: Optional[str] = None
    trigger_profile: Optional[str] = None
    background: Optional[bool] = None
    purpose: Optional[str] = None
    status: ScanStatus = "unknown"
    failure_reason: Optional[str] = None
    devices: list[str] = Field(default_factory=list)
    has_scan_info: bool = False

    @property
    def label(self) -> str:
        """Return the folder-style identifier, e.g. ``"Scan005"``."""
        return f"Scan{self.number:03d}"


class DaySummary(BaseModel):
    """Every scan folder present for one date.

    Attributes
    ----------
    day : date
        The date this summary covers.
    experiment : str
        The experiment whose share was read.
    folder : str
        The ``scans/`` directory the summary was read from, for display and
        for the "derived — not stored here" provenance line.
    exists : bool
        Whether that directory exists. A date with no scans is a normal,
        empty day rather than an error.
    scans : list of ScanSummary
        Present scans, ordered by scan number.
    """

    day: date
    experiment: str
    folder: str
    exists: bool = False
    scans: list[ScanSummary] = Field(default_factory=list)

    @property
    def failed(self) -> int:
        """Return how many scans on this day ended in a failure."""
        return sum(1 for s in self.scans if s.status == "failed")
