"""The scan-folder facts, path-free: one home for what a scan folder says.

Every consumer of the share — the data portal's folder catalog, the scan
logbook, ScanAnalysis, GEECS-LogTriage — needs the same handful of facts
about a ``scans/ScanNNN`` folder, and before this module each wrote its
own, and they disagreed (#1061 rows 3.6/3.7/3.8):

- :func:`parse_scan_folder_name` / :func:`scan_folder_name` — the
  ``ScanNNN`` name.  **Parse leniently, write padded**: ``Scan7`` and
  ``Scan0007`` both parse to 7, while every folder this repo builds is
  ``Scan{n:03d}`` — a reader that starts from a real folder keeps its real
  name rather than re-deriving one from the number.
- :func:`analysis_folder_for` — the sibling ``{day}/analysis/ScanNNN`` of a
  ``{day}/scans/ScanNNN`` folder: pure path arithmetic, nothing created.
- :func:`read_scan_info_file` — the raw ``[Scan Info]`` parse;
  :class:`ScanInfoRecord` — the same facts typed (finite floats only);
  :func:`is_noscan` — "no variable was stepped".
- :func:`classify_end_info` / :func:`end_info_reason` — the scan's outcome
  from ``ScanEndInfo`` plus the run-closed evidence.

Pure and read-only by construction (the repo's scan-folder invariant):
nothing here creates, writes or deletes a path.
"""

from __future__ import annotations

import logging
import re
from configparser import ConfigParser, Error as ConfigParserError, NoSectionError
from pathlib import Path
from typing import Literal, Mapping, Optional, Union

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

#: A scan folder's name.  Lenient on padding (``Scan7`` is scan 7); a
#: ``ScanData…`` file or a ``Scan007_old`` folder is not a scan folder.
SCAN_FOLDER_RE = re.compile(r"^Scan(?P<number>\d+)$")

#: ``Scan Parameter`` values that mean "no variable was stepped"
#: (compared stripped and case-insensitively).
NOSCAN_PARAMETERS = frozenset({"", "shotnumber", "noscan", "none"})

#: A scan's outcome, as its folder reports it.
ScanEndStatus = Literal["success", "failed", "aborted", "incomplete", "unknown"]


def parse_scan_folder_name(name: str) -> Optional[int]:
    """Return the scan number a ``ScanNNN`` folder name carries, else ``None``.

    Parameters
    ----------
    name : str
        A directory name (not a path).

    Returns
    -------
    int or None
        The number (``"Scan042"`` → 42, ``"Scan7"`` → 7), or ``None`` when
        *name* is not a scan folder's name.
    """
    match = SCAN_FOLDER_RE.match(name)
    return int(match.group("number")) if match else None


def scan_folder_name(number: int) -> str:
    """The canonical folder name for scan *number*: ``Scan{number:03d}``."""
    return f"Scan{number:03d}"


def analysis_folder_for(scan_folder: Path, *, validate: bool = False) -> Path:
    """``{day}/analysis/ScanNNN`` for a ``{day}/scans/ScanNNN`` folder.

    Pure path construction — nothing is created, here or by the caller's
    behalf; whoever writes into the analysis folder creates it at write
    time.

    Parameters
    ----------
    scan_folder : Path
        A ``scans/ScanNNN`` folder path.
    validate : bool, optional
        Also check the disk: the scan folder must exist, and the analysis
        folder must not resolve (through a symlink) into the raw ``scans/``
        tree.  For writers, which must never put outputs beside raw data.

    Returns
    -------
    Path
        The sibling analysis folder (same ``ScanNNN`` name as the input).

    Raises
    ------
    ValueError
        When *scan_folder* is not shaped ``.../scans/ScanNNN`` (always
        checked), or, with *validate*, when the target resolves into the
        raw tree.
    FileNotFoundError
        With *validate*, when the scan folder does not exist.
    """
    scan_folder = Path(scan_folder)
    if (
        parse_scan_folder_name(scan_folder.name) is None
        or scan_folder.parent.name != "scans"
    ):
        raise ValueError(f"{scan_folder} is not a scans/ScanNNN folder")
    target = scan_folder.parent.parent / "analysis" / scan_folder.name
    if validate:
        if not scan_folder.is_dir():
            raise FileNotFoundError(scan_folder)
        if target.resolve().is_relative_to(scan_folder.parent.resolve()):
            raise ValueError("Analysis outputs cannot point into the raw scans tree")
    return target


def read_scan_info_file(ini_path: Union[Path, str]) -> dict[str, str]:
    """Parse a ``ScanInfoScanNNN.ini`` file into its ``[Scan Info]`` mapping.

    The one implementation of this parse: :meth:`ScanPaths.load_scan_info`,
    the folder catalog and the scan logbook all call it.  Keys keep their
    case (``"Scan Parameter"``); values lose their surrounding quotes.

    Parameters
    ----------
    ini_path : Path or str
        The ScanInfo file. Need not exist.

    Returns
    -------
    dict of str to str
        The ``[Scan Info]`` section, or an empty dict when the file is
        missing, sectionless, or unreadable.  Never raises: one unparsable
        scan must not take down a caller iterating a whole day.
    """
    parser = ConfigParser()
    parser.optionxform = str  # keep key case
    try:
        parser.read(ini_path)
        return {key: value.strip("'\"") for key, value in parser.items("Scan Info")}
    except NoSectionError:
        logger.warning('ScanInfo file %s has no "Scan Info" section', ini_path)
    except (ConfigParserError, OSError, UnicodeDecodeError) as exc:
        logger.warning("unreadable ScanInfo file %s: %s", ini_path, exc)
    return {}


def finite_float(raw: Optional[str]) -> Optional[float]:
    """Parse a finite float from a ScanInfo or s-file value, else ``None``.

    ``inf``/``nan`` are refused: they are unusable downstream and escape
    ``int()`` as ``OverflowError``/``ValueError`` — a whole day must not
    fail over one malformed field.
    """
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if -1e308 < value < 1e308 else None


def _as_int(raw: Optional[str]) -> Optional[int]:
    value = finite_float(raw)
    return None if value is None else int(value)


def _as_bool(raw: Optional[str]) -> Optional[bool]:
    if raw is None:
        return None
    text = raw.strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def _as_text(raw: Optional[str]) -> Optional[str]:
    text = (raw or "").strip()
    return text or None


def is_noscan(parameter: Optional[str]) -> bool:
    """Whether a ``Scan Parameter`` value means no variable was stepped.

    ``None``, empty, ``Shotnumber``, ``noscan`` and ``none`` (any case,
    surrounding whitespace ignored) — see :data:`NOSCAN_PARAMETERS`.
    """
    return (parameter or "").strip().lower() in NOSCAN_PARAMETERS


class ScanInfoRecord(BaseModel):
    """A ``ScanInfoScanNNN.ini``, typed.

    Built with :meth:`from_ini` from :func:`read_scan_info_file`'s mapping.
    Every field is optional — an archive scan or a half-written ini lacks
    some — and an unusable value (unparsable, ``inf``, ``nan``) reads as
    ``None`` rather than raising.

    Attributes
    ----------
    parameter : str or None
        ``Scan Parameter`` — the scanned variable (see :func:`is_noscan`).
    start, end, step_size : float or None
        The scan range.
    shots_per_step : int or None
        ``Shots per step``.
    mode, plan, scanner, trigger_profile : str or None
        ``ScanMode``, ``Plan``, ``Scanner``, ``Trigger profile``.
    background : bool or None
        ``Background``.
    purpose : str or None
        ``ScanStartInfo`` — the operator's stated reason for the scan.
    end_info : str or None
        ``ScanEndInfo`` verbatim (see :func:`classify_end_info`); ``None``
        when the key is absent.
    """

    model_config = ConfigDict(frozen=True)

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
    end_info: Optional[str] = None

    @classmethod
    def from_ini(cls, info: Mapping[str, str]) -> "ScanInfoRecord":
        """Type a :func:`read_scan_info_file` mapping."""
        return cls(
            parameter=_as_text(info.get("Scan Parameter")),
            start=finite_float(info.get("Start")),
            end=finite_float(info.get("End")),
            step_size=finite_float(info.get("Step size")),
            shots_per_step=_as_int(info.get("Shots per step")),
            mode=_as_text(info.get("ScanMode")),
            plan=_as_text(info.get("Plan")),
            scanner=_as_text(info.get("Scanner")),
            trigger_profile=_as_text(info.get("Trigger profile")),
            background=_as_bool(info.get("Background")),
            purpose=_as_text(info.get("ScanStartInfo")),
            end_info=info.get("ScanEndInfo"),
        )

    @property
    def noscan(self) -> bool:
        """Whether no variable was stepped (:func:`is_noscan`)."""
        return is_noscan(self.parameter)


def classify_end_info(end_info: Optional[str], *, closed: bool) -> ScanEndStatus:
    """A scan's outcome from its ``ScanEndInfo`` and the run-closed evidence.

    ``ScanEndInfo`` decides whenever it says something: ``success``
    (exactly, any case), ``fail…`` → ``failed``, ``abort…`` → ``aborted``,
    any other text → ``unknown`` (never promoted to success).  When it is
    empty or absent (no ScanInfo at all), the files decide: the native
    scanner writes it empty when it claims the folder and fills it at the
    stop document, while LabVIEW Master Control leaves it empty on every
    scan — so an empty value is ``success`` once the run closed (*closed*:
    ``ScanDataScanNNN.txt`` or the analysis s-file exists, see
    :func:`~geecs_data_utils.data.sfile.run_closed_evidence`) and
    ``incomplete`` before.

    Parameters
    ----------
    end_info : str or None
        The raw ``ScanEndInfo``; ``None`` when there is no ScanInfo.
    closed : bool
        Whether a file that exists only once the run closed was found.

    Returns
    -------
    ScanEndStatus
        One of ``success``, ``failed``, ``aborted``, ``incomplete``,
        ``unknown``.
    """
    text = (end_info or "").strip().lower()
    if text == "success":
        return "success"
    if text.startswith("fail"):
        return "failed"
    if text.startswith("abort"):
        return "aborted"
    if text:
        return "unknown"
    return "success" if closed else "incomplete"


def end_info_reason(end_info: Optional[str]) -> Optional[str]:
    """The text of a ``fail…``/``abort…`` ``ScanEndInfo`` without its ``verdict:`` prefix.

    ``"fail: TimeoutError(...)"`` → ``"TimeoutError(...)"``; a bare
    ``"abort"`` stays ``"abort"``; ``None`` for any other outcome.
    """
    if classify_end_info(end_info, closed=False) not in ("failed", "aborted"):
        return None
    text = (end_info or "").strip()
    return re.sub(r"^(fail|abort)[a-z]*:\s*", "", text, flags=re.IGNORECASE) or None
