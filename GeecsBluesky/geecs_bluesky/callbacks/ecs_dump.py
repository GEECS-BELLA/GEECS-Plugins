r"""The ECS live dump: every settable's value, read once at a run's start.

LabVIEW Master Control wrote one per scan, and the readers in
``geecs_data_utils`` (``ScanPaths.get_ecs_dump_file``,
``ScanData.load_ecs_live_dump``) still look for it:
``<day>/ECS Live dumps/Scan<N>.txt`` beside ``scans/``, ``N`` the claimed
scan number unpadded.  The format is Master Control's, byte for byte where
the data allows — a LabVIEW config file, CRLF line endings, no newline
after the last line::

    [Experiment]
    Expt Name = "<experiment>"
    No of Devices = "<count>"

    [Device 1]
    Device Name = "<device>"
    shot # = "0"
    <variable> = "<value>"
    ...

Devices are every device of the experiment's GEECS DB roster, ordered as
the DB collates names (case-insensitively); each lists the settable
variables the CA gateway serves, in the roster's order.  A value is
written as text: a numeric readback as ``%.6f`` (LabVIEW's ``%f``, the
form most Master Control values take — the gateway serves numerics as
doubles, so a device's own digits are not recoverable), an enum as its
choice string, a string or path verbatim with LabVIEW's escapes (``\\``
for a backslash).  A variable that cannot be read — no answer within the
budget, or the gateway's INVALID mark on a down device's stale readback —
is left out, as Master Control leaves out a variable it holds no value
for; a device with none read keeps its section with its name alone.

The reads are one concurrent batch on the shared one-shot loop
(:func:`~geecs_bluesky.devices.ca.oneshot.try_caget_many`), bounded by
one budget, at the start document — before the plan's first move, since
the RunEngine dispatches the start document before it goes on.  Every
failure is logged and never raised: a dump that cannot be written never
costs the scan.  An existing file is never overwritten (Master Control,
where it still runs, writes the same name for the scan it expects next).

Scanner side only: the dump's folder is created here when missing (one
level, never with ``parents``) beside the day's ``scans/``; a scan folder
is never created.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from geecs_core.pv_naming import pv_name

from geecs_bluesky.callbacks._base import _RunCallback, _scan_folder

if TYPE_CHECKING:  # pragma: no cover - typing only
    from geecs_bluesky.namespace import DeviceRoster

logger = logging.getLogger(__name__)

#: The dump folder's name under the day folder (Master Control's).
ECS_DUMP_FOLDER = "ECS Live dumps"

#: Budget (seconds) for the whole batch of reads: concurrent, so every
#: dead PV costs this once, not once each.
DEFAULT_READ_TIMEOUT_S = 2.0

#: EPICS alarm severity INVALID: the gateway's mark on a down device's
#: stale readbacks.
INVALID_SEVERITY = 3

#: What the dump covers: ``(device, [settable variable, ...])`` per device,
#: in file order.
EcsTargets = Sequence[tuple[str, Sequence[str]]]

#: Reads the readings of ``[(device, variable), ...]``, positionally, each
#: ``None`` when unread: ``reader(experiment, pairs, timeout=...)``.
EcsReader = Callable[..., list[Any]]


def ecs_dump_targets(roster: DeviceRoster) -> list[tuple[str, list[str]]]:
    """Every roster device and its served settable variables, in dump order.

    Devices ordered case-insensitively (the DB's collation, which Master
    Control's dumps follow); variables in the roster's row order, each
    spelling once, only those the gateway serves (a non-scalar settable has
    no PV).
    """
    targets: list[tuple[str, list[str]]] = []
    for device in sorted(roster.variables, key=lambda d: (d.upper(), d)):
        served = roster.served_for(device)
        names: list[str] = []
        for row in roster.variables[device]:
            name = str(row.get("name", ""))
            if (
                name
                and row.get("settable")
                and name.lower() in served
                and name not in names
            ):
                names.append(name)
        targets.append((device, names))
    return targets


def read_settables(
    experiment: str | None,
    pairs: Sequence[tuple[str, str]],
    *,
    timeout: float = DEFAULT_READ_TIMEOUT_S,
) -> list[Any]:
    """One concurrent ``FORMAT_CTRL`` read of each pair's gateway readback.

    Positional with *pairs*; an unread PV is ``None`` (fail-open, the whole
    batch when ``aioca`` is missing).
    """
    from geecs_bluesky.devices.ca.oneshot import try_caget_many

    pvs = [pv_name(experiment, device, variable) for device, variable in pairs]
    return try_caget_many(pvs, timeout=timeout, ctrl=True)


def warm_up(experiment: str | None, targets: EcsTargets, *, timeout: float) -> int:
    """Connect the dump's channels once, at environment open; the count that answered.

    The one-shot loop keeps every channel it creates, so the first run's
    batch finds them connected and its budget covers the reads alone (a
    cold batch of a few thousand channels can spend it on searches).
    Bounded by *timeout*; a channel that does not answer is simply read
    again at the next run.
    """
    pairs = [(device, name) for device, names in targets for name in names]
    if not pairs:
        return 0
    answered = sum(
        r is not None for r in read_settables(experiment, pairs, timeout=timeout)
    )
    logger.info(
        "ECS live dump: %d of %d settables answered at warm-up", answered, len(pairs)
    )
    return answered


def ecs_value(reading: Any) -> str | None:
    """A reading as the dump's value text; ``None`` when it is not a value.

    ``None`` for an unread PV or an INVALID one.  An enum (its ``enums``
    present) is its choice string; a char array (a long-string path PV) is
    its text up to the first NUL; a string is itself; an integer is its
    digits; a float is ``%.6f`` (``NaN``, ``Inf`` and ``-Inf`` as LabVIEW
    writes them).
    """
    if reading is None:
        return None
    if getattr(reading, "severity", 0) >= INVALID_SEVERITY:
        return None
    enums = getattr(reading, "enums", None)
    if enums is not None:
        index = int(reading)
        return str(enums[index]) if 0 <= index < len(enums) else None
    if isinstance(reading, str):
        return str(reading)
    dtype = getattr(reading, "dtype", None)
    if dtype is not None and getattr(reading, "ndim", 0) == 1:
        return _char_array_text(reading)
    return _number_text(reading)


def _char_array_text(array: Any) -> str | None:
    """A long-string PV's char array as its text up to the first NUL; else ``None``."""
    if array.dtype.kind not in "iu" or array.dtype.itemsize != 1:
        return None
    raw = bytes(array.astype("uint8"))
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def _number_text(reading: Any) -> str | None:
    """An integer's digits, a float's ``%.6f`` (LabVIEW's ``NaN``/``Inf``); else ``None``."""
    dtype = getattr(reading, "dtype", None)
    if isinstance(reading, (bool, int)) or (dtype is not None and dtype.kind in "biu"):
        return str(int(reading))
    try:
        number = float(reading)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "Inf" if number > 0 else "-Inf"
    return f"{number:.6f}"


def escape_value(text: str) -> str:
    r"""Escape *text* as LabVIEW's config files do: ``\\``, ``\"``, ``\r``, ``\n``, ``\t``, else ``\XX``."""
    out: list[str] = []
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\{ord(ch):02X}")
        else:
            out.append(ch)
    return "".join(out)


def render_ecs_dump(
    experiment: str,
    sections: Sequence[tuple[str, Sequence[tuple[str, str]]]],
) -> str:
    """The dump's text: ``sections`` is ``(device, [(variable, value), ...])`` in order.

    Lines joined by CRLF, no line ending after the last, as Master Control
    writes it.
    """
    lines = [
        "[Experiment]",
        f'Expt Name = "{escape_value(experiment)}"',
        f'No of Devices = "{len(sections)}"',
    ]
    for index, (device, values) in enumerate(sections, start=1):
        lines += [
            "",
            f"[Device {index}]",
            f'Device Name = "{escape_value(device)}"',
            'shot # = "0"',
        ]
        lines += [f'{name} = "{escape_value(value)}"' for name, value in values]
    return "\r\n".join(lines)


def ecs_dump_path(scan_folder: Path, scan_number: int) -> Path:
    """``<day>/ECS Live dumps/Scan<N>.txt`` for ``<day>/scans/ScanNNN``."""
    return scan_folder.parent.parent / ECS_DUMP_FOLDER / f"Scan{int(scan_number)}.txt"


def write_ecs_dump(
    scan_folder: Path,
    scan_number: int,
    experiment: str,
    targets: EcsTargets,
    *,
    reader: EcsReader = read_settables,
    timeout: float = DEFAULT_READ_TIMEOUT_S,
) -> Path | None:
    """Read every target once and write the run's dump; the path, or ``None`` when not written.

    Not written: the scan folder is not ``<day>/scans/ScanNNN`` (the dump's
    place is unknown), or the dump already exists (kept, never replaced).

    Raises
    ------
    OSError
        From creating the folder or writing the file (the caller logs it).
    """
    if scan_folder.parent.name != "scans":
        logger.warning(
            "ECS live dump: %s is not under a day's scans/ folder; no dump written",
            scan_folder,
        )
        return None
    path = ecs_dump_path(scan_folder, scan_number)
    if path.exists():
        logger.warning(
            "ECS live dump: %s already exists (Master Control's?); kept, not replaced",
            path,
        )
        return None
    pairs = [(device, name) for device, names in targets for name in names]
    readings = reader(experiment, pairs, timeout=timeout) if pairs else []
    if len(readings) != len(pairs):
        logger.warning(
            "ECS live dump: %d readings for %d variables; none used",
            len(readings),
            len(pairs),
        )
        readings = [None] * len(pairs)
    values = iter(ecs_value(reading) for reading in readings)
    sections: list[tuple[str, list[tuple[str, str]]]] = []
    unread: dict[str, int] = {}
    for device, names in targets:
        read: list[tuple[str, str]] = []
        for name in names:
            value = next(values)
            if value is None:
                unread[device] = unread.get(device, 0) + 1
            else:
                read.append((name, value))
        sections.append((device, read))
    text = render_ecs_dump(experiment, sections)
    path.parent.mkdir(exist_ok=True)
    try:
        with open(path, "x", encoding="cp1252", errors="replace", newline="") as f:
            f.write(text)
    except FileExistsError:
        logger.warning(
            "ECS live dump: %s appeared while reading; kept, not replaced", path
        )
        return None
    missing = sum(unread.values())
    logger.info(
        "ECS live dump: wrote %s (%d devices, %d of %d settables read)",
        path,
        len(sections),
        len(pairs) - missing,
        len(pairs),
    )
    if unread:
        logger.info(
            "ECS live dump: unread (left out): %s",
            ", ".join(f"{device} ({count})" for device, count in unread.items()),
        )
    return path


class EcsDumpCallback(_RunCallback):
    """Write the run's ECS live dump at its start document; best-effort.

    Parameters
    ----------
    targets :
        ``(device, [settable variable, ...])`` per device, in file order
        (:func:`ecs_dump_targets` over the namespace's roster).
    reader :
        The batch read (:func:`read_settables`; a fake in tests).
    timeout :
        The batch's read budget, seconds.
    """

    def __init__(
        self,
        targets: EcsTargets,
        *,
        reader: EcsReader = read_settables,
        timeout: float = DEFAULT_READ_TIMEOUT_S,
    ) -> None:
        super().__init__()
        self._targets = [(device, list(names)) for device, names in targets]
        self._reader = reader
        self._timeout = timeout
        self.last_path: Path | None = None

    def on_start(self, start: dict[str, Any]) -> None:
        """Read and write the dump; the base class logs any failure."""
        self.last_path = None
        folder = _scan_folder(start)
        number = start.get("scan_number")
        if folder is None or number is None:
            return
        experiment = str(start.get("experiment") or "")
        self.last_path = write_ecs_dump(
            folder,
            int(number),
            experiment,
            self._targets,
            reader=self._reader,
            timeout=self._timeout,
        )
