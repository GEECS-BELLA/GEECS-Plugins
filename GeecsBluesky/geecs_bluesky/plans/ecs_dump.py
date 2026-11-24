r"""The ECS live dump: Master Control's per-scan file of every settable, read at a run's start.

``<day>/ECS Live dumps/Scan<N>.txt`` beside ``scans/`` (``N`` the claimed
number, unpadded), which ``geecs_data_utils`` reads (``ScanPaths.get_ecs_dump_file``,
``parse_ecs_dump``), in Master Control's format: a LabVIEW config file, CRLF,
no final line ending, an ``[Experiment]`` header, one ``[Device i]`` section
per roster device with ``Device Name`` and ``shot # = "0"``.

:func:`ecs_dump_preprocessor` inserts the dump right after each claimed
``open_run``: one concurrent ``FORMAT_CTRL`` batch over every served
settable readback, awaited on the RunEngine's loop (never blocking it) and
bounded, so the values precede the plan's first move.  Values the batch
cannot read, or reads INVALID, are left out, as Master Control leaves out a
variable it holds none for.  Nothing here fails or stalls a run beyond the
bound.  The dump replaces a ``Scan<N>.txt`` Master Control wrote earlier (it
writes the *next* scan's name when a scan ends, so its file is stale), moved
aside as ``Scan<N>.master_control.txt``, never deleted.

Scanner side only: ``ECS Live dumps/`` is created (one level) in the day the
claim made; a scan folder is never created.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import tempfile
from collections.abc import Awaitable, Callable, Generator, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import bluesky.plan_stubs as bps
from bluesky.preprocessors import plan_mutator
from bluesky.utils import Msg

from geecs_core.pv_naming import pv_name

if TYPE_CHECKING:  # pragma: no cover - typing only
    from geecs_bluesky.namespace import DeviceRoster

logger = logging.getLogger(__name__)

#: The dump folder's name under the day folder (Master Control's).
ECS_DUMP_FOLDER = "ECS Live dumps"

#: Budget (seconds) for the whole batch of reads: concurrent, so every
#: dead PV costs this once, not once each.
DEFAULT_READ_TIMEOUT_S = 2.0

#: Bound (seconds) on the file write, a share that hangs included.
WRITE_TIMEOUT_S = 10.0

#: Grace (seconds) past the read budget before the whole batch is abandoned.
_BATCH_GRACE_S = 1.0

#: EPICS alarm severity INVALID: the gateway's mark on a down device's
#: stale readbacks.
INVALID_SEVERITY = 3

#: What the dump covers: ``(device, [settable variable, ...])`` per device,
#: in file order.
EcsTargets = Sequence[tuple[str, Sequence[str]]]

#: The batch read: ``await reader(experiment, pairs, timeout=...)`` → the
#: readings of ``[(device, variable), ...]``, positionally, ``None`` unread.
EcsReader = Callable[..., Awaitable[list[Any]]]


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


async def read_settables(
    experiment: str | None,
    pairs: Sequence[tuple[str, str]],
    *,
    timeout: float = DEFAULT_READ_TIMEOUT_S,
) -> list[Any]:
    """One concurrent ``FORMAT_CTRL`` read of each pair's gateway readback, on the running loop.

    ``FORMAT_CTRL`` carries each value's alarm ``severity`` and an enum's
    ``enums`` (its choice strings) in one read.  Positional with *pairs*;
    an unread PV is ``None`` — every one when ``aioca`` is missing or the
    batch outlives its budget.
    """
    if not pairs:
        return []
    try:
        from aioca import FORMAT_CTRL, caget
    except Exception:  # noqa: BLE001 — the dump is fail-open
        return [None] * len(pairs)

    async def _one(pv: str) -> Any:
        try:
            return await caget(pv, format=FORMAT_CTRL, timeout=timeout)
        except Exception:  # noqa: BLE001 — an unread PV is left out
            return None

    pvs = [pv_name(experiment, device, variable) for device, variable in pairs]
    try:
        return list(
            await asyncio.wait_for(
                asyncio.gather(*(_one(pv) for pv in pvs)), timeout + _BATCH_GRACE_S
            )
        )
    except Exception:  # noqa: BLE001 — a batch past its budget reads nothing
        return [None] * len(pairs)


def warm_up(run_engine: Any, experiment: str | None, targets: EcsTargets) -> None:
    """Start connecting the dump's channels on the RunEngine's loop; return at once.

    aioca keeps a channel per loop, so the first run's batch finds them
    connected and its budget covers the reads, not thousands of searches.
    Fire and forget: environment open never waits for it.
    """
    pairs = [(device, name) for device, names in targets for name in names]
    if not pairs:
        return

    async def _warm() -> None:
        readings = await read_settables(experiment, pairs, timeout=10.0)
        answered = sum(r is not None for r in readings)
        logger.info(
            "ECS live dump: %d of %d settables answered at warm-up",
            answered,
            len(pairs),
        )

    asyncio.run_coroutine_threadsafe(_warm(), run_engine.loop)


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


def dump_sections(
    targets: EcsTargets, readings: Sequence[Any]
) -> tuple[list[tuple[str, list[tuple[str, str]]]], dict[str, int]]:
    """The dump's sections from *readings* (positional over *targets*), and the unread count per device."""
    values = iter([ecs_value(r) for r in readings])
    sections: list[tuple[str, list[tuple[str, str]]]] = []
    unread: dict[str, int] = {}
    for device, names in targets:
        read: list[tuple[str, str]] = []
        for name in names:
            value = next(values, None)
            if value is None:
                unread[device] = unread.get(device, 0) + 1
            else:
                read.append((name, value))
        sections.append((device, read))
    return sections, unread


def write_ecs_dump(scan_folder: Path, scan_number: int, text: str) -> Path | None:
    """Write *text* as the scan's dump, atomically; the path, or ``None`` when it has no place.

    A scan folder not under a day's ``scans/`` has none.  An existing dump
    (Master Control's) is moved aside to ``Scan<N>.master_control.txt``,
    never deleted: an aside already there is never replaced.

    Raises
    ------
    OSError
        From creating the folder, writing, or renaming (the caller logs it).
    """
    if scan_folder.parent.name != "scans" or not scan_folder.is_dir():
        logger.warning(
            "ECS live dump: %s is not a scan folder under a day's scans/; "
            "no dump written",
            scan_folder,
        )
        return None
    path = ecs_dump_path(scan_folder, scan_number)
    path.parent.mkdir(exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="cp1252", errors="replace", newline="") as f:
            f.write(text)
        os.chmod(temporary, 0o644)
        aside = path.with_name(f"{path.stem}.master_control.txt")
        if path.exists() and not aside.exists():
            # Master Control's.  With an aside already there, the file at
            # the name is this scan's own earlier dump: replaced, and the
            # aside — Master Control's — never touched.
            os.replace(path, aside)
            logger.warning(
                "ECS live dump: %s existed (Master Control's, written before this "
                "scan started); moved aside to %s",
                path.name,
                aside.name,
            )
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


async def dump_run(
    scan_folder: Path,
    scan_number: int,
    experiment: str,
    targets: EcsTargets,
    *,
    reader: EcsReader = read_settables,
    timeout: float = DEFAULT_READ_TIMEOUT_S,
) -> Path | None:
    """Read every target once and write the dump; never raises, the write bounded."""
    try:
        pairs = [(device, name) for device, names in targets for name in names]
        readings = await reader(experiment, pairs, timeout=timeout)
        if len(readings) != len(pairs):
            logger.warning(
                "ECS live dump: %d readings for %d variables; none used",
                len(readings),
                len(pairs),
            )
            readings = [None] * len(pairs)
        sections, unread = dump_sections(targets, readings)
        text = render_ecs_dump(experiment, sections)
        path = await asyncio.wait_for(
            asyncio.to_thread(write_ecs_dump, scan_folder, scan_number, text),
            WRITE_TIMEOUT_S,
        )
    except Exception:  # noqa: BLE001 — the dump never fails a run
        logger.warning(
            "ECS live dump for scan %s not written", scan_number, exc_info=True
        )
        return None
    if path is None:
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


def ecs_dump_preprocessor(
    plan: Generator[Msg, Any, Any],
    *,
    targets: EcsTargets,
    reader: EcsReader = read_settables,
    timeout: float = DEFAULT_READ_TIMEOUT_S,
) -> Generator[Msg, Any, Any]:
    """Insert the run's dump right after every claimed ``open_run``.

    Installed outside the claim preprocessor (it reads the ``scan_folder``,
    ``scan_number`` and ``experiment`` the claim put on the message); an
    ``open_run`` without them passes untouched.  The dump is one
    ``wait_for`` on the RunEngine's loop: awaited, so it precedes the
    plan's next message, and never pending, so it never fails the run.
    """
    targets = [(device, list(names)) for device, names in targets]
    # Once per claimed scan: a resume before the plan's first checkpoint
    # replays the wait_for (never the open_run), and the dump is the
    # scan's start state, not the resume's.
    dumped: set[str] = set()

    def _after_open(msg: Msg):
        if msg.command != "open_run":
            return None, None
        folder = msg.kwargs.get("scan_folder")
        number = msg.kwargs.get("scan_number")
        if not folder or number is None:
            return None, None
        once = str(folder)
        factory = partial(
            dump_run,
            Path(str(folder)),
            int(number),
            str(msg.kwargs.get("experiment") or ""),
            targets,
            reader=reader,
            timeout=timeout,
        )

        async def _once() -> Path | None:
            if once in dumped:
                return None
            dumped.add(once)
            return await factory()

        def _tail():
            yield from bps.wait_for([_once])

        return None, _tail()

    return (yield from plan_mutator(plan, _after_open))
