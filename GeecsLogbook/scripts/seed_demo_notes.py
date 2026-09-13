#!/usr/bin/env python
"""Fill a notes database with worked-example entries, for looking at.

Why this exists
---------------
The logbook's own store is authoritative: what people wrote exists nowhere
else, so it must never be seeded with invented content — six months later
nobody can tell a demo entry from a real one. But a page that renders a
day with no notes in it shows none of the thing it was built for, which is
how an unstyled composer and an orphaned container both shipped unnoticed.

So: a *separate* database, written by a script that is obviously a script,
carrying entries plainly marked as examples. Point a development server at
it with ``--notes-db``; never point it at the deployed one.

The day it describes is real (the native-Bluesky phase-2b acceptance run),
because test data shaped like the real thing exercises the real cases —
long purposes that must ellipsize, a failure reason that must not, notes
whose timestamps sit between two scans rather than on either.

Usage
-----
::

    poetry run python scripts/seed_demo_notes.py /tmp/demo-notes.db
    poetry run geecs-portal --scan-log --notes-db /tmp/demo-notes.db

The scan folders for ``DAY`` must be reachable for the scan-anchored
entries to appear: an anchor naming a scan the day does not contain is
stored and counted but never drawn. The day-level entry always renders,
so it is the check that seeding worked.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from geecs_logbook.store import NotesStore

#: What a deployment names its store (GeecsLogbook/CLAUDE.md, the portal's
#: state directory). Refused outright: on a fresh host the file does not
#: exist yet, so an existence check alone lets the worst case through.
DEPLOYED_DB_NAME = "logbook.db"

#: The day these entries belong to, and the scans they hang off.
DAY = "2026-09-12"

#: ``(scan, after, author, body)`` — exactly one of ``scan``/``after`` is
#: set, which the store enforces: an entry is anchored to a scan or after
#: one, never both.
ENTRIES: list[tuple[int | None, int | None, str, str]] = [
    # A day-level entry, anchored to neither a scan nor a gap. It is the one
    # shape that renders whether or not the share is reachable, so it is the
    # signal that seeding worked: every other entry here hangs off a scan
    # number, and an anchor naming a scan the day folder does not contain is
    # accepted, counted in "Notes N", and never drawn.
    (
        None,
        None,
        "demo",
        "> [!NOTE] Seeded example day\n"
        "> Written by `scripts/seed_demo_notes.py`, not by a person.\n\n"
        "Native-Bluesky phase-2b acceptance. If the scan blocks below are "
        "missing, the data share for this day is not reachable from here — "
        "the notes are stored either way.",
    ),
    # --- the A1 gap: two failures, then a pass, with 18 minutes in between
    (
        None,
        2,
        "demo",
        "> [!NOTE] Example entry\n"
        "> Seeded by `scripts/seed_demo_notes.py` — not a real record.\n\n"
        "A1 failed twice on `UC_Amp3_IR_input`: first the save directory "
        "did not exist, then the HDF capture did not arm inside 10 s.\n\n"
        "Created the per-scan directory on the camera server and restarted "
        "the plugin. Third attempt took it.",
    ),
    (
        3,
        None,
        "demo",
        "Clean pass. Both plugin cameras armed on the first trigger.",
    ),
    # --- A3 fails, gets parked, comes back much later with a different clock
    (
        None,
        6,
        "demo",
        "> [!WARNING] Example entry\n"
        "> Seeded by `scripts/seed_demo_notes.py` — not a real record.\n\n"
        "A3 wants a triggered scalar as the shot clock and `U_HP_Daq` never "
        "produced one inside 3 s.\n\n"
        "Parking it and moving to A4 rather than burning scan numbers on "
        "retries. Worth trying a different device as the clock.",
    ),
    (
        None,
        14,
        "demo",
        "> [!TIP] Example entry\n"
        "> Seeded by `scripts/seed_demo_notes.py` — not a real record.\n\n"
        "Back to A3 with `U_BCaveICT` as the clock instead of `U_HP_Daq`. "
        "It ticks on every shot, which is the property the sampler needs.\n\n"
        "This is the note the scan record structurally cannot hold: the "
        "28-minute gap above is invisible in `ScanInfo`, and the only "
        "difference between the failure and the pass is a decision.",
    ),
    (
        15,
        None,
        "demo",
        "A3 passes. Clocked by `U_BCaveICT`.",
    ),
    # --- a code bug, which belongs against the scan that hit it
    (
        17,
        None,
        "demo",
        "`unsupported operand type(s) for +: 'SignalR' and 'float'` — "
        "`rel_scan` is adding a float to the signal object rather than to "
        "its value. A real bug, not a hardware fault.",
    ),
    # --- and a note with no scan on either side of it, late in the day
    (
        None,
        21,
        "demo",
        "Broader-set scans all green. Stopping here; the two plugin cameras "
        "on separate servers were the case worth proving.",
    ),
]


def seed(db_path: Path) -> int:
    """Write the example entries into ``db_path`` and return how many.

    Refuses anything that looks like a real store. The checks run *before*
    :class:`NotesStore` opens the file, because constructing one runs the
    schema and its column migration on whatever path it is handed — so a
    guard that consults the store has already touched the thing it meant
    to protect.

    Two holes an earlier version had, both reachable by import rather than
    through the CLI: it asked only whether :data:`DAY` had entries, so a
    store holding a human's notes for every *other* day sailed through;
    and it ignored tombstones, so re-seeding after a delete doubled the
    rows.
    """
    if db_path.name == DEPLOYED_DB_NAME:
        raise ValueError(
            f"{db_path} is named {DEPLOYED_DB_NAME}, which is what a deployment "
            "calls its real store — seed a differently named file"
        )
    if db_path.exists() and _is_a_store(db_path):
        raise ValueError(f"{db_path} is already a notes store — seed a fresh file")

    store = NotesStore(db_path)
    for scan, after, author, body in ENTRIES:
        store.create(day=DAY, author=author, body_md=body, scan=scan, after=after)
    return len(ENTRIES)


def _is_a_store(db_path: Path) -> bool:
    """Whether this file is a notes store — not whether it has rows in it.

    Asking about rows was not enough. Mounting ``--notes-db some.db`` on a
    fresh host creates the file and the ``entries`` table with zero rows,
    so a row check waved through the live store of a deployment that had
    started but not yet been written in — exactly the case the reserved
    filename was added to catch, except ``--notes-db`` takes any path.

    Read-only and schema-free: it opens the file directly rather than
    through :class:`NotesStore`, whose constructor would create and
    migrate the very thing this is protecting.
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:  # pragma: no cover - unreadable path
        return True  # refuse what we cannot inspect
    try:
        return bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='entries'"
            ).fetchone()
        )
    except sqlite3.DatabaseError:
        # Not a SQLite file at all. Refusing beats overwriting it.
        return True
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    """Seed a demo database named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "db",
        type=Path,
        help="Where to write. Its parent must exist; a name with 'demo' or "
        "'test' in it is a good habit.",
    )
    args = parser.parse_args(argv)

    if args.db.exists():
        print(f"{args.db} already exists — refusing to add to it", file=sys.stderr)
        return 1
    if not args.db.parent.is_dir():
        print(f"{args.db.parent} is not a directory", file=sys.stderr)
        return 1

    n = seed(args.db)
    print(f"wrote {n} example entries for {DAY} to {args.db}")
    print("point a development server at it with --notes-db; never the deployed one")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
