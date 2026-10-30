"""Several scans' per-shot tables as one: GEECSplotter's "merge s files".

:func:`merge_scan_frames` stacks one frame per scan, in the order given,
and tags each row with the scan it came from (:data:`SCAN_COLUMN`,
:data:`DAY_COLUMN`). The columns are the union: a column only some scans
have is NaN on the others' rows. Nothing is renumbered: each shot keeps
its own ``Shotnumber`` and ``Bin #``, so a shot is named by its scan and
its number together, and ``Bin #`` means nothing across scans (bin a
merged table by a value, never by ``Bin #``).

Pure: frames in, a frame out; no file is read or written.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

#: A scan as the merge names it: its day and its number.
ScanRef = tuple[date, int]

#: The scan number of each merged row.
SCAN_COLUMN = "scan"
#: The day of each merged row, as ISO text (``"2026-10-26"``).
DAY_COLUMN = "day"


def merge_scan_frames(frames: Mapping[ScanRef, "pd.DataFrame"]) -> "pd.DataFrame":
    """Stack the scans' frames into one, each row tagged with its scan.

    Parameters
    ----------
    frames : Mapping[tuple[datetime.date, int], pandas.DataFrame]
        ``(day, scan number) -> that scan's per-shot table``, in the order
        the rows are wanted.

    Returns
    -------
    pandas.DataFrame
        Every row of every frame, in order, on a fresh ``0..n-1`` index
        (labels are unique across scans). :data:`SCAN_COLUMN` (int) and
        :data:`DAY_COLUMN` (ISO text) come first, then the union of the
        frames' columns in the order first seen; a column a scan lacks is
        NaN on its rows.

    Raises
    ------
    ValueError
        No frames, or a frame that already has a ``scan`` or ``day``
        column (the tag would overwrite it).
    """
    import pandas as pd

    if not frames:
        raise ValueError("merge at least one scan")
    parts = []
    for (day, scan), frame in frames.items():
        taken = [c for c in (SCAN_COLUMN, DAY_COLUMN) if c in frame.columns]
        if taken:
            raise ValueError(
                f"Scan{scan:03d} on {day.isoformat()} already has a column "
                f"named {taken[0]!r}; the merge cannot tag its rows"
            )
        tags = pd.DataFrame(
            {SCAN_COLUMN: int(scan), DAY_COLUMN: day.isoformat()},
            index=frame.index,
        )
        parts.append(pd.concat([tags, frame], axis=1))
    # sort=False keeps the columns in first-seen order (the union).
    return pd.concat(parts, ignore_index=True, sort=False)
