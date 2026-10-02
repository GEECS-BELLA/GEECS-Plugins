"""THE s-file reader and path convention — one home, not three.

The GEECS scanner exports a per-scan scalar summary ("s-file"),
``s{number}.txt`` (number **unpadded**), tab-separated, living in the
*day's* ``analysis/`` folder — a sibling of ``scans/``, not inside the
scan folder.  Its headers are verbatim LabVIEW column names
(``"Bin #"``, ``"Shotnumber"``, ``"Device Variable"`` spellings), a
namespace disjoint from the Bluesky event schema — which is why the
union frame (:mod:`geecs_data_utils.scan_frame`) keeps both without
reconciliation.

Before this module the read and the path convention were duplicated in
``ScanData.load_scalars`` and twice in ScanAnalysis (``base.py``).
Consolidate here; the writer-side sites
(``copy_fresh_sfile_to_analysis``, ``tiled_export``) are deliberately
out of scope.  Strictly read-only (repo scan-folder invariant).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers
    import pandas as pd

_SCAN_FOLDER_RE = re.compile(r"^Scan(?P<number>\d{3,})$")


def sfile_path_for_scan(scan_folder: Path) -> Path:
    """The s-file path for a canonical ``scans/ScanNNN`` folder.

    Pure path construction (nothing is touched on disk — the caller
    checks existence): ``{day}/analysis/s{N}.txt`` with the scan number
    taken from the folder name, **unpadded** per the LabVIEW convention.

    Parameters
    ----------
    scan_folder : Path
        A ``scans/ScanNNN`` folder path (existing or not).

    Returns
    -------
    Path
        The conventional s-file path for that scan.

    Raises
    ------
    ValueError
        When *scan_folder* is not shaped ``.../scans/ScanNNN``.
    """
    match = _SCAN_FOLDER_RE.match(scan_folder.name)
    if match is None or scan_folder.parent.name != "scans":
        raise ValueError(f"{scan_folder} is not a canonical scans/ScanNNN folder")
    number = int(match.group("number"))
    return scan_folder.parent.parent / "analysis" / f"s{number}.txt"


def scan_data_txt_path_for(scan_folder: Path) -> Path:
    """The scanner-written scalar table inside a scan folder: ``ScanDataScanNNN.txt``.

    Pure path construction from the folder's own name (``ScanData`` +
    ``ScanNNN`` + ``.txt``); nothing is touched on disk.  The native
    scanner writes it at the stop document, the legacy scanner wrote it
    at the end of every scan — either way it exists only once the run
    closed, which is what :func:`run_closed_evidence` reads off it.
    """
    scan_folder = Path(scan_folder)
    return scan_folder / f"ScanData{scan_folder.name}.txt"


def stream_table_parquet_path_for(scan_folder: Path, stream: str) -> Path:
    """The Tiled writer's table for one event stream: ``ScanDataScanNNN-<stream>.parquet``.

    The s-file's sibling (GeecsBluesky 0.110.0): ``geecs-tiled-writer`` writes
    each stream's rows here at the run's close and registers the file in the
    Tiled catalog, so the scan folder is the record and the catalog can be
    rebuilt from it.  Pure path construction, as :func:`scan_data_txt_path_for`;
    the stream name must be a plain name (``primary``, ``shots``, ``baseline``,
    ``<device>_stream``) — a path separator or a hidden-file dot is refused.
    """
    scan_folder = Path(scan_folder)
    if not stream or "/" in stream or "\\" in stream or stream.startswith("."):
        raise ValueError(f"not a stream name: {stream!r}")
    return scan_folder / f"ScanData{scan_folder.name}-{stream}.parquet"


def run_closed_evidence(scan_folder: Path) -> Optional[Path]:
    """A file that exists only once the scanner closed the run, or ``None``.

    :func:`scan_data_txt_path_for` first, then the analysis tree's s-file
    (:func:`sfile_path_for_scan`) — both are written when the run ends.
    Read-only: nothing is written or created.  A folder that is not a
    canonical ``scans/ScanNNN`` path has only the first candidate.
    """
    scan_folder = Path(scan_folder)
    candidates = [scan_data_txt_path_for(scan_folder)]
    try:
        candidates.append(sfile_path_for_scan(scan_folder))
    except ValueError:  # not a canonical scans/ScanNNN folder
        pass
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def read_sfile(path: Path) -> "pd.DataFrame":
    r"""Read one s-file into a DataFrame, headers verbatim.

    One ``read_csv(sep="\t")`` with pandas' default dtype inference —
    numeric columns come back numeric, string columns stay strings (the
    dtype-tolerant contract downstream consumers already assume).

    Parameters
    ----------
    path : Path
        The s-file (raises ``FileNotFoundError`` naturally when absent).

    Returns
    -------
    pandas.DataFrame
        The scalar table, one row per shot, columns as LabVIEW wrote
        them.
    """
    import pandas as pd

    return pd.read_csv(path, delimiter="\t")
