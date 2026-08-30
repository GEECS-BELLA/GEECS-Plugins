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
``ScanData.load_scalars`` and twice in ScanAnalysis
(``base.py`` — see ``Planning/data_portal/03_analysis_tabs_design.md``,
finding 2).  Consolidate here; the writer-side sites
(``copy_fresh_sfile_to_analysis``, ``tiled_export``) are deliberately
out of scope.  Strictly read-only (repo scan-folder invariant).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

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
