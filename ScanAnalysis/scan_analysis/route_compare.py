"""Snapshot and compare the analysis trees two factory routes wrote for one scan.

One definition of "the same outputs", shared by the in-suite differential test
and ``scripts/analysis_scan_compare.py``, so neither drifts when a product kind
or a tolerance changes. HDF5 payloads and tab-separated scalar tables are
decoded and compared; every other file (the PNG figures) counts by presence
only, because the two renderers draw different pixels by design.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

#: Noscan averages carry this marker in their file name; see ``compare_snapshots``.
AVERAGE_MARKER = "_average_processed"


def snapshot_analysis_tree(analysis: Path) -> dict[str, tuple]:
    """Decode every file under an ``analysis/`` folder, keyed by relative path.

    Entries are ``("h5", dataset, array, dtype)``, ``("table", DataFrame)`` for
    ``.txt`` files, or ``("file", size)`` for anything else.
    """
    files: dict[str, tuple] = {}
    for path in sorted(Path(analysis).rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(analysis).as_posix()
        if path.suffix == ".h5":
            with h5py.File(path) as handle:
                (key,) = list(handle)
                files[name] = ("h5", key, handle[key][:], handle[key].dtype)
        elif path.suffix == ".txt":
            files[name] = ("table", pd.read_csv(path, sep="\t"))
        else:
            files[name] = ("file", path.stat().st_size)
    return files


def compare_snapshots(
    legacy: dict[str, tuple], core: dict[str, tuple], *, average_ulps: int = 4
) -> list[str]:
    """Return one line per difference; an empty list means the trees match.

    Arrays and tables must match exactly, with one explicit exception: noscan
    average arrays (``AVERAGE_MARKER`` in the name). The legacy wrapper sums
    shots in directory-listing order, whatever the filesystem returns, while
    the core sums in scalar-row order; same per-shot inputs and formula, so
    they differ only by summation rounding, bounded here by ``average_ulps``
    of the stored dtype. Other files are compared by presence only.
    """
    problems = []
    for name in sorted(set(legacy) - set(core)):
        problems.append(f"only legacy wrote {name}")
    for name in sorted(set(core) - set(legacy)):
        problems.append(f"only core wrote {name}")
    for name in sorted(set(legacy) & set(core)):
        expected, actual = legacy[name], core[name]
        if expected[0] != actual[0]:
            problems.append(f"{name}: kind {expected[0]} vs {actual[0]}")
        elif expected[0] == "h5":
            _, key, data, dtype = expected
            _, key2, data2, dtype2 = actual
            if (key, dtype, data.shape) != (key2, dtype2, data2.shape):
                problems.append(
                    f"{name}: {key}/{dtype}/{data.shape} vs {key2}/{dtype2}/{data2.shape}"
                )
                continue
            if AVERAGE_MARKER in name:
                tolerance = average_ulps * np.finfo(dtype).eps
                same = np.allclose(data, data2, rtol=tolerance, atol=0, equal_nan=True)
            else:
                same = np.array_equal(data, data2, equal_nan=True)
            if not same:
                gap = np.nanmax(np.abs(data.astype(float) - data2.astype(float)))
                problems.append(f"{name}: arrays differ (max |delta| = {gap:.3g})")
        elif expected[0] == "table":
            try:
                pd.testing.assert_frame_equal(expected[1], actual[1], check_exact=True)
            except AssertionError as exc:
                problems.append(f"{name}: {str(exc).splitlines()[0]}")
    return problems
