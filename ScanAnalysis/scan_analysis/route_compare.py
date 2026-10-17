"""Snapshot and compare the analysis trees two runs wrote for one scan.

One definition of "the same outputs", shared by every differential test in
the suite (a run against its committed golden tree, a pooled run against the
serial one), so none drifts when a product kind or a tolerance changes. HDF5
payloads and tab-separated scalar tables are decoded and compared; every
other file (the PNG figures) counts by presence only, because figures are the
operator's review, not a pin. The files a run adds inside the scan folder
itself (a measure's sibling outputs) are no figures: the magspec
``-interpSpec`` spectra are decoded and compared like the analysis tree's
tables, the ``-interp`` 16-bit images as their pixels (the core's own
reader; integer arrays are always compared exactly).

Exact is the default. A golden generated on one platform and checked on
another sees float drift (BLAS/SIMD summation order, scipy builds): a
magspec peak energy, a bow-tie fit, a beam width differ in the last digits.
``compare_snapshots(rtol=...)`` states that allowance once — relative to the
array's or column's own scale, so zeros beside large values do not fail
spuriously — and every reported difference carries the max relative delta,
so the margin is visible where it fails.

A snapshot can be stored as a **golden** (:func:`write_golden`: a JSON
manifest beside one ``.npz`` of the arrays and numeric table columns) and
read back (:func:`read_golden`) in the shape :func:`compare_snapshots`
takes, so a test pins a run against a tree generated once — the legacy
analyzers' outputs, captured before those analyzers were deleted (#1039).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
from geecs_data_utils.io.images import read_imaq_png_image

#: Noscan averages carry this marker in their file name; see ``compare_snapshots``.
AVERAGE_MARKER = "_average_processed"

#: The scalar columns the ``line`` and ``beam`` measures' widths land in; see
#: ``compare_snapshots(nonuniform_axis=True)``.
WIDTH_COLUMN_SUFFIXES = ("_rms", "_fwhm")


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


def snapshot_scan_products(scan: Path, inputs: Iterable[str]) -> dict[str, tuple]:
    """Decode every file a run added inside a ``scans/ScanNNN`` folder.

    Keyed by path relative to the scan folder's parent (``ScanNNN/...``, so
    the keys never collide with an analysis tree's). A ``.tsv`` / ``.txt``
    is ``("table", DataFrame)``, a PNG its decoded pixels
    ``("h5", "pixels", array, dtype)`` (a data product, never a figure:
    ``read_imaq_png_image`` keeps the 16-bit depth), anything else
    ``("bytes", sha256 hex)``. ``inputs`` names the top-level entries that
    were there before the run (the copied device folder, the ScanInfo ini)
    and are skipped.
    """
    scan = Path(scan)
    skipped = set(inputs)
    files: dict[str, tuple] = {}
    for path in sorted(scan.rglob("*")):
        relative = path.relative_to(scan)
        if not path.is_file() or relative.parts[0] in skipped:
            continue
        name = path.relative_to(scan.parent).as_posix()
        if path.suffix in {".tsv", ".txt"}:
            files[name] = ("table", pd.read_csv(path, sep="\t"))
        elif path.suffix.lower() == ".png":
            pixels = read_imaq_png_image(path)
            files[name] = ("h5", "pixels", pixels, pixels.dtype)
        else:
            files[name] = ("bytes", hashlib.sha256(path.read_bytes()).hexdigest())
    return files


def _close(expected: np.ndarray, actual: np.ndarray, rtol: float) -> tuple[bool, float]:
    """Whether ``actual`` is within ``rtol`` of ``expected``, and the relative delta seen.

    The allowance is relative to the expected array's own scale
    (``rtol * max|expected|``, so a zero beside large values is held to the
    same absolute margin as its neighbours), NaNs match NaNs, and the
    returned delta is ``max|actual - expected| / max|expected|`` (0 when
    both are all zero), whatever the verdict. An integer array (a 16-bit
    image product) is exact whatever ``rtol`` says: integers do not drift.
    """
    if (
        np.asarray(expected).dtype.kind in "iub"
        and np.asarray(actual).dtype.kind in "iub"
    ):
        rtol = 0.0
    a = np.asarray(expected, dtype=float)
    b = np.asarray(actual, dtype=float)
    if a.shape != b.shape:
        return False, float("inf")
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if not np.array_equal(nan_a, nan_b):
        return False, float("inf")
    finite = ~nan_a
    if not finite.any():
        return True, 0.0
    scale = float(np.max(np.abs(a[finite])))
    gap = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
    if scale == 0.0:
        return gap == 0.0, (0.0 if gap == 0.0 else float("inf"))
    relative = gap / scale
    return relative <= rtol, relative


def compare_snapshots(  # noqa: C901, PLR0912
    legacy: dict[str, tuple],
    core: dict[str, tuple],
    *,
    average_ulps: int = 4,
    nonuniform_axis: bool = False,
    rtol: float = 0.0,
) -> list[str]:
    """Return one line per difference; an empty list means the trees match.

    Arrays and tables must match exactly, with three explicit allowances.
    Noscan average arrays (``AVERAGE_MARKER`` in the name): the legacy
    wrapper summed shots in directory-listing order, whatever the filesystem
    returned, while the core sums in scalar-row order; same per-shot inputs
    and formula, so they differ only by summation rounding, bounded here by
    ``average_ulps`` of the stored dtype. When the caller declares the
    diagnostic's trace axis nonuniform (``nonuniform_axis``: a trace
    stitched from sibling cameras, a nonlinear calibration), the tables'
    ``*_rms`` / ``*_fwhm`` columns (``WIDTH_COLUMN_SUFFIXES``) are compared
    by presence only: the core measures those widths over x (GEECS-Analysis
    0.26.0, #1029) where the legacy analyzer scaled index-space widths by
    the one spacing at the centroid, so they differ by design. And ``rtol``
    — **cross-platform float drift**: a golden generated on one platform
    and checked on another (BLAS/SIMD summation order, scipy builds) —
    bounds every array and numeric table column to ``rtol`` of its own
    scale (``_close``; the pooled-vs-serial pins keep the default 0, they
    run on one machine). Every array or column difference reports the max
    relative delta seen, so the margin is visible; an integer array (a
    16-bit image product) is exact regardless. Scan-folder products
    (``snapshot_scan_products``) follow the same rules by kind; raw bytes
    must be identical; other files are compared by presence only.
    """
    problems = []
    for name in sorted(set(legacy) - set(core)):
        problems.append(f"only legacy wrote {name}")  # noqa: PERF401 — clearer as a loop
    for name in sorted(set(core) - set(legacy)):
        problems.append(f"only core wrote {name}")  # noqa: PERF401 — clearer as a loop
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
            if AVERAGE_MARKER in name and average_ulps:
                tolerance = max(rtol, average_ulps * float(np.finfo(dtype).eps))
                same = np.allclose(data, data2, rtol=tolerance, atol=0, equal_nan=True)
                _, relative = _close(data, data2, rtol)
            else:
                same, relative = _close(data, data2, rtol)
            if not same:
                gap = np.nanmax(np.abs(data.astype(float) - data2.astype(float)))
                problems.append(
                    f"{name}: arrays differ (max |delta| = {gap:.3g}, "
                    f"relative {relative:.3g} > rtol {rtol:.3g})"
                )
        elif expected[0] == "bytes":
            if expected[1] != actual[1]:
                problems.append(f"{name}: bytes differ")
        elif expected[0] == "table":
            old_table, new_table = expected[1], actual[1]
            if nonuniform_axis:
                old_widths = _width_columns(old_table)
                new_widths = _width_columns(new_table)
                if old_widths != new_widths:
                    problems.append(
                        f"{name}: width columns {old_widths} vs {new_widths}"
                    )
                    continue
                old_table = old_table.drop(columns=old_widths)
                new_table = new_table.drop(columns=new_widths)
            problems.extend(
                f"{name}: {line}"
                for line in _table_problems(old_table, new_table, rtol)
            )
    return problems


def _table_problems(old: pd.DataFrame, new: pd.DataFrame, rtol: float) -> list[str]:
    """Column names, dtypes and non-numeric cells exact; numeric columns within ``rtol``."""
    if list(old.columns) != list(new.columns):
        return [f"columns {list(old.columns)} vs {list(new.columns)}"]
    if len(old) != len(new):
        return [f"{len(old)} rows vs {len(new)}"]
    problems = []
    for column in old.columns:
        a, b = old[column].to_numpy(), new[column].to_numpy()
        numeric = np.issubdtype(a.dtype, np.number) and np.issubdtype(
            b.dtype, np.number
        )
        if numeric and a.dtype.kind == "f" and rtol:
            same, relative = _close(a, b, rtol)
            if not same:
                problems.append(
                    f'column "{column}" differs (relative {relative:.3g} > rtol {rtol:.3g})'
                )
        else:
            try:
                pd.testing.assert_series_equal(
                    old[column], new[column], check_exact=True, check_names=False
                )
            except AssertionError as exc:
                problems.append(f'column "{column}": {str(exc).splitlines()[0]}')
    return problems


def _width_columns(table: pd.DataFrame) -> list[str]:
    """The table's ``*_rms`` / ``*_fwhm`` columns, in table order."""
    return [c for c in table.columns if str(c).endswith(WIDTH_COLUMN_SUFFIXES)]


# ------------------------------------------------------------------- goldens

MANIFEST_NAME = "manifest.json"
ARRAYS_NAME = "arrays.npz"
#: The one allowance a committed golden gets: cross-platform float drift,
#: relative to each array's / column's own scale (``compare_snapshots(rtol=)``).
#: Generated on macOS, checked on Linux CI; a ×1.0001 (1e-4) perturbation of
#: a core numeric must still fail — ``tests/test_route_compare.py`` pins it.
GOLDEN_RTOL = 1e-6


@dataclass(frozen=True)
class Golden:
    """A stored snapshot: the files in ``compare_snapshots``' shape, plus notes."""

    files: dict[str, tuple]
    #: Whatever the writer recorded beside the tree (a display-file list).
    extra: dict[str, Any] = field(default_factory=dict)


def write_golden(
    snapshot: dict[str, tuple], folder: Path, *, extra: dict[str, Any] | None = None
) -> None:
    """Store a snapshot under ``folder`` as ``manifest.json`` + ``arrays.npz``.

    HDF5 payloads keep their dtype in the ``.npz``; a table's numeric
    columns go there too (so a float survives bit for bit, where a text
    round trip may not) and its other columns into the manifest; a
    scan-folder product keeps its digest; any other file keeps only its
    name, as the comparison counts it by presence alone.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, Any] = {}

    def store(data: np.ndarray) -> str:
        key = f"a{len(arrays)}"
        arrays[key] = np.asarray(data)
        return key

    for name in sorted(snapshot):
        entry = snapshot[name]
        kind = entry[0]
        if kind == "h5":
            _, key, data, dtype = entry
            manifest[name] = {
                "kind": kind,
                "key": key,
                "dtype": np.dtype(dtype).str,
                "array": store(data),
            }
        elif kind == "table":
            table: pd.DataFrame = entry[1]
            columns = []
            for column in table.columns:
                values = table[column].to_numpy()
                if np.issubdtype(values.dtype, np.number) or values.dtype == bool:
                    columns.append({"name": column, "array": store(values)})
                else:
                    columns.append(
                        {
                            "name": column,
                            "values": [
                                None if (isinstance(v, float) and np.isnan(v)) else v
                                for v in values.tolist()
                            ],
                        }
                    )
            manifest[name] = {"kind": kind, "columns": columns}
        elif kind == "bytes":
            manifest[name] = {"kind": kind, "sha256": entry[1]}
        else:
            manifest[name] = {"kind": "file"}
    payload = {"files": manifest, "extra": extra or {}}
    (folder / MANIFEST_NAME).write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n"
    )
    np.savez_compressed(folder / ARRAYS_NAME, **arrays)


def read_golden(folder: Path) -> Golden:
    """Read a tree :func:`write_golden` stored; ``("file", None)`` for presence-only files."""
    folder = Path(folder)
    payload = json.loads((folder / MANIFEST_NAME).read_text())
    with np.load(folder / ARRAYS_NAME) as handle:
        arrays = {key: handle[key] for key in handle.files}
    files: dict[str, tuple] = {}
    for name, entry in payload["files"].items():
        kind = entry["kind"]
        if kind == "h5":
            data = arrays[entry["array"]]
            files[name] = ("h5", entry["key"], data, np.dtype(entry["dtype"]))
        elif kind == "table":
            columns = {}
            for column in entry["columns"]:
                if "array" in column:
                    columns[column["name"]] = arrays[column["array"]]
                else:
                    columns[column["name"]] = column["values"]
            files[name] = ("table", pd.DataFrame(columns))
        elif kind == "bytes":
            files[name] = ("bytes", entry["sha256"])
        else:
            files[name] = ("file", None)
    return Golden(files, dict(payload.get("extra", {})))
