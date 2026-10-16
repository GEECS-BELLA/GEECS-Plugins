"""Run one diagnostic on a completed scan through both factory routes and diff the outputs.

Run from the repository's root Poetry environment::

    poetry run python scripts/analysis_scan_compare.py \
        --diagnostic /configs/analyzers/HTU/Amp4Input.yaml \
        --scan /data/Undulator/Y2025/02-Feb/25_0220/scans/Scan014 \
        --set scan.data_format=per_shot_files --output /tmp/compare

Nothing is written next to the real scan. The scan's ScanInfo ini, the
diagnostic's device folder and the s-file are copied into two private trees in
the GEECS layout, ``<output>/legacy`` and ``<output>/core``; the legacy wrapper
route runs in one and the core route in the other. Their ``analysis/ScanNNN``
trees are then compared by ``scan_analysis.route_compare``, the same rules the
in-suite differential test uses: file lists, HDF5 dataset names/dtypes/payloads,
the s-file and sidecar tables, PNG presence. Comparison is exact except for
noscan average arrays, where the legacy wrapper sums shots in directory-listing
order; ``--average-ulps`` bounds that difference (default 4 ulps of the stored
dtype) — and, with ``--nonuniform-axis``, for the ``*_rms``/``*_fwhm`` columns,
which the core measures over x on a nonuniform trace axis (GEECS-Analysis
0.26.0, #1029) where the legacy analyzer scaled index-space widths by one
spacing. The files a run adds inside the scan folder itself (a measure's
sibling outputs: the magspec ``-interp`` / ``-interpSpec`` folders) are
compared byte for byte.

``--core-diagnostic`` runs the core route on another document — the v3
recipe converted from the v2 diagnostic the legacy route runs (a magspec
document, whose v2 form the core never routes) — so a regenerated document
is accepted against the analyzer it replaces.

A dark-scan background (``scan.background_source.scan_number``) is copied
too: that scan's device folder and ScanInfo ini, never its cached
background, so each route computes its own inside its private tree. Other
background sources are refused before anything is copied.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

# A file-writing comparison never opens a window; the legacy wrappers render
# from worker threads, which the interactive macOS backend aborts on.
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
from typing import Sequence


def _copy_scan(scan: Path, device: str, base: Path, *, rows: bool = True) -> Path:
    """Copy one scan's inputs for ``device`` into ``base`` in the GEECS layout.

    ``rows=False`` (a dark scan) copies the ScanInfo ini and the device
    folder only.
    """
    from geecs_data_utils import ScanPaths

    tag = ScanPaths(folder=scan).get_tag()
    target = ScanPaths.get_scan_folder_path(tag=tag, base_directory=base)
    target.mkdir(parents=True)  # a private copy, never the share
    ini = scan / f"ScanInfo{scan.name}.ini"
    if not ini.is_file():
        raise SystemExit(f"missing {ini}")
    shutil.copy2(ini, target / ini.name)
    source = scan / device
    if not source.is_dir():
        raise SystemExit(f"missing device folder {source}")
    shutil.copytree(source, target / device)
    if not rows:
        return target
    sfile = scan.parent.parent / "analysis" / f"s{tag.number}.txt"
    if not sfile.is_file():
        raise SystemExit(f"missing s-file {sfile}")
    analysis = target.parent.parent / "analysis"
    analysis.mkdir()
    shutil.copy2(sfile, analysis / sfile.name)
    return target


def _inputs_of(document) -> tuple[str, int | None]:
    """The device folder a document reads and the dark scan it averages, if any.

    Either format: a v2 ``scan.background_source.scan_number`` or a v3
    ``from_scan`` input with a scan number. Any other scan-computed
    background (the current scan, autodetect) is refused: the two routes
    cache those under names they do not share.
    """
    from geecs_schemas.analysis import AnalysisRecipe

    if isinstance(document, AnalysisRecipe):
        darks = {i.from_scan.scan for i in document.inputs.values() if i.from_scan}
        folder = document.data_folder
    else:
        source = document.scan.background_source
        darks = set() if source is None else {source.scan_number}
        folder = document.scan.device or document.name
    if None in darks or len(darks) > 1:
        raise SystemExit(
            "only one dark-scan background (a scan number) is copied; other "
            "scan-computed backgrounds are refused"
        )
    return folder, next(iter(darks), None)


def _apply_overrides(document, overrides: Sequence[str]):
    """Apply ``section.field=value`` overrides to a diagnostic copy."""
    for item in overrides:
        key, _, value = item.partition("=")
        section, _, field = key.partition(".")
        if section != "scan" or not field or not value:
            raise SystemExit(f"--set expects scan.<field>=<value>, got {item!r}")
        payload = document.scan.model_dump()
        payload[field] = value
        document = document.model_copy(
            update={"scan": type(document.scan).model_validate(payload)}
        )
    return document


def _run(
    route: str, document, base: Path, scan: Path, *, serial: bool = False
) -> tuple[list[str], float]:
    """Run one route on its private tree; return display files and seconds."""
    import geecs_data_utils
    import scan_analysis.base as scan_base
    from geecs_data_utils import ScanPaths
    from scan_analysis.config import create_scan_analyzer

    class PrivateScanPaths(ScanPaths):
        """``ScanPaths`` rooted in the private tree unless told otherwise."""

        def __init__(self, *args, **kwargs):
            kwargs.setdefault("base_directory", base)
            super().__init__(*args, **kwargs)

    # Construction never resolves paths; only run_analysis does, so the
    # ScanPaths patch wraps exactly that call and is restored on every exit.
    # The legacy wrapper imports ScanPaths at call time to find a dark scan,
    # so the package attribute is patched too.
    analyzer = create_scan_analyzer(document, route=route)
    if serial:
        analyzer.max_workers = 1
    tag = ScanPaths(folder=scan).get_tag()
    started = time.perf_counter()
    scan_base.ScanPaths = PrivateScanPaths
    geecs_data_utils.ScanPaths = PrivateScanPaths
    try:
        display = analyzer.run_analysis(tag) or []
    finally:
        scan_base.ScanPaths = ScanPaths
        geecs_data_utils.ScanPaths = ScanPaths
        analyzer.cleanup()
    return [str(p) for p in display], time.perf_counter() - started


def main(argv: Sequence[str] | None = None) -> int:
    """Run both routes on private copies of the scan and report the differences."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--diagnostic", type=Path, required=True, help="diagnostic YAML path"
    )
    parser.add_argument(
        "--core-diagnostic",
        type=Path,
        help="the document the core route runs (default: --diagnostic), e.g. "
        "the v3 recipe converted from it; give it the same file stem, which "
        "names the scalar sidecar",
    )
    parser.add_argument(
        "--scan", type=Path, required=True, help="…/scans/ScanNNN folder"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="empty or missing directory for the two private trees",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="scan.FIELD=VALUE",
        help="override a scan: field, e.g. scan.data_format=per_shot_files",
    )
    parser.add_argument("--average-ulps", type=int, default=4)
    parser.add_argument(
        "--legacy-serial",
        action="store_true",
        help="run the legacy wrapper's shots one at a time: the legacy magspec "
        "analyzer writes each shot's teslameter field into one shared "
        "calibration, so its 16 threads can measure a shot with another's",
    )
    parser.add_argument(
        "--nonuniform-axis",
        action="store_true",
        help="the trace axis is nonuniform (stitched siblings, a nonlinear "
        "calibration): the core measures rms/fwhm over x (GEECS-Analysis 0.26.0, "
        "#1029) where legacy scaled index widths, so those columns are excluded",
    )
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"--output must be empty: {args.output}")
    from image_analysis.config import load_diagnostic
    from geecs_data_utils.scan_info import scan_folder_name
    from scan_analysis.route_compare import (
        compare_snapshots,
        snapshot_analysis_tree,
        snapshot_scan_products,
    )

    documents = {"legacy": _apply_overrides(load_diagnostic(args.diagnostic), args.set)}
    documents["core"] = (
        _apply_overrides(load_diagnostic(args.core_diagnostic), args.set)
        if args.core_diagnostic is not None
        else documents["legacy"]
    )
    sources = {_inputs_of(document) for document in documents.values()}
    if len(sources) != 1:
        raise SystemExit(f"the two documents read different inputs: {sources}")
    ((device, dark),) = sources
    scan = args.scan.resolve()
    results = {}
    for route in ("legacy", "core"):
        base = args.output / route
        if dark is not None:
            _copy_scan(scan.parent / scan_folder_name(dark), device, base, rows=False)
        target = _copy_scan(scan, device, base)
        display, seconds = _run(
            route,
            documents[route],
            base,
            scan,
            serial=args.legacy_serial and route == "legacy",
        )
        snapshot = snapshot_analysis_tree(target.parent.parent / "analysis")
        snapshot.update(
            snapshot_scan_products(target, {device, f"ScanInfo{target.name}.ini"})
        )
        results[route] = (snapshot, display, seconds)
        print(
            f"{route:7s} {seconds:7.1f} s  {len(results[route][0])} files  display: {[Path(p).name for p in display]}"
        )
    problems = compare_snapshots(
        results["legacy"][0],
        results["core"][0],
        average_ulps=args.average_ulps,
        nonuniform_axis=args.nonuniform_axis,
    )
    legacy_display = [
        Path(p).relative_to(args.output / "legacy").as_posix()
        for p in results["legacy"][1]
    ]
    core_display = [
        Path(p).relative_to(args.output / "core").as_posix() for p in results["core"][1]
    ]
    if legacy_display != core_display:
        problems.append(f"display files differ: {legacy_display} vs {core_display}")
    for line in problems:
        print("DIFF", line)
    print("MATCH" if not problems else f"{len(problems)} difference(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
