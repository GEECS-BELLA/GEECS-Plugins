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
dtype). Recipes with ``scan.background_source`` are refused before anything is
copied: the legacy wrapper would resolve the reference scan through the real
share and cache a background beside it, and the core does not run them.
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
from functools import partial
from pathlib import Path
from typing import Sequence


def _copy_scan(scan: Path, device: str, base: Path) -> Path:
    """Copy one scan's inputs for ``device`` into ``base`` in the GEECS layout."""
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
    sfile = scan.parent.parent / "analysis" / f"s{tag.number}.txt"
    if not sfile.is_file():
        raise SystemExit(f"missing s-file {sfile}")
    analysis = target.parent.parent / "analysis"
    analysis.mkdir()
    shutil.copy2(sfile, analysis / sfile.name)
    return target


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


def _run(route: str, document, base: Path, scan: Path) -> tuple[list[str], float]:
    """Run one route on its private tree; return display files and seconds."""
    import scan_analysis.base as scan_base
    from geecs_data_utils import ScanPaths
    from scan_analysis.config import create_scan_analyzer

    # Construction never resolves paths; only run_analysis does, so the
    # ScanPaths patch wraps exactly that call and is restored on every exit.
    analyzer = create_scan_analyzer(document, route=route)
    tag = ScanPaths(folder=scan).get_tag()
    original = scan_base.ScanPaths
    started = time.perf_counter()
    scan_base.ScanPaths = partial(ScanPaths, base_directory=base)
    try:
        display = analyzer.run_analysis(tag) or []
    finally:
        scan_base.ScanPaths = original
        analyzer.cleanup()
    return [str(p) for p in display], time.perf_counter() - started


def main(argv: Sequence[str] | None = None) -> int:
    """Run both routes on private copies of the scan and report the differences."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--diagnostic", type=Path, required=True, help="diagnostic YAML path"
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
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"--output must be empty: {args.output}")
    from image_analysis.config import load_diagnostic
    from scan_analysis.route_compare import compare_snapshots, snapshot_analysis_tree

    document = _apply_overrides(load_diagnostic(args.diagnostic), args.set)
    if document.scan.background_source is not None:
        # The legacy wrapper resolves the reference scan through the real
        # share (its own ScanPaths, outside the private tree) and caches a
        # background there, and the core cannot run the recipe at all.
        raise SystemExit(
            "recipes with scan.background_source are refused: they would write "
            "next to the archived scan and the core does not run them"
        )
    device = document.scan.device or document.name
    scan = args.scan.resolve()
    results = {}
    for route in ("legacy", "core"):
        base = args.output / route
        _copy_scan(scan, device, base)
        display, seconds = _run(route, document, base, scan)
        results[route] = (
            snapshot_analysis_tree(next(base.rglob("analysis"))),
            display,
            seconds,
        )
        print(
            f"{route:7s} {seconds:7.1f} s  {len(results[route][0])} files  display: {[Path(p).name for p in display]}"
        )
    problems = compare_snapshots(
        results["legacy"][0], results["core"][0], average_ulps=args.average_ulps
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
