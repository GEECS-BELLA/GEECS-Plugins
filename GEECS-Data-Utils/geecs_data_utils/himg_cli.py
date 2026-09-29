"""``geecs-himg`` — convert and verify HASO ``.himg`` folders from a shell.

The backlog tool.  The ``himg_to_stack`` analyzer kind converts one scan
at a click in the Data Portal; this command walks a scan folder (every
device folder holding ``.himg`` files, or the ones named) or one device
folder, and writes each device's capture stack through
:func:`geecs_data_utils.io.himg_stack.convert_himg_folder` — the same
function, the same checks.  ``verify`` re-checks an existing stack
against its manifest, or against the ``.himg`` files still on disk.

Nothing is deleted and no directory is created: converting adds one
``<device>.h5`` beside the sources and nothing else.

Examples
--------
::

    geecs-himg convert /data/Undulator/Y2026/03-Mar/26_0310/scans/Scan012
    geecs-himg convert .../scans/Scan012 --device U_HasoLift --overwrite
    geecs-himg verify  .../scans/Scan012/U_HasoLift --against-files
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from geecs_data_utils.data.sfile import read_sfile, sfile_path_for_scan
from geecs_data_utils.io.himg_stack import (
    DEFAULT_COMPRESSION_LEVEL,
    HimgStackError,
    convert_himg_folder,
    list_himg_files,
    stack_path_for,
    verify_himg_stack,
)

logger = logging.getLogger(__name__)

__all__ = ["main"]


def _scan_folder_of(path: Path) -> Path | None:
    """The ``scans/ScanNNN`` folder *path* is, or the one it lies in; else ``None``."""
    if path.parent.name == "scans":
        return path
    if path.parent.parent.name == "scans":
        return path.parent
    return None


def _scan_rows(scan_folder: Path | None):
    """The scan's scalar rows (for legacy shot-numbered names), or ``None``.

    The scanner-written table inside the scan folder first, then the
    analysis tree's s-file copy; both carry the device ``acq_timestamp``
    columns.
    """
    if scan_folder is None:
        return None
    candidates = [scan_folder / f"ScanData{scan_folder.name}.txt"]
    try:
        candidates.append(sfile_path_for_scan(scan_folder))
    except ValueError:
        pass
    for candidate in candidates:
        if candidate.is_file():
            return read_sfile(candidate)
    return None


def _device_dirs(path: Path, devices: Sequence[str] | None) -> list[Path]:
    """The device folders to work on: *path* itself, or its ``.himg``-holding children."""
    if not path.is_dir():
        raise HimgStackError(f"{path} is not a directory")
    if list_himg_files(path):
        return [path]
    children = {p.name: p for p in sorted(path.iterdir()) if p.is_dir()}
    if devices:
        unknown = [d for d in devices if d not in children]
        if unknown:
            raise HimgStackError(
                f"no such device folder in {path}: {', '.join(unknown)}"
            )
        return [children[d] for d in devices]
    return [p for p in children.values() if list_himg_files(p)]


def _convert(args: argparse.Namespace) -> int:
    rows = _scan_rows(_scan_folder_of(args.path))
    failures = 0
    for device_dir in _device_dirs(args.path, args.device):
        try:
            report = convert_himg_folder(
                device_dir,
                rows=rows,
                verify=not args.no_verify,
                overwrite=args.overwrite,
                compression_level=args.level,
            )
        except HimgStackError as exc:
            failures += 1
            print(f"{device_dir.name}: FAILED — {exc}")
            continue
        print(f"{device_dir.name}: {report.summary()}")
    return 1 if failures else 0


def _verify(args: argparse.Namespace) -> int:
    if args.path.is_file():
        stacks = [args.path]
    else:
        stacks = [stack_path_for(d) for d in _device_dirs(args.path, args.device)]
    failures = 0
    for stack in stacks:
        if not stack.is_file():
            failures += 1
            print(f"{stack.parent.name}: no stack at {stack}")
            continue
        try:
            report = verify_himg_stack(stack, against_files=args.against_files)
        except HimgStackError as exc:
            failures += 1
            print(f"{stack.parent.name}: FAILED — {exc}")
            continue
        if not report.ok:
            failures += 1
        print(f"{stack.parent.name}: {report.summary()}")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    """The ``geecs-himg`` command line."""
    parser = argparse.ArgumentParser(
        prog="geecs-himg",
        description="Convert HASO .himg device folders into per-scan capture "
        "stacks (<device>/<device>.h5), and verify them. The .himg files stay.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser(
        "convert", help="write each device folder's stack from its .himg files"
    )
    convert.add_argument(
        "path",
        type=Path,
        help="a device folder holding .himg files, or a scans/ScanNNN folder "
        "(every device folder holding .himg files)",
    )
    convert.add_argument(
        "--device",
        action="append",
        metavar="NAME",
        help="only this device folder of the scan folder (repeatable)",
    )
    convert.add_argument(
        "--overwrite", action="store_true", help="replace an existing stack"
    )
    convert.add_argument(
        "--no-verify",
        action="store_true",
        help="skip rebuilding every frame against its SHA-256 after writing",
    )
    convert.add_argument(
        "--level",
        type=int,
        default=DEFAULT_COMPRESSION_LEVEL,
        help=f"gzip level for the frames (default {DEFAULT_COMPRESSION_LEVEL})",
    )
    convert.set_defaults(run=_convert)

    verify = sub.add_parser("verify", help="check an existing stack")
    verify.add_argument(
        "path",
        type=Path,
        help="a stack file, a device folder, or a scans/ScanNNN folder",
    )
    verify.add_argument(
        "--device", action="append", metavar="NAME", help="as for convert"
    )
    verify.add_argument(
        "--against-files",
        action="store_true",
        help="also compare every rebuilt frame with the .himg file on disk",
    )
    verify.set_defaults(run=_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: parse, run, return the exit status (1 on any failure)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    try:
        return args.run(args)
    except HimgStackError as exc:
        print(f"geecs-himg: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
