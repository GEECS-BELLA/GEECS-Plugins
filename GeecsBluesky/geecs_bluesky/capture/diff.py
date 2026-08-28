"""The dual-write diff: capture stacks vs native PNGs, per scan.

The Phase-6 evidence engine. For every device folder in a scan that holds
a capture stack, join the
stack's per-frame ``acq_timestamp``s against the native per-shot files'
filename timestamps (the same canonical-millisecond keys the analysis join
uses) and pixel-compare every matched pair (IMAQ-decoded PNG vs stack
frame — proven bit-identical on healthy dual-writes). One verdict line per
device, optionally appended to a JSONL evidence log; a non-zero exit on
any mismatch. Weeks of clean log entries are the PNG-deprecation gate
(``Planning/data_capture/01_central_pva_capture_scope.md`` Phase 6).

Vocabulary (per device):

- ``matched`` — timestamps present on both sides; each is pixel-compared.
- ``stack_only`` — frames only the daemon captured (pre-save-window extras,
  or every frame on a toggle-off scan) — attributable, never a failure.
- ``png_only`` — **the bad bucket**: a frame LV saved that capture missed.
- verdicts: ``pass`` (no png_only, all matched pixel-identical),
  ``capture_only`` (no PNGs at all — toggle-off scan, nothing to diff),
  ``no_stack`` (PNGs but no stack — not captured; informational),
  ``mismatch`` otherwise.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from geecs_data_utils.io.scan_stack import (
    ShotRef,
    find_stack_file,
    read_shot,
    read_stack_timestamps,
)
from geecs_data_utils.native_files import filename_timestamp_regex, timestamp_key

logger = logging.getLogger(__name__)

# Signature: path -> ndarray (the IMAQ-decoding native-file reader).
PngReader = Callable[[Path], "object"]


@dataclass
class DeviceDiff:
    """One device's dual-write reconciliation for one scan."""

    scan: str
    device: str
    matched: int
    pixel_identical: int
    png_only: int
    stack_only: int
    verdict: str


def _default_png_reader(path: Path):
    from geecs_data_utils.io.images import read_imaq_image

    return read_imaq_image(path)


def diff_device_dir(
    device_dir: Path, *, png_reader: PngReader | None = None
) -> DeviceDiff | None:
    """Diff one scan device folder; ``None`` when there is nothing to say.

    A folder with neither a stack nor native files (a scalar-only device)
    returns ``None``; every other combination gets a verdict.
    """
    import numpy as np

    png_reader = png_reader or _default_png_reader
    scan = device_dir.parent.name
    stack = find_stack_file(device_dir)
    regex = filename_timestamp_regex(".png")
    pngs_by_key: dict[int, Path] = {}
    for f in device_dir.iterdir():
        m = regex.search(f.name)
        if m:
            pngs_by_key[timestamp_key(float(m.group("ts")))] = f

    if stack is None:
        if not pngs_by_key:
            return None
        return DeviceDiff(scan, device_dir.name, 0, 0, len(pngs_by_key), 0, "no_stack")

    stack_ts = read_stack_timestamps(stack, labview_epoch=True)
    stack_by_key = {timestamp_key(float(ts)): i for i, ts in enumerate(stack_ts)}

    if not pngs_by_key:
        return DeviceDiff(
            scan, device_dir.name, 0, 0, 0, len(stack_by_key), "capture_only"
        )

    matched_keys = sorted(set(stack_by_key) & set(pngs_by_key))
    png_only = len(set(pngs_by_key) - set(stack_by_key))
    stack_only = len(set(stack_by_key) - set(pngs_by_key))
    identical = 0
    for key in matched_keys:
        frame = read_shot(ShotRef(stack, stack_by_key[key]))
        png = np.asarray(png_reader(pngs_by_key[key]))
        if frame.shape == png.shape and np.array_equal(frame, png.astype(frame.dtype)):
            identical += 1
        else:
            logger.error(
                "%s/%s: pixel mismatch at timestamp key %d",
                scan,
                device_dir.name,
                key,
            )
    verdict = "pass" if png_only == 0 and identical == len(matched_keys) else "mismatch"
    return DeviceDiff(
        scan,
        device_dir.name,
        len(matched_keys),
        identical,
        png_only,
        stack_only,
        verdict,
    )


def diff_scan(
    scan_dir: Path, *, png_reader: PngReader | None = None
) -> list[DeviceDiff]:
    """Diff every device folder of one scan directory."""
    results: list[DeviceDiff] = []
    for child in sorted(scan_dir.iterdir()):
        if not child.is_dir() or child.name == "analysis_status":
            continue
        result = diff_device_dir(child, png_reader=png_reader)
        if result is not None:
            results.append(result)
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI: diff scan folder(s); append the evidence log; exit 1 on mismatch."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scans", nargs="+", help="scan folder path(s) (ScanNNN dirs)")
    ap.add_argument(
        "--log", default=None, help="JSONL evidence log to append verdicts to"
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    failed = False
    records: list[dict] = []
    for scan in args.scans:
        for result in diff_scan(Path(scan)):
            row = {"checked_at": time.time(), **asdict(result)}
            records.append(row)
            print(
                f"{result.scan} {result.device}: {result.verdict} "
                f"(matched={result.matched} identical={result.pixel_identical} "
                f"png_only={result.png_only} stack_only={result.stack_only})"
            )
            if result.verdict == "mismatch":
                failed = True
    if args.log and records:
        log_path = Path(args.log).expanduser()
        with log_path.open("a") as f:
            for row in records:
                f.write(json.dumps(row) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
