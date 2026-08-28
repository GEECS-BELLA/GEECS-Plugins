"""The dual-write diff: capture stacks vs native PNGs, per scan.

The Phase-6 evidence engine. For every device folder in a scan that holds
a capture stack, join the
stack's per-frame ``acq_timestamp``s against the native per-shot files'
filename timestamps — the analysis join's exact contract: canonical
millisecond keys plus its \u00b11 ms candidate tolerance — and pixel-compare every matched pair (IMAQ-decoded PNG vs stack
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
from geecs_data_utils.native_files import (
    filename_timestamp_regex,
    timestamp_key,
    timestamp_key_candidates,
)

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
    stack_by_key: dict[int, int] = {}
    for i, ts in enumerate(stack_ts):
        # keep-first on duplicates — parity with the analysis join
        stack_by_key.setdefault(timestamp_key(float(ts)), i)

    if not pngs_by_key:
        return DeviceDiff(
            scan, device_dir.name, 0, 0, 0, len(stack_by_key), "capture_only"
        )

    # The join mirrors the analysis join exactly: exact canonical-ms keys
    # first, then ±1 ms candidate pairing for the rounding-boundary class
    # (review of this PR Monte-Carlo'd the epoch/wire float round-trip:
    # without candidates, boundary keys become false mismatch pairs).
    pairs: list[tuple[int, int]] = []  # (png_key, stack_index)
    residual_stack = dict(stack_by_key)
    residual_png = dict(pngs_by_key)
    for key in sorted(set(stack_by_key) & set(pngs_by_key)):
        pairs.append((key, stack_by_key[key]))
        residual_stack.pop(key, None)
        residual_png.pop(key, None)
    for key in sorted(residual_png):
        for candidate in timestamp_key_candidates(key):
            if candidate in residual_stack:
                pairs.append((key, residual_stack.pop(candidate)))
                residual_png.pop(key)
                break
    png_only = len(residual_png)
    stack_only = len(residual_stack)
    identical = 0
    for png_key, stack_index in pairs:
        frame = read_shot(ShotRef(stack, stack_index))
        png = np.asarray(png_reader(pngs_by_key[png_key]))
        # No dtype cast: a cast can wrap out-of-range values and mask a
        # real difference; numpy's promotion compares values correctly.
        if frame.shape == png.shape and np.array_equal(frame, png):
            identical += 1
        else:
            logger.error(
                "%s/%s: pixel mismatch at timestamp key %d",
                scan,
                device_dir.name,
                png_key,
            )
    verdict = "pass" if png_only == 0 and identical == len(pairs) else "mismatch"
    return DeviceDiff(
        scan,
        device_dir.name,
        len(pairs),
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

    mismatch = False
    op_error = False
    log_file = Path(args.log).expanduser().open("a") if args.log else None
    try:
        for scan in args.scans:
            try:
                results = diff_scan(Path(scan))
            except OSError as exc:
                # Operational failure (missing path, unmounted share) is NOT
                # a mismatch — distinct exit code, sweep continues.
                logger.error("%s: unreadable (%s)", scan, exc)
                op_error = True
                continue
            for result in results:
                row = {"checked_at": time.time(), **asdict(result)}
                print(
                    f"{result.scan} {result.device}: {result.verdict} "
                    f"(matched={result.matched} "
                    f"identical={result.pixel_identical} "
                    f"png_only={result.png_only} "
                    f"stack_only={result.stack_only})"
                )
                if result.verdict == "mismatch":
                    mismatch = True
                if log_file is not None:
                    log_file.write(json.dumps(row) + "\n")
                    log_file.flush()
    finally:
        if log_file is not None:
            log_file.close()
    return 1 if mismatch else (2 if op_error else 0)


if __name__ == "__main__":
    sys.exit(main())
