"""Compute a scan background — a per-pixel statistic over a scan's frames — for the core.

The legacy wrapper's ``scan.background_source`` (and a recipe's ``from_scan``
frame input): the mean of another scan's frames (a dark scan), or the median
or a percentile of the scan's own frames. The analysis core never reads a
scan, so the host computes the frame here before the run and binds it for the
``background_frame`` step.

The statistic is exact — no subsampling, no approximation — and memory is
bounded however long the scan is:

- **mean** is a float64 running sum over the frames in order, which is the
  order numpy reduces a stack along its first axis, so it equals the legacy
  ``np.mean(np.stack(frames), axis=0)`` bit for bit.
- **median / percentile** need every value of a pixel at once. The frames
  are written once, at their native dtype, into a scratch array on disk
  (in memory when small), then reduced a horizontal strip of rows at a
  time: ``np.percentile(stack[:, r0:r1, :], q, axis=0)``. A pixel's value
  depends only on its own column of samples, so the strips reproduce the
  whole-stack result exactly — memory is frames × one strip.

The result is cached in the analysis tree beside that scan's other outputs,
named by the statistic, so a dark scan is averaged once for every scan that
uses it; a missing scan folder is an error, never created.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from geecs_analysis.compat.v2 import ScanBackground
from geecs_data_utils.io.images import read_imaq_image
from geecs_data_utils.io.scan_stack import (
    FRAMES_DATASET,
    find_stack_file,
    open_stack,
    read_frame,
)

logger = logging.getLogger(__name__)

__all__ = [
    "STRIP_BUDGET_BYTES",
    "background_cache_path",
    "resolve_scan_background",
    "scan_statistic",
]

#: The float64 working set one strip may use (frames × rows × width × 8 B);
#: the on-disk scratch copy is used once the whole stack would exceed it.
STRIP_BUDGET_BYTES = 256 * 1024 * 1024


def _reduce(block: np.ndarray, statistic: str, percentile: Optional[float]):
    values = block.astype(np.float64)
    if statistic == "median":
        return np.median(values, axis=0)
    return np.percentile(values, percentile, axis=0)


def scan_statistic(
    loaders: Sequence[Callable[[], np.ndarray]],
    statistic: str,
    percentile: Optional[float] = None,
    *,
    budget_bytes: int = STRIP_BUDGET_BYTES,
    scratch_dir: Optional[Path] = None,
) -> np.ndarray:
    """The per-pixel ``statistic`` over the frames the ``loaders`` return, in order.

    A loader that raises is skipped with a warning (the legacy aggregation
    skipped unreadable files); frames of different shapes are an error.
    ``percentile`` is in [0, 100] and required for ``"percentile"``.
    """
    if statistic not in {"mean", "median", "percentile"}:
        raise ValueError(f"Unknown scan statistic {statistic!r}")
    if statistic == "percentile" and (
        percentile is None or not 0.0 <= percentile <= 100.0
    ):
        raise ValueError(f"percentile must be in [0, 100]; got {percentile}")

    def frames():
        for load in loaders:
            try:
                frame = np.asarray(load())
            except Exception as exc:  # noqa: BLE001 — legacy skips unreadable files
                logger.warning("Skipping a background source frame: %s", exc)
                continue
            if frame.ndim != 2:
                raise ValueError(f"A background frame must be 2D, got {frame.shape}")
            yield frame

    iterator = frames()
    first = next(iterator, None)
    if first is None:
        raise ValueError("No background source frame could be read")
    shape = first.shape

    if statistic == "mean":
        total = first.astype(np.float64)
        count = 1
        for frame in iterator:
            if frame.shape != shape:
                raise ValueError(
                    f"Background frames differ in shape: {frame.shape} != {shape}"
                )
            total += frame
            count += 1
        return total / count

    rows_per_strip = max(1, budget_bytes // max(1, len(loaders) * shape[1] * 8))
    in_memory = len(loaders) * shape[0] * shape[1] * 8 <= budget_bytes
    with tempfile.TemporaryDirectory(dir=scratch_dir) as scratch:
        if in_memory:
            stack = np.empty((len(loaders), *shape), dtype=first.dtype)
        else:
            stack = np.lib.format.open_memmap(
                Path(scratch) / "stack.npy",
                mode="w+",
                dtype=first.dtype,
                shape=(len(loaders), *shape),
            )
        stack[0] = first
        count = 1
        for frame in iterator:
            if frame.shape != shape:
                raise ValueError(
                    f"Background frames differ in shape: {frame.shape} != {shape}"
                )
            stack[count] = frame
            count += 1
        used = stack[:count]
        result = np.empty(shape, dtype=np.float64)
        for r0 in range(0, shape[0], rows_per_strip):
            r1 = min(shape[0], r0 + rows_per_strip)
            result[r0:r1] = _reduce(used[:, r0:r1, :], statistic, percentile)
        del used, stack
    return result


def _cache_name(device: str, request: ScanBackground) -> str:
    if request.statistic == "percentile":
        tag = f"p{request.percentile:g}"
    else:
        tag = {"mean": "avg", "median": "median"}[request.statistic]
    return f"{device}_background_{tag}.npy"


def background_cache_path(
    request: ScanBackground, data_dir: Path, device: str
) -> tuple[Path, Path]:
    """``(the source scan's device folder, the cached frame's path)``.

    The source scan is the analyzed one (``scan_number`` unset) or that
    scan of the same day; the cache sits in its analysis folder under the
    device — for a dark scan's mean, the legacy wrapper's own file
    (``<device>_background_avg.npy``), so both routes share one compute.
    """
    scan_folder = Path(data_dir).parent
    if request.scan_number is not None:
        scan_folder = scan_folder.parent / f"Scan{request.scan_number:03d}"
    analysis = scan_folder.parent.parent / "analysis" / scan_folder.name
    return scan_folder / Path(data_dir).name, analysis / device / _cache_name(
        device, request
    )


def _frame_loaders(
    device_dir: Path, file_tail: str, prefer_stack: bool
) -> list[Callable[[], np.ndarray]]:
    """Every frame of the device folder: the stack's frames, or its files in order."""
    stack = find_stack_file(device_dir) if prefer_stack else None
    if stack is not None:
        with open_stack(stack) as handle:
            count = handle[FRAMES_DATASET].shape[0]

        def frame(index: int) -> Callable[[], np.ndarray]:
            def load() -> np.ndarray:
                with open_stack(stack) as handle:
                    return read_frame(handle, index)

            return load

        return [frame(i) for i in range(count)]
    files = sorted(device_dir.glob(f"*{file_tail}"))
    return [lambda f=f: read_imaq_image(f) for f in files]


def resolve_scan_background(
    request: ScanBackground,
    *,
    data_dir: Path,
    device: str,
    file_tail: Optional[str],
    prefer_stack: bool,
    compute: bool = True,
) -> np.ndarray:
    """The requested scan background as a float64 frame, computed once and cached.

    ``compute=False`` (a preview) only reads an existing cache and raises
    ``LookupError`` otherwise: a per-request view never reads a whole scan
    or writes a file. A missing source scan or device folder raises
    ``FileNotFoundError``; nothing is ever created under ``scans/``.
    """
    device_dir, cache = background_cache_path(request, data_dir, device)
    if cache.is_file():
        logger.info("Using cached scan background: %s", cache)
        return np.load(cache).astype(np.float64)
    if not compute:
        raise LookupError(f"Scan background not computed yet: {cache}")
    if not device_dir.parent.is_dir():
        raise FileNotFoundError(
            f"Background source scan does not exist: {device_dir.parent}"
        )
    if not device_dir.is_dir():
        raise FileNotFoundError(
            f"Background source device folder does not exist: {device_dir}"
        )
    loaders = _frame_loaders(device_dir, file_tail or ".png", prefer_stack)
    if not loaders:
        raise FileNotFoundError(
            f"No background source frames in {device_dir} matching *{file_tail or '.png'}"
        )
    background = scan_statistic(loaders, request.statistic, request.percentile)
    # The analysis tree may be created; the scan folder above was checked.
    cache.parent.mkdir(parents=True, exist_ok=True)
    partial = cache.with_suffix(".partial.npy")
    np.save(partial, background)
    partial.replace(cache)
    logger.info(
        "Saved scan background to %s (%s of %d frames)",
        cache,
        request.statistic,
        len(loaders),
    )
    return background
