"""Resource viewer: (scan folder, device, shot) → a browser-displayable image.

The portal's phase-4 join over the two image stores (the scope doc's
tiering): a device's shot is served from its capture-daemon HDF5 frame
stack when one exists (Tier A, `geecs_data_utils.io.scan_stack`), else
from its native per-shot file (Tier B, the GEECS filename convention via
`ScanPaths.build_asset_path` — never re-derived here), while
vendor-SDK-only formats (Tier C, e.g. HASO ``.himg``) are reported as a
path instead of rendered.

Strictly read-only: resolution + reads only; nothing on the scans path
is ever created (repo scan-folder invariant).  Every lookup validates
the device name against the scan folder's actual subfolders, so URL
input can never traverse outside it.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from geecs_data_utils.io.scan_stack import (
    LABVIEW_EPOCH_OFFSET,
    find_stack_file,
    read_shot,
)
from geecs_data_utils.scan_paths import ScanPaths

logger = logging.getLogger(__name__)

#: Extensions the portal renders from native per-shot files.
_RENDERABLE_EXTS = {"png", "tif", "tiff", "h5", "npy"}

#: Vendor-SDK formats: never rendered, reported as a path (Tier C).
_VENDOR_EXTS = {"himg", "has"}

#: Subfolders of a scan folder that are never image devices.
_NON_DEVICE_DIRS = {"analysis_status"}

#: Display normalization percentiles (robust to hot pixels).
_P_LO, _P_HI = 1.0, 99.7

#: Max |file stamp − expected stamp| for a timestamp-named file match (s).
_TS_MATCH_TOLERANCE_S = 1.0


@dataclass(frozen=True)
class ShotImage:
    """One resolved shot: rendered PNG bytes, or a tiered refusal."""

    kind: str  # "stack" | "native" | "vendor" | "missing"
    png: Optional[bytes] = None
    path: Optional[Path] = None
    reason: str = ""


def image_devices(scan_folder: Path) -> list[str]:
    """Device subfolders of *scan_folder* (read-only listing, sorted).

    Parameters
    ----------
    scan_folder : Path
        An existing ``scans/ScanNNN`` folder.

    Returns
    -------
    list of str
        Subdirectory names that can hold per-shot device files.
    """
    try:
        return sorted(
            entry.name
            for entry in scan_folder.iterdir()
            if entry.is_dir() and entry.name not in _NON_DEVICE_DIRS
        )
    except OSError as exc:
        logger.warning("cannot list %s: %s", scan_folder, exc)
        return []


def to_display_png(array: np.ndarray) -> bytes:
    """Render a 2D (or RGB) array to 8-bit PNG bytes with robust scaling.

    Percentile windowing (1–99.7) maps the camera's dynamic range into
    display range — the raw 16-bit files render near-black in a browser
    otherwise.  A flat image renders black rather than dividing by zero.

    Parameters
    ----------
    array : numpy.ndarray
        The image data.

    Returns
    -------
    bytes
        PNG-encoded 8-bit image.
    """
    from PIL import Image

    data = np.asarray(array, dtype=np.float64)
    finite = data[np.isfinite(data)]
    if finite.size:
        lo, hi = np.percentile(finite, [_P_LO, _P_HI])
    else:
        lo, hi = 0.0, 0.0
    if hi <= lo:
        scaled = np.zeros(data.shape, dtype=np.uint8)
    else:
        scaled = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
        scaled = (scaled * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(scaled).save(buffer, format="PNG")
    return buffer.getvalue()


def _timestamped_files(device_dir: Path, device: str, ext: str) -> list:
    """Bluesky-native files ``<device>_<labview_seconds>.<ext>``, time order.

    The native saver names files by LabVIEW acquisition timestamp rather
    than the legacy ``ScanNNN_device_shot`` convention (the still-open
    filename-compatibility question recorded in
    ``GeecsBluesky/TILED_SETUP.md``) — production Bluesky scans use this
    form today.

    Returns
    -------
    list of (float, Path)
        ``(labview_seconds, path)`` sorted by timestamp.
    """
    prefix = f"{device}_"
    out = []
    try:
        entries = sorted(device_dir.iterdir())
    except OSError:
        return []
    for entry in entries:
        name = entry.name
        if not (name.startswith(prefix) and name.endswith(f".{ext}")):
            continue
        stamp_text = name[len(prefix) : -(len(ext) + 1)]
        try:
            out.append((float(stamp_text), entry))
        except ValueError:
            continue
    out.sort(key=lambda pair: pair[0])
    return out


def load_shot_image(
    scan_folder: Path,
    device: str,
    shot: int,
    acq_timestamp: Optional[float] = None,
) -> ShotImage:
    """Resolve and render one device shot from an existing scan folder.

    Parameters
    ----------
    scan_folder : Path
        The run's existing ``scans/ScanNNN`` folder.
    device : str
        Device subfolder name; must be one of :func:`image_devices`
        (path-traversal guard).
    shot : int
        1-based shot number (the GEECS filename convention; stack frame
        index is ``shot - 1`` — dual-written stacks follow the same
        trigger sequence).
    acq_timestamp : float, optional
        The event row's device ``acq_timestamp`` (LabVIEW or Unix
        epoch — both accepted).  Used
        to join Bluesky-native timestamp-named files exactly; without
        it those fall back to ordinal order (orphan between-step frames
        can shift the ordinal join — pass the timestamp when the event
        table is at hand).

    Returns
    -------
    ShotImage
        Rendered PNG bytes, or the tiered refusal (vendor path /
        missing reason).
    """
    if device not in image_devices(scan_folder) or shot < 1:
        return ShotImage(kind="missing", reason="unknown device or bad shot")
    device_dir = scan_folder / device

    stack = find_stack_file(device_dir)
    if stack is not None:
        try:
            frame = read_shot(stack, shot - 1)
            return ShotImage(kind="stack", png=to_display_png(frame), path=stack)
        except (IndexError, OSError, ValueError) as exc:
            return ShotImage(kind="missing", path=stack, reason=f"stack: {exc}")

    paths = ScanPaths(folder=scan_folder)
    ext = paths.infer_device_ext(device)
    native = paths.build_asset_path(shot=shot, device=device, ext=ext)
    if ext in _VENDOR_EXTS:
        return ShotImage(kind="vendor", path=native, reason="vendor SDK format")
    if ext not in _RENDERABLE_EXTS:
        return ShotImage(kind="vendor", path=native, reason=f"no renderer for .{ext}")
    if not native.is_file():
        stamped = _timestamped_files(device_dir, device, ext)
        chosen = None
        if stamped and acq_timestamp:
            # Event rows carry the GEECS wire convention (LabVIEW epoch,
            # matching the filenames directly — verified live); Unix-epoch
            # sources (the capture stack) need the offset. Accept either.
            expected = (acq_timestamp, acq_timestamp + LABVIEW_EPOCH_OFFSET)

            def _distance(pair):
                return min(abs(pair[0] - value) for value in expected)

            best = min(stamped, key=_distance)
            if _distance(best) <= _TS_MATCH_TOLERANCE_S:
                chosen = best[1]
        elif stamped and len(stamped) >= shot:
            chosen = stamped[shot - 1][1]
        if chosen is None:
            return ShotImage(kind="missing", path=native, reason="file not found")
        native = chosen
    try:
        from geecs_data_utils.io.images import read_imaq_image

        return ShotImage(
            kind="native", png=to_display_png(read_imaq_image(native)), path=native
        )
    except Exception as exc:  # noqa: BLE001 — corrupt file must not 500
        return ShotImage(kind="missing", path=native, reason=f"read failed: {exc}")


def device_kind(scan_folder: Path, device: str) -> tuple[str, Optional[Path]]:
    """Cheap tier probe — no pixel reads — for rendering the gallery UI.

    Parameters
    ----------
    scan_folder : Path
        The run's existing scan folder.
    device : str
        A device subfolder name (validated against the folder).

    Returns
    -------
    tuple of (str, Path or None)
        ``("stack", stack_path)`` when a capture stack exists,
        ``("vendor", device_dir)`` for vendor-SDK formats,
        ``("native", device_dir)`` otherwise; ``("missing", None)`` for
        an unknown device.
    """
    if device not in image_devices(scan_folder):
        return ("missing", None)
    device_dir = scan_folder / device
    stack = find_stack_file(device_dir)
    if stack is not None:
        return ("stack", stack)
    ext = ScanPaths(folder=scan_folder).infer_device_ext(device)
    if ext in _VENDOR_EXTS or ext not in _RENDERABLE_EXTS:
        return ("vendor", device_dir)
    return ("native", device_dir)
