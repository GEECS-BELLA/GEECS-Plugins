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
    find_stack_file,
    read_shot,
    read_stack_timestamps,
)
from geecs_data_utils.native_files import (
    filename_timestamp_regex,
    native_file_name_from_key,
    timestamp_key,
    timestamp_key_candidates,
)
from geecs_data_utils.scan_paths import ScanPaths

logger = logging.getLogger(__name__)

#: Extensions the portal renders from native per-shot files.
_RENDERABLE_EXTS = {"png", "tif", "tiff", "h5"}

#: Vendor-SDK formats: never rendered, reported as a path (Tier C).
_VENDOR_EXTS = {"himg", "has"}

#: Subfolders of a scan folder that are never image devices.
_NON_DEVICE_DIRS = {"analysis_status"}

#: Display normalization percentiles (robust to hot pixels).
_P_LO, _P_HI = 1.0, 99.7


@dataclass(frozen=True)
class ShotImage:
    """One resolved shot: rendered PNG bytes, or a tiered refusal."""

    kind: str  # "stack" | "native" | "vendor" | "unrenderable" | "missing"
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

    data = np.asarray(array, dtype=np.float32)
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


def _vendor_only(device_dir: Path) -> bool:
    """Whether the device saves only vendor-SDK files (Tier C, e.g. HASO).

    Needed because ``infer_device_ext`` only recognises its own accepted
    extension set — a ``.has``-only folder would otherwise read as a
    missing-png device instead of getting its Tier C path card.
    """
    saw_vendor = False
    try:
        entries = device_dir.iterdir()
    except OSError:
        return False
    for entry in entries:
        if not entry.is_file():
            continue
        ext = entry.suffix.lstrip(".").lower()
        if ext in _RENDERABLE_EXTS:
            return False
        if ext in _VENDOR_EXTS:
            saw_vendor = True
    return saw_vendor


def _native_file_for_timestamp(
    device_dir: Path, stem: str, ext: str, acq_timestamp: float
) -> Optional[Path]:
    """Exact native-file probe for one row timestamp — never a neighbour.

    Direct stat probes over `geecs_data_utils.native_files`'
    millisecond-canonical candidate names (the ±1 ms neighbours are
    ``%.3f`` rendering canonicalisation, not a tolerance — a missing
    shot must read as missing, never as the adjacent shot's image).
    Filenames carry the device's own ``acq_timestamp`` double verbatim,
    the same value the event row records, so no epoch conversion applies.
    """
    for key in timestamp_key_candidates(timestamp_key(acq_timestamp)):
        candidate = device_dir / native_file_name_from_key(stem, key, f".{ext}")
        if candidate.is_file():
            return candidate
    return None


def _ordinal_native_file(device_dir: Path, ext: str, shot: int) -> Optional[Path]:
    """No-metadata fallback: the *shot*-th timestamp-named file in order.

    Only used when the run's event table offers no ``acq_timestamp`` for
    the device — ordinal order can misalign on free-run orphan frames,
    which is why the timestamp join is always preferred.
    """
    pattern = filename_timestamp_regex(f".{ext}")
    stamped = []
    try:
        entries = list(device_dir.iterdir())
    except OSError:
        return None
    for entry in entries:
        match = pattern.search(entry.name)
        if match:
            stamped.append((float(match.group("ts")), entry))
    stamped.sort(key=lambda pair: pair[0])
    if len(stamped) >= shot:
        return stamped[shot - 1][1]
    return None


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
        The event row's device ``acq_timestamp`` double (the GEECS wire
        convention — the same value native filenames render).  Used
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
            if acq_timestamp is not None:
                # The canonical-millisecond join (ScanAnalysis parity):
                # /acq_timestamp is the stack's universal join key — the
                # stack stores Unix epoch, the event row the device's
                # LabVIEW-epoch double, so read converted.
                stamps = read_stack_timestamps(stack, labview_epoch=True)
                keys = {timestamp_key(float(s)): i for i, s in enumerate(stamps)}
                index = None
                for key in timestamp_key_candidates(timestamp_key(acq_timestamp)):
                    if key in keys:
                        index = keys[key]
                        break
                if index is None:
                    return ShotImage(
                        kind="missing",
                        path=stack,
                        reason="no stack frame for this shot",
                    )
            else:
                index = shot - 1
            frame = read_shot(stack, index)
            return ShotImage(kind="stack", png=to_display_png(frame), path=stack)
        # KeyError/TypeError: a malformed-but-schema-valid stack (missing
        # or mistyped /acq_timestamp) — same enumeration ScanAnalysis
        # defends against (PR #693 review); must 404, never 500.
        except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
            return ShotImage(kind="missing", path=stack, reason=f"stack: {exc}")

    if _vendor_only(device_dir):
        return ShotImage(kind="vendor", path=device_dir, reason="vendor SDK format")
    try:
        paths = ScanPaths(folder=scan_folder)
    except ValueError as exc:
        # A recorded folder that exists but doesn't follow the canonical
        # layout (dev/scratch runs) — degrade, never 500.
        return ShotImage(kind="missing", path=scan_folder, reason=f"layout: {exc}")
    ext = paths.infer_device_ext(device)
    native = paths.build_asset_path(shot=shot, device=device, ext=ext)
    if ext in _VENDOR_EXTS:
        return ShotImage(kind="vendor", path=native, reason="vendor SDK format")
    if ext not in _RENDERABLE_EXTS:
        return ShotImage(
            kind="unrenderable", path=native, reason=f"no renderer for .{ext}"
        )
    if not native.is_file():
        if acq_timestamp is not None:
            chosen = _native_file_for_timestamp(device_dir, device, ext, acq_timestamp)
        else:
            chosen = _ordinal_native_file(device_dir, ext, shot)
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
        ``("unrenderable", device_dir)`` for non-image native formats
        (trace/array files — findable, not rendered),
        ``("native", device_dir)`` otherwise; ``("missing", None)`` for
        an unknown device.
    """
    if device not in image_devices(scan_folder):
        return ("missing", None)
    device_dir = scan_folder / device
    stack = find_stack_file(device_dir)
    if stack is not None:
        return ("stack", stack)
    if _vendor_only(device_dir):
        return ("vendor", device_dir)
    try:
        ext = ScanPaths(folder=scan_folder).infer_device_ext(device)
    except ValueError:
        # Non-canonical folder layout — degrade to the missing card.
        return ("missing", None)
    if ext in _VENDOR_EXTS:
        return ("vendor", device_dir)
    if ext not in _RENDERABLE_EXTS:
        return ("unrenderable", device_dir)
    return ("native", device_dir)
