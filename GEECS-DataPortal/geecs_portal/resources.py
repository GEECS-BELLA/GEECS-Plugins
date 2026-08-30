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
from typing import NamedTuple, Optional

import numpy as np

from geecs_data_utils.io.images import DISPLAYABLE_IMAGE_EXTS
from geecs_data_utils.io.scan_stack import (
    find_stack_file,
    read_shot,
    read_shot_for_acq_timestamp,
)
from geecs_data_utils.native_files import (
    filename_timestamp_regex,
    probe_native_file,
)
from geecs_data_utils.scan_paths import (
    VENDOR_ONLY_EXTS,
    ScanPaths,
    infer_device_dir_ext,
)

logger = logging.getLogger(__name__)

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
    #: False when the resolution depended on a directory listing (the
    #: ordinal native fallback) — an SMB visibility blip can shift that
    #: join, so the result must never be long-cached by the browser.
    cacheable: bool = True


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
    kind = device_kind(scan_folder, device)
    if kind.kind == "missing":
        return ShotImage(kind="missing", reason=kind.reason or "unknown device")
    if shot < 1:
        return ShotImage(kind="missing", reason="bad shot")
    device_dir = scan_folder / device

    if kind.kind == "stack":
        stack = kind.path
        try:
            if acq_timestamp is not None:
                # The canonical-millisecond join, ONE file open — the
                # shared keep-first contract lives in
                # geecs_data_utils.io.scan_stack (ScanAnalysis parity):
                # the stack stores Unix epoch, the event row the device's
                # LabVIEW-epoch double, converted inside the helper.
                joined = read_shot_for_acq_timestamp(stack, acq_timestamp)
                if joined is None:
                    return ShotImage(
                        kind="missing",
                        path=stack,
                        reason="no stack frame for this shot",
                    )
                _, frame = joined
            else:
                frame = read_shot(stack, shot - 1)
            return ShotImage(kind="stack", png=to_display_png(frame), path=stack)
        # KeyError/TypeError: a malformed-but-schema-valid stack (missing
        # or mistyped /acq_timestamp) — same enumeration ScanAnalysis
        # defends against (PR #693 review); must 404, never 500.
        except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
            return ShotImage(kind="missing", path=stack, reason=f"stack: {exc}")

    if kind.kind == "vendor":
        return ShotImage(kind="vendor", path=kind.path, reason="vendor SDK format")
    if kind.kind == "unrenderable":
        return ShotImage(
            kind="unrenderable",
            path=kind.path,
            reason=f"no renderer for .{kind.ext}",
        )

    ext = kind.ext or "png"
    try:
        native = ScanPaths(folder=scan_folder).build_asset_path(
            shot=shot, device=device, ext=ext
        )
    except ValueError as exc:
        # A folder that exists but fails the canonical-layout validation
        # (dev/scratch runs, or a share blip between probes) — degrade,
        # never 500.
        return ShotImage(kind="missing", path=scan_folder, reason=f"layout: {exc}")
    cacheable = True
    if not native.is_file():
        if acq_timestamp is not None:
            # Exact stat probe from the shared naming contract — direct
            # stats bypass stale SMB listing caches, and a missing shot
            # must read as missing, never as a neighbouring shot's image.
            chosen = probe_native_file(device_dir, device, f".{ext}", acq_timestamp)
        else:
            chosen = _ordinal_native_file(device_dir, ext, shot)
            cacheable = False  # listing-order join: never long-cache
        if chosen is None:
            return ShotImage(kind="missing", path=native, reason="file not found")
        native = chosen
    try:
        from geecs_data_utils.io.images import read_imaq_image

        return ShotImage(
            kind="native",
            png=to_display_png(read_imaq_image(native)),
            path=native,
            cacheable=cacheable,
        )
    except Exception as exc:  # noqa: BLE001 — corrupt file must not 500
        return ShotImage(kind="missing", path=native, reason=f"read failed: {exc}")


class DeviceKind(NamedTuple):
    """One device's gallery tier, resolved by :func:`device_kind`."""

    kind: str  # "stack" | "native" | "vendor" | "unrenderable" | "missing"
    path: Optional[Path] = None
    ext: Optional[str] = None
    reason: str = ""


def device_kind(
    scan_folder: Path, device: str, devices: Optional[list[str]] = None
) -> DeviceKind:
    """THE tier ladder — one probe, no pixel reads, shared by every caller.

    :func:`load_shot_image` dispatches on this same probe, so the gallery
    UI's badge and the image endpoint can never disagree about a device's
    tier.  Tier vocabulary comes from the shared taxonomy
    (``scan_paths.VENDOR_ONLY_EXTS`` / ``io.images.DISPLAYABLE_IMAGE_EXTS``)
    — never a local extension set.

    Parameters
    ----------
    scan_folder : Path
        The run's existing scan folder.
    device : str
        A device subfolder name (validated against the folder — the
        path-traversal guard).
    devices : list of str, optional
        An already-computed :func:`image_devices` listing, to spare a
        second directory scan when the caller just listed the folder.

    Returns
    -------
    DeviceKind
        ``("stack", stack_path, None)`` when a capture stack exists,
        ``("vendor", device_dir, ext)`` for vendor-SDK formats (Tier C),
        ``("unrenderable", device_dir, ext)`` for non-image native
        formats (trace/array files — findable, not rendered),
        ``("native", device_dir, ext)`` otherwise;
        ``("missing", None, None)`` for an unknown device.
    """
    if device not in (devices if devices is not None else image_devices(scan_folder)):
        return DeviceKind("missing", reason="unknown device")
    device_dir = scan_folder / device
    stack = find_stack_file(device_dir)
    if stack is not None:
        return DeviceKind("stack", stack)
    # Module-level inference — no ScanPaths construction, so a vendor
    # device classifies correctly even inside a non-canonical dev/scratch
    # scan folder (layout validation belongs to the native path builder).
    ext = infer_device_dir_ext(device_dir)
    if ext in VENDOR_ONLY_EXTS:
        return DeviceKind("vendor", device_dir, ext)
    if ext not in DISPLAYABLE_IMAGE_EXTS:
        return DeviceKind("unrenderable", device_dir, ext)
    return DeviceKind("native", device_dir, ext)
