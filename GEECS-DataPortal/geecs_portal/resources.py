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


@dataclass(frozen=True)
class ShotArray:
    """One resolved shot's raw pixels, or a tiered refusal.

    The array-level result :func:`load_shot_array` returns — consumers
    that combine shots (per-bin averaging) work on arrays and render
    once; :func:`load_shot_image` is the render-one-shot wrapper.
    """

    kind: str  # "stack" | "native" | "vendor" | "unrenderable" | "missing"
    array: Optional[np.ndarray] = None
    path: Optional[Path] = None
    reason: str = ""
    #: Same contract as :attr:`ShotImage.cacheable`.
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


def safe_cmap(cmap: Optional[str]) -> Optional[str]:
    """A known matplotlib colormap name, else None (value-degrade: grayscale)."""
    if not cmap:
        return None
    import matplotlib as mpl

    return str(cmap) if str(cmap) in mpl.colormaps else None


def window_percentiles(plo, phi) -> tuple[float, float]:
    """Public alias of :func:`_window_percentiles` (the rendered view shares it)."""
    return _window_percentiles(plo, phi)


def _window_percentiles(plo, phi) -> tuple[float, float]:
    """The display window's percentile pair — value-degrade semantics.

    Display values ride shared links and must never make a link fail
    (the ``display`` doctrine): anything that isn't a sane
    ``0 <= lo < hi <= 100`` pair falls back to the defaults.
    """
    try:
        # Per-field defaulting: a one-sided override ({"plo": 5}, the
        # popup's stores-defaults-as-absent shape) must apply, not
        # silently no-op the pair.
        lo = _P_LO if plo is None else float(plo)
        hi = _P_HI if phi is None else float(phi)
    except (TypeError, ValueError):
        return _P_LO, _P_HI
    if not (0.0 <= lo < hi <= 100.0):
        return _P_LO, _P_HI
    return lo, hi


def to_display_png(
    array: np.ndarray,
    *,
    cmap: Optional[str] = None,
    plo: Optional[float] = None,
    phi: Optional[float] = None,
) -> bytes:
    """Render a 2D (or RGB) array to 8-bit PNG bytes with robust scaling.

    Percentile windowing (default 1–99.7, overridable per request via
    the ``display`` state) maps the camera's dynamic range into display
    range — the raw 16-bit files render near-black in a browser
    otherwise.  A flat image renders at the scale's floor rather than
    dividing by zero (black in grayscale; the colormap's lowest color
    under a ``cmap``).  ``cmap`` names a matplotlib colormap applied to the windowed
    image (2D input only — an RGB input keeps its own colors); an
    unknown name degrades to grayscale, same value-degrade rule as the
    window (display values ride shared links and must never fail one).

    Parameters
    ----------
    array : numpy.ndarray
        The image data.
    cmap : str, optional
        Matplotlib colormap name (``viridis``, ``magma``, …); ``None``
        renders grayscale.
    plo, phi : float, optional
        Percentile window overrides (defaults 1 / 99.7).

    Returns
    -------
    bytes
        PNG-encoded 8-bit image (RGB when a colormap applied).
    """
    from PIL import Image

    data = np.asarray(array, dtype=np.float32)
    lo_pct, hi_pct = (
        _window_percentiles(plo, phi)
        if (plo is not None or phi is not None)
        else (_P_LO, _P_HI)
    )
    finite = data[np.isfinite(data)]
    if finite.size:
        lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    else:
        lo, hi = 0.0, 0.0
    if hi <= lo:
        norm = np.zeros(data.shape, dtype=np.float32)
    else:
        norm = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    colormap = None
    if cmap and data.ndim == 2:
        import matplotlib as mpl

        try:
            colormap = mpl.colormaps[str(cmap)]
        except KeyError:
            colormap = None  # unknown name: grayscale, never a failure
    if colormap is not None:
        # bytes=True skips the H×W×4 float64 RGBA intermediate — same
        # output bytes at ~1/8 the transient memory on full frames.
        scaled = colormap(norm, bytes=True)[..., :3]
    else:
        scaled = (norm * 255).astype(np.uint8)
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


def _stack_shot_from_memory(
    index_map: dict, frames, shot: int, acq_timestamp: Optional[float], path=None
) -> ShotArray:
    """Resolve one shot's pixels from cached stack data (no filesystem access)."""
    from geecs_data_utils.io.scan_stack import frame_index_for_timestamp

    if acq_timestamp is not None:
        index = frame_index_for_timestamp(index_map, acq_timestamp)
        if index is None:
            return ShotArray(
                kind="missing", path=path, reason="no stack frame for this shot"
            )
    else:
        index = shot - 1
        if not 0 <= index < len(frames):
            return ShotArray(
                kind="missing",
                path=path,
                reason=f"stack: shot {shot} outside {len(frames)} frames",
            )
    return ShotArray(kind="stack", array=frames[index], path=path)


def load_shot_array(
    scan_folder: Path,
    device: str,
    shot: int,
    acq_timestamp: Optional[float] = None,
    data_cache=None,
    cache_key: Optional[tuple[str, str]] = None,
) -> ShotArray:
    """Resolve one device shot's pixel array from an existing scan folder.

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

    data_cache : ShotDataCache, optional
        The per-``(uid, device)`` pixel cache (``geecs_portal.cache``).
        Pass ONLY for completed runs — their data is immutable.
    cache_key : tuple of (str, str), optional
        The cache key (``uid``, ``device``); required with *data_cache*.

    Returns
    -------
    ShotArray
        The decoded pixels, or the tiered refusal (vendor path /
        missing reason).
    """
    caching = data_cache is not None and cache_key is not None
    if caching and shot >= 1:
        # Fast paths: a cached entry was created after full validation,
        # so a hit serves with zero filesystem access.
        stack_hit = data_cache.stack_entry(cache_key)
        if stack_hit is not None:
            return _stack_shot_from_memory(*stack_hit, shot, acq_timestamp)
        cached = data_cache.native_shot(cache_key, shot)
        if cached is not None:
            return ShotArray(kind="native", array=cached)

    kind = device_kind(scan_folder, device)
    if kind.kind == "missing":
        return ShotArray(kind="missing", reason=kind.reason or "unknown device")
    if shot < 1:
        return ShotArray(kind="missing", reason="bad shot")
    device_dir = scan_folder / device

    if kind.kind == "stack":
        stack = kind.path
        try:
            if caching:
                # Eager within-scan load (owner doctrine): the whole
                # frames array in one open; navigation never reopens.
                # None = not admissible (un-finalized tail-race window,
                # or over the per-entry cap) — serve per shot from disk.
                admitted = data_cache.stack_frames(cache_key, stack)
                if admitted is not None:
                    index_map, frames = admitted
                    return _stack_shot_from_memory(
                        index_map, frames, shot, acq_timestamp, path=stack
                    )
            if acq_timestamp is not None:
                # The canonical-millisecond join, ONE file open — the
                # shared keep-first contract lives in
                # geecs_data_utils.io.scan_stack (ScanAnalysis parity):
                # the stack stores Unix epoch, the event row the device's
                # LabVIEW-epoch double, converted inside the helper.
                joined = read_shot_for_acq_timestamp(stack, acq_timestamp)
                if joined is None:
                    return ShotArray(
                        kind="missing",
                        path=stack,
                        reason="no stack frame for this shot",
                    )
                _, frame = joined
            else:
                frame = read_shot(stack, shot - 1)
            return ShotArray(kind="stack", array=frame, path=stack)
        # KeyError/TypeError: a malformed-but-schema-valid stack (missing
        # or mistyped /acq_timestamp) — same enumeration ScanAnalysis
        # defends against (PR #693 review); must 404, never 500.
        except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
            return ShotArray(kind="missing", path=stack, reason=f"stack: {exc}")

    if kind.kind == "vendor":
        return ShotArray(kind="vendor", path=kind.path, reason="vendor SDK format")
    if kind.kind == "unrenderable":
        return ShotArray(
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
        return ShotArray(kind="missing", path=scan_folder, reason=f"layout: {exc}")
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
            return ShotArray(kind="missing", path=native, reason="file not found")
        native = chosen
    try:
        from geecs_data_utils.io.images import read_imaq_image

        array = read_imaq_image(native)
        if caching and cacheable:
            # Never cache an ordinal (listing-order) resolution — the
            # same rule as the no-long-cache header.
            data_cache.store_native_shot(cache_key, shot, array)
        return ShotArray(
            kind="native",
            array=array,
            path=native,
            cacheable=cacheable,
        )
    except Exception as exc:  # noqa: BLE001 — corrupt file must not 500
        return ShotArray(kind="missing", path=native, reason=f"read failed: {exc}")


def load_shot_image(
    scan_folder: Path,
    device: str,
    shot: int,
    acq_timestamp: Optional[float] = None,
    data_cache=None,
    cache_key: Optional[tuple[str, str]] = None,
    cmap: Optional[str] = None,
    plo: Optional[float] = None,
    phi: Optional[float] = None,
) -> ShotImage:
    """Resolve and render one device shot — :func:`load_shot_array` + PNG.

    Same parameters and tier ladder as :func:`load_shot_array` (which
    carries the full docs); this wrapper only adds the display
    rendering (``cmap``/``plo``/``phi`` per :func:`to_display_png`),
    so single-shot serving and per-bin averaging share one resolution
    path.
    """
    resolved = load_shot_array(
        scan_folder,
        device,
        shot,
        acq_timestamp=acq_timestamp,
        data_cache=data_cache,
        cache_key=cache_key,
    )
    png = None
    if resolved.array is not None:
        try:
            png = to_display_png(resolved.array, cmap=cmap, plo=plo, phi=phi)
        except Exception as exc:  # noqa: BLE001 — unrenderable shape must not 500
            # A readable-but-unrenderable array (e.g. a stacked .npy or
            # an odd-shaped h5 in a dev/scratch folder) degrades to the
            # missing card, same as a corrupt file always has.
            return ShotImage(
                kind="missing",
                path=resolved.path,
                reason=f"render failed: {exc}",
                cacheable=resolved.cacheable,
            )
    return ShotImage(
        kind=resolved.kind,
        png=png,
        path=resolved.path,
        reason=resolved.reason,
        cacheable=resolved.cacheable,
    )


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
