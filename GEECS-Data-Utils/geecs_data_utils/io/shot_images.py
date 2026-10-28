"""(scan folder, device, shot) → that shot's pixels, one shot at a time.

The join over the two image stores, as a ladder of tiers a device sits on:
a device's shot is read from the PVA gateway file plugin's HDF5 capture
stack when one exists (``stack``, :mod:`geecs_data_utils.io.scan_stack`,
joined by ``acq_timestamp``), else from its native per-shot file
(``native``, the GEECS filename convention via
:meth:`~geecs_data_utils.scan_paths.ScanPaths.build_asset_path` and
:mod:`geecs_data_utils.native_files` — never re-derived here), while
vendor-SDK formats (``vendor``, e.g. HASO ``.himg``) and non-image native
files (``unrenderable``, e.g. ``.tdms``) are reported as a path instead of
read. :func:`device_kind` is that ladder; every loader dispatches on it.

Not every stack is pixels: the same file plugin captures a device's ARRAY
variables (scope traces, spectra) through the same layout, so ``stack``
covers both. :func:`stack_content` says which a device's stack holds, and
:func:`load_shot_trace` reads one shot of an array stack as x-vs-y. A viewer
checks :func:`stack_content` before rendering pixels, or it draws a
``(16384, 2)`` lineout as a two-pixel-wide image.

The event row decides which file is the shot's: :func:`row_acq_timestamp`
reads the device's ``acq_timestamp`` (and its ``valid`` flag) from a scan's
table, :func:`row_file_shot` the shot number in its ``ScanNNN_<device>_SSS``
file names. A device that missed the shot is refused — **a missing shot
never shows a neighbour's image**.

Lazy per shot: nothing here lists or reads a whole scan's images; a caller
holding a :class:`ShotPixelCache` can keep a completed scan's pixels.

Strictly read-only: nothing on the scans path is ever created (the repo's
scan-folder invariant). Every lookup validates the device name against the
scan folder's real subfolders, so a device name from a URL can never
traverse outside it.

Moved from the Data Portal's ``resources.py`` so every viewer resolves a
shot the same way.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Optional, Protocol

import numpy as np

from geecs_data_utils import tiled_schema
from geecs_data_utils.io.display_png import to_display_png
from geecs_data_utils.io.array1d import (
    Data1DConfig,
    Data1DResult,
    Data1DType,
    read_1d_data,
)
from geecs_data_utils.io.images import DISPLAYABLE_IMAGE_EXTS, read_imaq_image
from geecs_data_utils.io.scan_stack import (
    ShotRef,
    find_stack_file,
    frame_index_for_acq_timestamp,
    frame_index_for_timestamp,
    read_shot,
    read_shot_for_acq_timestamp,
    stack_content_kind,
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

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

#: Subfolders of a scan folder that are never devices.
_NON_DEVICE_DIRS = {"analysis_status"}


@dataclass(frozen=True)
class ShotArray:
    """One resolved shot's raw pixels, or a tiered refusal.

    Consumers that combine shots (per-bin averaging) work on arrays and
    render once; :func:`load_shot_image` is the render-one-shot wrapper.
    """

    kind: str  # "stack" | "native" | "vendor" | "unrenderable" | "missing"
    array: Optional[np.ndarray] = None
    path: Optional[Path] = None
    reason: str = ""
    #: False when the resolution depended on a directory listing (the
    #: ordinal native fallback) — an SMB visibility blip can shift that
    #: join, so the result must never be long-cached by a browser.
    cacheable: bool = True


@dataclass(frozen=True)
class ShotImage:
    """One resolved shot rendered to PNG bytes, or a tiered refusal."""

    kind: str  # "stack" | "native" | "vendor" | "unrenderable" | "missing"
    png: Optional[bytes] = None
    path: Optional[Path] = None
    reason: str = ""
    #: Same contract as :attr:`ShotArray.cacheable`.
    cacheable: bool = True


class DeviceKind(NamedTuple):
    """One device's tier, resolved by :func:`device_kind`."""

    kind: str  # "stack" | "native" | "vendor" | "unrenderable" | "missing"
    path: Optional[Path] = None
    ext: Optional[str] = None
    reason: str = ""


class ShotPixelCache(Protocol):
    """A per-``(scan, device)`` pixel cache :func:`load_shot_array` can ride.

    Pass one ONLY for completed scans — their data is immutable. The Data
    Portal's ``ShotDataCache`` is the one implementation.
    """

    def stack_entry(self, key: tuple[str, str]) -> Optional[tuple[dict, np.ndarray]]:
        """The cached ``(index_map, frames)`` for *key*, or ``None``."""

    def stack_frames(
        self, key: tuple[str, str], stack_path: Path
    ) -> Optional[tuple[dict, np.ndarray]]:
        """Load and keep the stack's ``(index_map, frames)``; ``None`` = not admitted."""

    def native_shot(self, key: tuple[str, str], shot: int) -> Optional[np.ndarray]:
        """A cached decoded native shot, or ``None``."""

    def store_native_shot(
        self, key: tuple[str, str], shot: int, array: np.ndarray
    ) -> bool:
        """Keep one decoded native shot."""


def image_devices(scan_folder: Path) -> list[str]:
    """Device subfolders of *scan_folder* (read-only listing, sorted).

    Parameters
    ----------
    scan_folder : Path
        An existing ``scans/ScanNNN`` folder.

    Returns
    -------
    list of str
        Subdirectory names that can hold per-shot device files; empty when
        the folder cannot be listed.
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


def device_kind(
    scan_folder: Path, device: str, devices: Optional[list[str]] = None
) -> DeviceKind:
    """THE tier ladder — one probe, no pixel reads, shared by every caller.

    :func:`load_shot_array` dispatches on this same probe, so a viewer's
    device badge and its image can never disagree about a device's tier.
    Tier vocabulary comes from the shared taxonomy
    (``scan_paths.VENDOR_ONLY_EXTS`` / ``io.images.DISPLAYABLE_IMAGE_EXTS``).

    Parameters
    ----------
    scan_folder : Path
        The scan's existing folder.
    device : str
        A device subfolder name (validated against the folder — the
        path-traversal guard).
    devices : list of str, optional
        An already-computed :func:`image_devices` listing, to spare a
        second directory scan.

    Returns
    -------
    DeviceKind
        ``("stack", stack_path)`` when a capture stack exists,
        ``("vendor", device_dir, ext)`` for vendor-SDK formats,
        ``("unrenderable", device_dir, ext)`` for non-image native formats,
        ``("native", device_dir, ext)`` otherwise; ``("missing", …)`` for
        an unknown device.
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


def row_acq_timestamp(
    frame: Optional[pd.DataFrame], device: str, shot: int
) -> tuple[Optional[float], bool]:
    """The scan table's ``acq_timestamp`` for *device* at 1-based *shot*.

    Column matching goes through
    :func:`~geecs_data_utils.tiled_schema.device_acq_timestamp_column` and
    :func:`~geecs_data_utils.tiled_schema.device_valid_column`.

    Parameters
    ----------
    frame : pandas.DataFrame or None
        The scan's per-shot table (event rows or s-file), one row per shot.
    device : str
        The device name.
    shot : int
        1-based shot number (the row).

    Returns
    -------
    tuple of (float or None, bool)
        ``(value, column_present)``. No column (or no such row) →
        ``(None, False)``, and a loader may fall back to file order.
        Column present but the row invalid (NaN or non-positive: the
        device missed this shot; or its ``valid`` companion false: the
        frame belongs to a different physical shot) → ``(None, True)`` —
        the caller must refuse rather than show a neighbouring shot.
    """
    if frame is None or shot < 1 or shot > len(frame):
        return (None, False)
    columns = [str(c) for c in frame.columns]
    column = tiled_schema.device_acq_timestamp_column(columns, device)
    if column is None:
        return (None, False)
    valid = tiled_schema.device_valid_column(columns, device)
    if valid is not None:
        try:
            if not bool(frame[valid].iloc[shot - 1]):
                return (None, True)
        except (TypeError, ValueError):  # pd.NA: unknown, not false
            pass
    try:
        value = float(frame[column].iloc[shot - 1])
    except (TypeError, ValueError):
        return (None, True)
    if not math.isfinite(value) or value <= 0:
        return (None, True)
    return (value, True)


def row_file_shot(frame: Optional[pd.DataFrame], shot: int) -> Optional[int]:
    """The number in the row's shot-numbered file names, when it is not *shot*.

    A scan-mode run's devices save ``ScanNNN_<device>_SSS`` under the shot
    the run announced (:func:`~geecs_data_utils.tiled_schema.announced_shots`),
    which is not always the row index.

    Parameters
    ----------
    frame : pandas.DataFrame or None
        The scan's per-shot table.
    shot : int
        1-based shot number (the row).

    Returns
    -------
    int or None
        ``None`` for every run that announces no shot numbers (then the
        file number is *shot*). A scan-mode row without a usable value
        gives ``0``, a number no file carries, so the native tier misses
        rather than serving the row index's file.
    """
    if frame is None or shot < 1 or shot > len(frame):
        return None
    numbers = tiled_schema.announced_shots(frame)
    if numbers is None:
        return None
    number = numbers.iloc[shot - 1]
    return 0 if number != number else int(number)


def _ordinal_native_file(device_dir: Path, ext: str, shot: int) -> Optional[Path]:
    """No-metadata fallback: the *shot*-th timestamp-named file in order.

    Only used when the scan table offers no ``acq_timestamp`` for the
    device — ordinal order can misalign on free-run orphan frames, which is
    why the timestamp join is always preferred.
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


def load_shot_array(  # noqa: C901, PLR0912
    scan_folder: Path,
    device: str,
    shot: int,
    acq_timestamp: Optional[float] = None,
    data_cache: Optional[ShotPixelCache] = None,
    cache_key: Optional[tuple[str, str]] = None,
    file_shot: Optional[int] = None,
) -> ShotArray:
    """Resolve one device shot's pixel array from an existing scan folder.

    Parameters
    ----------
    scan_folder : Path
        The scan's existing ``scans/ScanNNN`` folder.
    device : str
        Device subfolder name; must be one of :func:`image_devices`
        (path-traversal guard).
    shot : int
        1-based shot number (the GEECS filename convention; stack frame
        index is ``shot - 1`` without a timestamp).
    acq_timestamp : float, optional
        The row's device ``acq_timestamp`` (:func:`row_acq_timestamp`; the
        LabVIEW-epoch double native filenames render). Joins stacks and
        timestamp-named files exactly; without it those fall back to
        ordinal order (orphan between-step frames can shift that join).
    data_cache : ShotPixelCache, optional
        A per-``(scan, device)`` pixel cache. Pass ONLY for completed scans.
    cache_key : tuple of (str, str), optional
        The cache key; required with *data_cache*.
    file_shot : int, optional
        The number in the shot's ``ScanNNN_<device>_SSS`` file name when it
        is not *shot* (:func:`row_file_shot`). ``None`` means *shot*.

    Returns
    -------
    ShotArray
        The decoded pixels, or the tiered refusal (vendor path / missing
        reason).
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
                # Eager within-scan load: the whole frames array in one
                # open; navigation never reopens. None = not admissible
                # (un-finalized tail-race window, or over the per-entry
                # cap) — serve per shot from disk.
                admitted = data_cache.stack_frames(cache_key, stack)
                if admitted is not None:
                    index_map, frames = admitted
                    return _stack_shot_from_memory(
                        index_map, frames, shot, acq_timestamp, path=stack
                    )
            if acq_timestamp is not None:
                # The canonical-millisecond join, ONE file open — the
                # shared keep-first contract lives in io.scan_stack: the
                # stack stores Unix epoch, the row the device's
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
        # or mistyped /acq_timestamp) — a refusal, never a crash.
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
            shot=shot if file_shot is None else file_shot, device=device, ext=ext
        )
    except ValueError as exc:
        # A folder that exists but fails the canonical-layout validation
        # (dev/scratch runs, or a share blip between probes) — degrade.
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
        array = read_imaq_image(native)
        if caching and cacheable:
            # Never cache an ordinal (listing-order) resolution.
            data_cache.store_native_shot(cache_key, shot, array)
        return ShotArray(
            kind="native",
            array=array,
            path=native,
            cacheable=cacheable,
        )
    except Exception as exc:  # noqa: BLE001 — a corrupt file is a refusal
        return ShotArray(kind="missing", path=native, reason=f"read failed: {exc}")


def load_shot_image(
    scan_folder: Path,
    device: str,
    shot: int,
    acq_timestamp: Optional[float] = None,
    data_cache: Optional[ShotPixelCache] = None,
    cache_key: Optional[tuple[str, str]] = None,
    cmap: Optional[str] = None,
    plo: Optional[float] = None,
    phi: Optional[float] = None,
    file_shot: Optional[int] = None,
) -> ShotImage:
    """Resolve and render one device shot — :func:`load_shot_array` + PNG.

    Same parameters and tier ladder as :func:`load_shot_array`; this
    wrapper only adds the display rendering (``cmap``/``plo``/``phi`` per
    :func:`~geecs_data_utils.io.display_png.to_display_png`), so one-shot
    serving and per-bin averaging share one resolution path.
    """
    resolved = load_shot_array(
        scan_folder,
        device,
        shot,
        acq_timestamp=acq_timestamp,
        data_cache=data_cache,
        cache_key=cache_key,
        file_shot=file_shot,
    )
    png = None
    if resolved.array is not None:
        try:
            png = to_display_png(resolved.array, cmap=cmap, plo=plo, phi=phi)
        except Exception as exc:  # noqa: BLE001 — an odd shape is a refusal
            # A readable-but-unrenderable array (e.g. a stacked .npy or an
            # odd-shaped h5 in a dev/scratch folder) degrades to missing,
            # same as a corrupt file.
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


@dataclass(frozen=True)
class ShotTrace:
    """One resolved shot of an ARRAY stack — x-vs-y, or a refusal.

    The :class:`ShotImage` of the trace path.  ``result`` carries the
    values the way every other 1D consumer in the repo receives them
    (:class:`~geecs_data_utils.io.array1d.Data1DResult`: an Nx2 array
    plus whatever the file could say about its axes), already trimmed to
    the trace's true length.
    """

    kind: str  # "stack" | "missing"
    result: Optional[Data1DResult] = None
    #: ``"lineout"`` or ``"waveform"`` — see ``stack_content_kind``.
    content: str = ""
    path: Optional[Path] = None
    reason: str = ""


#: The one reader config for array capture stacks: the stack
#: itself says whether a frame is a lineout or a waveform, and the axis
#: columns are the reader's defaults.
_TRACE_CONFIG = Data1DConfig(data_type=Data1DType.PVA_STACK)


def stack_content(probe: DeviceKind) -> str:
    """What a probed device's shots hold: ``"image"``, ``"lineout"`` or ``"waveform"``.

    Takes the :func:`device_kind` probe the caller already has, so a
    gallery costs one directory scan.  Anything that is not a
    readable array stack reads as ``"image"`` — the answer that keeps a
    caller on the path it has always taken, so a damaged or unfamiliar
    stack degrades to the image path rather than to nothing.
    """
    if probe.kind != "stack" or probe.path is None:
        return "image"
    try:
        return stack_content_kind(probe.path)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        logger.warning("cannot classify stack %s: %s", probe.path, exc)
        return "image"


def load_shot_trace(
    scan_folder: Path,
    device: str,
    shot: int,
    acq_timestamp: Optional[float] = None,
    devices: Optional[list[str]] = None,
) -> ShotTrace:
    """Resolve one device shot of an ARRAY capture stack as x-vs-y.

    The trace twin of :func:`load_shot_array`, and deliberately thinner:
    array stacks are small, so there is no frame cache and no native-file
    tier — a device whose shots are traces has no per-shot file fallback.

    The shot→frame join is the same one the image path uses — its
    index-only form
    (:func:`~geecs_data_utils.io.scan_stack.frame_index_for_acq_timestamp`,
    the same ``_joined_index`` arithmetic, never a second copy), because
    the reader addresses the frame by ``ShotRef`` and reading it here
    would read it twice.  Without a timestamp it falls back to
    ``shot - 1`` exactly as the image path does.

    Parameters
    ----------
    scan_folder : Path
        The run's existing ``scans/ScanNNN`` folder.
    device : str
        Device subfolder name; validated against the folder listing.
    shot : int
        1-based shot number.
    acq_timestamp : float, optional
        The event row's device ``acq_timestamp`` (LabVIEW-epoch double).
    devices : list of str, optional
        An already-computed :func:`image_devices` listing.

    Returns
    -------
    ShotTrace
        The trimmed trace, or the refusal and its reason.
    """
    probe = device_kind(scan_folder, device, devices=devices)
    content = stack_content(probe)
    if content == "image":
        return ShotTrace(
            kind="missing",
            path=probe.path,
            reason=probe.reason or "device has no array capture stack",
        )
    if shot < 1:
        return ShotTrace(kind="missing", path=probe.path, reason="bad shot")
    stack = probe.path
    try:
        if acq_timestamp is not None:
            index = frame_index_for_acq_timestamp(stack, acq_timestamp)
            if index is None:
                return ShotTrace(
                    kind="missing",
                    path=stack,
                    content=content,
                    reason="no stack frame for this shot",
                )
        else:
            index = shot - 1
        result = read_1d_data(ShotRef(stack, index), _TRACE_CONFIG)
    # Same enumeration the image path degrades on: a malformed stack must
    # 404, never 500.
    except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
        return ShotTrace(
            kind="missing", path=stack, content=content, reason=f"stack: {exc}"
        )
    return ShotTrace(kind="stack", result=result, content=content, path=stack)
