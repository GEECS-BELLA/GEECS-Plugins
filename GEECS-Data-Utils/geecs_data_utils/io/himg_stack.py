"""Write a device's capture stack from its folder of HASO ``.himg`` files.

The HASO wavefront sensor saves natively — one ``.himg`` per shot into
``scans/ScanNNN/<device>/`` — and no file plugin writes a stack for it.
This module is the converter that gives it one after the fact:
``<device>/<device>.h5`` in the areaDetector NDFileHDF5 layout that
:mod:`geecs_data_utils.io.scan_stack` reads and every stack consumer (the
shot mapper, the data portal's gallery, the analysis core's source) already
prefers over per-shot files.  Lossless and about five times smaller:

- ``/entry/data/data`` — the ``(N, H, W)`` ``uint16`` frames, one frame per
  chunk, gzip + shuffle (the file plugin's own ``zlib`` choice).
- ``/entry/instrument/NDAttributes/<device>-hdf-himg-frame_acq_timestamp``
  — the per-frame stamps in Unix seconds, the shot join key.  From the
  native filename (``<device>_<stamp>.himg``), or for legacy shot-numbered
  names (``ScanNNN_<device>_NNN.himg``) from the scan's scalar rows
  (the device's ``acq_timestamp`` column).
- ``/entry/instrument/himg/`` — the provenance that makes the stack
  lossless: each file's ``header`` bytes (everything before the pixels),
  its ``source_name``, ``source_sha256`` and ``source_size``.
  :func:`verify_himg_stack` rebuilds every frame with
  :func:`~geecs_data_utils.io.himg.himg_bytes` and checks it against that
  SHA-256, so "byte-identical" is a checked property, not a promise.

The stack is written to ``<device>.h5.part`` and renamed into place only
when complete and (by default) verified, so a reader never finds a
half-written stack under the name it looks for.  The ``.himg`` files are
never touched: converting only adds the stack.  Deleting the sources is
the separate, explicit, verify-first step of
:mod:`geecs_data_utils.io.himg_compact`, and never this module's.

Every long loop here takes a ``progress`` callback — ``(done, total,
phase)`` after each frame — so a host running the conversion out of
process (:mod:`geecs_data_utils.io.himg_worker`) can show frames
done/total instead of a silent wait.  Source files are read through
:func:`read_source_bytes`, which asks the kernel to drop them from the
page cache once read: a 44 GB scan streamed through a service's cgroup
otherwise counts against its memory limit as cache.

Scan-folder invariant: this module creates no directory.  The device
folder must already exist with its ``.himg`` files; nothing is created
above it, beside it, or in its absence.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import h5py
import numpy as np

from geecs_data_utils.io.himg import HimgFormatError, himg_bytes, parse_himg
from geecs_data_utils.io.scan_stack import (
    ATTRIBUTES_GROUP,
    FRAMES_DATASET,
    LABVIEW_EPOCH_OFFSET,
    TIMESTAMP_SUFFIX,
    open_stack,
)
from geecs_data_utils.native_files import (
    filename_timestamp_regex,
    legacy_filename_regex,
)
from geecs_data_utils.tiled_schema import device_acq_timestamp_column, normalize_token

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_COMPRESSION_LEVEL",
    "HEADER_DATASET",
    "HIMG_GROUP",
    "HIMG_SUFFIX",
    "PART_SUFFIX",
    "SOURCE_NAME_DATASET",
    "SOURCE_SHA256_DATASET",
    "SOURCE_SIZE_DATASET",
    "STAMP_VARIABLE",
    "HimgSource",
    "HimgSourcesDeleted",
    "HimgStackError",
    "HimgStackExists",
    "HimgStackReport",
    "HimgStampsUnavailable",
    "HimgVerificationFailed",
    "HimgVerifyReport",
    "NoHimgFiles",
    "Progress",
    "convert_himg_folder",
    "forget_pages",
    "stack_header",
    "himg_sources",
    "is_himg_stack",
    "list_himg_files",
    "package_version",
    "part_path_for",
    "read_source_bytes",
    "stack_path_for",
    "stamp_attribute_name",
    "verify_himg_stack",
    "write_himg_stack",
]

#: A progress callback: ``(done, total, phase)`` after each unit of work
#: — ``phase`` is a short verb for the host's status line (``"writing"``,
#: ``"verifying"``, ``"deleting"``, ``"restoring"``).
Progress = Callable[[int, int, str], None]

#: The source files' suffix, exactly as the sensor writes it.
HIMG_SUFFIX = ".himg"
#: The in-progress stack: renamed to ``<device>.h5`` only once complete.
PART_SUFFIX = ".h5.part"
#: The ``<variable>`` token of the stamp attribute name — the HASO image
#: reaches no GEECS stream variable, so the file format names it.
STAMP_VARIABLE = "himg"
#: The per-frame provenance group — beside, not inside, ``NDAttributes``
#: (which the readers scan for numeric per-frame attributes).
HIMG_GROUP = "/entry/instrument/himg"
#: ``(N, L)`` uint8: each file's header bytes (``header_length`` attribute = L).
HEADER_DATASET = f"{HIMG_GROUP}/header"
#: ``(N,)`` str: each source file's name.
SOURCE_NAME_DATASET = f"{HIMG_GROUP}/source_name"
#: ``(N,)`` 64-char ASCII: the SHA-256 of each source file as read.
SOURCE_SHA256_DATASET = f"{HIMG_GROUP}/source_sha256"
#: ``(N,)`` int64: each source file's length in bytes.
SOURCE_SIZE_DATASET = f"{HIMG_GROUP}/source_size"
#: gzip level for the frames: 4 gives ~10 % smaller files than the file
#: plugin's live-speed level 1 at twice the CPU, and this runs offline
#: (5.8x on a real HASO4 LIFT frame; reads cost the same at every level).
DEFAULT_COMPRESSION_LEVEL = 4

_HEADER_LENGTH_ATTRIBUTE = "header_length"


class HimgStackError(RuntimeError):
    """A conversion or verification could not proceed; nothing was left behind."""


class NoHimgFiles(HimgStackError):
    """The device folder holds no ``.himg`` files."""


class HimgStackExists(HimgStackError):
    """The device already has a stack; converting again needs ``overwrite``."""

    def __init__(self, stack_path: Path):
        super().__init__(f"{stack_path} already exists")
        self.stack_path = stack_path


class HimgStampsUnavailable(HimgStackError):
    """A legacy shot-numbered file has no per-shot stamp to carry into the stack."""


class HimgVerificationFailed(HimgStackError):
    """A rebuilt frame did not match its source (or the file on disk).

    The converter removes the failed stack (its default *outcome*); a
    compaction or a restore that finds a mismatch leaves everything as it
    was and says so.
    """

    def __init__(
        self,
        stack_path: Path,
        mismatches: Sequence[str],
        *,
        outcome: str = "the stack was removed",
    ):
        shown = ", ".join(mismatches[:5]) + (" …" if len(mismatches) > 5 else "")
        super().__init__(
            f"{len(mismatches)} frame(s) of {stack_path} did not rebuild "
            f"byte-identically ({shown}); {outcome}"
        )
        self.stack_path = stack_path
        self.mismatches = tuple(mismatches)


class HimgSourcesDeleted(HimgStackError):
    """The existing stack holds frames whose ``.himg`` files are gone (a compacted folder).

    Overwriting it would rebuild the stack from the files still on disk
    and replace the only copy of the deleted frames — refused; restore
    the folder first.
    """

    def __init__(self, stack_path: Path, missing: Sequence[str]):
        shown = ", ".join(missing[:5]) + (" …" if len(missing) > 5 else "")
        super().__init__(
            f"{stack_path} holds {len(missing)} frame(s) whose {HIMG_SUFFIX} files "
            f"are no longer in the folder ({shown}) — the folder was compacted; "
            "restore it (geecs-himg restore) before converting with overwrite"
        )
        self.stack_path = stack_path
        self.missing = tuple(missing)


@dataclass(frozen=True)
class HimgSource:
    """One ``.himg`` file and the stamp it enters the stack with.

    Attributes
    ----------
    path : Path
        The file.
    acq_timestamp : float
        The device's acquisition stamp in LabVIEW-epoch seconds — the
        convention of native filenames and s-file columns; the stack
        stores it in Unix seconds.
    shot_number : int or None
        The shot number a legacy name carries; ``None`` for a native name.
    """

    path: Path
    acq_timestamp: float
    shot_number: int | None = None


@dataclass
class HimgStackReport:
    """What one conversion wrote."""

    stack_path: Path
    frames: int
    source_bytes: int
    stack_bytes: int
    seconds: float
    #: ``True``/``False`` after a verification pass; ``None`` when not run.
    verified: bool | None = None

    @property
    def ratio(self) -> float:
        """Source bytes per stack byte."""
        return (
            self.source_bytes / self.stack_bytes if self.stack_bytes else float("nan")
        )

    def summary(self) -> str:
        """One line for a log or a task record."""
        checked = {True: ", verified", False: ", VERIFICATION FAILED", None: ""}
        return (
            f"{self.stack_path.name}: {self.frames} frames, "
            f"{self.source_bytes / 1e9:.2f} GB -> {self.stack_bytes / 1e9:.2f} GB "
            f"({self.ratio:.1f}x){checked[self.verified]}, {self.seconds:.0f} s"
        )


@dataclass
class HimgVerifyReport:
    """What a verification pass found.

    Two kinds of disagreement are kept apart because they call for
    opposite remedies: a frame that does not rebuild its recorded hash
    means the *stack* is damaged (reconvert from the files), while a
    file on disk that differs from an intact frame means the *file*
    changed after conversion (decide which copy is right; never
    reconvert over the stack).
    """

    stack_path: Path
    frames: int
    #: Source names whose rebuilt bytes did not hash to the recorded SHA-256.
    mismatches: tuple[str, ...] = ()
    #: Source names absent from the folder (``against_files`` only).
    missing: tuple[str, ...] = ()
    #: Source names whose file on disk differs from the (intact) frame
    #: (``against_files`` only).
    changed: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Every frame rebuilt byte-identically (and, if asked, every source was there and equal)."""
        return not self.mismatches and not self.missing and not self.changed

    def summary(self) -> str:
        """One line for a log or a task record."""
        if self.ok:
            return f"{self.stack_path.name}: {self.frames} frames verified"
        return (
            f"{self.stack_path.name}: {len(self.mismatches)} of {self.frames} frames "
            f"mismatched, {len(self.missing)} source file(s) missing, "
            f"{len(self.changed)} changed on disk"
        )


def stack_path_for(device_dir: Path) -> Path:
    """Where the device's stack lives: ``<device>/<device>.h5`` (the reader's rule)."""
    return device_dir / f"{device_dir.name}.h5"


def part_path_for(device_dir: Path) -> Path:
    """The in-progress stack a writer owns: ``<device>/<device>.h5.part``."""
    return device_dir / f"{device_dir.name}{PART_SUFFIX}"


def is_himg_stack(stack_path: Path) -> bool:
    """Whether *stack_path* is a stack this module wrote (frames + provenance group)."""
    try:
        with open_stack(Path(stack_path)) as f:
            return all(
                dataset in f
                for dataset in (
                    FRAMES_DATASET,
                    HEADER_DATASET,
                    SOURCE_NAME_DATASET,
                    SOURCE_SHA256_DATASET,
                )
            )
    except OSError:
        return False


def read_source_bytes(path: Path) -> bytes:
    """Read a whole file and ask the kernel to forget it.

    A conversion, a verification or a compaction streams every ``.himg``
    of a scan through the process once; left in the page cache, tens of
    gigabytes count against the service's memory cgroup (the portal sat
    at its ``MemoryHigh`` for the length of a 1806-shot conversion).
    ``posix_fadvise(DONTNEED)`` after the read drops the clean pages
    (Linux; a no-op where the call is missing or refused).
    """
    with open(path, "rb") as handle:
        data = handle.read()
        forget_pages(handle.fileno())
    return data


def forget_pages(fd: int) -> None:
    """``posix_fadvise(fd, 0, 0, DONTNEED)`` where available; silent otherwise."""
    advise = getattr(os, "posix_fadvise", None)
    if advise is None:
        return
    with contextlib.suppress(OSError):
        advise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def stamp_attribute_name(device: str) -> str:
    """The per-frame stamp dataset's name: ``<device>-hdf-himg-frame_acq_timestamp``."""
    return f"{normalize_token(device)}-hdf-{STAMP_VARIABLE}-{TIMESTAMP_SUFFIX}"


def list_himg_files(device_dir: Path) -> list[Path]:
    """The ``.himg`` files of *device_dir* (exact suffix, as the sensor writes it), by name; empty for a missing folder."""
    try:
        entries = sorted(Path(device_dir).iterdir())
    except OSError:
        return []
    return [p for p in entries if p.is_file() and p.suffix == HIMG_SUFFIX]


def himg_sources(
    device_dir: Path,
    *,
    rows: "pd.DataFrame | None" = None,
    device: str | None = None,
) -> list[HimgSource]:
    """Every ``.himg`` of the folder with its stamp, in stack (stamp) order.

    A native name (``<device>_<stamp>.himg``) carries its own stamp.  A
    legacy name (``ScanNNN_<device>_<shot>.himg``) takes the stamp of its
    shot's row in *rows* — the device's ``acq_timestamp`` column, the same
    value the legacy scanner recorded for the shot.

    Parameters
    ----------
    device_dir : Path
        The device folder.
    rows : pandas.DataFrame, optional
        The scan's scalar rows (``Shotnumber`` plus the device's
        ``acq_timestamp`` column); needed only for legacy names.
    device : str, optional
        The device whose column is read; the folder name by default.

    Raises
    ------
    HimgStackError
        A file whose name is neither shape.
    HimgStampsUnavailable
        A legacy name with no rows, no column, no row, or no finite stamp.
    """
    device_dir = Path(device_dir)
    device = device or device_dir.name
    native = filename_timestamp_regex(HIMG_SUFFIX)
    legacy = legacy_filename_regex(HIMG_SUFFIX)
    stamps_by_shot: dict[int, float] | None = None
    sources: list[HimgSource] = []
    for path in list_himg_files(device_dir):
        match = native.search(path.name)
        if match:
            sources.append(HimgSource(path, float(match.group("ts"))))
            continue
        match = legacy.match(path.name)
        if not match:
            raise HimgStackError(
                f"{path.name}: neither a native (<device>_<stamp>.himg) nor a "
                "legacy (ScanNNN_<device>_NNN.himg) name"
            )
        shot = int(match.group("shot_number"))
        if stamps_by_shot is None:
            stamps_by_shot = _legacy_stamps(rows, device, device_dir)
        stamp = stamps_by_shot.get(shot)
        if stamp is None or not np.isfinite(stamp) or stamp <= 0:
            raise HimgStampsUnavailable(
                f"{path.name}: shot {shot} has no finite {device} acq_timestamp "
                "in the scan's scalar rows"
            )
        sources.append(HimgSource(path, float(stamp), shot))
    sources.sort(key=lambda s: (s.acq_timestamp, s.path.name))
    return sources


def _legacy_stamps(
    rows: "pd.DataFrame | None", device: str, device_dir: Path
) -> dict[int, float]:
    """``{shot number: acq_timestamp}`` from the scan's rows, for legacy names."""
    if rows is None:
        raise HimgStampsUnavailable(
            f"{device_dir} holds legacy shot-numbered .himg names, which need "
            "the scan's scalar rows for their stamps (none given)"
        )
    if "Shotnumber" not in rows.columns:
        raise HimgStampsUnavailable("the scalar rows carry no Shotnumber column")
    column = device_acq_timestamp_column(list(rows.columns), device)
    if column is None:
        raise HimgStampsUnavailable(
            f"the scalar rows carry no {device} acq_timestamp column"
        )
    stamps: dict[int, float] = {}
    for shot, stamp in zip(rows["Shotnumber"], rows[column]):
        try:
            stamps[int(shot)] = float(stamp)
        except (TypeError, ValueError):
            continue
    return stamps


def package_version() -> str:
    """The installed version of this package, for the stack's and the manifest's ``writer``."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("geecs-data-utils")
    except PackageNotFoundError:  # a source checkout that was never installed
        return "unknown"


def write_himg_stack(  # noqa: C901, PLR0912, PLR0915
    device_dir: Path,
    sources: Sequence[HimgSource],
    *,
    device: str | None = None,
    overwrite: bool = False,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    progress: Optional[Progress] = None,
) -> HimgStackReport:
    """Write ``<device>/<device>.h5`` from *sources*, one frame per file.

    Reads each file once (hashing it as read), writes its frame, header
    and stamp, and renames the finished ``.part`` file into place.  Any
    failure removes the part file and raises; the folder is left as found.

    Parameters
    ----------
    device_dir : Path
        The existing device folder — never created here.
    sources : sequence of HimgSource
        The files in stack order (:func:`himg_sources`).
    device : str, optional
        The device name in the stamp attribute; the folder name by default.
    overwrite : bool
        Replace an existing stack (atomically, at the final rename).
    compression_level : int
        gzip level for the frames.
    progress : callable, optional
        ``(done, total, "writing")`` after each frame.

    Raises
    ------
    HimgStackError
        The folder is missing, a file is not a ``.himg``, or the frames
        disagree in shape or header length (two sensors in one folder).
    HimgStackExists
        A stack is already there and *overwrite* is false.
    NoHimgFiles
        *sources* is empty.
    """
    device_dir = Path(device_dir)
    if not device_dir.is_dir():
        raise HimgStackError(f"{device_dir} is not an existing directory")
    if not sources:
        raise NoHimgFiles(f"no {HIMG_SUFFIX} files in {device_dir}")
    device = device or device_dir.name
    stack = stack_path_for(device_dir)
    if stack.exists() and not overwrite:
        raise HimgStackExists(stack)
    part = part_path_for(device_dir)
    if overwrite and stack.exists() and is_himg_stack(stack):
        # A compacted folder: the stack is the ONLY copy of frames whose
        # .himg files were deleted. Rebuilding it from the files on disk
        # would silently drop them — refuse, and name the way back.
        with open_stack(stack) as f:
            recorded = [str(n) for n in f[SOURCE_NAME_DATASET].asstr()[:]]
        on_disk = {s.path.name for s in sources}
        gone = [name for name in recorded if name not in on_disk]
        if gone:
            raise HimgSourcesDeleted(stack, gone)
    if overwrite:
        part.unlink(missing_ok=True)  # a previous attempt that died mid-write
    try:
        # Exclusive creation: a second writer on the same folder (the backlog
        # CLI while the portal converts the same scan) must refuse, not race
        # this one to the rename — a stack under the reader's name is always
        # one its own writer verified.
        with open(part, "x"):
            pass
    except FileExistsError:
        raise HimgStackError(
            f"{part} exists: a conversion is in progress, or one died mid-write. "
            "Remove it once nothing is converting this folder, or convert with "
            "overwrite."
        ) from None

    count = len(sources)
    names: list[str] = []
    digests: list[str] = []
    sizes: list[int] = []
    source_bytes = 0
    started = time.time()
    try:
        with h5py.File(part, "w", locking=False) as f:
            f.attrs["device"] = device
            f.attrs["variable"] = STAMP_VARIABLE
            f.attrs["source_format"] = "himg"
            f.attrs["writer"] = f"geecs-data-utils {package_version()}"
            f.attrs["created"] = started
            stamps = f.create_dataset(
                f"{ATTRIBUTES_GROUP}/{stamp_attribute_name(device)}",
                shape=(count,),
                dtype="f8",
            )
            frames = headers = None
            for index, source in enumerate(sources):
                data = read_source_bytes(source.path)
                names.append(source.path.name)
                digests.append(hashlib.sha256(data).hexdigest())
                sizes.append(len(data))
                source_bytes += len(data)
                try:
                    header, pixels = parse_himg(data)
                except HimgFormatError as exc:
                    raise HimgStackError(f"{source.path.name}: {exc}") from exc
                del data
                if frames is None:
                    frames = f.create_dataset(
                        FRAMES_DATASET,
                        shape=(count, *pixels.shape),
                        dtype=pixels.dtype,
                        chunks=(1, *pixels.shape),
                        compression="gzip",
                        compression_opts=compression_level,
                        shuffle=True,
                    )
                    headers = f.create_dataset(
                        HEADER_DATASET, shape=(count, len(header)), dtype="u1"
                    )
                    headers.attrs[_HEADER_LENGTH_ATTRIBUTE] = len(header)
                elif pixels.shape != frames.shape[1:]:
                    raise HimgStackError(
                        f"{source.path.name}: frame {pixels.shape} differs from "
                        f"the first frame's {frames.shape[1:]}"
                    )
                elif len(header) != headers.shape[1]:
                    raise HimgStackError(
                        f"{source.path.name}: {len(header)}-byte header differs "
                        f"from the first file's {headers.shape[1]} bytes"
                    )
                frames[index] = pixels
                headers[index] = np.frombuffer(header, dtype=np.uint8)
                stamps[index] = source.acq_timestamp - LABVIEW_EPOCH_OFFSET
                if (index + 1) % 100 == 0 or index + 1 == count:
                    logger.info(
                        "%s: %d/%d frames written", device_dir.name, index + 1, count
                    )
                if progress is not None:
                    progress(index + 1, count, "writing")
            f.create_dataset(SOURCE_NAME_DATASET, data=names, dtype=h5py.string_dtype())
            f.create_dataset(SOURCE_SHA256_DATASET, data=np.array(digests, dtype="S64"))
            f.create_dataset(SOURCE_SIZE_DATASET, data=np.array(sizes, dtype="i8"))
            f.attrs["finalized"] = True
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, stack)
    return HimgStackReport(
        stack_path=stack,
        frames=count,
        source_bytes=source_bytes,
        stack_bytes=stack.stat().st_size,
        seconds=time.time() - started,
    )


def stack_header(stack_path: Path, index: int = 0) -> bytes:
    """One source file's header bytes from a ``.himg`` stack's provenance group.

    Any header of a sensor lets WaveKit read that sensor's pixels (the
    per-shot header differs only in its timestamp), so the ``haso`` measure's
    host takes the stack's first header to rebuild the temporary ``.himg``
    it hands the SDK for every frame.  Reads only that row.

    Raises
    ------
    HimgStackError
        The file is not a stack this module wrote (no header dataset), or
        *index* is out of range.
    """
    stack_path = Path(stack_path)
    with open_stack(stack_path) as f:
        if HEADER_DATASET not in f:
            raise HimgStackError(
                f"{stack_path}: not a .himg stack (no {HEADER_DATASET})"
            )
        headers = f[HEADER_DATASET]
        if not 0 <= index < headers.shape[0]:
            raise HimgStackError(
                f"{stack_path}: header {index} out of range ({headers.shape[0]} frames)"
            )
        return headers[index].tobytes()


def verify_himg_stack(  # noqa: C901
    stack_path: Path,
    *,
    against_files: bool = False,
    progress: Optional[Progress] = None,
) -> HimgVerifyReport:
    """Rebuild every frame of a stack and check it against the manifest.

    Each frame plus its header is rebuilt with
    :func:`~geecs_data_utils.io.himg.himg_bytes` and hashed; a SHA-256
    equal to the one recorded when the source was read means the bytes
    are the source's.  This reads only the stack.  ``against_files`` also
    compares the rebuilt bytes with the ``.himg`` still in the folder (a
    second read of every source — the audit a compaction runs before it
    deletes).  ``progress`` gets ``(done, total, "verifying")`` per frame.

    Raises
    ------
    HimgStackError
        The file is not a stack this module wrote (no provenance group).
    """
    stack_path = Path(stack_path)
    mismatches: list[str] = []
    missing: list[str] = []
    changed: list[str] = []
    with open_stack(stack_path) as f:
        for dataset in (FRAMES_DATASET, HEADER_DATASET, SOURCE_SHA256_DATASET):
            if dataset not in f:
                raise HimgStackError(f"{stack_path}: not a .himg stack (no {dataset})")
        frames = f[FRAMES_DATASET]
        headers = f[HEADER_DATASET]
        names = [str(n) for n in f[SOURCE_NAME_DATASET].asstr()[:]]
        digests = [str(d) for d in f[SOURCE_SHA256_DATASET].asstr()[:]]
        count = frames.shape[0]
        if not len(names) == len(digests) == headers.shape[0] == count:
            raise HimgStackError(
                f"{stack_path}: manifest and frames disagree in length"
            )
        for index in range(count):
            rebuilt = himg_bytes(headers[index].tobytes(), np.asarray(frames[index]))
            if hashlib.sha256(rebuilt).hexdigest() != digests[index]:
                mismatches.append(names[index])
            elif against_files:
                source = stack_path.parent / names[index]
                try:
                    on_disk = read_source_bytes(source)
                except FileNotFoundError:
                    missing.append(names[index])
                except OSError as exc:
                    raise HimgStackError(
                        f"{source.name}: unreadable while verifying ({exc})"
                    ) from exc
                else:
                    if on_disk != rebuilt:
                        changed.append(names[index])
            if progress is not None:
                progress(index + 1, count, "verifying")
    return HimgVerifyReport(
        stack_path=stack_path,
        frames=count,
        mismatches=tuple(mismatches),
        missing=tuple(missing),
        changed=tuple(changed),
    )


def convert_himg_folder(
    device_dir: Path,
    *,
    rows: "pd.DataFrame | None" = None,
    device: str | None = None,
    verify: bool = True,
    overwrite: bool = False,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    progress: Optional[Progress] = None,
) -> HimgStackReport:
    """Convert one device folder: discover, write, verify.

    :func:`himg_sources` + :func:`write_himg_stack` + (by default)
    :func:`verify_himg_stack`.  A stack that fails verification is removed
    and :class:`HimgVerificationFailed` raised, so a stack under the
    reader's name is always one whose every frame rebuilds its source.

    Parameters
    ----------
    device_dir : Path
        The existing device folder.
    rows : pandas.DataFrame, optional
        The scan's scalar rows, for legacy shot-numbered names.
    device : str, optional
        The device name (stamp column and attribute); the folder name by default.
    verify : bool
        Rebuild and hash every frame after writing.
    overwrite : bool
        Replace an existing stack.
    compression_level : int
        gzip level for the frames.
    progress : callable, optional
        ``(done, total, phase)`` per frame — ``"writing"``, then ``"verifying"``.
    """
    started = time.time()
    device_dir = Path(device_dir)
    if not device_dir.is_dir():
        raise NoHimgFiles(f"no {HIMG_SUFFIX} files in {device_dir} (not a directory)")
    if stack_path_for(device_dir).exists() and not overwrite:
        # Before discovery: a converted folder is an answer on its own, and
        # a file that landed after the conversion must not hide it.
        raise HimgStackExists(stack_path_for(device_dir))
    sources = himg_sources(device_dir, rows=rows, device=device)
    if not sources:
        raise NoHimgFiles(f"no {HIMG_SUFFIX} files in {device_dir}")
    report = write_himg_stack(
        device_dir,
        sources,
        device=device,
        overwrite=overwrite,
        compression_level=compression_level,
        progress=progress,
    )
    if verify:
        try:
            check = verify_himg_stack(report.stack_path, progress=progress)
        except BaseException:
            report.stack_path.unlink(missing_ok=True)
            raise
        report.verified = check.ok
        if not check.ok:
            report.stack_path.unlink(missing_ok=True)
            raise HimgVerificationFailed(report.stack_path, check.mismatches)
        report.seconds = time.time() - started
    logger.info(report.summary())
    return report
