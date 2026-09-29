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
never touched: converting only adds the stack (deleting the sources is a
separate, explicit step — ``himg_compact`` — that verifies first).

Scan-folder invariant: this module creates no directory.  The device
folder must already exist with its ``.himg`` files; nothing is created
above it, beside it, or in its absence.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

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
from geecs_data_utils.tiled_schema import normalize_token

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
    "HimgStackError",
    "HimgStackExists",
    "HimgStackReport",
    "HimgStampsUnavailable",
    "HimgVerificationFailed",
    "HimgVerifyReport",
    "NoHimgFiles",
    "convert_himg_folder",
    "himg_sources",
    "list_himg_files",
    "stack_path_for",
    "stamp_attribute_name",
    "verify_himg_stack",
    "write_himg_stack",
]

#: The source files' suffix (matched case-insensitively).
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
    """A rebuilt frame did not match its source; the stack was removed."""

    def __init__(self, stack_path: Path, mismatches: Sequence[str]):
        shown = ", ".join(mismatches[:5]) + (" …" if len(mismatches) > 5 else "")
        super().__init__(
            f"{len(mismatches)} frame(s) of {stack_path} did not rebuild "
            f"byte-identically ({shown}); the stack was removed"
        )
        self.stack_path = stack_path
        self.mismatches = tuple(mismatches)


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
    """What a verification pass found."""

    stack_path: Path
    frames: int
    #: Source names whose rebuilt bytes did not hash (or compare) to the source.
    mismatches: tuple[str, ...] = ()
    #: Source names absent from the folder (``against_files`` only).
    missing: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Every frame rebuilt byte-identically (and, if asked, every source was there)."""
        return not self.mismatches and not self.missing

    def summary(self) -> str:
        """One line for a log or a task record."""
        if self.ok:
            return f"{self.stack_path.name}: {self.frames} frames verified"
        return (
            f"{self.stack_path.name}: {len(self.mismatches)} of {self.frames} frames "
            f"mismatched, {len(self.missing)} source file(s) missing"
        )


def stack_path_for(device_dir: Path) -> Path:
    """Where the device's stack lives: ``<device>/<device>.h5`` (the reader's rule)."""
    return device_dir / f"{device_dir.name}.h5"


def stamp_attribute_name(device: str) -> str:
    """The per-frame stamp dataset's name: ``<device>-hdf-himg-frame_acq_timestamp``."""
    return f"{normalize_token(device)}-hdf-{STAMP_VARIABLE}-{TIMESTAMP_SUFFIX}"


def list_himg_files(device_dir: Path) -> list[Path]:
    """The ``.himg`` files of *device_dir*, by name; empty for a missing folder."""
    try:
        entries = sorted(Path(device_dir).iterdir())
    except OSError:
        return []
    return [p for p in entries if p.is_file() and p.suffix.lower() == HIMG_SUFFIX]


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
    # The shared device<->column rule (normalize_token on both sides), the
    # spelling-tolerant way ScanAnalysis's shot mapper finds the column:
    # "<Device> acq_timestamp" (s-file), "<Device>:acq_timestamp",
    # "<device>-acq_timestamp" all collapse to "<token>_acq_timestamp".
    wanted = f"{normalize_token(device)}_acq_timestamp"
    column = next(
        (str(c) for c in rows.columns if normalize_token(str(c)) == wanted), None
    )
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


def _package_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("geecs-data-utils")
    except PackageNotFoundError:  # a source checkout that was never installed
        return "unknown"


def write_himg_stack(
    device_dir: Path,
    sources: Sequence[HimgSource],
    *,
    device: str | None = None,
    overwrite: bool = False,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
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
    part = device_dir / f"{device_dir.name}{PART_SUFFIX}"
    part.unlink(missing_ok=True)  # a previous attempt that died mid-write

    count = len(sources)
    names: list[str] = []
    digests: list[str] = []
    sizes: list[int] = []
    source_bytes = 0
    started = time.time()
    try:
        with h5py.File(part, "w", libver="latest", locking=False) as f:
            f.attrs["device"] = device
            f.attrs["variable"] = STAMP_VARIABLE
            f.attrs["source_format"] = "himg"
            f.attrs["writer"] = f"geecs-data-utils {_package_version()}"
            f.attrs["created"] = started
            stamps = f.create_dataset(
                f"{ATTRIBUTES_GROUP}/{stamp_attribute_name(device)}",
                shape=(count,),
                dtype="f8",
            )
            frames = headers = None
            for index, source in enumerate(sources):
                data = source.path.read_bytes()
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


def verify_himg_stack(
    stack_path: Path, *, against_files: bool = False
) -> HimgVerifyReport:
    """Rebuild every frame of a stack and check it against the manifest.

    Each frame plus its header is rebuilt with
    :func:`~geecs_data_utils.io.himg.himg_bytes` and hashed; a SHA-256
    equal to the one recorded when the source was read means the bytes
    are the source's.  This reads only the stack.  ``against_files`` also
    compares the rebuilt bytes with the ``.himg`` still in the folder (a
    second read of every source — the audit a compaction runs before it
    deletes).

    Raises
    ------
    HimgStackError
        The file is not a stack this module wrote (no provenance group).
    """
    stack_path = Path(stack_path)
    mismatches: list[str] = []
    missing: list[str] = []
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
                continue
            if against_files:
                source = stack_path.parent / names[index]
                if not source.is_file():
                    missing.append(names[index])
                elif source.read_bytes() != rebuilt:
                    mismatches.append(names[index])
    return HimgVerifyReport(
        stack_path=stack_path,
        frames=count,
        mismatches=tuple(mismatches),
        missing=tuple(missing),
    )


def convert_himg_folder(
    device_dir: Path,
    *,
    rows: "pd.DataFrame | None" = None,
    device: str | None = None,
    verify: bool = True,
    overwrite: bool = False,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
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
    )
    if verify:
        check = verify_himg_stack(report.stack_path)
        report.verified = check.ok
        if not check.ok:
            report.stack_path.unlink(missing_ok=True)
            raise HimgVerificationFailed(report.stack_path, check.mismatches)
        report.seconds = time.time() - started
    logger.info(report.summary())
    return report
