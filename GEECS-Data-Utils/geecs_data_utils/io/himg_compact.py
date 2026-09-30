"""Compact a HASO device folder — delete verified ``.himg`` sources — and restore it.

Once :mod:`geecs_data_utils.io.himg_stack` has written a device's capture
stack, the ``.himg`` files are a second copy of the same bytes at five
times the size (2026: 1.65 TB of them).  This module is the one place
that removes them, and the one that puts them back:

- :func:`compact_himg_folder` runs the stack's own audit
  (:func:`~geecs_data_utils.io.himg_stack.verify_himg_stack` with
  ``against_files``: every frame rebuilt and checked against the SHA-256
  recorded at conversion **and** against the file still on disk), and
  only then deletes the ``.himg`` files, leaving a manifest
  (:data:`MANIFEST_NAME`) beside the stack that says what was removed
  and when.  The stack itself is never rewritten — its per-shot header
  rows are what the ``haso`` measure reads its sensor header from.
- :func:`restore_himg_folder` rebuilds each ``.himg`` from the stack,
  byte-identical to the file the converter read (checked against the
  same SHA-256 before it is written), and removes the manifest.

Guards live here, not in the callers, because the callers are a click in
the data portal and a shell command and both must refuse the same things:

- **the scan may still be written** — any ``.himg`` younger than
  :data:`MIN_SOURCE_AGE_S`, or no evidence that the scanner closed the
  run (:func:`~geecs_data_utils.data.sfile.run_closed_evidence`: the
  ``ScanDataScanNNN.txt`` table the stop document writes, else the
  analysis s-file);
- **the stack does not cover the folder** — a ``.himg`` on disk that the
  stack has no frame for (a file that landed after the conversion);
- **another writer owns the folder** — a ``<device>.h5.part`` in progress;
- **a disagreement anywhere** — nothing is deleted (and nothing
  overwritten on restore) unless every frame verifies first.  A frame
  that does not rebuild its recorded hash (:class:`HimgVerificationFailed`,
  the stack is damaged) and a file that changed on disk after conversion
  (:class:`HimgSourceChanged`, the stack is intact) are reported apart,
  because they call for opposite remedies.

The converter's side of the same contract: once a folder is compacted,
``convert`` with ``overwrite`` refuses
(:class:`~geecs_data_utils.io.himg_stack.HimgSourcesDeleted`) — the
stack is the only copy of the deleted frames, so the way to reconvert is
restore first.

Scan-folder invariant, and its one deliberate exception: this module
touches only the files inside the one device folder it is given — it
deletes ``.himg`` files, writes the manifest, rewrites ``.himg`` files on
restore — and never creates a directory or reaches outside that folder
(the run-closed check reads the scan folder and writes nothing).  The
repo's "analysis only adds files" rule names these two functions as its
exception.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from geecs_data_utils.data.sfile import run_closed_evidence
from geecs_data_utils.io.himg import himg_bytes
from geecs_data_utils.io.himg_stack import (
    HEADER_DATASET,
    HIMG_SUFFIX,
    SOURCE_NAME_DATASET,
    SOURCE_SHA256_DATASET,
    SOURCE_SIZE_DATASET,
    HimgStackError,
    HimgVerificationFailed,
    NoHimgFiles,
    Progress,
    forget_pages,
    is_himg_stack,
    list_himg_files,
    package_version,
    part_path_for,
    read_source_bytes,
    stack_path_for,
    verify_himg_stack,
)
from geecs_data_utils.io.scan_stack import (
    FRAMES_DATASET,
    open_stack,
    read_stack_timestamps,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MANIFEST_FORMAT",
    "MANIFEST_NAME",
    "MIN_SOURCE_AGE_S",
    "HimgCompactReport",
    "HimgFolderActive",
    "HimgRestoreReport",
    "HimgSourceChanged",
    "HimgStackIncomplete",
    "NoHimgStack",
    "compact_himg_folder",
    "manifest_path_for",
    "read_manifest",
    "restore_himg_folder",
]

#: The record a compaction leaves in the device folder: what was deleted,
#: when, by which writer, and the stack that holds it.
MANIFEST_NAME = "himg_manifest.json"
MANIFEST_FORMAT = "geecs-himg-manifest"
MANIFEST_VERSION = 1
#: A ``.himg`` younger than this is a scan that may still be writing:
#: refuse.  A minute covers the sensor's ~1 Hz cadence with room for a
#: clock offset between the acquisition PC and the share.
MIN_SOURCE_AGE_S = 60.0


class NoHimgStack(HimgStackError):
    """The device folder has no stack to compact against or restore from."""


class HimgFolderActive(HimgStackError):
    """The scan may still be writing this folder: a young file, or no closed run."""


class HimgStackIncomplete(HimgStackError):
    """A ``.himg`` on disk has no frame in the stack — restore (if compacted) and reconvert first."""


class HimgSourceChanged(HimgStackError):
    """A ``.himg`` on disk differs from the stack's intact frame — the file changed after conversion.

    The stack is fine; the two copies disagree.  Nothing is deleted or
    overwritten: someone decides which copy is right.
    """

    def __init__(self, stack_path: Path, changed: Sequence[str], *, outcome: str):
        shown = ", ".join(changed[:5]) + (" …" if len(changed) > 5 else "")
        super().__init__(
            f"{len(changed)} {HIMG_SUFFIX} file(s) on disk differ from their frame in "
            f"{stack_path} ({shown}); the stack is intact and the file changed after "
            f"conversion — decide which copy is right; {outcome}"
        )
        self.stack_path = stack_path
        self.changed = tuple(changed)


@dataclass
class HimgCompactReport:
    """What one compaction did (or found already done)."""

    device_dir: Path
    stack_path: Path
    #: Frames in the stack — every one verified before anything was deleted.
    frames: int
    #: ``.himg`` files deleted by this run (fewer than *frames* when an
    #: earlier run was interrupted after deleting some).
    files_deleted: int
    #: Bytes those files held.
    bytes_freed: int
    stack_bytes: int
    seconds: float
    manifest_path: Path
    #: The folder held no ``.himg`` files and a manifest already: nothing done.
    already_compacted: bool = False

    @property
    def files_before(self) -> int:
        """Files in the folder that mattered before this run: the ``.himg`` files plus the stack."""
        return self.files_deleted + 1

    @property
    def bytes_before(self) -> int:
        """Bytes those files held."""
        return self.bytes_freed + self.stack_bytes

    def summary(self) -> str:
        """One line for a log or a task record: files and GB before and after."""
        name = self.device_dir.name
        if self.already_compacted:
            return (
                f"{name}: already compacted — {self.frames} frames in "
                f"{self.stack_path.name} ({self.stack_bytes / 1e9:.2f} GB), "
                f"no {HIMG_SUFFIX} files"
            )
        ratio = self.bytes_before / self.stack_bytes if self.stack_bytes else 0.0
        return (
            f"{name}: {self.files_deleted} {HIMG_SUFFIX} files verified against "
            f"{self.stack_path.name} and deleted — {self.files_before} files / "
            f"{self.bytes_before / 1e9:.2f} GB -> 1 file / "
            f"{self.stack_bytes / 1e9:.2f} GB ({self.bytes_freed / 1e9:.2f} GB "
            f"freed, {ratio:.1f}x), {self.seconds:.0f} s"
        )


@dataclass
class HimgRestoreReport:
    """What one restore wrote."""

    device_dir: Path
    stack_path: Path
    frames: int
    #: ``.himg`` files written by this run.
    files_restored: int
    #: Files already on disk and byte-identical to the stack (left alone).
    files_present: int
    bytes_written: int
    stack_bytes: int
    seconds: float

    def summary(self) -> str:
        """One line for a log or a task record: files and GB before and after."""
        present = (
            f", {self.files_present} already present" if self.files_present else ""
        )
        after_files = self.files_restored + self.files_present + 1
        after_bytes = (
            self.bytes_written + self.stack_bytes
        )  # present files not re-counted
        return (
            f"{self.device_dir.name}: {self.files_restored} {HIMG_SUFFIX} files "
            f"restored from {self.stack_path.name}, verified{present} — "
            f"{1 + self.files_present} file(s) / "
            f"{self.stack_bytes / 1e9:.2f} GB -> {after_files} files / "
            f"{after_bytes / 1e9:.2f} GB ({self.bytes_written / 1e9:.2f} GB "
            f"written), {self.seconds:.0f} s"
        )


def manifest_path_for(device_dir: Path) -> Path:
    """Where a compaction leaves its record: ``<device>/himg_manifest.json``."""
    return Path(device_dir) / MANIFEST_NAME


def read_manifest(device_dir: Path) -> Optional[dict]:
    """The folder's compaction manifest as a dict, or ``None`` when there is none."""
    path = manifest_path_for(device_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise HimgStackError(f"{path}: unreadable manifest ({exc})") from exc


def _open_checked(device_dir: Path) -> Path:
    """The folder's stack, once the folder, the part file and the stack itself check out."""
    device_dir = Path(device_dir)
    if not device_dir.is_dir():
        raise HimgStackError(f"{device_dir} is not an existing directory")
    part = part_path_for(device_dir)
    if part.exists():
        raise HimgStackError(
            f"{part} exists: a conversion is in progress, or one died mid-write; "
            "nothing was touched"
        )
    stack = stack_path_for(device_dir)
    if not stack.is_file():
        raise NoHimgStack(f"{device_dir} has no stack ({stack.name}); convert first")
    if not is_himg_stack(stack):
        raise NoHimgStack(
            f"{stack} is not a {HIMG_SUFFIX} stack (no provenance group); "
            "convert with overwrite first"
        )
    return stack


def _provenance(f) -> tuple[list[str], list[str], list[int]]:
    """``(names, sha256s, sizes)`` of the stack's sources, checked for one length."""
    names = [str(n) for n in f[SOURCE_NAME_DATASET].asstr()[:]]
    digests = [str(d) for d in f[SOURCE_SHA256_DATASET].asstr()[:]]
    sizes = [int(s) for s in f[SOURCE_SIZE_DATASET][:]]
    count = f[FRAMES_DATASET].shape[0]
    if (
        not len(names)
        == len(digests)
        == len(sizes)
        == f[HEADER_DATASET].shape[0]
        == count
    ):
        raise HimgStackError("manifest and frames disagree in length")
    return names, digests, sizes


def _rebuild(f, index: int) -> bytes:
    """Frame *index* of an open stack as the bytes of its source file."""
    return himg_bytes(
        f[HEADER_DATASET][index].tobytes(), np.asarray(f[FRAMES_DATASET][index])
    )


def _write_manifest(
    device_dir: Path,
    stack: Path,
    names: list[str],
    digests: list[str],
    sizes: list[int],
    stamps: list[float],
    *,
    compacted: float,
    updated: Optional[float] = None,
) -> Path:
    """Write the manifest atomically (``.part`` + rename) and return its path.

    *compacted* is the first compaction's stamp — kept by a rerun that
    finishes an interrupted pass, which records its own time as *updated*.
    """
    record = {
        "format": MANIFEST_FORMAT,
        "version": MANIFEST_VERSION,
        "device": device_dir.name,
        "stack": stack.name,
        "compacted": compacted,
        "compacted_iso": datetime.fromtimestamp(compacted, timezone.utc).isoformat(),
        "updated": updated,
        "writer": f"geecs-data-utils {package_version()}",
        "frames": len(names),
        "source_bytes": int(sum(sizes)),
        "stack_bytes": stack.stat().st_size,
        "files": [
            {"name": name, "size": size, "sha256": digest, "acq_timestamp": stamp}
            for name, size, digest, stamp in zip(names, sizes, digests, stamps)
        ],
    }
    target = manifest_path_for(device_dir)
    part = target.with_name(target.name + ".part")
    part.write_text(json.dumps(record, indent=1))
    os.replace(part, target)
    return target


def compact_himg_folder(
    device_dir: Path,
    *,
    min_age: float = MIN_SOURCE_AGE_S,
    require_closed: bool = True,
    progress: Optional[Progress] = None,
) -> HimgCompactReport:
    """Verify every frame of the stack against its source, then delete the sources.

    Order: guards, the full audit (:func:`verify_himg_stack` with
    ``against_files`` — every frame rebuilt and hashed; every ``.himg``
    still on disk compared byte for byte), the manifest, and only then
    the deletions.  A failure anywhere before the deletions leaves the
    folder as found.  An interrupted deletion pass leaves the manifest
    and the remaining files; running again verifies and finishes,
    keeping the first compaction's stamp.

    Parameters
    ----------
    device_dir : Path
        The existing device folder holding the stack and the ``.himg`` files.
    min_age : float
        Refuse while any ``.himg`` is younger than this many seconds.
    require_closed : bool
        Refuse without :func:`~geecs_data_utils.data.sfile.run_closed_evidence`.
        The shell command's escape hatch for a dead scan that never closed;
        the portal never turns it off.
    progress : callable, optional
        ``(done, total, phase)`` per frame — ``"verifying"``, then ``"deleting"``.

    Raises
    ------
    NoHimgStack
        No stack, or not a ``.himg`` stack.
    NoHimgFiles
        No ``.himg`` files and no manifest — nothing to do and nothing done.
    HimgStackIncomplete
        A ``.himg`` on disk that the stack has no frame for.
    HimgFolderActive
        A young ``.himg``, or no evidence the run closed.
    HimgVerificationFailed
        A frame that does not rebuild its recorded hash (the stack is
        damaged); nothing was deleted.
    HimgSourceChanged
        A file on disk that differs from its intact frame; nothing was deleted.
    HimgStackError
        A ``.part`` file (another writer), a missing folder, or a file
        that vanished or became unreadable mid-run (another process
        changing the folder).
    """
    started = time.time()
    device_dir = Path(device_dir)
    stack = _open_checked(device_dir)
    manifest = manifest_path_for(device_dir)
    on_disk = list_himg_files(device_dir)
    with open_stack(stack) as f:
        names, digests, sizes = _provenance(f)
    stack_bytes = stack.stat().st_size
    if not on_disk:
        if manifest.is_file():
            report = HimgCompactReport(
                device_dir=device_dir,
                stack_path=stack,
                frames=len(names),
                files_deleted=0,
                bytes_freed=0,
                stack_bytes=stack_bytes,
                seconds=time.time() - started,
                manifest_path=manifest,
                already_compacted=True,
            )
            logger.info(report.summary())
            return report
        raise NoHimgFiles(
            f"no {HIMG_SUFFIX} files in {device_dir} and no {MANIFEST_NAME}: "
            "nothing to compact"
        )

    # --- guards, before any read of the frames -----------------------------
    known = set(names)
    extra = sorted(p.name for p in on_disk if p.name not in known)
    if extra:
        shown = ", ".join(extra[:5]) + (" …" if len(extra) > 5 else "")
        raise HimgStackIncomplete(
            f"{len(extra)} {HIMG_SUFFIX} file(s) in {device_dir.name} have no frame "
            f"in {stack.name} ({shown}); restore the folder first if it was "
            "compacted (geecs-himg restore), then reconvert with overwrite, "
            "then compact"
        )
    now = time.time()
    try:
        youngest = min(now - p.stat().st_mtime for p in on_disk)
    except OSError as exc:
        raise HimgStackError(
            f"{device_dir.name}: a {HIMG_SUFFIX} file vanished while checking "
            f"({exc}); another process is changing the folder"
        ) from exc
    if youngest < min_age:
        raise HimgFolderActive(
            f"{device_dir.name}: a {HIMG_SUFFIX} file was written {youngest:.0f} s "
            f"ago (under {min_age:.0f} s); the scan may still be running"
        )
    if require_closed and run_closed_evidence(device_dir.parent) is None:
        raise HimgFolderActive(
            f"{device_dir.name}: no evidence the scan closed (no ScanData table "
            "and no s-file); refusing to delete"
        )

    # --- the stack's own audit, before deleting anything --------------------
    # ``missing`` sources are the ones an interrupted earlier pass already
    # deleted: their frames verified against the recorded hash like every
    # other, so they are fine.
    check = verify_himg_stack(stack, against_files=True, progress=progress)
    if check.mismatches:
        raise HimgVerificationFailed(
            stack, check.mismatches, outcome="nothing was deleted"
        )
    if check.changed:
        raise HimgSourceChanged(stack, check.changed, outcome="nothing was deleted")

    # --- the record first, then the deletions --------------------------------
    existing = read_manifest(device_dir)
    first = existing.get("compacted") if existing else None
    stamps = [float(s) for s in read_stack_timestamps(stack)]
    manifest = _write_manifest(
        device_dir,
        stack,
        names,
        digests,
        sizes,
        stamps,
        compacted=float(first) if isinstance(first, (int, float)) else time.time(),
        updated=time.time() if existing else None,
    )
    freed = 0
    deleted = 0
    for done, path in enumerate(on_disk, start=1):
        try:
            size = path.stat().st_size
            path.unlink()
        except FileNotFoundError:
            continue  # another compaction of the same folder got there first
        except OSError as exc:
            raise HimgStackError(
                f"{path.name}: could not delete ({exc}); {deleted} of "
                f"{len(on_disk)} deleted, the manifest is in place — run again"
            ) from exc
        freed += size
        deleted += 1
        if progress is not None:
            progress(done, len(on_disk), "deleting")
    report = HimgCompactReport(
        device_dir=device_dir,
        stack_path=stack,
        frames=check.frames,
        files_deleted=deleted,
        bytes_freed=freed,
        stack_bytes=stack_bytes,
        seconds=time.time() - started,
        manifest_path=manifest,
    )
    logger.info(report.summary())
    return report


def restore_himg_folder(
    device_dir: Path, *, progress: Optional[Progress] = None
) -> HimgRestoreReport:
    """Rebuild every ``.himg`` of the stack into the device folder, byte-identical.

    Each frame is rebuilt, checked against the SHA-256 recorded at
    conversion, and written as ``<name>.part`` then renamed — so a file
    under a source's name is always complete and verified.  A file already
    there is left alone when it matches, and stops the restore when it
    does not.  The manifest is removed at the end.  File modification
    times are not restored (the stack does not record them); the bytes are.

    Raises
    ------
    NoHimgStack
        No stack, or not a ``.himg`` stack.
    HimgVerificationFailed
        A frame that does not rebuild its recorded hash (the stack is
        damaged); restore stopped, nothing overwritten.
    HimgSourceChanged
        A file on disk that differs from its intact frame; restore
        stopped, nothing overwritten.
    HimgStackError
        A ``.part`` file (another writer), a missing folder, or a file
        that became unreadable mid-run.
    """
    started = time.time()
    device_dir = Path(device_dir)
    stack = _open_checked(device_dir)
    restored = present = written = 0
    with open_stack(stack) as f:
        names, digests, _ = _provenance(f)
        count = len(names)
        for index in range(count):
            name = names[index]
            rebuilt = _rebuild(f, index)
            if hashlib.sha256(rebuilt).hexdigest() != digests[index]:
                raise HimgVerificationFailed(
                    stack, [name], outcome="restore stopped; nothing overwritten"
                )
            target = device_dir / name
            try:
                on_disk = read_source_bytes(target)
            except FileNotFoundError:
                on_disk = None
            except OSError as exc:
                raise HimgStackError(
                    f"{name}: unreadable while restoring ({exc}); restore stopped"
                ) from exc
            if on_disk is not None:
                if on_disk != rebuilt:
                    raise HimgSourceChanged(
                        stack, [name], outcome="restore stopped, nothing overwritten"
                    )
                present += 1
            else:
                part = device_dir / f"{name}.part"
                with open(part, "wb") as handle:
                    handle.write(rebuilt)
                    handle.flush()
                    os.fsync(handle.fileno())
                    forget_pages(handle.fileno())
                os.replace(part, target)
                restored += 1
                written += len(rebuilt)
            if progress is not None:
                progress(index + 1, count, "restoring")
    manifest_path_for(device_dir).unlink(missing_ok=True)
    report = HimgRestoreReport(
        device_dir=device_dir,
        stack_path=stack,
        frames=count,
        files_restored=restored,
        files_present=present,
        bytes_written=written,
        stack_bytes=stack.stat().st_size,
        seconds=time.time() - started,
    )
    logger.info(report.summary())
    return report
