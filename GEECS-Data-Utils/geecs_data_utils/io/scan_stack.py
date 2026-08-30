"""Reader for per-device capture frame stacks (``geecs-capture/*``).

The capture daemon (GeecsBluesky ``geecs_bluesky.capture``) writes one
frame-stack file per device per scan — ``scans/ScanNNN/<device>/<device>.h5``
— whose contract is ``GeecsBluesky/geecs_bluesky/capture/FORMAT.md``: an
``(N, H, W)`` ``/frames`` dataset chunked one-frame-per-chunk plus an aligned
``/acq_timestamp`` dataset (Unix seconds; LabVIEW epoch minus
:data:`LABVIEW_EPOCH_OFFSET`), self-described by a ``schema`` root attribute.

This module is the read side of that contract, deliberately small:

- :func:`find_stack_file` — locate + validate a device's stack in a scan
  device folder (dispatch on the ``schema`` attribute, never the extension).
- :func:`read_stack_timestamps` — the join key array, one read.
- :func:`read_shot` — one frame by index (a single chunk read).
- :class:`ShotRef` — a :class:`pathlib.Path` subclass carrying a frame
  index, so per-shot analysis pipelines can pass "this shot inside that
  stack" anywhere a per-shot file path travels today (including through
  pickling into process pools).

Never writes: producing stacks is the capture daemon's job alone.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# LabVIEW timestamps count from 1904-01-01; Unix from 1970-01-01. The stack
# stores Unix seconds (the PVA timestamp); GEECS s-files and native filenames
# carry LabVIEW seconds. lv = unix + OFFSET.
LABVIEW_EPOCH_OFFSET = 2_082_844_800

STACK_SCHEMA_PREFIX = "geecs-capture/"

_PathBase = type(Path())


class ShotRef(_PathBase):
    """A path to a frame stack plus the index of one frame inside it.

    Behaves as the stack file's path everywhere a ``Path`` is expected
    (logging, ``aux["file_path"]``, parent lookups), while carrying
    ``shot_index`` for the loader that resolves it to pixels. Pickles
    correctly (process-pool analysis workers receive real ``ShotRef``
    objects).

    Two deliberate limits: *derived* paths (``ref.parent``,
    ``ref.with_suffix(...)``) are plain paths semantically — they carry no
    ``shot_index`` and must not be fed to :func:`read_shot`; and equality/
    hash are the path's (two refs to different frames of one stack compare
    equal) — never key a cache by ``ShotRef`` alone.
    """

    __slots__ = ("shot_index",)

    def __new__(cls, path: "str | Path", shot_index: int) -> "ShotRef":
        """Create a ref to frame *shot_index* of the stack at *path*."""
        # 3.11 parses the path in __new__; 3.12+ accepts-and-ignores args
        # there (parsing moved to __init__, handled below).
        self = super().__new__(cls, path)
        self.shot_index = int(shot_index)
        return self

    def __init__(self, path: "str | Path", shot_index: int) -> None:
        """Forward only the path to pathlib (3.12+ parses in __init__)."""
        try:
            super().__init__(path)  # type: ignore[call-arg]
        except TypeError:
            super().__init__()  # 3.11: object.__init__ — parsing already done

    def with_segments(self, *segments):  # pragma: no cover - 3.12+ cloning
        """Derive plain Paths (3.12+ clone hook) — the index dies with the ref."""
        return _PathBase(*segments)

    def __reduce__(self):
        """Pickle as (path, shot_index) — Path's own reduce drops the index."""
        return (type(self), (str(self), self.shot_index))

    def __repr__(self) -> str:  # noqa: D105 - trivial
        return f"ShotRef({str(self)!r}, shot_index={self.shot_index})"


def is_stack_file(path: Path) -> bool:
    """Return whether *path* is a readable capture frame stack.

    Dispatches on the ``schema`` root attribute per the format contract;
    a partially-written (un-finalized) stack still qualifies — its
    ``/frames`` tail is valid.
    """
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as f:
            schema = f.attrs.get("schema", "")
            if isinstance(schema, bytes):
                schema = schema.decode()
            return str(schema).startswith(STACK_SCHEMA_PREFIX) and "frames" in f
    except OSError:
        return False


def find_stack_file(device_dir: Path) -> Path | None:
    """Locate the capture stack for the device folder *device_dir*.

    The daemon names the file after the device folder
    (``<device>/<device>.h5``). Returns ``None`` when absent or not a valid
    stack — per the contract, an absent stack means "not captured", never
    an error.
    """
    candidate = device_dir / f"{device_dir.name}.h5"
    if is_stack_file(candidate):
        return candidate
    return None


def read_stack_timestamps(path: Path, *, labview_epoch: bool = False) -> np.ndarray:
    """Return the stack's per-frame ``acq_timestamp`` array.

    Parameters
    ----------
    path : Path
        The stack file.
    labview_epoch : bool
        When true, convert from the stored Unix seconds to LabVIEW-epoch
        seconds (the convention of s-file columns and native filenames).
    """
    with h5py.File(path, "r") as f:
        ts = np.asarray(f["acq_timestamp"][:], dtype=float)
    return ts + LABVIEW_EPOCH_OFFSET if labview_epoch else ts


def read_shot(ref: "ShotRef | Path", shot_index: int | None = None) -> np.ndarray:
    """Read one frame from a stack — a single chunk read.

    Accepts a :class:`ShotRef` (index carried on the ref) or a plain path
    plus an explicit *shot_index*.
    """
    if shot_index is None:
        # getattr: a path *derived* from a ShotRef (ref.parent / name) keeps
        # the type on 3.11 but has no index — refuse it cleanly.
        shot_index = getattr(ref, "shot_index", None)
        if shot_index is None:
            raise TypeError("read_shot needs a ShotRef or an explicit shot_index")
    with h5py.File(ref, "r") as f:
        frames = f["frames"]
        if not 0 <= shot_index < frames.shape[0]:
            raise IndexError(
                f"shot_index {shot_index} outside stack of {frames.shape[0]} "
                f"frames: {ref}"
            )
        return np.asarray(frames[shot_index])


def stack_frame_index_map(
    stamps: np.ndarray,
) -> "dict[int, int]":
    """Millisecond-key → frame-index map over a stack's timestamp array.

    THE stack side of the canonical-millisecond join, shared by every
    consumer (ScanAnalysis's per-shot mapper, the data portal's gallery,
    the capture-diff audit): **keep-first on duplicate millisecond keys**
    — the deterministic contract (the daemon dedupes identical timestamps
    upstream; two consumers resolving duplicates differently would serve
    different frames for the same shot).

    Parameters
    ----------
    stamps : numpy.ndarray
        The per-frame timestamp array from :func:`read_stack_timestamps`
        (in whichever epoch the row-side keys use — LabVIEW for event
        rows/s-files).

    Returns
    -------
    dict of int to int
        ``timestamp_key(ts) → frame index``, keep-first.
    """
    from geecs_data_utils.native_files import timestamp_key

    keys: dict[int, int] = {}
    for index, ts in enumerate(stamps):
        keys.setdefault(timestamp_key(float(ts)), index)
    return keys


def frame_index_for_timestamp(
    index_map: "dict[int, int]", acq_timestamp: float
) -> "int | None":
    """Resolve one row timestamp against a stack index map — exact keys only.

    Probes :func:`~geecs_data_utils.native_files.timestamp_key_candidates`
    (``%.3f`` rounding canonicalisation, never a tolerance window) in
    order; ``None`` means the shot has no frame — the caller must refuse,
    never fall back to a neighbouring frame.

    Parameters
    ----------
    index_map : dict of int to int
        From :func:`stack_frame_index_map`.
    acq_timestamp : float
        The event row's device ``acq_timestamp`` (same epoch as the map).

    Returns
    -------
    int or None
        The frame index, or ``None`` when no candidate key matches.
    """
    from geecs_data_utils.native_files import timestamp_key, timestamp_key_candidates

    for key in timestamp_key_candidates(timestamp_key(acq_timestamp)):
        index = index_map.get(key)
        if index is not None:
            return index
    return None


def read_shot_for_acq_timestamp(
    path: Path, acq_timestamp: float, *, labview_epoch: bool = True
) -> "tuple[int, np.ndarray] | None":
    """Join one row timestamp to its frame and read it — one file open.

    The single-shot composition of :func:`read_stack_timestamps` +
    :func:`stack_frame_index_map` + :func:`frame_index_for_timestamp` +
    :func:`read_shot`, folded into one ``h5py.File`` open (each open is
    several protocol round trips over SMB — the gallery's hot path).

    Parameters
    ----------
    path : Path
        The stack file.
    acq_timestamp : float
        The event row's device ``acq_timestamp`` double.
    labview_epoch : bool
        When true (the event-row/s-file convention), convert the stored
        Unix-epoch stack timestamps before keying.

    Returns
    -------
    tuple of (int, numpy.ndarray) or None
        ``(frame_index, frame)``, or ``None`` when the shot has no frame
        (the caller must refuse — never serve a neighbour).
    """
    with h5py.File(path, "r") as f:
        stamps = np.asarray(f["acq_timestamp"][:], dtype=float)
        if labview_epoch:
            stamps = stamps + LABVIEW_EPOCH_OFFSET
        index = frame_index_for_timestamp(stack_frame_index_map(stamps), acq_timestamp)
        if index is None:
            return None
        return index, np.asarray(f["frames"][index])
