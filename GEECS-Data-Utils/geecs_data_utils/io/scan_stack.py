"""Reader for per-device image stacks (the areaDetector NDFileHDF5 layout).

The PVA gateway's file plugin (GeecsPvaGateway ``geecs_pva_gateway.file_plugin``,
#806) writes one frame-stack file per device per scan —
``scans/ScanNNN/<device>/<device>.h5`` — in the layout areaDetector's
NDFileHDF5 plugin uses and Tiled's stock HDF5 adapter reads: the ``(N, H, W)``
frames at :data:`FRAMES_DATASET`, chunked one frame per chunk, plus the
aligned per-frame attribute datasets under ``/entry/instrument/NDAttributes``,
of which :data:`TIMESTAMPS_DATASET` (Unix seconds; LabVIEW epoch minus
:data:`LABVIEW_EPOCH_OFFSET`) is the shot join key.

This module is the read side of that contract, deliberately small:

- :func:`find_stack_file` — locate + validate a device's stack in a scan
  device folder (dispatch on the datasets, never the extension).
- :func:`read_stack_timestamps` — the join key array, one read.
- :func:`read_stack_attributes` / :func:`parse_attribute_name` /
  :func:`stack_scalar_variables` — every per-frame attribute (the stamps
  and, since GeecsPvaGateway 0.9, the device's subscribed numeric
  scalars), keyed by dataset name, and the raw names behind them.
- :func:`read_shot` — one frame by index (a single chunk read).
- :func:`stack_content_kind` — whether the frames are pixels or an
  x-vs-y array (the gateway serves both through one file plugin), which
  is what a renderer and the 1-D reader dispatch on.
- :class:`ShotRef` — a :class:`pathlib.Path` subclass carrying a frame
  index, so per-shot analysis pipelines can pass "this shot inside that
  stack" anywhere a per-shot file path travels today (including through
  pickling into process pools).

Never writes: producing stacks is the file plugin's job alone, and a
stack is read only after its scan has closed (never during a write —
HDF5 across SMB).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Mapping, TypeVar

import h5py
import numpy as np

from geecs_data_utils.io.arrays import WAVEFORM_ATTRIBUTE_SUFFIXES

# LabVIEW timestamps count from 1904-01-01; Unix from 1970-01-01. The stack
# stores Unix seconds (the PVA timestamp); GEECS s-files and native filenames
# carry LabVIEW seconds. lv = unix + OFFSET.
# File-format constant owned here independently of Core's wire-format constant.
# Keep the data layer installable without the hardware access library.
LABVIEW_EPOCH_OFFSET = 2_082_844_800

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

#: The frame stack — areaDetector's NDFileHDF5 dataset path.  ``(N, H, W)``
#: for a camera; since GeecsPvaGateway 0.13 also ``(N, n)`` for a stack of
#: waveforms and ``(N, n, 2)`` for one of lineouts (:func:`stack_content_kind`).
FRAMES_DATASET = "/entry/data/data"
#: The per-frame attribute datasets' group (NDFileHDF5's ``NDAttributes``).
ATTRIBUTES_GROUP = "/entry/instrument/NDAttributes"
#: The suffix of the per-frame stamp attribute dataset, ``(N,)`` float64
#: Unix s.  The plugin names it ``<device>-hdf-<variable>-frame_acq_timestamp``
#: (GeecsPvaGateway >= 0.8, GEECS-Plugins#829: unique across the cameras of a
#: run and never an event column's name); stacks written before that carry
#: the bare ``acq_timestamp`` — :func:`timestamps_dataset` resolves either.
TIMESTAMP_SUFFIX = "frame_acq_timestamp"
#: The bare-name spelling (stacks written by GeecsPvaGateway < 0.8).
TIMESTAMPS_DATASET = f"{ATTRIBUTES_GROUP}/acq_timestamp"


def timestamps_dataset(f: "h5py.File") -> str | None:
    """The path of the open stack's per-frame stamp dataset, or ``None``.

    ``…-frame_acq_timestamp`` (the current layout) or the bare
    ``acq_timestamp`` (the layout before GeecsPvaGateway 0.8); the first
    match in the attributes group.
    """
    group = f.get(ATTRIBUTES_GROUP)
    if group is None:
        return None
    for key in group:
        if key == "acq_timestamp" or key.endswith(f"-{TIMESTAMP_SUFFIX}"):
            return f"{ATTRIBUTES_GROUP}/{key}"
    return None


#: The plugin-child token in an attribute name: ``<device>-hdf-<variable>-<suffix>``.
ATTRIBUTE_PLUGIN_TOKEN = "-hdf-"


def parse_attribute_name(name: str) -> "tuple[str, str, str] | None":
    """Split ``<device>-hdf-<variable>-<suffix>`` into its three parts.

    The suffix is ``frame_acq_timestamp`` / ``frame_recv_timestamp`` for
    the stamps and the normalized variable name for a subscribed scalar
    (``uc_cam-hdf-image-maxcounts`` → ``("uc_cam", "image", "maxcounts")``).
    Every part went through ``normalize_component`` on the writing side,
    so neither the device nor the variable contains ``-``.  ``None`` for a
    name of another shape (the bare ``acq_timestamp`` of pre-0.8 stacks).
    """
    device, token, rest = name.partition(ATTRIBUTE_PLUGIN_TOKEN)
    if not token or not device:
        return None
    variable, dash, suffix = rest.partition("-")
    if not dash or not variable or not suffix:
        return None
    return device, variable, suffix


def read_stack_attributes(path: "str | Path") -> "dict[str, np.ndarray]":
    """Every numeric per-frame attribute dataset of the stack, keyed by name.

    The stamps and the device's subscribed scalars alike, each ``(N,)``
    float64 aligned with the frames (``NaN`` where the device did not
    send that variable with the frame).  Non-numeric members of the group
    (a string attribute from another writer) are skipped, never raised
    on.  One file open; use :func:`parse_attribute_name` to split a key
    and :func:`stack_scalar_variables` for the raw GEECS names.
    """
    with open_stack(path) as f:
        group = f.get(ATTRIBUTES_GROUP)
        if group is None:
            return {}
        return {
            key: np.asarray(group[key][:], dtype=float)
            for key in group
            if isinstance(group[key], h5py.Dataset)
            and np.issubdtype(group[key].dtype, np.number)
        }


def stack_scalar_variables(path: "str | Path") -> "dict[str, str]":
    """``{attribute dataset name: raw GEECS variable name}`` for the scalars.

    The manifest GeecsPvaGateway >= 0.9 writes as the root attributes
    ``scalar_attributes`` / ``scalar_variables`` (normalization is
    one-way, so the file carries the names back).  Empty for a stack
    without it (0.8 stacks: stamps only).
    """
    with open_stack(path) as f:
        names = [str(n) for n in f.attrs.get("scalar_attributes", [])]
        variables = [str(v) for v in f.attrs.get("scalar_variables", [])]
    if len(names) != len(variables):
        # Half a manifest (a third-party or damaged file): no names, like
        # read_stack_attributes skips rather than raises on foreign members.
        logger.warning("%s: scalar manifest attributes disagree in length", path)
        return {}
    return dict(zip(names, variables, strict=True))


def open_stack(path: "str | Path", mode: str = "r") -> "h5py.File":
    """Open a stack for reading with HDF5 file locking **off**.

    The stacks live on an SMB share written from Windows; the HDF5 lock is
    the known failure mode across it, so every reader in this module opens
    through here.  Read only after the scan closed (the plugin's ``finalized``
    root attribute).
    """
    return h5py.File(path, mode, locking=False)


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


def _timestamps(f: "h5py.File") -> str:
    dataset = timestamps_dataset(f)
    if dataset is None:
        raise KeyError(
            f"{f.filename}: no acq_timestamp dataset under {ATTRIBUTES_GROUP}"
        )
    return dataset


def is_stack_file(path: Path) -> bool:
    """Return whether *path* is a readable frame stack.

    Dispatches on the two datasets of the layout, never on the extension;
    a partially-written (un-finalized) stack still qualifies — its frames
    tail is valid.
    """
    if not path.is_file():
        return False
    try:
        with open_stack(path) as f:
            return FRAMES_DATASET in f and timestamps_dataset(f) is not None
    except OSError:
        return False


def find_stack_file(device_dir: Path) -> Path | None:
    """Locate the capture stack for the device folder *device_dir*.

    The plugin names the file after the device folder
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
    with open_stack(path) as f:
        ts = np.asarray(f[_timestamps(f)][:], dtype=float)
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
    with open_stack(ref) as f:
        frames = f[FRAMES_DATASET]
        if not 0 <= shot_index < frames.shape[0]:
            raise IndexError(
                f"shot_index {shot_index} outside stack of {frames.shape[0]} "
                f"frames: {ref}"
            )
        return np.asarray(frames[shot_index])


#: What one frame of a stack holds.  The gateway serves images and arrays
#: through the same file plugin, so the stack layout alone does not say
#: which — ``(N, H, W)`` pixels and ``(N, M, 2)`` lineout rows are both
#: rank 3.  What does say is the plugin's own declaration: it writes the
#: waveform axis attributes (:data:`~geecs_data_utils.io.arrays.WAVEFORM_ATTRIBUTE_SUFFIXES`)
#: for an array variable and never for an image one.
StackContent = Literal["image", "lineout", "waveform"]

#: The attribute whose presence marks a stack as an array stack (any of the
#: three would do; the plugin writes them together).
_ARRAY_MARKER_SUFFIX = WAVEFORM_ATTRIBUTE_SUFFIXES[1]  # "wave_dx"


def _content_kind(f: "h5py.File") -> StackContent:
    """:func:`stack_content_kind` against an already-open stack."""
    group = f.get(ATTRIBUTES_GROUP)
    if group is None or not any(
        key.endswith(f"-{_ARRAY_MARKER_SUFFIX}") for key in group
    ):
        return "image"
    shape = f[FRAMES_DATASET].shape
    if len(shape) == 2:
        return "waveform"
    if len(shape) == 3 and shape[2] == 2:
        return "lineout"
    raise ValueError(
        f"{f.filename}: array stack of frame shape {shape[1:]} is neither a "
        "waveform (n,) nor a lineout (n, 2)"
    )


def stack_content_kind(stack: "str | Path | h5py.File") -> StackContent:
    """What one frame of *stack* (a path, or an already-open stack) holds.

    ``"image"`` for a camera stack, ``"waveform"`` for a stack of 1-D
    arrays (a scope trace: values only, its axis in the per-frame
    ``wave_*`` attributes) and ``"lineout"`` for a stack of ``(n, 2)``
    rows (a spectrum: its axis in column 0).

    The distinction is the renderer's and the reader's: an image is
    pixels, an array is x-vs-y, and a ``(2048, 2)`` lineout drawn as
    pixels is a two-pixel-wide strip.  Use
    :func:`~geecs_data_utils.io.array1d.read_1d_data` with
    :attr:`~geecs_data_utils.io.array1d.Data1DType.PVA_STACK` to read one
    shot of an array stack as x-vs-y.

    Raises
    ------
    ValueError
        The stack declares itself an array stack but its frames are
        neither ``(n,)`` nor ``(n, 2)``.
    """
    if isinstance(stack, h5py.File):
        return _content_kind(stack)
    with open_stack(stack) as f:
        return _content_kind(f)


def stack_frame_index_map(
    stamps: np.ndarray,
) -> "dict[int, int]":
    """Millisecond-key → frame-index map over a stack's timestamp array.

    THE stack side of the canonical-millisecond join, shared by its
    consumers (ScanAnalysis's per-shot mapper, the data portal's gallery;
    the capture-diff audit's bulk reconciliation deliberately keeps its
    own equivalent map build for now): **keep-first on duplicate
    millisecond keys**
    — the deterministic contract (the file plugin dedupes identical timestamps
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
    index_map: "Mapping[int, _T]", acq_timestamp: float
) -> "_T | None":
    """Resolve one row timestamp against a millisecond-key map — exact keys only.

    Generic over the map's value type so listing-based maps (key → file
    path) share the same probe as stack index maps (key → frame index).

    Probes :func:`~geecs_data_utils.native_files.timestamp_key_candidates`
    (``%.3f`` rounding canonicalisation, never a tolerance window) in
    order; ``None`` means the shot has no frame — the caller must refuse,
    never fall back to a neighbouring frame.

    Parameters
    ----------
    index_map : Mapping of int to T
        From :func:`stack_frame_index_map` (or any keep-first
        millisecond-key map over the same epoch).
    acq_timestamp : float
        The event row's device ``acq_timestamp`` (same epoch as the map).

    Returns
    -------
    T or None
        The mapped value, or ``None`` when no candidate key matches.
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
    with open_stack(path) as f:
        index = _joined_index(f, acq_timestamp, labview_epoch)
        if index is None:
            return None
        return index, np.asarray(f[FRAMES_DATASET][index])


def _joined_index(
    f: "h5py.File", acq_timestamp: float, labview_epoch: bool
) -> "int | None":
    """The join itself, against an open stack — ONE copy for both callers."""
    stamps = np.asarray(f[_timestamps(f)][:], dtype=float)
    if labview_epoch:
        stamps = stamps + LABVIEW_EPOCH_OFFSET
    return frame_index_for_timestamp(stack_frame_index_map(stamps), acq_timestamp)


def frame_index_for_acq_timestamp(
    path: Path, acq_timestamp: float, *, labview_epoch: bool = True
) -> "int | None":
    """:func:`read_shot_for_acq_timestamp` without reading the frame.

    The same join (one open, the same arithmetic — both go through
    ``_joined_index``), for a caller that wants to address the frame
    rather than receive it: the array readers take a
    :class:`ShotRef`, so handing them an index costs nothing while
    reading the frame here would read it twice.

    Returns
    -------
    int or None
        The frame index, or ``None`` when the shot has no frame (the
        caller must refuse — never serve a neighbour).
    """
    with open_stack(path) as f:
        return _joined_index(f, acq_timestamp, labview_epoch)
