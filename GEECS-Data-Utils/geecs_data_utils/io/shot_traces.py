"""A scan's trace devices: (scan folder, device, row) → x-vs-y, one row at a time.

The trace twin of :mod:`geecs_data_utils.io.shot_images`' row loaders.
A device records traces in one of two ways:

- an **array capture stack** (the PVA gateway's file plugin: lineouts and
  waveforms, ``shot_images.stack_content``), one channel;
- **per-shot native scope files**: an NI-HWS waveform HDF5 (what the
  LeCroy and Tektronix scope VIs save, ``ScanNNN_<device>_SSS.dat``),
  read by :func:`~geecs_data_utils.io.array1d.read_1d_data` as
  ``tek_scope_hdf5``, one channel of the file's several (``trace_index``).

Each comes back as a :class:`~geecs_data_utils.io.shot_images.ShotTrace`
(an Nx2 :class:`~geecs_data_utils.io.array1d.Data1DResult`, or the
refusal and why). A row the device missed is refused, **never a
neighbour's trace**, as for images.

A bin's traces average sample by sample (:func:`average_traces`) only when
they share one x axis (the same length and the same x values); otherwise
the bin has no average and says so. Nothing is resampled: a scope's time
base is fixed within a scan, and a bin whose axes differ is something to
see, not to paper over.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from geecs_data_utils.io.array1d import (
    Data1DConfig,
    Data1DResult,
    Data1DType,
    read_1d_data,
)
from geecs_data_utils.io.shot_images import (
    ShotTrace,
    device_kind,
    load_shot_trace,
    native_shot_file,
    row_acq_timestamp,
    row_file_shot,
    stack_content,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

#: Native per-shot file extensions that may hold a scope waveform. Only
#: an HDF5 file with the NI-HWS ``wfm_group0`` group is taken as one.
NATIVE_TRACE_EXTS = frozenset({"dat"})

#: How close two shots' x values must be to share an axis (relative to
#: the axis' span): float rounding in a stored start and increment, no more.
_SAME_AXIS_RTOL = 1e-9


@dataclass(frozen=True)
class TraceDevice:
    """What a device's traces are, from one probe: no trace is read.

    ``kind`` is ``"stack"`` (an array capture stack), ``"native"`` (per-shot
    scope files) or ``""`` (no traces: a camera, a vendor format, an
    unknown name).
    """

    kind: str = ""
    #: ``"lineout"`` / ``"waveform"`` for a stack, ``"waveform"`` for a scope file.
    content: str = ""
    #: The native files' extension (``"dat"``); ``""`` for a stack.
    ext: str = ""
    #: The channels' names, in ``trace_index`` order (one, unnamed, for a stack).
    channels: tuple[str, ...] = ()
    path: Optional[Path] = None


def scope_channels(path: Path) -> Optional[tuple[str, ...]]:
    """The channel names of an NI-HWS waveform HDF5 file; ``None`` if it is not one.

    ``wfm_group0/traces/traceN`` each carry a ``name`` (``"Channel 0"``);
    a trace without one is named by its number. Opens the file once.
    """
    try:
        import h5py

        if not h5py.is_hdf5(path):
            return None
        with h5py.File(path, "r", locking=False) as f:
            group = f.get("wfm_group0")
            if group is None or "axes" not in group:
                return None
            traces = group.get("traces")
            if traces is None:
                count = len(group["axes"]) // 2
                return tuple(f"Channel {i}" for i in range(count))
            names = []
            for i in range(len(traces)):
                raw = traces[f"trace{i}"].attrs.get("name", b"")
                name = (
                    raw.decode(errors="replace") if isinstance(raw, bytes) else str(raw)
                )
                names.append(name or f"Channel {i}")
            return tuple(names)
    except (ImportError, KeyError, OSError, TypeError, ValueError) as exc:
        logger.warning("cannot read %s as a scope file: %s", path, exc)
        return None


def _first_file(device_dir: Path, ext: str) -> Optional[Path]:
    """One file of *ext* in *device_dir* (the first listed), or ``None``."""
    try:
        for entry in device_dir.iterdir():
            if entry.is_file() and entry.suffix.lower() == f".{ext}":
                return entry
    except OSError as exc:
        logger.warning("cannot list %s: %s", device_dir, exc)
    return None


def trace_device(
    scan_folder: Path, device: str, devices: Optional[list[str]] = None
) -> TraceDevice:
    """Whether *device* records traces, and how (:class:`TraceDevice`).

    The image ladder's probe (``shot_images.device_kind``) first: an
    array stack is a trace device; a device whose native files are an
    extension in :data:`NATIVE_TRACE_EXTS` is one when its first file is
    an NI-HWS waveform (its channels named from that file). Everything
    else is not (``kind == ""``).
    """
    probe = device_kind(scan_folder, device, devices=devices)
    if probe.kind == "stack":
        content = stack_content(probe)
        if content == "image":
            return TraceDevice()
        return TraceDevice("stack", content, channels=("",), path=probe.path)
    if probe.kind != "unrenderable" or probe.ext not in NATIVE_TRACE_EXTS:
        return TraceDevice()
    sample = _first_file(scan_folder / device, probe.ext)
    channels = scope_channels(sample) if sample is not None else None
    if not channels:
        return TraceDevice()
    return TraceDevice("native", "waveform", probe.ext, channels, probe.path)


def load_row_trace(
    scan_folder: Path,
    frame: Optional["pd.DataFrame"],
    device: str,
    row: int,
    channel: int = 0,
    probe: Optional[TraceDevice] = None,
) -> ShotTrace:
    """One row of the scan's table as *device*'s trace (one *channel*), or why not.

    The row decides the shot as it does for an image
    (``shot_images.load_row_array``): its ``acq_timestamp`` and its
    announced shot; a row past the table or one the device missed is
    refused.

    Parameters
    ----------
    scan_folder : Path
        The scan's existing ``scans/ScanNNN`` folder.
    frame : pandas.DataFrame or None
        The scan's WHOLE per-shot table, one row per shot (never filtered).
    device : str
        The device folder.
    row : int
        1-based row.
    channel : int
        The scope channel (``trace_index``); a stack has one, and ignores it.
    probe : TraceDevice, optional
        :func:`trace_device` already made, to spare the probe.

    Returns
    -------
    ShotTrace
    """
    if probe is None:
        probe = trace_device(scan_folder, device)
    if not probe.kind:
        return ShotTrace(kind="missing", reason=f"{device} records no traces")
    if frame is not None and not 1 <= row <= len(frame):
        return ShotTrace(kind="missing", reason=f"no shot {row} in this scan")
    stamp, present = row_acq_timestamp(frame, device, row)
    if present and stamp is None:
        return ShotTrace(kind="missing", reason="the device missed this shot")
    if probe.kind == "stack":
        return load_shot_trace(scan_folder, device, row, acq_timestamp=stamp)
    if not 0 <= channel < len(probe.channels):
        return ShotTrace(
            kind="missing",
            reason=f"no channel {channel}: the scope has {len(probe.channels)}",
        )
    path, _, reason = native_shot_file(
        scan_folder, device, probe.ext, row, stamp, row_file_shot(frame, row)
    )
    if reason:
        return ShotTrace(kind="missing", path=path, reason=reason)
    config = Data1DConfig(data_type=Data1DType.TEK_SCOPE_HDF5, trace_index=channel)
    try:
        result = read_1d_data(path, config)
    except KeyError:
        # The scope saved the file's frame but no waveform in it (a 2 KB
        # stub when it was not armed): say so, not h5py's words.
        return ShotTrace(kind="missing", path=path, reason="the file holds no trace")
    except (ImportError, OSError, TypeError, ValueError) as exc:
        return ShotTrace(kind="missing", path=path, reason=f"read failed: {exc}")
    return ShotTrace(kind="native", result=result, content="waveform", path=path)


def average_traces(traces: Sequence[Data1DResult]) -> Optional[Data1DResult]:
    """The traces' sample-by-sample mean, or ``None`` when their x axes differ.

    One trace is itself. Several must have the same length and the same x
    values (to float rounding of the axis' span); the mean of the y
    values is ``nanmean``, so a NaN sample in one shot does not blank the
    average. The first trace's labels and units are the average's.
    """
    if not traces:
        return None
    first = traces[0]
    if len(traces) == 1:
        return first
    x = first.data[:, 0]
    span = float(np.nanmax(x) - np.nanmin(x)) if x.size else 0.0
    tolerance = _SAME_AXIS_RTOL * max(span, np.finfo(float).tiny)
    for other in traces[1:]:
        if other.data.shape != first.data.shape:
            return None
        if not np.allclose(other.data[:, 0], x, rtol=0.0, atol=tolerance):
            return None
    ys = np.stack([t.data[:, 1] for t in traces])
    with warnings.catch_warnings():
        # A sample NaN in every shot stays NaN, without numpy's warning.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(ys, axis=0)
    return Data1DResult(
        data=np.column_stack((x, mean)),
        x_units=first.x_units,
        y_units=first.y_units,
        x_label=first.x_label,
        y_label=first.y_label,
    )


def average_trace_rows(
    scan_folder: Path,
    frame: Optional["pd.DataFrame"],
    device: str,
    rows: Sequence[int],
    channel: int = 0,
    probe: Optional[TraceDevice] = None,
) -> ShotTrace:
    """*device*'s trace over *rows*: one row's trace, or their mean.

    Rows with no trace (missed, missing or unreadable) are left out of the
    mean; the rest average by :func:`average_traces`. A bin whose shots'
    x axes differ has no average (a refusal saying so).

    Returns
    -------
    ShotTrace
        The trace (``kind`` the rows'), or the refusal: the last row's
        reason when no row had a trace.
    """
    if probe is None:
        probe = trace_device(scan_folder, device)
    found: list[ShotTrace] = []
    last = ShotTrace(kind="missing", reason="no shots")
    for row in rows:
        one = load_row_trace(scan_folder, frame, device, row, channel, probe)
        if one.result is None:
            last = one
            continue
        found.append(one)
    if not found:
        return ShotTrace(kind="missing", path=last.path, reason=last.reason)
    mean = average_traces([t.result for t in found if t.result is not None])
    if mean is None:
        return ShotTrace(
            kind="missing",
            reason="the shots' x axes differ, so they are not averaged",
        )
    return ShotTrace(kind=found[0].kind, result=mean, content=found[0].content)
