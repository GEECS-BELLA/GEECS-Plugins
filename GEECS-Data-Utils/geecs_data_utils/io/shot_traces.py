"""A scan's trace devices: (scan folder, device, row) → x-vs-y, one row at a time.

The trace twin of :mod:`geecs_data_utils.io.shot_images`' row loaders.
A device records traces in one of three ways:

- an **array capture stack** (the PVA gateway's file plugin: lineouts and
  waveforms, ``shot_images.stack_content``), one channel;
- **per-shot native scope files**: an NI-HWS waveform HDF5 (what the
  LeCroy and Tektronix scope VIs save, ``ScanNNN_<device>_SSS.dat``),
  read by :func:`~geecs_data_utils.io.array1d.read_1d_data` as
  ``tek_scope_hdf5``, one channel of the file's several (``trace_index``);
- **per-shot text columns**: a tab-separated ``.txt`` / ``.tsv`` file of
  numbers, x then y (a spectrometer's wavelength and counts), header or
  none, read as ``tsv``; or a Thorlabs FTS spectrometer's ``.csv`` export
  (:func:`~geecs_data_utils.io.array1d.is_thorlabs_fts`), read as
  ``thorlabs_fts``; one channel;
- an **analysis output's text table** (``analysis:<folder>``,
  :mod:`~geecs_data_utils.io.analysis_outputs`): a tab-separated file of
  two or more numeric columns under a header line, x first, each further
  column a channel named from the header (``Charge_fC``: label ``Charge``,
  units ``fC``). Raw device folders keep the two-column rule: a sidecar
  of four columns beside a camera's files is not a trace.

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
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from geecs_data_utils.io.array1d import (
    Data1DConfig,
    Data1DResult,
    Data1DType,
    is_thorlabs_fts,
    read_1d_data,
)
from geecs_data_utils.io import analysis_outputs
from geecs_data_utils.io.shot_images import (
    ScanRows,
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

#: Native per-shot text extensions that may hold x/y columns. Only a file
#: of two tab-separated numeric columns, or a Thorlabs FTS export, is
#: taken as a trace.
TEXT_TRACE_EXTS = frozenset({"txt", "tsv", "csv"})

#: How close two shots' x values must be to share an axis (relative to
#: the axis' span): float rounding in a stored start and increment, no more.
_SAME_AXIS_RTOL = 1e-9


@dataclass(frozen=True)
class TraceDevice:
    """What a device's traces are, from one probe: no trace is read.

    ``kind`` is ``"stack"`` (an array capture stack), ``"native"`` (per-shot
    files: scope waveforms, or x/y text columns) or ``""`` (no traces: a
    camera, a vendor format, an unknown name).
    """

    kind: str = ""
    #: ``"lineout"`` / ``"waveform"`` for a stack, ``"waveform"`` for a scope
    #: file, ``"lineout"`` for text columns.
    content: str = ""
    #: The native files' extension (``"dat"``, ``"txt"``); ``""`` for a stack.
    ext: str = ""
    #: The channels' names, in ``trace_index`` order (one, unnamed, for a stack).
    channels: tuple[str, ...] = ()
    path: Optional[Path] = None
    #: The files are column tables (an analysis output's): the channel is
    #: the y column, after the x one.
    table: bool = False


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


#: How many of a device's files the probe reads for its channels before
#: deciding it records none (an unarmed scope leaves empty stub files).
_PROBE_FILES = 5


def _files(device_dir: Path, ext: str) -> list[Path]:
    """The files of *ext* in *device_dir*, sorted by name.

    A listing's own order differs between file systems.
    """
    try:
        return sorted(
            entry
            for entry in device_dir.iterdir()
            if entry.is_file() and entry.suffix.lower() == f".{ext}"
        )
    except OSError as exc:
        logger.warning("cannot list %s: %s", device_dir, exc)
        return []


def _channels(device_dir: Path, ext: str) -> Optional[tuple]:
    """The scope's channels from the first of its files that names any."""
    for sample in _files(device_dir, ext)[:_PROBE_FILES]:
        channels = scope_channels(sample)
        if channels:
            return channels
    return None


def _two_columns(path: Path) -> bool:
    """Whether *path* starts as x/y columns: two lines of exactly two numbers.

    The fields are tab-separated; one header line before them is allowed.
    Anything else (prose, an empty file, a single row, one column, a
    sidecar's four, a 2-D matrix saved as text) is not an x/y trace, and is
    refused before the reader sees it.
    """
    rows = 0
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle):
                fields = line.strip().split("\t")
                if fields == [""]:
                    continue
                try:
                    [float(v) for v in fields]
                except ValueError:
                    if number == 0:
                        continue  # a header
                    return False
                if len(fields) != 2:
                    return False
                rows += 1
                if rows == 2:
                    return True
    except OSError:
        return False
    return False


def read_text_trace(path: Path) -> Optional[Data1DResult]:
    """*path* as x/y text columns (x first); ``None`` if it is not.

    It is when it is a Thorlabs FTS export (read as ``thorlabs_fts``, its
    header naming the axes), or opens as two tab-separated columns
    (:func:`_two_columns`; a header line, if any, names them) — and has a
    row with both numbers finite.
    """
    if is_thorlabs_fts(path):
        kind = Data1DType.THORLABS_FTS
    elif _two_columns(path):
        kind = Data1DType.TSV
    else:
        return None
    try:
        result = read_1d_data(path, Data1DConfig(data_type=kind))
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    if not np.isfinite(result.data).all(axis=1).any():
        return None
    return result


def read_text_table(path: Path) -> Optional[tuple[tuple[str, ...], np.ndarray]]:
    """*path* as a header line over tab-separated numeric columns; ``None`` if not.

    Returns the column names (``column N`` where the header has none, or
    with no header at all) and the ``(rows, columns)`` values, at least two
    columns and one row. A row of another width, or a field that is not a
    number after the first line, makes it no table.
    """
    names: tuple[str, ...] = ()
    rows: list[list[float]] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle):
                fields = line.rstrip("\r\n").split("\t")
                if not line.strip():
                    continue
                try:
                    rows.append([float(v) for v in fields])
                except ValueError:
                    if number == 0 and not rows:
                        names = tuple(f.strip() for f in fields)
                        continue
                    return None
                if len(rows[-1]) != len(rows[0]):
                    return None
    except OSError:
        return None
    if not rows or len(rows[0]) < 2:
        return None
    width = len(rows[0])
    names = tuple(
        (names[i] if i < len(names) and names[i] else f"column {i}")
        for i in range(width)
    )
    return names, np.asarray(rows, dtype=float)


def _axis_name(header: str) -> tuple[str, Optional[str]]:
    """A column header as ``(label, units)``.

    ``Charge [fC]`` and ``Charge_fC`` (the units after the last ``_``) both
    give ``("Charge", "fC")``; a header with neither is all label.
    """
    match = re.match(r"^(?P<label>.*?)\s*\[(?P<units>[^\]]*)\]$", header)
    if match:
        return match.group("label"), match.group("units") or None
    label, sep, units = header.rpartition("_")
    if sep and label and units:
        return label, units
    return header, None


def read_table_trace(path: Path, channel: int) -> Optional[Data1DResult]:
    """Column *channel* + 1 of a text table (:func:`read_text_table`) against column 0.

    ``None`` when the file is no table, has no such column, or has no row
    with both numbers finite.
    """
    table = read_text_table(path)
    if table is None:
        return None
    names, values = table
    if not 0 <= channel < values.shape[1] - 1:
        return None
    data = values[:, [0, channel + 1]]
    if not np.isfinite(data).all(axis=1).any():
        return None
    x_label, x_units = _axis_name(names[0])
    y_label, y_units = _axis_name(names[channel + 1])
    return Data1DResult(
        data=data, x_units=x_units, y_units=y_units, x_label=x_label, y_label=y_label
    )


def _output_trace_device(scan_folder: Path, name: str) -> TraceDevice:
    """:func:`trace_device` for an analysis output: a series of text tables."""
    series = analysis_outputs.resolve_output(scan_folder, name)
    if series is None or series.ext not in TEXT_TRACE_EXTS:
        return TraceDevice()
    for sample in series.files()[:_PROBE_FILES]:
        table = read_text_table(sample)
        if table is not None:
            return TraceDevice(
                "native", "lineout", series.ext, table[0][1:], series.folder, True
            )
    return TraceDevice()


def _text_device(device_dir: Path, ext: str) -> bool:
    """Whether one of the device's first files (by name) is a text trace."""
    return any(
        read_text_trace(f) is not None for f in _files(device_dir, ext)[:_PROBE_FILES]
    )


def trace_device(
    scan_folder: Path, device: str, devices: Optional[list[str]] = None
) -> TraceDevice:
    """Whether *device* records traces, and how (:class:`TraceDevice`).

    The image ladder's probe (``shot_images.device_kind``) first: an
    array stack is a trace device; a device whose native files are an
    extension in :data:`NATIVE_TRACE_EXTS` is one when one of its first files
    (by name) is an NI-HWS waveform (its channels named from that file), and
    one in :data:`TEXT_TRACE_EXTS` when one of its first files is x/y columns
    (:func:`read_text_trace`; one unnamed channel). Everything else is not
    (``kind == ""``). An analysis output's name is a series of text tables,
    one channel per y column (:func:`read_text_table`).
    """
    if analysis_outputs.is_output_name(device):
        return _output_trace_device(scan_folder, device)
    probe = device_kind(scan_folder, device, devices=devices)
    if probe.kind == "stack":
        content = stack_content(probe)
        if content == "image":
            return TraceDevice()
        return TraceDevice("stack", content, channels=("",), path=probe.path)
    if probe.kind == "unrenderable" and probe.ext in TEXT_TRACE_EXTS:
        if not _text_device(scan_folder / device, probe.ext):
            return TraceDevice()
        return TraceDevice("native", "lineout", probe.ext, ("",), probe.path)
    if probe.kind != "unrenderable" or probe.ext not in NATIVE_TRACE_EXTS:
        return TraceDevice()
    channels = _channels(scan_folder / device, probe.ext)
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
    stamp, present = row_acq_timestamp(frame, device, row, scan_folder)
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
    if probe.table:
        return _read_table(path, channel)
    return _read_native(path, probe.ext, channel)


def _read_table(path: Path, channel: int) -> ShotTrace:
    """One analysis output's text table, its column *channel* + 1."""
    result = read_table_trace(path, channel)
    if result is None:
        return ShotTrace(
            kind="missing", path=path, reason=f"the file has no column {channel + 1}"
        )
    return ShotTrace(kind="native", result=result, content="lineout", path=path)


def _read_native(path: Path, ext: str, channel: int) -> ShotTrace:
    """One native trace file: text columns, or one channel of a scope file."""
    if ext in TEXT_TRACE_EXTS:
        result = read_text_trace(path)
        if result is None:
            return ShotTrace(
                kind="missing", path=path, reason="the file holds no x/y columns"
            )
        return ShotTrace(kind="native", result=result, content="lineout", path=path)
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


def crop_trace(
    trace: Data1DResult, low: Optional[float] = None, high: Optional[float] = None
) -> Optional[Data1DResult]:
    """The samples of *trace* with x from *low* to *high* (both kept); ``None`` if none.

    Either bound may be ``None`` (no limit on that side), and the order of
    the samples is kept. A window the trace does not reach has no samples,
    so the caller can say why rather than draw nothing. Labels and units
    are the trace's.
    """
    if low is None and high is None:
        return trace
    x = trace.data[:, 0]
    keep = np.ones(x.shape, dtype=bool)
    if low is not None:
        keep &= x >= low
    if high is not None:
        keep &= x <= high
    if not keep.any():
        return None
    return Data1DResult(
        data=trace.data[keep],
        x_units=trace.x_units,
        y_units=trace.y_units,
        x_label=trace.x_label,
        y_label=trace.y_label,
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
    return average_trace_scan_rows(
        [ScanRows(scan_folder, frame, tuple(rows))],
        device,
        channel,
        probes=None if probe is None else [probe],
    )


def average_trace_scan_rows(
    parts: Sequence[ScanRows],
    device: str,
    channel: int = 0,
    probes: Optional[Sequence[TraceDevice]] = None,
) -> ShotTrace:
    """*device*'s trace over rows of one or several scans: one trace, or their mean.

    :func:`average_trace_rows` across scans (a pooled bin's rows, per
    scan): each shot is read from its own scan's folder, and the mean is
    still sample by sample on one shared x axis, or refused.

    Parameters
    ----------
    parts : sequence of ScanRows
        Each scan's folder, table and rows.
    device : str
        The device folder (in every scan).
    channel : int
        The scope channel.
    probes : sequence of TraceDevice, optional
        :func:`trace_device` per part, already made.

    Returns
    -------
    ShotTrace
    """
    found: list[ShotTrace] = []
    last = ShotTrace(kind="missing", reason="no shots")
    for i, part in enumerate(parts):
        probe = (
            probes[i] if probes is not None else trace_device(part.scan_folder, device)
        )
        for row in part.rows:
            one = load_row_trace(
                part.scan_folder, part.frame, device, row, channel, probe
            )
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
