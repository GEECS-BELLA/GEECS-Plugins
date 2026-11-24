"""A magnetic spectrometer's per-shot image read with its axes and charge unit.

A magspec analysis writes three series per shot into the day's analysis
tree (``analysis/ScanNNN/``, :mod:`~geecs_data_utils.io.analysis_outputs`):

- ``<folder>/ScanNNN_<folder>_SSS.png``: the charge image, 16-bit
  greyscale, rows the divergence angle and columns the momentum. A pixel
  is the charge in that pixel in counts; the PNG's ``Comment`` text says
  what a count is (``1 aC/count``), and one count is one attocoulomb when
  it says nothing.
- ``<folder>-interpSpec/ScanNNN_<folder>-interpSpec_SSS.txt``: the
  momentum axis, one row per image **column** in column order, with the
  spectrum lineouts (``Momentum_GeV/c  Charge_fC  ChargeDen_pC/GeV/c``).
- ``<folder>-interpDiv/ScanNNN_<folder>-interpDiv_SSS.txt``: the angle
  axis, one row per image **row** in row order, with the divergence
  lineouts (``Angle_mrad  Charge_fC  ChargeDen_fC/mrad``).

Each table is tab-separated under one header line whose names are
``<label>_<unit>``. The axes are read from every shot's own tables, never
shared: the momentum axis moves with the dipole field shot to shot. A table
whose length is not the image's matching dimension is refused, never
resampled or cut.

:func:`charge_density` turns the charge per pixel into charge per unit of
each axis (``pC/mrad/(GeV/c)``), and :func:`average_shots` averages shots
index-wise with their axes, as the analysis core averages a measure whose
axes are per shot.

Strictly read-only: nothing is created.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import png

from geecs_data_utils.frames import Axis, Frame, ShotMeta
from geecs_data_utils.io import analysis_outputs
from geecs_data_utils.io.analysis_outputs import OutputSeries
from geecs_data_utils.io.shot_traces import column_name, read_text_table

#: The output folder suffixes that carry the image's two axes.
MOMENTUM_SUFFIX = "-interpSpec"
ANGLE_SUFFIX = "-interpDiv"
#: What one count is when the image does not say: one attocoulomb.
DEFAULT_CHARGE_PER_COUNT: tuple[float, str] = (1.0, "aC")

#: The PNG comment naming a count: ``1 aC/count``.
_PER_COUNT = re.compile(r"^\s*(?P<scale>[-+0-9.eE]+)\s*(?P<unit>\S+?)\s*/\s*count\s*$")


@dataclass(frozen=True)
class LineoutTable:
    """One axis table: the axis, and each further column as a trace on it."""

    #: The first column: ``Axis(values, unit="GeV/c", label="Momentum")``.
    axis: Axis
    #: The other columns by label (``Charge``, ``ChargeDen``), each a 1D frame on :attr:`axis`.
    channels: Mapping[str, Frame]


@dataclass(frozen=True)
class MagspecShot:
    """One shot's charge image with real axes, and the two lineout tables."""

    #: The image in charge per pixel: axes ``(angle, momentum)``, unit ``aC``.
    frame: Frame
    #: The ``-interpSpec`` table: the momentum axis and the spectrum.
    spectrum: LineoutTable
    #: The ``-interpDiv`` table: the angle axis and the divergence profile.
    divergence: LineoutTable


def read_lineout_table(path: Union[Path, str]) -> LineoutTable:
    """Read an axis table: the first column the axis, each further one a trace on it.

    The table and its headers are read as the trace path reads every
    analysis-output table (:func:`~geecs_data_utils.io.shot_traces.read_text_table`,
    :func:`~geecs_data_utils.io.shot_traces.column_name`: ``Charge_fC`` and
    ``Charge [fC]`` are both label ``Charge``, unit ``fC``).

    Raises
    ------
    ValueError
        Not a table of two or more numeric columns under at most one header
        line (rows of one width), a repeated label, or an axis that is not
        finite.
    """
    path = Path(path)
    table = read_text_table(path)
    if table is None:
        raise ValueError(
            f"{path.name}: not a tab-separated table of two or more numeric columns"
        )
    headers, data = table
    names = [column_name(header) for header in headers]
    labels = [label for label, _ in names]
    if len(set(labels)) != len(labels):
        raise ValueError(f"{path.name}: column labels must be distinct")
    (axis_label, axis_unit), *others = [(label, unit or "") for label, unit in names]
    try:
        axis = Axis(data[:, 0], unit=axis_unit, label=axis_label)
    except ValueError as exc:
        raise ValueError(f"{path.name}: {exc}") from exc
    channels = {
        label: Frame(data[:, column], (axis,), unit=unit, label=label)
        for column, (label, unit) in enumerate(others, start=1)
    }
    return LineoutTable(axis, MappingProxyType(channels))


def charge_per_count(comment: Optional[str]) -> Optional[tuple[float, str]]:
    """``"1 aC/count"`` → ``(1.0, "aC")``; ``None`` for anything else."""
    if comment is None:
        return None
    match = _PER_COUNT.match(comment)
    if match is None:
        return None
    try:
        return float(match.group("scale")), match.group("unit")
    except ValueError:
        return None


def read_charge_image(path: Union[Path, str]) -> tuple[np.ndarray, str]:
    """A magspec PNG as charge per pixel: ``(float64 image, unit)``.

    The counts are scaled by the PNG's ``Comment`` (``<scale> <unit>/count``),
    else by :data:`DEFAULT_CHARGE_PER_COUNT`. One read of the file.

    Raises
    ------
    ValueError
        A PNG that is not single-channel greyscale, or one pypng cannot
        decode (truncated or empty: a file its producer is still writing).
    """
    raw = Path(path).read_bytes()
    comment = None
    try:
        for kind, data in png.Reader(bytes=raw).chunks():
            if kind == b"tEXt":
                key, _, text = data.partition(b"\x00")
                if key == b"Comment":
                    comment = text.decode("latin-1")
        width, height, rows, info = png.Reader(bytes=raw).read()
        counts = np.array(list(rows), dtype=np.float64)
    except (png.Error, EOFError) as exc:
        raise ValueError(f"{Path(path).name}: not a readable PNG ({exc})") from exc
    if not info.get("greyscale") or info.get("alpha"):
        raise ValueError(f"{Path(path).name}: a magspec image is greyscale")
    counts = counts.reshape(height, width)
    scale, unit = charge_per_count(comment) or DEFAULT_CHARGE_PER_COUNT
    return counts * scale, unit


def read_magspec_shot(
    image_path: Union[Path, str],
    momentum_path: Union[Path, str],
    angle_path: Union[Path, str],
    *,
    shot: Optional[ShotMeta] = None,
) -> MagspecShot:
    """Join one shot's image and its two axis tables into a frame with real axes.

    Raises
    ------
    ValueError
        The momentum table's length is not the image's width, or the angle
        table's not its height (nothing is resampled or cut), or either
        table or the image is malformed.
    """
    image, unit = read_charge_image(image_path)
    spectrum = read_lineout_table(momentum_path)
    divergence = read_lineout_table(angle_path)
    height, width = image.shape
    if spectrum.axis.values.size != width:
        raise ValueError(
            f"{Path(momentum_path).name}: {spectrum.axis.values.size} rows for an "
            f"image {width} columns wide"
        )
    if divergence.axis.values.size != height:
        raise ValueError(
            f"{Path(angle_path).name}: {divergence.axis.values.size} rows for an "
            f"image {height} rows high"
        )
    frame = Frame(image, (divergence.axis, spectrum.axis), shot, unit, "Charge")
    return MagspecShot(frame, spectrum, divergence)


#: Coulombs per charge unit a magspec image may carry.
CHARGE_UNITS: Mapping[str, float] = MappingProxyType(
    {"aC": 1e-18, "fC": 1e-15, "pC": 1e-12, "nC": 1e-9}
)


def _per(unit: str) -> str:
    """*unit* as a divisor: ``mrad`` stays, ``GeV/c`` becomes ``(GeV/c)``."""
    return f"({unit})" if "/" in unit else unit


def pixel_widths(axis: Axis) -> np.ndarray:
    """Each sample's width along *axis*: the spacing to its neighbours, positive.

    Raises
    ------
    ValueError
        An axis of one sample (no width), or two samples at one coordinate.
    """
    if axis.values.size < 2:
        raise ValueError(f"the {axis.label or 'axis'} axis has one sample: no width")
    widths = np.abs(np.gradient(axis.values))
    if not (widths > 0).all():
        raise ValueError(f"the {axis.label or 'axis'} axis repeats a coordinate")
    return widths


def charge_density(frame: Frame, unit: str = "pC") -> Frame:
    """A charge-per-pixel image as charge per unit of each axis.

    Each pixel's charge is divided by its widths along the rows' and the
    columns' axes (:func:`pixel_widths`) and converted to *unit*: an image
    in aC per pixel on mrad rows and GeV/c columns becomes
    ``pC/mrad/(GeV/c)``. The axes are kept.

    Raises
    ------
    ValueError
        Not a 2D frame, a charge unit or *unit* not in :data:`CHARGE_UNITS`,
        or an axis with no width.
    """
    if frame.data.ndim != 2:
        raise ValueError("a charge density is of an image")
    if frame.unit not in CHARGE_UNITS or unit not in CHARGE_UNITS:
        raise ValueError(
            f"charge units are {', '.join(CHARGE_UNITS)}; got {frame.unit!r} → {unit!r}"
        )
    rows, columns = frame.axes
    area = np.outer(pixel_widths(rows), pixel_widths(columns))
    scale = CHARGE_UNITS[frame.unit] / CHARGE_UNITS[unit]
    return replace(
        frame,
        data=frame.data * scale / area,
        unit=f"{unit}/{_per(rows.unit)}/{_per(columns.unit)}",
        label="Charge density",
    )


def average_shots(frames: Sequence[Frame]) -> Frame:
    """Shots' images averaged pixel by pixel, and their axes coordinate by coordinate.

    A magspec's momentum axis follows each shot's field, so the shots of a
    bin differ slightly; they average index-wise, as the analysis core
    averages a measure whose axes are per shot. One frame is returned as it
    is. The unit and label are the first frame's.

    Raises
    ------
    ValueError
        No frames, or frames of different shapes.
    """
    if not frames:
        raise ValueError("no shots to average")
    if len(frames) == 1:
        return frames[0]
    first = frames[0]
    if any(f.data.shape != first.data.shape for f in frames):
        raise ValueError("the shots' images differ in size")
    axes = tuple(
        replace(axis, values=np.mean([f.axes[i].values for f in frames], axis=0))
        for i, axis in enumerate(first.axes)
    )
    data = np.mean([f.data for f in frames], axis=0)
    return Frame(data, axes, None, first.unit, first.label)


def magspec_series(
    scan_folder: Path, name: str
) -> Optional[tuple[OutputSeries, OutputSeries, OutputSeries]]:
    """The image, momentum and angle series of the magspec output *name*, or ``None``.

    *name* is the image's output (``analysis:py-magspec``); its axes are the
    sibling folders ``analysis:py-magspec-interpSpec`` and
    ``analysis:py-magspec-interpDiv``. ``None`` when *name* is not an
    output, its default series is not an image, or either sibling is absent.
    """
    if not analysis_outputs.is_output_name(name):
        return None
    image = analysis_outputs.resolve_output(scan_folder, name)
    if image is None or image.ext != "png":
        return None
    folder = image.folder.name
    prefix = analysis_outputs.OUTPUT_PREFIX
    momentum = analysis_outputs.resolve_output(
        scan_folder, prefix + folder + MOMENTUM_SUFFIX
    )
    angle = analysis_outputs.resolve_output(scan_folder, prefix + folder + ANGLE_SUFFIX)
    if momentum is None or angle is None:
        return None
    return image, momentum, angle


def magspec_shot(
    scan_folder: Path,
    name: str,
    number: int,
    *,
    series: Optional[tuple[OutputSeries, OutputSeries, OutputSeries]] = None,
) -> Optional[MagspecShot]:
    """File number *number* of the magspec output *name*, with its axes; ``None`` if absent.

    *number* is the files' shot number (the row's shot, or in a scan-mode
    run the shot it announced). A shot missing any of its three files is
    ``None``: never a neighbour's axis. Malformed or mismatched files raise
    as :func:`read_magspec_shot` does.

    Resolving *name* lists six folders; a caller reading many shots passes
    :func:`magspec_series`' answer once as *series*, leaving three stats
    and three reads per shot.
    """
    found = series if series is not None else magspec_series(scan_folder, name)
    if found is None:
        return None
    paths = [series.shot_file(number) for series in found]
    if not all(path.is_file() for path in paths):
        return None
    return read_magspec_shot(*paths, shot=ShotMeta(name, number))
