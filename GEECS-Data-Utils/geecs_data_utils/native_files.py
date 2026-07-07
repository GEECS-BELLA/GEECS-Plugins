"""THE native GEECS per-shot file naming contract.

A GEECS device server natively saves one file per acquisition, named by the
device's own ``acq_timestamp`` double rendered at millisecond precision::

    {stem}_{acq_timestamp:.3f}{file_tail}

living inside a per-device folder named by the same stem::

    {scan_folder}/{stem}/{stem}_{acq_timestamp:.3f}{file_tail}

where ``stem = {device_name}{directory_suffix}``.  ``directory_suffix`` is a
diagnostic-output discriminator (FROG ``-Temporal``, magspec ``-interpSpec``,
...): a suffixed diagnostic names **both** the subfolder and the filename — a
suffixed device never writes into the bare device folder.  ``file_tail`` is
the verbatim filename tail after the timestamp (usually just an extension
like ``.png``, but compound tails are legal and must round-trip untouched).

This module is the single source of truth for that contract.  Its consumers
(keep this list current):

- ``geecs_bluesky.assets.registry`` — the *producer* side: builds the paths
  devices are told to save to, and the expected paths external-asset
  documents point at.
- ``scan_analysis.analyzers.common.single_device_scan_analyzer`` — the
  *reader* side: joins s-file rows to natively saved files through the
  device's per-shot ``acq_timestamp`` column.
- ``geecs_scanner.optimization.session_bridge`` — the *waiter* side: blocks
  an optimization iteration until the bin's expected native files are
  visible on the data server.

Millisecond canonicalization
----------------------------
The timestamp appears twice per shot — as a raw double in the event row /
s-file, and as its ``%.3f`` rendering in the filename.  Joining the two
representations canonicalises both to integer milliseconds
(:func:`timestamp_key`).  When the row value has round-tripped through text
(s-file) it can differ from the double the device formatted, and at the
``%.3f`` rounding boundary (e.g. ``…59.5239997`` vs ``…59.524``) the two
canonicalise to *adjacent* integer keys.  :func:`timestamp_key_candidates`
therefore yields the exact key plus its ±1 ms neighbours, in probe order.
This is float-formatting canonicalisation, **not** a physical tolerance
window — never widen it.

Legacy Master Control convention
--------------------------------
The same contract family includes the legacy MC/file-mover naming
``Scan{NNN}_{device_subject}_{shot:03d}{file_tail}`` used by scans whose
files were renamed post-hoc by shot number; :func:`legacy_filename_regex`
builds its matcher.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    "render_timestamp",
    "native_file_stem",
    "native_file_name",
    "native_file_name_from_key",
    "native_file_path",
    "timestamp_key",
    "timestamp_key_candidates",
    "filename_timestamp_regex",
    "legacy_filename_regex",
]


def render_timestamp(acq_timestamp: float) -> str:
    """Render an ``acq_timestamp`` double exactly as native filenames do.

    Parameters
    ----------
    acq_timestamp : float
        The device's acquisition timestamp (epoch-style seconds).

    Returns
    -------
    str
        The ``%.3f`` (millisecond-precision, zero-padded) rendering — the
        one and only timestamp format that appears in native filenames.
    """
    return f"{acq_timestamp:.3f}"


def native_file_stem(device_name: str, directory_suffix: str = "") -> str:
    """Return the on-disk stem for a device's native files and folder.

    Parameters
    ----------
    device_name : str
        The GEECS device name (e.g. ``"U_FROG"``).
    directory_suffix : str, optional
        Diagnostic-output discriminator appended verbatim (e.g.
        ``"-Temporal"``).  Empty (the default) for a device's primary
        output.

    Returns
    -------
    str
        ``{device_name}{directory_suffix}`` — the stem naming both the
        per-device subfolder and every filename inside it.
    """
    return f"{device_name}{directory_suffix}"


def native_file_name(device_name: str, acq_timestamp: float, file_tail: str) -> str:
    """Return the native filename for one timestamped device file.

    Parameters
    ----------
    device_name : str
        The on-disk stem (device name, already carrying any directory
        suffix — compose with :func:`native_file_stem` if needed).
    acq_timestamp : float
        The device's acquisition timestamp double.
    file_tail : str
        Verbatim filename tail after the timestamp, including the leading
        dot of the extension (e.g. ``".png"``).

    Returns
    -------
    str
        ``{device_name}_{acq_timestamp:.3f}{file_tail}``.
    """
    return f"{device_name}_{render_timestamp(acq_timestamp)}{file_tail}"


def native_file_name_from_key(device_name: str, key: int, file_tail: str) -> str:
    """Return the native filename for an integer-millisecond timestamp key.

    Used to reconstruct candidate filenames from :func:`timestamp_key` /
    :func:`timestamp_key_candidates` values when probing for a file whose
    exact ``%.3f`` rendering may sit at a rounding boundary.

    Parameters
    ----------
    device_name : str
        The on-disk stem (see :func:`native_file_name`).
    key : int
        Integer-millisecond timestamp key.
    file_tail : str
        Verbatim filename tail (see :func:`native_file_name`).

    Returns
    -------
    str
        ``{device_name}_{key / 1000:.3f}{file_tail}``.
    """
    return native_file_name(device_name, key / 1000, file_tail)


def native_file_path(
    scan_folder: Path | str,
    device_name: str,
    acq_timestamp: float,
    file_tail: str,
    directory_suffix: str = "",
) -> Path:
    """Return the full expected native file path inside a scan folder.

    Parameters
    ----------
    scan_folder : Path or str
        The ``scans/Scan{NNN}`` folder (not the device subfolder).
    device_name : str
        The GEECS device name.  If it already carries the suffix (callers
        holding a pre-composed ``data_device_name``), leave
        ``directory_suffix`` empty.
    acq_timestamp : float
        The device's acquisition timestamp double.
    file_tail : str
        Verbatim filename tail (see :func:`native_file_name`).
    directory_suffix : str, optional
        Diagnostic-output discriminator folded into the stem (see
        :func:`native_file_stem`).

    Returns
    -------
    Path
        ``{scan_folder}/{stem}/{stem}_{acq_timestamp:.3f}{file_tail}``.
    """
    stem = native_file_stem(device_name, directory_suffix)
    return Path(scan_folder) / stem / native_file_name(stem, acq_timestamp, file_tail)


def timestamp_key(acq_timestamp: float) -> int:
    """Canonicalise an ``acq_timestamp`` double to an integer-millisecond key.

    Both representations of a shot's timestamp — the raw double from the
    event row / s-file and the ``%.3f``-rendered value parsed back out of a
    filename — canonicalise through this function, making the row↔file join
    an exact integer comparison.

    Parameters
    ----------
    acq_timestamp : float
        Timestamp in seconds.

    Returns
    -------
    int
        ``round(acq_timestamp * 1000)``.
    """
    return round(acq_timestamp * 1000)


def timestamp_key_candidates(key: int) -> tuple[int, int, int]:
    """Return the millisecond keys a row timestamp may have rendered to.

    A row double sitting at the ``%.3f`` rounding boundary (e.g.
    ``…59.5239997`` after an s-file text round-trip, where the device
    formatted ``…59.524``) can canonicalise one integer away from the
    filename's key.  Probing the exact key plus both neighbours covers
    every rendering the same physical timestamp can produce.  This is
    float-formatting canonicalisation, not a tolerance window.

    Parameters
    ----------
    key : int
        The row-side integer-millisecond key from :func:`timestamp_key`.

    Returns
    -------
    tuple of int
        ``(key, key - 1, key + 1)`` — in probe order, exact match first.
    """
    return (key, key - 1, key + 1)


def filename_timestamp_regex(file_tail: str) -> re.Pattern[str]:
    r"""Compile the timestamp-extraction pattern for native filenames.

    The returned pattern is meant for :meth:`re.Pattern.search` against a
    bare filename; group ``"ts"`` captures the rendered timestamp (parse
    with :class:`float`, canonicalise with :func:`timestamp_key`).

    Parameters
    ----------
    file_tail : str
        Verbatim filename tail the device writes (escaped and anchored at
        the end of the name, so only exact-tail files match).

    Returns
    -------
    re.Pattern
        ``_(?P<ts>\\d+\\.\\d+){file_tail}$``.
    """
    return re.compile(r"_(?P<ts>\d+\.\d+)" + re.escape(file_tail) + r"$")


def legacy_filename_regex(file_tail: str) -> re.Pattern[str]:
    """Compile the legacy Master Control filename pattern.

    Matches ``Scan{NNN}_{device_subject}_{shot:03d}{file_tail}`` names
    produced by MC / the legacy file-mover, which renames native files by
    shot number after the scan.  Meant for :meth:`re.Pattern.match`
    against a bare filename; named groups are ``scan_number``,
    ``device_subject`` and ``shot_number``.

    Parameters
    ----------
    file_tail : str
        Verbatim filename tail (escaped, end-anchored).

    Returns
    -------
    re.Pattern
        The compiled legacy-convention pattern.
    """
    return re.compile(
        r"Scan(?P<scan_number>\d{3,})_"  # scan number
        r"(?P<device_subject>.*?)_"  # non-greedy subject
        r"(?P<shot_number>\d{3,})"  # shot number
        + re.escape(file_tail)
        + r"$"  # literal suffix+format
    )
