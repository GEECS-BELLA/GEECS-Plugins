"""Test fixtures for the GEECS data layout: scan-folder skeletons and forged payloads.

**Test code only. Never import this module from production code.** It
creates ``scans/ScanNNN/`` folders, which only the scanner side may do in
production (root ``CLAUDE.md`` § "Analysis code is a consumer of scan
folders, never a producer"); here it plays the scanner, building a
fixture tree under a test's ``tmp_path``.

Nothing here imports pytest or pandas: a table argument is anything with
pandas' ``to_csv`` (a ``DataFrame``), or the file's text.
"""

from __future__ import annotations

import os
import struct
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Protocol

from geecs_data_utils.data.sfile import scan_data_txt_path_for, sfile_path_for_scan
from geecs_data_utils.io.images import _IMAQ_CLASS_MARKER


class _Table(Protocol):
    """A table written the way the scanner writes it (a pandas ``DataFrame``)."""

    def to_csv(self, path: Path, *, sep: str, index: bool) -> object:
        """Write the table to *path*."""


def _write_table(path: Path, rows: _Table | str) -> Path:
    if isinstance(rows, str):
        path.write_text(rows)
    else:
        rows.to_csv(path, sep="\t", index=False)
    return path


def scan_info_text(fields: Mapping[str, object]) -> str:
    """The ``[Scan Info]`` ini text, every value quoted: ``Key = "value"``."""
    return "[Scan Info]\n" + "".join(f'{k} = "{v}"\n' for k, v in fields.items())


def write_sfile(scan: Path, rows: _Table | str) -> Path:
    """Write the scan's s-file, ``{day}/analysis/s{N}.txt``, tab-separated."""
    path = sfile_path_for_scan(scan)
    path.parent.mkdir(exist_ok=True)
    return _write_table(path, rows)


def make_scan_folder(
    scan: Path,
    *,
    scan_info: Mapping[str, object] | str | None = None,
    devices: Iterable[str] = (),
    scan_data: _Table | str | None = None,
    sfile: _Table | str | None = None,
) -> Path:
    """Build a scan folder skeleton at *scan* (``…/scans/ScanNNN``), as the scanner would.

    Written in this order, each only when given: the folder (``parents=True``;
    it must not exist yet), the device folders, ``ScanInfoScanNNN.ini``, the
    raw ``ScanDataScanNNN.txt`` table, the s-file. A test that needs the
    s-file written after its device files calls :func:`write_sfile` itself.

    Parameters
    ----------
    scan : Path
        The scan folder, e.g. from ``ScanPaths.get_scan_folder_path``; any
        layout the path encodes (``data_root_experiment`` included) is kept.
    scan_info : mapping or str, optional
        The ScanInfo ini: a mapping rendered by :func:`scan_info_text`, or
        the file's text verbatim.
    devices : iterable of str
        Device folders to create inside the scan folder.
    scan_data : DataFrame or str, optional
        The scanner-written ``ScanDataScanNNN.txt``.
    sfile : DataFrame or str, optional
        The s-file (see :func:`write_sfile`).

    Returns
    -------
    Path
        *scan*.
    """
    scan.mkdir(parents=True)
    for device in devices:
        (scan / device).mkdir()
    if scan_info is not None:
        text = scan_info if isinstance(scan_info, str) else scan_info_text(scan_info)
        (scan / f"ScanInfo{scan.name}.ini").write_text(text)
    if scan_data is not None:
        _write_table(scan_data_txt_path_for(scan), scan_data)
    if sfile is not None:
        write_sfile(scan, sfile)
    return scan


def write_closed(folder: Path, *, age: float = 60.0) -> None:
    """What the scanner leaves at the end of a run: the table, then the s-file.

    Both are dated *age* seconds in the past, so the run reads as closed
    and, with the default, settled.
    """
    table = scan_data_txt_path_for(folder)
    table.write_text("Shotnumber\n1\n")
    sfile = sfile_path_for_scan(folder)
    sfile.parent.mkdir(parents=True, exist_ok=True)
    sfile.write_text("Shotnumber\n1\n")
    then = time.time() - age
    for path in (table, sfile):
        os.utime(path, (then, then))


def imaq_struct(width: int, height: int, border: int = 0, pixel_type: int = 0) -> bytes:
    """A 64-byte IMAQ image struct: type, width, height and border, little-endian."""
    sb = bytearray(64)
    struct.pack_into("<i", sb, 36, pixel_type)
    struct.pack_into("<i", sb, 40, width)
    struct.pack_into("<i", sb, 48, height)
    struct.pack_into("<i", sb, 56, border)
    return bytes(sb)


def forge_imaq_blob(
    payload: bytes,
    struct_bytes: bytes = b"",
    *,
    device: bytes,
    before_payload: bytes | None = None,
) -> bytes:
    """Forge a LabVIEW NI-IMAQ flattened-image byte stream around *payload*.

    The layout the decoder reads: the big-endian length-prefixed device
    name, the class marker, 8 wrapper bytes and the IMAQ struct, then the
    prefixed device name again immediately before the pixels. Cameras whose
    wrapper carries something else there (a compression-mode field) are
    forged by passing those bytes as *before_payload*.
    """
    prefixed = struct.pack(">I", len(device)) + device
    wrapper = b"nivissvc.*" + _IMAQ_CLASS_MARKER + b"\x00" * 8 + struct_bytes
    gap = prefixed if before_payload is None else before_payload
    return prefixed + wrapper + gap + payload
