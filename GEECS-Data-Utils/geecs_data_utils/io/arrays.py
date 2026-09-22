"""Decoders for the array payloads GEECS devices push over TCP (not files).

Beside the IMAQ image (:func:`geecs_data_utils.io.images.decode_imaq_image_string`),
GEECS devices push their ``1darray``-typed variables in three text-or-binary
shapes, each established byte-exact on the reference deployment (2026-09):

- **Nested pairs** — ``[[x,y], [x,y], ...]``: the MagSpec lineouts
  (``interpSpec``, ``interpDiv``).  One variable carries the axis (column 0)
  *and* the values (column 1); the row count varies with the device state
  (the energy span over a fixed ``dE``), and a single row is a valid value
  (the magnet-off default).  → ``(n, 2)`` ``float64``.
- **CSV values** — ``v,v,v,...``, CRLF-terminated: the Hamamatsu spectrometer
  (``counts``, ``wavelength``) and the MagSpec axes (``EnergyAxis``,
  ``AngleAxis``).  → ``(n,)`` ``float64``.
- **LabVIEW flattened waveform** — the scopes (PicoscopeV2, DaqPad_NI6009)::

      "<actualSamples>,<relativeInitialX>,<xIncrement>,<offset>,<gain>,<name>|"
        + uint32 big-endian sample count
        + count × int16 big-endian raw samples
      volts[i] = offset + gain * raw[i]      t[i] = relativeInitialX + i * xIncrement

  Self-describing: the axis and the scaling ride in the header.  Decoded to
  **physical units** (volts; the axis parameters returned beside the values
  in seconds) so no consumer needs the raw counts.  → ``(n,)`` ``float64``.

:func:`decode_array_payload` tells the three apart by **sniffing the
payload**, never by devicetype: a leading ``[[`` is pairs, a six-field
header ending in ``|`` is a waveform, anything else numeric is CSV.  Every
decoder raises :class:`ValueError` on a payload it cannot account for byte
by byte (a ragged row, a stray bracket, a leftover byte, a count that
disagrees with the header, an empty record) — a truncated, empty or misread
array must never come back looking valid.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from typing import Literal, Union

import numpy as np

from geecs_data_utils.io.images import _flatten_string_to_bytes

ArrayKind = Literal["pairs", "csv", "waveform"]

#: The keys a waveform's :attr:`DecodedArray.attributes` carries — the axis
#: (``x0``, ``dx`` in seconds, ``samples``) and the raw scaling — spelled
#: here once for every consumer that stores or displays them.
WAVEFORM_AXIS_KEYS: tuple[str, ...] = ("x0", "dx", "samples")
WAVEFORM_ATTRIBUTE_KEYS: tuple[str, ...] = (
    *WAVEFORM_AXIS_KEYS,
    "offset",
    "gain",
    "name",
)

_PAIR = re.compile(r"\[([^\[\]]*)\]")
#: The whole pairs payload: ``[`` rows ``]`` with rows ``[..]`` separated by
#: commas — nothing else between, before or after (a stray bracket or a
#: missing comma is a malformed payload, not a shorter one).
_PAIRS_PAYLOAD = re.compile(r"^\[\s*\[[^\[\]]*\](?:\s*,\s*\[[^\[\]]*\])*\s*\]$")
#: Five numeric fields, a name, then the ``|`` that ends the waveform header.
_WAVEFORM_HEADER = re.compile(
    rb"^([^,|]+),([^,|]+),([^,|]+),([^,|]+),([^,|]+),([^|]*)\|"
)


@dataclass(frozen=True)
class DecodedArray:
    """One decoded array payload.

    Attributes
    ----------
    values :
        ``(n,)`` or ``(n, 2)`` ``float64``.
    kind :
        Which wire shape it was.
    attributes :
        What the payload said about itself beyond the values — for a
        waveform ``x0`` (s), ``dx`` (s), ``samples``, ``offset``, ``gain`` and
        ``name``; empty for the text shapes.
    """

    values: np.ndarray
    kind: ArrayKind
    attributes: dict[str, Union[float, int, str]] = field(default_factory=dict)


def _text(blob: Union[str, bytes]) -> str:
    return blob if isinstance(blob, str) else bytes(blob).decode("latin-1")


def decode_nested_pairs(blob: Union[str, bytes]) -> np.ndarray:
    """Decode ``[[x,y], [x,y], ...]`` to an ``(n, 2)`` ``float64`` array, ``n >= 1``."""
    text = _text(blob).strip()
    if not _PAIRS_PAYLOAD.match(text):
        raise ValueError("nested pairs: payload is not [[x,y], [x,y], ...]")
    rows = _PAIR.findall(text[1:-1])
    try:
        values = np.array(
            [[float(v) for v in row.split(",")] for row in rows], dtype=np.float64
        )
    except ValueError as exc:
        raise ValueError(f"nested pairs: non-numeric cell ({exc})") from exc
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(
            f"nested pairs: every row must have exactly 2 columns, got shape {values.shape}"
        )
    return values


def decode_csv_values(blob: Union[str, bytes]) -> np.ndarray:
    """Decode a comma-separated list of numbers (any line ending) to ``(n,)`` ``float64``."""
    text = _text(blob).strip()
    if not text:
        raise ValueError("csv values: empty payload")
    try:
        return np.array([float(v) for v in text.split(",")], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"csv values: non-numeric cell ({exc})") from exc


def decode_labview_waveform(blob: Union[str, bytes]) -> DecodedArray:
    """Decode a LabVIEW flattened waveform to volts, with its axis parameters.

    Raises :class:`ValueError` when the header is not the six-field shape,
    when the header's sample count and the binary count disagree, or when
    the payload is longer or shorter than the count implies.
    """
    data = _flatten_string_to_bytes(blob)
    match = _WAVEFORM_HEADER.match(data)
    if match is None:
        raise ValueError(
            "waveform: header is not '<n>,<x0>,<dx>,<offset>,<gain>,<name>|'"
        )
    try:
        samples = int(float(match.group(1)))
        x0, dx, offset, gain = (float(match.group(i)) for i in range(2, 6))
    except ValueError as exc:
        raise ValueError(f"waveform: non-numeric header field ({exc})") from exc
    name = match.group(6).decode("latin-1")
    body = data[match.end() :]
    if len(body) < 4:
        raise ValueError("waveform: no sample count after the header")
    (count,) = struct.unpack(">I", body[:4])
    if count == 0:
        raise ValueError(
            "waveform: empty record (0 samples) — no value, not a short one"
        )
    if count != samples:
        raise ValueError(
            f"waveform: header says {samples} samples, count field {count}"
        )
    expected = 4 + 2 * count
    if len(body) != expected:
        raise ValueError(
            f"waveform: {len(body)} payload bytes for {count} samples (expected {expected})"
        )
    raw = np.frombuffer(body, dtype=">i2", count=count, offset=4).astype(np.float64)
    return DecodedArray(
        values=offset + gain * raw,
        kind="waveform",
        attributes=dict(
            zip(
                WAVEFORM_ATTRIBUTE_KEYS,
                (x0, dx, count, offset, gain, name),
                strict=True,
            )
        ),
    )


def decode_array_payload(blob: Union[str, bytes]) -> DecodedArray:
    """Decode any of the three array shapes, chosen by sniffing the payload."""
    data = _flatten_string_to_bytes(blob)
    head = data.lstrip()[:2]
    if head.startswith(b"[["):
        return DecodedArray(values=decode_nested_pairs(data), kind="pairs")
    if _WAVEFORM_HEADER.match(data):
        return decode_labview_waveform(data)
    return DecodedArray(values=decode_csv_values(data), kind="csv")
