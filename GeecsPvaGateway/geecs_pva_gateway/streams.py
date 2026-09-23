"""Decode and shape one pushed value for a served stream: an image, or an array.

The gateway serves two kinds of non-scalar variable and treats them alike
downstream (one ``NTNDArray`` PV, one file plugin each).  What differs is
here: how the wire payload becomes an array, and what shape it is posted in.

- **Images** decode with :func:`geecs_data_utils.io.decode_imaq_image_string`
  and are posted as they come (``(H, W)``, the camera's dtype).
- **Arrays** decode with :func:`geecs_data_utils.io.decode_array_payload`
  (the payload says which of the three wire shapes it is) to ``float64`` in
  physical units, then are **padded along axis 0 to the devicetype's
  ceiling** with NaN when one is declared
  (:func:`geecs_core.db.device_streams.array_ceiling`): the MagSpec lineouts'
  row count moves with the magnet current, and a Bluesky descriptor, a PV
  monitor and an HDF5 stack all want one shape per run.  Longer than the
  ceiling raises :class:`ArrayTooLongError` — the frame is dropped and
  counted by its consumer, never truncated, because a truncated spectrum is
  indistinguishable downstream from a real one that ends early.  With no
  ceiling the array is posted at its native length (a scope trace's length
  is its configured record).

A waveform's axis parameters (``x0``, ``dx`` in seconds, ``samples``, the
raw ``offset``/``gain`` and the channel ``name``) come back as the
attributes to attach to the posted ``NTNDArray``, so a live client can
rebuild the time axis without a companion PV.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from geecs_data_utils.io import decode_array_payload, decode_imaq_image_string

Attributes = dict[str, Union[float, int, str]]


class ArrayTooLongError(ValueError):
    """An array had more rows than its devicetype's ceiling; the frame is dropped, not cut."""


def pad_rows(values: np.ndarray, ceiling: int) -> np.ndarray:
    """*values* (1-D or 2-D) padded along axis 0 to *ceiling* rows with NaN, as ``float64``."""
    n = values.shape[0]
    if n > ceiling:
        raise ArrayTooLongError(
            f"array of {n} rows exceeds the devicetype ceiling of {ceiling}; "
            "dropped, never truncated"
        )
    out = np.full((ceiling, *values.shape[1:]), np.nan, dtype=np.float64)
    out[:n] = values
    return out


def decode_image(blob: Union[str, bytes]) -> tuple[np.ndarray, Attributes]:
    """An image variable's payload → ``(frame, {})``."""
    return decode_imaq_image_string(blob), {}


def decode_array(
    blob: Union[str, bytes], ceiling: int | None
) -> tuple[np.ndarray, Attributes]:
    """An array variable's payload → ``(values, attributes)``, padded to *ceiling* when given."""
    decoded = decode_array_payload(blob)
    values = decoded.values.astype(np.float64, copy=False)
    if ceiling is not None:
        values = pad_rows(values, ceiling)
    return values, dict(decoded.attributes)
