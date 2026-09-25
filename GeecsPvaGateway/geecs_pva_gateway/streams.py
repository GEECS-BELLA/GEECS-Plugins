"""Decode and shape one pushed value for a served stream: an image, or an array.

The gateway serves two kinds of non-scalar variable and treats them alike
downstream (one ``NTNDArray`` PV, one file plugin each).  What differs is
here: how the wire payload becomes an array, and what shape it is posted in.

- **Images** decode with :func:`geecs_data_utils.io.decode_imaq_image_string`
  and are posted as they come (``(H, W)``, the camera's dtype).
- **Arrays** decode with :func:`geecs_data_utils.io.decode_array_payload`
  (the payload says which of the three wire shapes it is) to ``float64`` in
  physical units, and are posted at their native length — never padded.
  A variable-length array (the MagSpec lineouts, whose row count is the
  energy span over the configured ΔE) is fixed for a scan the way an image
  is: the file plugin takes the stack shape at the arm and drops and counts
  any frame of another shape.

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


def decode_image(blob: Union[str, bytes]) -> tuple[np.ndarray, Attributes]:
    """An image variable's payload → ``(frame, {})``."""
    return decode_imaq_image_string(blob), {}


def decode_array(blob: Union[str, bytes]) -> tuple[np.ndarray, Attributes]:
    """An array variable's payload → ``(values, attributes)`` at native length."""
    decoded = decode_array_payload(blob)
    return decoded.values.astype(np.float64, copy=False), dict(decoded.attributes)
