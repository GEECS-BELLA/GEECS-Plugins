"""Helpers for IMAQ camera devices.

This module is intentionally side-effect-free at import time (no config or
database access), so it can be imported and unit-tested without live hardware.
It provides on-demand decoding of the latest frame a camera device has streamed
into its ``state`` dict via the TCP subscription.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from geecs_data_utils.io import decode_imaq_image_string


def latest_image(device) -> Optional[np.ndarray]:
    """Decode a device's most recent streamed frame to a 2-D array.

    The frame is decoded lazily, only when this function is called — nothing is
    decoded on the subscription thread. The decode itself is handled by
    :func:`geecs_data_utils.io.decode_imaq_image_string`, which transparently
    supports compressed (JPEG) and uncompressed (8/16-bit) device modes for any
    camera/ROI.

    Parameters
    ----------
    device : GeecsDevice
        Any device exposing a ``state`` mapping. Only ``state["image"]`` is read,
        so non-camera devices are handled gracefully (see Returns).

    Returns
    -------
    Optional[np.ndarray]
        The decoded image as a 2-D array, or ``None`` if the device has no
        ``image`` in its state — i.e. it is not a camera, or no frame has arrived
        yet. This guardrail avoids assuming every device is a camera.
    """
    raw = getattr(device, "state", {}).get("image")
    return decode_imaq_image_string(raw) if raw else None
