"""Helpers for IMAQ camera devices.

This module is intentionally side-effect-free at import time (no config or
database access), so it can be imported and unit-tested without live hardware.
It offers two ways to get decoded frames from a camera device:

- :func:`latest_image` — *pull*: decode the current frame on demand.
- :func:`on_image` — *push*: register a callback invoked per incoming frame.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

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


def on_image(
    device,
    callback: Callable[[Union[np.ndarray, str]], None],
    *,
    name: str = "image",
    decode: bool = True,
) -> str:
    """Invoke ``callback`` with each new frame as it streams in (push model).

    This is the counterpart to :func:`latest_image` (pull): it registers an
    update listener on the device so ``callback`` fires once per incoming TCP
    message that carries an image. It returns the listener name; pass it to
    ``device.unregister_update_listener(name)`` to stop.

    The image is extracted from the raw message via the device's own subscription
    parser (the single source of truth for the wire format) and decoded with
    :func:`geecs_data_utils.io.decode_imaq_image_string`. This matters because the
    update listener fires *before* the device finishes parsing the message into
    ``state``, so reading ``state["image"]`` from inside a listener would return
    the *previous* frame.

    Parameters
    ----------
    device : GeecsDevice
        The camera device to listen to. Must expose ``register_update_listener``
        and ``_subscription_parser`` (i.e. a ``GeecsDevice`` or subclass).
    callback : Callable
        Called as ``callback(image)`` per frame — a 2-D :class:`numpy.ndarray`
        when ``decode`` is True (default), or the raw flattened ``str`` payload
        when ``decode`` is False. Frames without an image are skipped, so the
        callback only ever sees real frames.
    name : str, optional
        Listener name (used to unregister). Defaults to ``"image"``.
    decode : bool, optional
        If False, hand the callback the undecoded flattened payload instead of a
        decoded array (e.g. to decode elsewhere). Defaults to True.

    Returns
    -------
    str
        The listener ``name``, for ``device.unregister_update_listener(name)``.

    Notes
    -----
    The callback runs on the TCP listener thread, so keep it fast — hand the
    frame off to a queue or worker thread rather than doing display or disk I/O
    inline, which would stall frame reception.

    Examples
    --------
    >>> from queue import Queue
    >>> frames = Queue(maxsize=2)
    >>> on_image(cam, frames.put_nowait)   # producer stays on the recv thread
    'image'
    >>> img = frames.get()                 # consumer: GUI loop / worker thread
    """

    def _listener(message: str) -> None:
        _, _, values = device._subscription_parser(message)
        blob = values.get("image")
        if blob:  # guardrail: only fire for frames that actually carry an image
            callback(decode_imaq_image_string(blob) if decode else blob)

    device.register_update_listener(name, _listener)
    return name
