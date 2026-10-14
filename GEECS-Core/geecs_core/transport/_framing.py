"""The GEECS TCP framing: a 4-byte big-endian length, then the payload.

Shared by the ``Wait>>`` subscription's write side (:mod:`.tcp_subscriber`),
the scan-mode path channel (:mod:`.scan_mode`) and the fakes in
``geecs_core.testing`` — one spelling, so they cannot drift.  The one
deliberate holdout is the subscriber's listen loop, which reads its own
headers and *skips* a non-positive length (a long-standing tolerance on the
live device stream); :func:`read_frame` instead treats a negative length as
a corrupt stream and raises, since nothing after it can be trusted.
"""

from __future__ import annotations

import asyncio
import struct


def frame(payload: bytes) -> bytes:
    """Length-prefix *payload*: 4-byte big-endian signed length, then the bytes."""
    return struct.pack(">i", len(payload)) + payload


async def read_frame(reader: asyncio.StreamReader) -> bytes:
    """Read one frame's payload (``b""`` for an empty frame).

    Raises ``asyncio.IncompleteReadError`` at end of stream and ``ValueError``
    on a negative length: the stream is corrupt and cannot be resynchronised.
    """
    header = await reader.readexactly(4)
    (length,) = struct.unpack(">i", header)
    if length < 0:
        raise ValueError(f"corrupt frame header: length {length}")
    return await reader.readexactly(length)
