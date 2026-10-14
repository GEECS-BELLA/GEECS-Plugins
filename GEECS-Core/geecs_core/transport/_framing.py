"""The GEECS TCP framing: a 4-byte big-endian length, then the payload.

Shared by the ``Wait>>`` subscription (:mod:`.tcp_subscriber`), the scan-mode
path channel (:mod:`.scan_mode`) and the fakes in ``geecs_core.testing`` —
one spelling, so the three cannot drift.
"""

from __future__ import annotations

import asyncio
import struct


def frame(payload: bytes) -> bytes:
    """Length-prefix *payload*: 4-byte big-endian signed length, then the bytes."""
    return struct.pack(">i", len(payload)) + payload


async def read_frame(reader: asyncio.StreamReader) -> bytes:
    """Read one frame's payload; ``asyncio.IncompleteReadError`` at end of stream."""
    header = await reader.readexactly(4)
    (length,) = struct.unpack(">i", header)
    return await reader.readexactly(max(length, 0))
