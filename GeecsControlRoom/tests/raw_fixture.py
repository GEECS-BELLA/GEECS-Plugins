"""A test-only ``frame.raw`` route to Planning/raw_frames.md's contract, for the page's lane (R2).

Never mounted by the app: the server's own route is lane R1's. This one serves a
synthetic 600×600 frame (a dark floor of ~56 ± 3 counts and one hot pixel at
4000, which the 8-bit stream flattens and smears) exactly as the contract says
— uint16 little-endian, row-major, gzip framing from ``zlib.compressobj(1,
wbits=31)``, the geometry, full-scale and tag headers, ``ETag`` quoted strong,
a 304 on ``If-None-Match`` — so the page's decode path is driven against the
bytes the browser would get.
"""

from __future__ import annotations

import random
import sys
import zlib
from array import array
from dataclasses import dataclass, field

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

WIDTH = HEIGHT = 600
FLOOR, SPREAD = 56, 3
HOT = (412, 233, 4000)  # x, y, counts


def floor_frame(width: int = WIDTH, height: int = HEIGHT, seed: int = 7) -> array:
    """The dark floor with noise and one hot pixel, as ``ARR`` would serve it (uint16)."""
    rng = random.Random(seed)
    px = array("H", (max(0, min(4095, round(rng.gauss(FLOOR, SPREAD / 2)))) for _ in range(width * height)))  # fmt: skip
    x, y, v = HOT
    px[y * width + x] = v
    return px


def gzip_le(px: array) -> bytes:
    """The body: little-endian bytes, gzip framing (a browser decodes it; bare zlib it does not)."""
    if sys.byteorder != "little":
        px = array("H", px)
        px.byteswap()
    z = zlib.compressobj(1, wbits=31)
    return z.compress(px.tobytes()) + z.flush()


@dataclass
class Held:
    """The frame the fixture serves: its pixels, geometry, tag, and the full scale (None: unreadable)."""

    px: array = field(default_factory=floor_frame)
    width: int = WIDTH
    height: int = HEIGHT
    tag: int = 1
    full_scale: int | None = 4095
    silent: str | None = None  # a 503's words


def router(held: Held, name: str = "Cam1") -> APIRouter:
    """``GET /api/devices/{name}/frame.raw`` for one device; any other name has no ``raw_frame`` role (404)."""
    r = APIRouter()

    @r.get("/api/devices/{device}/frame.raw")
    async def frame_raw(request: Request, device: str) -> Response:
        if device != name:
            return JSONResponse({"code": "unavailable", "detail": f"the module names no raw_frame role for {device}"}, status_code=404)  # fmt: skip
        if held.silent:
            return JSONResponse({"code": "unavailable", "detail": held.silent}, status_code=503)  # fmt: skip
        etag = f'"{held.tag}"'
        tags = {"ETag": etag, "X-Frame-Tag": etag, "Cache-Control": "no-store"}
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=tags)
        body = gzip_le(held.px)
        headers = {
            **tags,
            "Content-Encoding": "gzip",
            "X-Frame-Width": str(held.width),
            "X-Frame-Height": str(held.height),
        }
        if held.full_scale is not None:
            headers["X-Frame-Full-Scale"] = str(held.full_scale)
        return Response(body, media_type="application/octet-stream", headers=headers)

    return r
