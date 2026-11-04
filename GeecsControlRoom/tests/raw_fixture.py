"""A test-only ``frame.raw``, served to the raw-frames contract (``Planning/raw_frames.md``), never the app's.

The page's raw mode is built against this route while the server's own lands
in its own slice: uint16 little-endian row-major, gzip framing from
``zlib.compressobj(1, wbits=31)``, ``X-Frame-Width``/``-Height``/``-Full-Scale``,
``X-Frame-Tag`` and a strong quoted ``ETag`` from the counter, a 304 with the
same validators on a matching ``If-None-Match``, ``Cache-Control: no-store``.
The frame is synthetic: a 600×600 dark floor of ~56 ± 3 counts with one hot
pixel at 4000, the two things the raw view exists to show.
"""

from __future__ import annotations

import array
import random
import sys
import zlib
from dataclasses import dataclass, field

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.responses import JSONResponse

WIDTH = HEIGHT = 600
FLOOR, NOISE = 56, 3
HOT, HOT_VALUE = (412, 187), 4000  # (x, y), in the driver's frame


def dark_frame(
    width: int = WIDTH, height: int = HEIGHT, seed: int = 1263
) -> array.array:
    """The fixture frame, row-major uint16: the floor's noise, the one hot pixel."""
    rng = random.Random(seed)
    data = array.array(
        "H", (max(0, round(rng.gauss(FLOOR, NOISE))) for _ in range(width * height))
    )
    data[HOT[1] * width + HOT[0]] = HOT_VALUE
    return data


def gzipped(data: array.array) -> bytes:
    """The body as the contract frames it: gzip (wbits=31), level 1, little-endian bytes."""
    if sys.byteorder == "big":  # the contract is little-endian on every host
        data = array.array("H", data)
        data.byteswap()
    z = zlib.compressobj(1, wbits=31)
    return z.compress(data.tobytes()) + z.flush()


@dataclass
class RawFixture:
    """The held frame and what the route answers; a test changes ``tag``, ``full_scale`` or ``fail``."""

    width: int = WIDTH
    height: int = HEIGHT
    tag: int = 12345
    #: None: the header omitted (the page assumes 4095 and says so)
    full_scale: int | None = 4095
    fail: str | None = None  # a 503's words
    data: array.array = field(default_factory=dark_frame)
    #: each request's If-None-Match
    requests: list[str | None] = field(default_factory=list)

    def router(self) -> APIRouter:
        """``GET /api/devices/{name}/frame.raw`` over this fixture."""
        r = APIRouter()

        @r.get("/api/devices/{name}/frame.raw")
        def frame_raw(name: str, request: Request) -> Response:
            asked = request.headers.get("if-none-match")
            self.requests.append(asked)
            if self.fail:
                return JSONResponse({"detail": self.fail}, status_code=503)
            etag = f'"{self.tag}"'
            head = {"ETag": etag, "X-Frame-Tag": etag, "Cache-Control": "no-store"}
            if asked == etag:
                return Response(status_code=304, headers=head)
            head |= {"X-Frame-Width": str(self.width), "X-Frame-Height": str(self.height), "Content-Encoding": "gzip"}  # fmt: skip
            if self.full_scale is not None:
                head["X-Frame-Full-Scale"] = str(self.full_scale)
            return Response(gzipped(self.data), media_type="application/octet-stream", headers=head)  # fmt: skip

        return r

    def app(self) -> FastAPI:
        """The route alone, for a test server."""
        app = FastAPI()
        app.include_router(self.router())
        return app
