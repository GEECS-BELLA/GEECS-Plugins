"""The page's raw decode path end to end: the server's ``frame.raw`` route over the fake CA, then camera_rules.js.

The frame is fake-CA data, as aioca delivers ``ARR:ArrayData``: a 600×600
dark floor of ~56 ± 3 counts with one hot pixel at 4000, as int32. The route
(images.py) casts, gzip-frames and tags it; the bytes are checked on the wire,
then httpx undoes the gzip framing as a browser's ``fetch`` does, and the body
and headers go through ``rawAnswer``, ``decodeRaw``, ``transformFrame``,
``histogram`` and ``autoLevels`` under node, as camera.js calls them. The
server and the page are pinned against each other, never against a second
copy of the contract.
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

import httpx
import numpy as np

from tests._node import needs_node, run
from tests.test_images import RAW_URL, Env, _raw

WIDTH = HEIGHT = 600
FLOOR, SPREAD = 56, 3
HOT = (412, 233, 4000)  # x, y, counts


def floor_frame(width: int = WIDTH, height: int = HEIGHT, seed: int = 7) -> np.ndarray:
    """The dark floor with noise and one hot pixel, as aioca delivers ``ARR:ArrayData`` (int32, flat)."""
    rng = np.random.default_rng(seed)
    px = np.clip(np.rint(rng.normal(FLOOR, SPREAD / 2, width * height)), 0, 4095).astype(np.int32)  # fmt: skip
    x, y, v = HOT
    px[y * width + x] = v
    return px


def _camera(full_scale: float | None = 4095.0) -> Env:
    env = Env()
    env.ca.raw = (floor_frame(), WIDTH, HEIGHT)
    env.ca.counters = [1.0]
    env.ca.full_scale = full_scale
    return env


async def test_the_floor_frame_arrives_exactly_then_304_until_the_camera_moves() -> (
    None
):
    # the headers, framing and cast are test_images.py's; this pins the sequence a poller sees on this frame
    env = _camera()
    async with env.client() as c:
        r, wire = await _raw(c)
        assert r.status_code == 200 and r.headers["etag"] == '"1"'
        body = gzip.decompress(wire)
        assert (
            body == floor_frame().astype("<u2").tobytes()
        )  # uint16 LE row-major, exact
        same = await c.get(RAW_URL, headers={"If-None-Match": '"1"'})
        assert (same.status_code, same.content, same.headers["x-frame-tag"]) == (304, b"", '"1"')  # fmt: skip
        env.ca.counters, env.clock[0] = [2.0], env.clock[0] + 1.0
        new = await c.get(RAW_URL, headers={"If-None-Match": '"1"'})
    assert new.status_code == 200 and new.headers["etag"] == '"2"' and new.content == body  # fmt: skip


_PAGE = """
const R = require(process.argv[1]);
const answers = JSON.parse(require("fs").readFileSync(0, "utf8"));
const out = answers.map((a) => {
  const what = R.rawAnswer(a.status);
  if (what !== "frame") return { what, detail: a.detail };
  const bytes = Buffer.from(a.body, "base64");
  const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const f = R.decodeRaw(buf, a.headers);
  if (!f.ok) return { what, why: f.why };
  const t = R.transformFrame(a.transform, f.data, f.width, f.height);
  const hist = R.histogram(t.data, f.full), l = R.autoLevels(hist, f.full);
  let hot = -1, max = 0;
  for (let i = 0; i < t.data.length; i++) if (t.data[i] > max) { max = t.data[i]; hot = i; }
  return { what, width: t.width, height: t.height, full: f.full, assumed: f.assumed, tag: f.tag, bins: hist.length, levels: l, hot: [hot % t.width, Math.floor(hot / t.width), max], first: Array.from(f.data.slice(0, 8)) };
});
console.log(JSON.stringify(out));
"""


def _answer(r: httpx.Response, transform: str = "None") -> dict:
    """What the page's ``fetch`` sees: the status, the decoded body, the headers it reads."""
    if r.status_code != 200:
        detail = r.json().get("detail") if r.content else None
        return {"status": r.status_code, "detail": detail}
    h = r.headers
    return {
        "status": 200,
        "body": base64.b64encode(r.content).decode(),
        "headers": {
            "width": h.get("x-frame-width"),
            "height": h.get("x-frame-height"),
            "fullScale": h.get("x-frame-full-scale"),
            "tag": h.get("x-frame-tag"),
        },  # fmt: skip
        "transform": transform,
    }


@needs_node
async def test_the_page_decodes_the_frame_the_route_serves() -> None:
    env = _camera()
    async with env.client() as c:
        first = await c.get(RAW_URL)
        answers = [_answer(first), _answer(first, "Rot90")]
        answers.append(_answer(await c.get(RAW_URL, headers={"If-None-Match": first.headers["etag"]})))  # fmt: skip
    unscaled = _camera(full_scale=None)
    async with unscaled.client() as c:
        answers.append(_answer(await c.get(RAW_URL)))
    silent = _camera()
    silent.ca.counter_fails = True
    async with silent.client() as c:
        answers.append(_answer(await c.get(RAW_URL)))
    got = run("camera_rules.js", _PAGE, json.dumps(answers))
    x, y, v = HOT
    plain, rotated, same, assumed, quiet = got
    assert plain["what"] == "frame" and (plain["width"], plain["height"], plain["bins"]) == (600, 600, 4096)  # fmt: skip
    assert (
        plain["first"] == floor_frame()[:8].tolist()
    )  # the pixels the camera sent, decoded exactly
    assert plain["hot"] == [x, y, v] and plain["tag"] == '"1"' and not plain["assumed"]
    # auto levels sit on the floor's noise, the hot pixel past the 99.9 % point
    assert FLOOR - 5 <= plain["levels"]["black"] < FLOOR < plain["levels"]["white"] <= FLOOR + 5  # fmt: skip
    # TRANS1's Rot90 (clockwise): the driver's (x, y) lands at (h - 1 - y, x)
    assert rotated["hot"] == [HEIGHT - 1 - y, x, v]
    assert same == {"what": "same", "detail": None}
    assert (assumed["full"], assumed["assumed"]) == (4095, True)
    assert quiet == {"what": "error", "detail": "the frame counter (bench:cam:ARR:ArrayCounter_RBV) could not be read"}  # fmt: skip


def test_the_camera_page_carries_the_raw_url_and_toggle() -> None:
    html = (Path(__file__).resolve().parents[1] / "geecs_control_room/templates/device_camera.html").read_text()  # fmt: skip
    assert (
        'data-raw-url="{{ root }}/api/devices/{{ name | urlencode }}/frame.raw"' in html
    )
    assert 'id="raw-mode"' in html
