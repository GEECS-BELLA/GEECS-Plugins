"""The page's raw decode path against a route to the contract (tests/raw_fixture.py), and the page's wiring.

httpx undoes the gzip framing as a browser's ``fetch`` does; the body and the
headers then go through camera_rules.js's ``rawAnswer``, ``decodeRaw``,
``transformFrame``, ``histogram`` and ``autoLevels`` under node, as camera.js
calls them.
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests._node import needs_node, run
from tests.raw_fixture import FLOOR, HEIGHT, HOT, WIDTH, Held, router

URL = "/api/devices/Cam1/frame.raw"


@pytest.fixture
def held() -> Held:
    return Held()


@pytest.fixture
def fixture(held: Held) -> TestClient:
    app = FastAPI()
    app.include_router(router(held))
    return TestClient(app)


def test_the_fixture_honours_the_contract(fixture: TestClient, held: Held) -> None:
    with fixture.stream("GET", URL) as r:
        # the bytes as sent, before httpx undoes the encoding
        wire = b"".join(r.iter_raw())
    assert r.status_code == 200 and r.headers["content-encoding"] == "gzip"
    assert r.headers["content-type"] == "application/octet-stream" and r.headers["content-length"] == str(len(wire))  # fmt: skip
    assert r.headers["etag"] == r.headers["x-frame-tag"] == '"1"' and r.headers["cache-control"] == "no-store"  # fmt: skip
    assert (r.headers["x-frame-width"], r.headers["x-frame-height"], r.headers["x-frame-full-scale"]) == ("600", "600", "4095")  # fmt: skip
    body = gzip.decompress(wire)  # gzip framing: bare zlib would raise here
    assert len(body) == WIDTH * HEIGHT * 2 and body == fixture.get(URL).content
    held.tag = 2
    assert fixture.get(URL, headers={"If-None-Match": '"1"'}).status_code == 200
    same = fixture.get(URL, headers={"If-None-Match": '"2"'})
    assert (same.status_code, same.content, same.headers["etag"], same.headers["x-frame-tag"]) == (304, b"", '"2"', '"2"')  # fmt: skip
    assert fixture.get("/api/devices/Other/frame.raw").status_code == 404


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
  return { what, width: t.width, height: t.height, full: f.full, assumed: f.assumed, tag: f.tag, bins: hist.length, levels: l, hot: [hot % t.width, Math.floor(hot / t.width), max] };
});
console.log(JSON.stringify(out));
"""


def _answer(r: httpx.Response, transform: str = "None") -> dict:
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
def test_the_page_decodes_the_fixture_frame(fixture: TestClient, held: Held) -> None:
    first = fixture.get(URL)
    answers = [_answer(first), _answer(first, "Rot90")]
    answers.append(_answer(fixture.get(URL, headers={"If-None-Match": first.headers["etag"]})))  # fmt: skip
    held.full_scale = None
    answers.append(_answer(fixture.get(URL)))
    held.silent = "the frame counter (bench:cam:ARR:ArrayCounter_RBV) could not be read"
    answers.append(_answer(fixture.get(URL)))
    got = run("camera_rules.js", _PAGE, json.dumps(answers))
    x, y, v = HOT
    plain, rotated, same, assumed, silent = got
    assert plain["what"] == "frame" and (plain["width"], plain["height"], plain["bins"]) == (600, 600, 4096)  # fmt: skip
    assert plain["hot"] == [x, y, v] and plain["tag"] == '"1"' and not plain["assumed"]
    # auto levels sit on the floor's noise, the hot pixel past the 99.9 % point
    assert FLOOR - 5 <= plain["levels"]["black"] < FLOOR < plain["levels"]["white"] <= FLOOR + 5  # fmt: skip
    # TRANS1's Rot90 (clockwise): the driver's (x, y) lands at (h - 1 - y, x)
    assert rotated["hot"] == [HEIGHT - 1 - y, x, v]
    assert same == {"what": "same", "detail": None}
    assert (assumed["full"], assumed["assumed"]) == (4095, True)
    assert silent == {"what": "error", "detail": "the frame counter (bench:cam:ARR:ArrayCounter_RBV) could not be read"}  # fmt: skip


def test_the_camera_page_carries_the_raw_url_and_toggle() -> None:
    html = (Path(__file__).resolve().parents[1] / "geecs_control_room/templates/device_camera.html").read_text()  # fmt: skip
    assert (
        'data-raw-url="{{ root }}/api/devices/{{ name | urlencode }}/frame.raw"' in html
    )
    assert 'id="raw-mode"' in html
