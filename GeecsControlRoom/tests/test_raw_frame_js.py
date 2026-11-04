"""The page's raw poll (camera_rules.js fetchRaw) against the test-only fixture route, over real HTTP.

Node's ``fetch`` decodes ``Content-Encoding: gzip`` as a browser's does, so
this is the page's decode path end to end: the contract's framing, headers,
``ETag``/304 and words, the frame as the page would hold it. Skipped without
node; the server is the fixture (``tests/raw_fixture.py``), never the app's.
"""

from __future__ import annotations

import gzip
import json
import socket
import subprocess
import threading
from collections.abc import Iterator

import pytest
import uvicorn
from fastapi.testclient import TestClient

from tests._node import STATIC, needs_node
from tests.raw_fixture import FLOOR, HOT, HOT_VALUE, RawFixture, gzipped

_POLL = """
const R = require(process.argv[1]);
const [url, etag, x, y] = JSON.parse(process.argv[2]);
(async () => {
  let got;
  try { got = await R.fetchRaw(fetch, url, etag || null); } catch (e) { console.log(JSON.stringify({ error: e.message })); return; }
  if (!got.fresh) { console.log(JSON.stringify(got)); return; }
  const f = got.frame, d = f.data;
  let sum = 0, max = 0;
  for (const v of d) { sum += v; if (v > max) max = v; }
  const hist = R.histogram(d, f.full), levels = R.autoLevels(hist, f.full);
  console.log(JSON.stringify({ fresh: true, width: f.width, height: f.height, full: f.full, assumed: f.assumed, tag: f.tag, etag: f.etag,
    n: d.length, hot: d[y * f.width + x], max, mean: sum / d.length, bins: hist.length, levels }));
})();
"""


@pytest.fixture
def served() -> Iterator[tuple[RawFixture, str]]:
    fixture = RawFixture()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(fixture.app(), host="127.0.0.1", port=port, log_level="warning"))  # fmt: skip
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        thread.join(0.01)
    yield fixture, f"http://127.0.0.1:{port}/api/devices/Cam1/frame.raw"
    server.should_exit = True
    thread.join(5)


def poll(url: str, etag: str | None = None) -> dict:
    """One poll as the page makes it, under node."""
    args = json.dumps([url, etag, *HOT])
    out = subprocess.run(["node", "-e", _POLL, str(STATIC / "camera_rules.js"), args], capture_output=True, text=True, check=True, timeout=20)  # fmt: skip
    return json.loads(out.stdout)


def test_the_fixture_is_gzip_framed_to_the_contract() -> None:
    fixture = RawFixture()
    body = gzipped(fixture.data)
    assert body[:2] == b"\x1f\x8b"  # gzip, not zlib framing
    assert gzip.decompress(body) == fixture.data.tobytes()
    with TestClient(fixture.app()) as c:
        r = c.get("/api/devices/Cam1/frame.raw")
        assert r.headers["content-encoding"] == "gzip" and r.headers["etag"] == r.headers["x-frame-tag"] == '"12345"'  # fmt: skip
        assert int(r.headers["content-length"]) == len(body) < len(r.content) == 2 * 600 * 600  # fmt: skip
        assert c.get("/api/devices/Cam1/frame.raw", headers={"If-None-Match": '"12345"'}).status_code == 304  # fmt: skip


@needs_node
def test_the_page_decodes_the_fixture_frame(served: tuple[RawFixture, str]) -> None:
    fixture, url = served
    got = poll(url)
    assert {k: got[k] for k in ("fresh", "width", "height", "full", "assumed", "tag", "etag", "n", "bins")} == {
        "fresh": True, "width": 600, "height": 600, "full": 4095, "assumed": False, "tag": '"12345"', "etag": '"12345"', "n": 360000, "bins": 4096,
    }  # fmt: skip
    assert got["hot"] == got["max"] == HOT_VALUE and abs(got["mean"] - FLOOR) < 0.1
    # auto levels on 0..4095 bracket the floor's noise; the hot pixel is one in 360000, past the 99.9 % point
    assert (
        FLOOR - 12
        <= got["levels"]["black"]
        < FLOOR
        < got["levels"]["white"]
        <= FLOOR + 12
    )


@needs_node
def test_a_304_is_nothing_new_and_a_new_tag_is_a_frame(
    served: tuple[RawFixture, str],
) -> None:
    fixture, url = served
    assert poll(url, '"12345"') == {"fresh": False}
    fixture.tag = 12346
    assert poll(url, '"12345"')["etag"] == '"12346"'
    assert fixture.requests == ['"12345"', '"12345"']


@needs_node
def test_full_scale_assumed_and_the_503_words(served: tuple[RawFixture, str]) -> None:
    fixture, url = served
    fixture.full_scale = None
    got = poll(url)
    assert (got["full"], got["assumed"], got["bins"]) == (4095, True, 4096)
    fixture.fail = "the raw frame (bench:cam:ARR:ArrayData) could not be read: timeout"
    assert poll(url) == {"error": fixture.fail}
