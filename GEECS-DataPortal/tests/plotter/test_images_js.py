"""Run the Images page's big view (images.js) under node: which tile it
opens on, which route draws it, and the trace device's stepping."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

IMAGES_JS = (
    Path(__file__).resolve().parents[2]
    / "geecs_portal"
    / "plotter"
    / "static"
    / "images.js"
)

#: A page with the big view and a strip of three tiles (bins 1, 1, 2 for a
#: shots strip; bins 1, 2, 3 for a bins strip), and stubs for what
#: images.js touches: fetch records its URLs; Plotly records the draws.
_HARNESS = r"""
const assert = require('node:assert/strict');
const fetched = [];
let drawn = 0;
const listeners = {};
const node = (extra) => Object.assign({
  hidden: false, textContent: "", value: "", dataset: {}, options: [],
  addEventListener(kind, fn) { (this._on = this._on || {})[kind] = fn; },
  toggleAttribute(name, on) { this[name] = on; },
  add(option) { this.options.push(option); },
  querySelector() { return this.img; },
}, extra || {});
const tiles = BINS.map((bin, i) => node({
  dataset: {index: String(MODE === "shots" ? i + 1 : i), bin, caption: `tile ${i}`},
  img: {getAttribute: () => `/plot/api/images/tile.png?index=${i}&cmap=gray`},
}));
const els = {
  "bigview": node({dataset: {mode: MODE, device: "UC_Cam", day: "2026-10-02", scan: "1", uid: "folder:x:1",
    shots: "5", cmap: "gray", inPortal: "true", portalRoot: "/portal"}}),
  "big-image": node(), "big-trace": node(), "big-note": node(), "big-caption": node(),
  "big-processing": node(), "big-processing-box": node({hidden: true}),
};
const document = {
  getElementById: (id) => els[id],
  querySelectorAll: (sel) => sel === "#strip .tile" ? tiles : [],
};
const location = {search: SEARCH};
const sessionStorage = new Map([["store", 1]]);
sessionStorage.getItem = () => null; sessionStorage.setItem = () => {};
const window = {addEventListener(kind, fn) { listeners[kind] = fn; }};
const fetch = (url) => { fetched.push(String(url));
  return Promise.resolve({ok: true, json: async () => (String(url).includes("/device?")
    ? {processing_options: ["UC_Crop"]} : {figure: {data: [], layout: {}}})}); };
let observed = null;
const MutationObserver = class { constructor(fn) { observed = fn; } observe() {} };
const Plotly = {react() { drawn++; }, purge() {}};
const Option = class { constructor(text, value) { this.text = text; this.value = value; } };
"""


def _run(tmp_path: Path, mode: str, bins: list[str], search: str, body: str) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required to run images.js")
    script = tmp_path / "images.cjs"
    script.write_text(
        f"const MODE = {json.dumps(mode)}, BINS = {json.dumps(bins)}, SEARCH = {json.dumps(search)};\n"
        + _HARNESS
        + IMAGES_JS.read_text().replace('"use strict";', "")
        + "\n(async () => {\nawait new Promise(setImmediate);\n"
        + body
        + "\n})().catch((e) => { console.error(e); process.exit(1); });\n"
    )
    result = subprocess.run(
        [node, str(script)], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_map_cell_opens_its_bin_tile(tmp_path: Path) -> None:
    _run(
        tmp_path,
        "bins",
        ["1", "2", "3"],
        "?bin=2",
        r"""
assert.equal(tiles[1]["aria-current"], true);
assert.equal(els["big-image"].src, "/plot/api/images/tile.png?index=1&cmap=gray");
""",
    )


def test_on_a_shots_strip_a_map_cell_opens_its_first_shot(tmp_path: Path) -> None:
    """The shot is its tile's picture (its ROI and levels) until a recipe
    is picked; then it is the portal's processing preview."""
    _run(
        tmp_path,
        "shots",
        ["1", "1", "2"],
        "?bin=2",
        r"""
assert.equal(tiles[2]["aria-current"], true);
assert.equal(els["big-image"].src, "/plot/api/images/tile.png?index=2&cmap=gray");
assert.ok(fetched.some((u) => u.startsWith("/portal/api/run/folder%3Ax%3A1/device?")));
assert.equal(els["big-processing-box"].hidden, false);
els["big-processing"].value = "UC_Crop";
els["big-processing"]._on.change();
const src = els["big-image"].src;
assert.ok(src.startsWith("/portal/run/folder%3Ax%3A1/image.png?"), src);
assert.ok(src.includes("shot=3") && src.includes("processing=UC_Crop"), src);
""",
    )


def test_the_big_view_follows_its_tiles_picture(tmp_path: Path) -> None:
    """A shared z scale read late, or a new colour map: the tile's picture
    changes, and the big view redraws from it (no timed retries)."""
    _run(
        tmp_path,
        "bins",
        ["1", "2", "3"],
        "",
        r"""
tiles[0].img.getAttribute = () => "";
tiles[0]._on.click();
assert.equal(els["big-note"].textContent, "Waiting for the strip's levels…");
tiles[0].img.getAttribute = () => "/plot/api/images/tile.png?index=0&zmin=1&zmax=9";
observed([{target: tiles[0].img}]);
assert.equal(els["big-image"].src, "/plot/api/images/tile.png?index=0&zmin=1&zmax=9");
""",
    )


def test_a_clicked_tile_is_the_one_shown(tmp_path: Path) -> None:
    _run(
        tmp_path,
        "bins",
        ["1", "2", "3"],
        "",
        r"""
assert.equal(tiles[0]["aria-current"], true);
tiles[2]._on.click();
assert.equal(tiles[2]["aria-current"], true);
""",
    )


def test_a_trace_device_draws_one_shot_from_the_trace_route(tmp_path: Path) -> None:
    _run(
        tmp_path,
        "trace",
        [],
        "",
        r"""
await new Promise(setImmediate);
assert.ok(fetched.some((u) => u.startsWith("/portal/api/run/folder%3Ax%3A1/trace?") && u.includes("shot=1")));
assert.equal(els["big-caption"].textContent, "shot 1 of 5");
assert.equal(drawn, 1);
assert.ok(!fetched.some((u) => u.includes("/device?")), "no processing for traces");
""",
    )
