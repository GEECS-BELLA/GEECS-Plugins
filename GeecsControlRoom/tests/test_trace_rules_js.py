"""trace_rules.js under node: each row one rule of the trace page (skipped without node).

The decode runs on the route's own bytes: ``test_the_page_decodes_the_trace_the_route_serves``
serves a scope capture through ``traces.py`` over the fake CA and reshapes it here.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx

from tests._node import call_all, needs_node, run
from tests.test_traces import capture, client, scope

VIEW = {"ok": True, "state": "live", "roles": {}}
CH = ["ch_a_enable", "ch_a_range", "ch_a_coupling", "ch_a_offset"]
KNOWN = CH + ["ch_a_present", "acquire"]

ROWS: list[tuple[str, str, list[Any], Any]] = [
    # the scale
    ("x range: the axis' extent", "xRange", [[3, 1, 2, None], None], [1, 3]),
    ("x range: the viewer's window wins", "xRange", [[0, 10], [2, 4]], [2, 4]),
    ("x range: one point opens one unit", "xRange", [[5], None], [5, 6]),
    ("ticks: 1, 2 or 5 × 10^k", "ticks", [0, 1, 5], [0, 0.2, 0.4, 0.6, 0.8, 1]),
    ("ticks: across zero, no float noise", "ticks", [-1e-6, 1e-6, 4], [-1e-6, -5e-7, 0, 5e-7, 1e-6]),
    ("log ticks: the decades inside", "logTicks", [0.5, 2000], [1, 10, 100, 1000]),
    ("si: microseconds", "si", [4e-6], {"factor": 1e-6, "prefix": "µ"}),
    ("si: millivolts", "si", [0.25], {"factor": 0.001, "prefix": "m"}),
    ("si: volts at 1", "si", [1], {"factor": 1, "prefix": ""}),
    ("si: nothing to scale", "si", [0], {"factor": 1, "prefix": ""}),
    ("tick label: in the prefix's units", "tickLabel", [2.5e-6, 1e-6], "2.5"),
    ("axis title: label, prefix, units", "axisTitle", ["time axis", "s", "µ"], "time axis (µs)"),
    ("axis title: no units, the label alone", "axisTitle", ["samples", "", "m"], "samples"),
    # the cursor and the window
    ("nearest: the closer sample", "nearest", [[0, 1, 2, 3], 1.6], 2),
    ("nearest: past the end, the last", "nearest", [[0, 1, 2], 9], 2),
    ("nearest: an empty axis", "nearest", [[], 1], -1),
    ("window: a drag over the drawn range", "windowFromDrag", [100, 20, 200, 0, 10], [1, 5]),
    ("window: a click is no window", "windowFromDrag", [50, 52, 200, 0, 10], None),
    ("window: a drag past the edge clamps", "windowFromDrag", [-30, 100, 200, 0, 10], [0, 5]),
    ("series name: one channel, the trace's label", "seriesName", [0, 1, "spectrum"], "spectrum"),
    ("series name: the scope's channel letters", "seriesName", [2, 4, "capture"], "C"),
    ("average: whole, 1..100", "averageOf", ["7.4"], 7),
    ("average: junk is 1", "averageOf", ["x"], 1),
    ("average: capped", "averageOf", [1000], 100),
    # the rail and the readings
    ("present: a channel's group, by its present role", "presentRole", [CH, KNOWN], "ch_a_present"),
    ("present: a channel's reading", "presentRole", [["ch_a_total"], KNOWN], "ch_a_present"),
    ("present: a mixed group is never hidden", "presentRole", [["acquire", "ch_a_range"], KNOWN], None),
    ("present: the device names no present role", "presentRole", [CH, CH], None),
    ("hidden: while present reads 0", "hidden", [{"value": 0}], True),
    ("hidden: present, or unknown, shows it", "hidden", [None], False),
    ("rate: first to last of the span's tags", "rate", [[0, 1000, 2000, 3000, 4000, 5000, 5900], 6000], {"seen": [1000, 2000, 3000, 4000, 5000, 5900], "text": "1.0 new/s"}),
    ("rate: one tag is no rate yet", "rate", [[4000], 6000], {"seen": [4000], "text": "1 new in 5 s"}),
    ("rate: none recent", "rate", [[0], 9000], {"seen": [], "text": "no new captures"}),
    # reused from camera_rules.js, never copied
    ("control: on the module's word", "control", [VIEW, {"role": "ch_a_range", "writable": True}, False], {"enabled": True, "why": ""}),
    ("control: read only", "control", [VIEW, {"role": "count", "writable": False}, False], {"enabled": False, "why": "read only (the module's policy)"}),
    ("format: an enum by its label", "fmt", [{"value": 1}, {"enumChoices": ["10 mV", "20 mV"]}], "20 mV"),
]  # fmt: skip


@needs_node
def test_every_row() -> None:
    got = call_all("trace_rules.js", [(fn, args) for _, fn, args, _ in ROWS])
    for (name, _, _, want), answer in zip(ROWS, got):
        assert answer == want, name


_HARNESS = """
const R = require(process.argv[1]);
const a = JSON.parse(require("fs").readFileSync(0, "utf8"));
const f32 = (xs) => Float32Array.from(xs.map((v) => (v === null ? NaN : v)));
const plain = (v) => Array.from(v, (x) => (Number.isFinite(x) ? Math.round(x * 1e6) / 1e6 : null));
const out = {};
// decode: the route's body, its headers
const bytes = Buffer.from(a.body, "base64");
const t = R.decodeTrace(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), a.headers);
out.decoded = { ok: t.ok, channels: t.channels, samples: t.samples, tag: t.tag, axis: plain(t.axis), series: t.series.map((s) => ({ off: s.off, values: plain(s.values) })) };
out.short = R.decodeTrace(new ArrayBuffer(12), a.headers);
out.unshaped = R.decodeTrace(new ArrayBuffer(12), { channels: null, samples: "3" });
// the viewer's average: last n of one shape, the finite values only
let tag = 0;
const cap = (vals, axis = [0, 1, 2], t = String(++tag)) => ({ channels: 1, samples: vals.length, axis: f32(axis), tag: t, series: [{ values: f32(vals), off: false }] });
let held = R.keep([], cap([1, 2, null]), 2);
held = R.keep(held, cap([3, 4, null]), 2);
held = R.keep(held, cap([5, 6, 9]), 2);
out.avg = { n: held.length, values: plain(R.averaged(held).series[0].values) };
out.again = R.keep(held, { ...held[1] }, 5).length; // the same tag asked again: not counted twice
out.untagged = R.keep(held, { ...held[1], tag: null }, 5).length;
out.reshaped = R.keep(held, { channels: 2, samples: 3, axis: f32([0, 1, 2]), tag: "x" }, 5).length;
out.moved = R.keep(held, cap([1, 1, 1], [0, 2, 4]), 5).length; // a new interval: the axis moved, start over
// y range: finite in the window, off channels out, log positive only
const axis = f32([0, 1, 2, 3]), s = [{ values: f32([1, null, 3, 100]), off: false }, { values: f32([null, null, null, null]), off: true }];
out.y = R.yRange(axis, s, 0, 2, false);
out.ylog = R.yRange(axis, [{ values: f32([-1, 0, 10, 1000]), off: false }], 0, 3, true);
out.ynone = R.yRange(axis, [s[1]], 0, 3, false);
// decimate: min and max per column, a NaN breaks the line
const many = f32([...Array(100).keys()].map((i) => (i === 50 ? null : i % 10)));
const pts = R.decimate(f32([...Array(100).keys()]), many, 0, 99, 10);
out.dec = { n: pts.length, gaps: pts.filter((p) => p === null).length, first: pts.slice(0, 2) };
out.decWindow = R.decimate(axis, f32([1, 2, 3, 4]), 1, 2, 100);
console.log(JSON.stringify(out));
"""


@needs_node
async def test_the_page_decodes_the_trace_the_route_serves() -> None:
    env = scope()
    async with client(env) as c:
        r = await c.get(
            "/api/devices/Cam/trace"
        )  # httpx undoes the gzip, as a browser's fetch does
    assert isinstance(r, httpx.Response) and r.status_code == 200
    headers = {"channels": r.headers["x-trace-channels"], "samples": r.headers["x-trace-samples"], "tag": r.headers["x-trace-tag"]}  # fmt: skip
    got = run("trace_rules.js", _HARNESS, json.dumps({"body": base64.b64encode(r.content).decode(), "headers": headers}))  # fmt: skip
    d = got["decoded"]
    assert (d["ok"], d["channels"], d["samples"], d["tag"]) == (True, 4, 5, '"7"')
    assert d["axis"] == [0, 1e-6, 2e-6, 3e-6, 4e-6]
    flat = capture()
    for c in range(
        4
    ):  # channel c is every fourth value from c: the IOC's channels-fastest layout
        want = [None if v != v else float(v) for v in flat[c::4]]
        assert d["series"][c] == {"off": c == 3, "values": want}
    assert d["series"][1]["values"] == [10, 11, 12, 13, 14]
    assert got["short"] == {"ok": False, "why": "shape disagrees: 12 bytes for 4×5"}
    assert got["unshaped"] == {"ok": False, "why": "the trace came without its shape"}
    # the last two captures, the finite values only; a new shape starts over
    assert (
        got["avg"] == {"n": 2, "values": [4, 5, 9]}
        and got["reshaped"] == got["moved"] == 1
    )
    assert (got["again"], got["untagged"]) == (2, 3)
    assert got["y"] == [0.9, 3.1] and got["ylog"] == [10, 1000] and got["ynone"] is None
    assert got["dec"]["gaps"] == 1 and got["dec"]["n"] <= 2 * 10 + 1 and got["dec"]["first"] == [[0, 0], [9, 9]]  # fmt: skip
    assert got["decWindow"] == [[1, 2], [2, 3]]
