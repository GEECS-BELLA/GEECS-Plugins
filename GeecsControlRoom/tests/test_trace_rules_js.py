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
    # what the plot is given
    ("x range: the axis' extent", "xRange", [[3, 1, 2, None], None], [1, 3]),
    ("x range: the viewer's window wins", "xRange", [[0, 10], [2, 4]], [2, 4]),
    ("x range: one point opens one unit", "xRange", [[5], None], [5, 6]),
    ("axis title: label and the family's units", "axisTitle", ["wavelength", "nm"], "wavelength (nm)"),
    ("axis title: no units, the label alone", "axisTitle", ["samples", ""], "samples"),
    ("window: a box zoom's x range, ordered", "windowFromRelayout", [{"xaxis.range[0]": 5, "xaxis.range[1]": 2, "yaxis.range[0]": 0}], [2, 5]),
    ("window: a range given whole", "windowFromRelayout", [{"xaxis.range": [1, 3]}], [1, 3]),
    ("window: autoscale or reset is the whole trace", "windowFromRelayout", [{"xaxis.autorange": True, "yaxis.autorange": True}], None),
    ("empty: before the first trace", "emptyWhy", [None, 0, False], "waiting for the first trace"),
    ("empty: drawing, no words", "emptyWhy", [{"series": [{"off": False}]}, 10, False], ""),
    ("empty: every channel off", "emptyWhy", [{"series": [{"off": True}]}, 0, True], "every channel is off or empty"),
    ("empty: nothing positive on a log scale", "emptyWhy", [{"series": [{"off": False}]}, 0, True], "nothing positive to draw on a log scale"),
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
// what Plotly is given: one trace per channel in channel order, an off channel empty and named so, a NaN a gap (null),
// the window's samples only, and on a log scale the positive extrema only
const axis = f32([0, 1, 2, 3]);
const three = { channels: 3, samples: 4, axis, series: [{ values: f32([1, null, 3, 100]), off: false }, { values: f32([null, null, null, null]), off: true }, { values: f32([-1, 0, 10, 1000]), off: false }] };
const pd = R.plotData(three, 0, 2, 100, false, "capture");
out.plot = { names: pd.traces.map((t) => t.name), x: pd.traces.map((t) => t.x), y: pd.traces.map((t) => t.y), drawn: pd.drawn, kind: [pd.traces[0].type, pd.traces[0].mode, pd.traces[0].connectgaps] };
out.plotLog = R.plotData(three, 0, 3, 100, true, "capture").traces[2].y;
out.plotOne = R.plotData({ channels: 1, samples: 4, axis, series: [three.series[0]] }, 0, 3, 100, false, "spectrum").traces.map((t) => t.name);
out.yOnly = R.windowFromRelayout({ "yaxis.range[0]": 0, "yaxis.range[1]": 1 }) === undefined && R.windowFromRelayout({ "hiddenlabels": [] }) === undefined;
// decimate: min and max per column, a NaN breaks the line
const many = f32([...Array(100).keys()].map((i) => (i === 50 ? null : i % 10)));
const pts = R.decimate(f32([...Array(100).keys()]), many, 0, 99, 10);
out.dec = { n: pts.length, gaps: pts.filter((p) => p === null).length, first: pts.slice(0, 2) };
out.decWindow = R.decimate(axis, f32([1, 2, 3, 4]), 1, 2, 100);
// log: each column's extrema over its positive samples only, as the render loop draws them (a segment between two
// drawn points in a row); a compressed bipolar trace must still draw (the review's reproduction)
const ramp = f32([...Array(1000).keys()]), bip = f32([...Array(1000).keys()].map((i) => [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9][i % 10]));
const drawn = (pts) => { let moves = 0, lines = 0, pen = false; for (const p of pts) { if (p === null || p[1] <= 0) { pen = false; continue; } if (pen) lines++; else moves++; pen = true; } return { moves, lines }; };
const logPts = R.decimate(ramp, bip, 0, 999, 100, true);
out.log = { ...drawn(logPts), positive: logPts.every((p) => p === null || p[1] > 0), lowest: Math.min(...logPts.filter(Boolean).map((p) => p[1])) };
out.logGap = R.decimate(f32([0, 1, 2, 3]), f32([1, -1, -2, 3]), 0, 3, 4, true); // a column with no positive sample: a gap
out.linear = R.decimate(ramp, bip, 0, 999, 100).slice(0, 2); // linear keeps the signed extrema, unchanged
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
    p = got["plot"]
    assert p["names"] == ["A", "B off", "C"] and p["kind"] == [
        "scatter",
        "lines",
        False,
    ]
    assert (
        p["x"] == [[0, None, 2], [None], [0, 1, 2]]
        and p["y"] == [[1, None, 3], [None], [-1, 0, 10]]
        and p["drawn"] == 5
    )
    assert got["plotLog"] == [10, 1000]  # the log scale's positive samples only
    assert got["plotOne"] == ["spectrum"] and got["yOnly"] is True
    assert got["dec"]["gaps"] == 1 and got["dec"]["n"] <= 2 * 10 + 1 and got["dec"]["first"] == [[0, 0], [9, 9]]  # fmt: skip
    assert got["decWindow"] == [[1, 2], [2, 3]]
    # log: 100 columns of [-1, 1..9] draw as one connected line over the positive extrema (1 and 9), never 100 lone moves
    assert got["log"] == {"moves": 1, "lines": 199, "positive": True, "lowest": 1}
    assert got["logGap"] == [[0, 1], None, [3, 3]]
    assert got["linear"] == [[0, -1], [9, 9]]
