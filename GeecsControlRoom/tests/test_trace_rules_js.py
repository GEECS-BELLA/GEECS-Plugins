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
YS = [{"role": "spectrum"}, {"role": "processed"}]

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
    # which array the plot shows (a spectrometer's raw and processed spectrum)
    ("shown: the processed array while its role reads 1", "shownY", [YS, {"processed": "background_enable"}, {"background_enable": 1}, None], "processed"),
    ("shown: the first array while that role reads 0", "shownY", [YS, {"processed": "background_enable"}, {"background_enable": 0}, None], "spectrum"),
    ("shown: an unknown role shows the first", "shownY", [YS, {"processed": "background_enable"}, {}, None], "spectrum"),
    ("shown: the viewer's choice wins", "shownY", [YS, {"processed": "background_enable"}, {"background_enable": 1}, "spectrum"], "spectrum"),
    ("shown: a choice no longer drawn falls back", "shownY", [YS, {}, {}, "gone"], "spectrum"),
    ("shown: none drawn", "shownY", [[], {}, {}, None], None),
    ("trace url: the array asked by name", "traceUrl", ["/room/api/devices/cct1/trace", "processed"], "/room/api/devices/cct1/trace?y=processed"),
    ("rate: first to last of the span's tags", "rate", [[0, 1000, 2000, 3000, 4000, 5000, 5900], 6000], {"seen": [1000, 2000, 3000, 4000, 5000, 5900], "text": "1.0 new/s"}),
    ("rate: one tag is no rate yet", "rate", [[4000], 6000], {"seen": [4000], "text": "1 new in 5 s"}),
    ("rate: none recent", "rate", [[0], 9000], {"seen": [], "text": "no new captures"}),
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


# The real trace.js in a VM over a fake page: the module's frames, Plotly, fetch and the poll timer are the
# harness's, the rules trace_rules.js's own. Load raw, switch to processed, fail its reads: the previous
# array's plot must not stay up under the new array's toggle and chip; the viewer's x window survives.
_PAGE = r"""
const vm = require("vm"), fs = require("fs"), path = require("path");
const dir = path.dirname(process.argv[1]);
const R = require(path.join(dir, "trace_rules.js"));
const flush = async () => { for (let i = 0; i < 8; i++) await new Promise((r) => setImmediate(r)); };
const timers = [];
function el(props = {}) {
  const on = {};
  return Object.assign({ dataset: {}, style: {}, hidden: false, textContent: "", value: "", disabled: false, title: "", attrs: {},
    setAttribute(k, v) { this.attrs[k] = v; }, getAttribute(k) { return this.attrs[k]; },
    addEventListener(t, f) { (on[t] = on[t] || []).push(f); }, fire(t, ev) { (on[t] || []).forEach((f) => f(ev)); },
    querySelector() { return el(); }, querySelectorAll() { return []; } }, props);
}
const ys = [{ role: "spectrum", label: "raw spectrum", units: "counts", counter: "arr_counter" },
            { role: "processed", label: "processed spectrum", units: "counts", counter: "proc_counter" }];
const yButtons = ys.map((y) => el({ dataset: { y: y.role } }));
const box = el({ dataset: { traceUrl: "/api/devices/cct1/trace", xLabel: "wavelength", xUnits: "nm" },
  querySelectorAll: (sel) => (sel === "#yrole [data-y]" ? yButtons : []) });
const relayout = [];
const host = el({ clientWidth: 400, on: (t, f) => { if (t === "plotly_relayout") relayout.push(f); } });
const ids = { trace: box, "trace-plot": host, "kit-state": el({ textContent: "{}" }), "trace-ys": el({ textContent: JSON.stringify(ys) }),
  "trace-show-while": el({ textContent: JSON.stringify({ processed: "background_enable" }) }), yscale: el({ value: "linear" }),
  average: el({ value: "1" }), alarms: el(), "trace-state": el(), "plot-note": el(), "trace-error": el(), "trace-error-text": el(), rate: el() };
const Plotly = { calls: [], react(h, data, layout) { this.calls.push({ data, layout }); } };
const answers = [], asked = [];
function answer(a) {
  if (a.status !== 200) return { status: a.status, headers: { get: () => null }, json: async () => ({ detail: "processed unreadable" }) };
  const body = Float32Array.from([...a.axis, ...a.y]);
  const hdr = { "X-Trace-Channels": "1", "X-Trace-Samples": String(a.axis.length), "X-Trace-Tag": `"${a.tag}"`, ETag: `"${a.tag}"` };
  return { status: 200, headers: { get: (k) => hdr[k] ?? null }, arrayBuffer: async () => body.buffer };
}
const window = {
  GeecsTraceRules: R,
  GeecsDevicePage: { frame: () => undefined, view: () => ({ ok: true, roles: {} }), entry: () => undefined, meta: () => ({}), word: () => "live", on() {} },
  GeecsDeviceWrites: { make: () => ({ write: async () => true, busy: () => false }) },
  GeecsDeviceRail: { make: () => ({ fill() {} }) },
  GeecsPlot: { axisFormat: () => ({}), layout: (o) => o, config: () => ({}), watch() {} },
  Plotly,
};
const document = { getElementById: (id) => ids[id] || null, hidden: false, activeElement: null, addEventListener() {} };
const fetch = async (url) => { asked.push(url); return answer(answers.shift()); };
const ctx = vm.createContext({ window, document, fetch, performance: { now: () => Date.now() }, console,
  setTimeout: (f) => { timers.push(f); return timers.length; }, clearTimeout() {} });
const shown = () => {
  const last = Plotly.calls[Plotly.calls.length - 1];
  const live = !!last && last.data.length > 0 && host.style.visibility !== "hidden";
  return { calls: Plotly.calls.length, live, name: live ? last.data[0].name : null, title: live ? last.layout.yaxis.title.text : null,
           xs: live ? last.data[0].x.filter((v) => v !== null) : null, error: ids["trace-error-text"].textContent };
};
const tick = async () => { timers.shift()(); await flush(); };
(async () => {
  const out = {};
  answers.push({ status: 200, axis: [0, 1, 2, 3], y: [10, 11, 12, 13], tag: 5 });
  vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), ctx);
  await flush();
  out.raw = shown();
  relayout.forEach((f) => f({ "xaxis.range[0]": 1, "xaxis.range[1]": 2 })); // the viewer's window
  out.window = shown();
  yButtons[1].fire("click");
  out.switched = shown();
  for (let i = 0; i < 3; i++) { answers.push({ status: 503 }); await tick(); }
  out.failing = shown();
  answers.push({ status: 200, axis: [0, 1, 2, 3], y: [20, 21, 22, 23], tag: 5 });
  await tick();
  out.processed = shown();
  out.asked = asked;
  out.pressed = yButtons.map((b) => b.getAttribute("aria-pressed"));
  console.log(JSON.stringify(out));
})();
"""


@needs_node
def test_switching_arrays_never_leaves_the_last_arrays_plot_up() -> None:
    page = run("trace.js", _PAGE)
    assert page["raw"]["live"] and page["raw"]["name"] == "raw spectrum"
    assert page["window"]["xs"] == [1, 2]  # the viewer's window, drawn
    # switched, and while every processed read fails: no plot shown, the toggle and the words say processed
    for state in ("switched", "failing"):
        assert not page[state]["live"], state
    assert page["failing"]["error"] == "processed unreadable"
    # the processed capture arrives: drawn under its own label, in the viewer's window
    after = page["processed"]
    assert after["live"] and (after["name"], after["title"]) == ("processed spectrum", "processed spectrum (counts)")  # fmt: skip
    assert after["xs"] == [1, 2] and after["error"] == ""
    assert page["asked"][0].endswith("?y=spectrum") and page["asked"][-1].endswith(
        "?y=processed"
    )
    assert page["pressed"] == ["false", "true"]
