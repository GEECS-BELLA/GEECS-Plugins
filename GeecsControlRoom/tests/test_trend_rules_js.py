"""trend_rules.js under node: each row one rule of the settables page's trends, then trend.js itself over a fake page (skipped without node)."""

from __future__ import annotations

from typing import Any

from tests._node import call_all, needs_node, run

T = 1_800_000_000_000  # a time, in ms
LIVE = {"connected": True, "severity": "NONE", "value": 2.5, "tsNs": (T - 400) * 1_000_000}  # fmt: skip
WANTED = [{"role": "actuator", "label": "actuator"}, {"role": "target", "label": "last target"}]  # fmt: skip
OK = {"ok": True, "to": T, "roles": {"actuator": [[T - 5, 1]], "target": [[T - 4, 2]]}, "why": {}}  # fmt: skip

ROWS: list[tuple[str, str, list[Any], Any]] = [
    # one of Hare's frames as a point
    ("point: the value at the record's own stamp", "point", [LIVE, T], [T - 400, 2.5]),
    ("point: a stamp after now is drawn now", "point", [LIVE | {"tsNs": (T + 900) * 1_000_000}, T], [T, 2.5]),
    ("point: no stamp is now", "point", [{"connected": True, "value": 3}, T], [T, 3]),
    ("point: no frame (the socket down, or nothing yet) is a gap", "point", [None, T], [T, None]),
    ("point: a disconnected record is a gap", "point", [LIVE | {"connected": False}, T], [T, None]),
    ("point: an INVALID record is a gap, never its last value", "point", [LIVE | {"severity": "INVALID"}, T], [T, None]),
    ("point: null is a gap, never a zero", "point", [LIVE | {"value": None}, T], [T, None]),
    ("point: a word is a gap", "point", [LIVE | {"value": "moving"}, T], [T, None]),
    ("point: zero is a value", "point", [LIVE | {"value": 0}, T], [T - 400, 0]),
    # the live buffer
    ("append: the first value", "append", [[], [10, 1], 50], [[10, 1]]),
    ("append: never a leading gap", "append", [[], [10, None], 50], []),
    ("append: a new value at its stamp", "append", [[[10, 1]], [20, 2], 50], [[10, 1], [20, 2]]),
    ("append: the same sample again is not a point", "append", [[[10, 1]], [10, 1], 50], [[10, 1]]),
    ("append: the same value at a later stamp is a point", "append", [[[10, 1]], [20, 1], 50], [[10, 1], [20, 1]]),
    ("append: a gap opens once", "append", [[[10, 1], [20, None]], [30, None], 50], [[10, 1], [20, None]]),
    ("append: a gap after a value", "append", [[[10, 1]], [30, None], 50], [[10, 1], [30, None]]),
    ("append: an old stamp after a gap is drawn when it arrived (the gap keeps its width)", "append", [[[10, 1], [20, None]], [10, 1], 50], [[10, 1], [20, None], [50, 1]]),
    ("append: an old stamp with a new value is drawn when it arrived", "append", [[[10, 1], [20, 2]], [15, 3], 50], [[10, 1], [20, 2], [50, 3]]),
    ("append: capped, the oldest dropped", "append", [[[1, 1], [2, 2], [3, 3]], [4, 4], 50, 3], [[2, 2], [3, 3], [4, 4]]),
    # the history as the route gives it
    ("past: oldest first, the malformed dropped", "past", [[[30, 3], [10, 1], [20, None], [None, 5], "x", [40, "y"], [50]]], [[10, 1], [20, None], [30, 3]]),
    ("past: nothing is nothing", "past", [None], []),
    ("merge: the history, then the live points after its last sample", "merge", [[[10, 1], [20, 2]], [[20, 2], [30, 3]]], [[10, 1], [20, 2], [30, 3]]),
    ("merge: no history is the live points", "merge", [[], [[30, 3]]], [[30, 3]]),
    ("merge: capped from the old end", "merge", [[[10, 1], [20, 2]], [[30, 3]], 2], [[20, 2], [30, 3]]),
    ("merge: a live buffer at its cap has a hole behind it: the history is not held across it", "merge", [[[10, 1]], [[500, 5], [600, 6]], 2], [[500, 5], [600, 6]]),
    ("holding: the newest live point is a value", "holding", [[[10, 1]]], True),
    ("holding: the newest live point is a gap", "holding", [[[10, 1], [20, None]]], False),
    ("holding: nothing live", "holding", [[]], False),
    # what a line draws inside the window
    ("series: the value held on entering, the points inside, held to now", "series", [[[5, 1], [120, 2], [150, 3]], 100, 200, True, 100], {"x": [100, 120, 150, 200], "y": [1, 2, 3, 3], "drawn": 4}),
    ("series: not held past the last point without a live value", "series", [[[5, 1], [120, 2]], 100, 200, False, 100], {"x": [100, 120], "y": [1, 2], "drawn": 2}),
    ("series: a gap is one null, and nothing is held across it", "series", [[[110, 1], [120, None], [130, None], [150, 3]], 100, 200, False, 100], {"x": [110, 120, 150], "y": [1, None, 3], "drawn": 2}),
    ("series: a line ending in a gap is not held to now", "series", [[[110, 1], [120, None]], 100, 200, True, 100], {"x": [110, 120], "y": [1, None], "drawn": 1}),
    ("series: a gap before the window enters as nothing", "series", [[[5, 1], [50, None], [150, 3]], 100, 200, False, 100], {"x": [150], "y": [3], "drawn": 1}),
    ("series: a quiet record is its last value, held across the window", "series", [[[5, 1]], 100, 200, True, 100], {"x": [100, 200], "y": [1, 1], "drawn": 2}),
    ("series: a point after now is left out", "series", [[[150, 1], [250, 2]], 100, 200, False, 100], {"x": [150], "y": [1], "drawn": 1}),
    ("series: nothing", "series", [[], 100, 200, True, 100], {"x": [], "y": [], "drawn": 0}),
    ("series: a column keeps its first, lowest, highest and last, so the level held is its last", "series", [[[101, 5], [102, 1], [103, 9], [104, 4], [105, 6], [150, 7]], 100, 200, False, 2], {"x": [101, 102, 103, 105, 150], "y": [5, 1, 9, 6, 7], "drawn": 5}),
    ("stamp: the viewer's wall clock, as Plotly's date axis reads it", "stampText", [T, 480], "2027-01-15 00:00:00.000"),
    ("stamp: UTC", "stampText", [T + 250, 0], "2027-01-15 08:00:00.250"),
    ("stamp: a gap stays a gap", "stampText", [None, 480], None),
    ("unit: shared", "sharedUnit", [["mm", "mm"]], "mm"),
    ("unit: two units are none", "sharedUnit", [["mm", "V"]], ""),
    ("unit: one line without is none", "sharedUnit", [["mm", ""]], ""),
    ("unit: no lines", "sharedUnit", [[]], ""),
    # the viewer's window and the ask
    ("window: a choice offered", "windowOf", ["3600", [120, 600, 3600], 600], 3600),
    ("window: anything else is the fallback", "windowOf", ["45", [120, 600, 3600], 600], 600),
    ("window: nothing pressed", "windowOf", [None, [120, 600, 3600], 120], 120),
    ("url: the window", "trendsUrl", ["/room/api/devices/lfs/trends", 600, None], "/room/api/devices/lfs/trends?seconds=600"),
    ("url: a per-axis family's axis", "trendsUrl", ["/api/devices/m/trends", 120, "m 1"], "/api/devices/m/trends?seconds=120&axis=m%201"),
    # the route's answer and the one line of words
    ("answer: the history per role, and why a role has none", "answer", [True, {"to": T, "roles": {"actuator": {"samples": [[T - 5, 1]], "why": None}, "target": {"samples": [], "why": "Vole did not answer"}}}, 200],
     {"ok": True, "to": T, "roles": {"actuator": [[T - 5, 1]], "target": []}, "why": {"target": "Vole did not answer"}}),
    ("answer: a refusal's words", "answer", [False, {"detail": "lfs serves no trend role"}, 404], {"ok": False, "why": "lfs serves no trend role"}),
    ("answer: a refusal without words", "answer", [False, None, 502], {"ok": False, "why": "answered 502"}),
    ("answer: a 200 that is no history", "answer", [True, [], 200], {"ok": False, "why": "no history in the answer"}),
    ("note: while the first ask is out", "historyNote", [None, WANTED], "reading the history"),
    ("note: every line has its history: no words", "historyNote", [OK, WANTED], ""),
    ("note: the Control Room silent", "historyNote", [{"ok": False, "why": "the Control Room did not answer"}, WANTED], "no earlier history (the Control Room did not answer): the plots start at page load"),
    ("note: Vole silent for every line", "historyNote", [{"ok": True, "roles": {"actuator": [], "target": []}, "why": {"actuator": "Vole did not answer", "target": "Vole did not answer"}}, WANTED], "no earlier history (Vole did not answer): the plots start at page load"),
    ("note: one line without", "historyNote", [OK | {"roles": OK["roles"] | {"target": []}, "why": {"target": "nothing buffered yet"}}, WANTED], "no earlier history for last target (nothing buffered yet): drawn from page load"),
]  # fmt: skip


@needs_node
def test_every_row() -> None:
    got = call_all("trend_rules.js", [(fn, args) for _, fn, args, _ in ROWS])
    for (name, _, _, want), answer in zip(ROWS, got):
        assert answer == want, name


@needs_node
def test_the_traces_are_one_held_line_per_role_in_the_panels_order() -> None:
    lines = [{"label": "actuator", "points": [[110, 1], [150, 2]], "hold": True}, {"label": "last target", "points": [], "hold": False}]  # fmt: skip
    one, none = call_all("trend_rules.js", [("traces", [lines, 100, 200, 100])])[0]
    assert one == {"type": "scatter", "mode": "lines", "name": "actuator", "x": [110, 150, 200], "y": [1, 2, 2], "connectgaps": False, "line": {"width": 1.25, "shape": "hv"}}  # fmt: skip
    # a line with nothing yet keeps its place (and so its colour) with one null point
    assert (none["name"], none["x"], none["y"]) == ("last target", [None], [None])


# The real trend.js in a VM over a fake page: the device page's frames, Plotly, fetch and the redraw timer are the
# harness's, the rules trend_rules.js's own.
_PAGE = r"""
const vm = require("vm"), fs = require("fs"), path = require("path");
const dir = path.dirname(process.argv[1]);
const a = JSON.parse(fs.readFileSync(0, "utf8"));
const flush = async () => { for (let i = 0; i < 8; i++) await new Promise((r) => setImmediate(r)); };
function el(props = {}) {
  const on = {};
  return Object.assign({ dataset: {}, textContent: "", attrs: {}, clientWidth: 0,
    setAttribute(k, v) { this.attrs[k] = v; }, getAttribute(k) { return this.attrs[k]; },
    addEventListener(t, f) { (on[t] = on[t] || []).push(f); }, fire(t) { (on[t] || []).forEach((f) => f({})); },
    querySelector() { return null; }, querySelectorAll() { return []; } }, props);
}
const hosts = a.trends.map(() => el({ clientWidth: a.width }));
const plots = a.trends.map((roles, i) => el({ dataset: { trend: JSON.stringify(roles) }, querySelector: () => hosts[i] }));
const buttons = [120, 600, 3600].map((s) => el({ dataset: { window: String(s) }, attrs: { "aria-pressed": String(s === 600) } }));
const note = el();
const box = el({ dataset: { trendsUrl: "/api/devices/lfs/trends" },
  querySelector: (sel) => (sel === ".trend-note" ? note : null),
  querySelectorAll: (sel) => (sel === ".trend" ? plots : sel === "[data-window]" ? buttons : []) });
const frames = {}, listeners = [], asked = [], timers = [], watched = [];
const Plotly = { calls: [], react(h, data, layout) { if (a.plotlyThrows) throw new Error("plotly"); this.calls.push({ host: hosts.indexOf(h), data, layout }); } };
let clock = a.now;
class FakeDate extends Date { constructor(...args) { if (args.length) super(...args); else super(clock); } static now() { return clock; } getTimezoneOffset() { return 0; } }
const window = {
  GeecsTrendRules: require(path.join(dir, "trend_rules.js")),
  GeecsDeviceRules: require(path.join(dir, "device_rules.js")),
  GeecsDevicePage: { frame: (role) => frames[role], meta: (role) => ({ units: a.egu[role] }), on: (f) => listeners.push(f) },
  GeecsPlot: a.noPlot ? undefined : { axisFormat: (u) => ({ fmt: u }), layout: (o) => o, config: () => ({}), watch: (h) => watched.push(hosts.indexOf(h)) },
  Plotly: a.noPlot ? undefined : Plotly,
};
const document = { hidden: false, querySelectorAll: (sel) => (sel === ".trends" ? [box] : []) };
const fetch = async (url) => {
  asked.push(url);
  const next = a.answers.shift();
  if (!next) throw new Error("down");
  return { ok: next.status === 200, status: next.status, json: async () => next.body };
};
const ctx = vm.createContext({ window, document, fetch, Date: FakeDate, console: { error() {} }, AbortController, Promise, Map, Set, JSON, Number, Math, String, Array, Object,
  setInterval: (f) => { timers.push(f); return timers.length; }, setTimeout, clearTimeout });
const last = (i) => { const c = Plotly.calls.filter((p) => p.host === i).pop(); return c && { names: c.data.map((t) => t.name), x: c.data.map((t) => t.x), y: c.data.map((t) => t.y), shape: c.data[0].line.shape, range: c.layout.xaxis.range, fixed: c.layout.xaxis.fixedrange, kind: c.layout.xaxis.type, unit: c.layout.yaxis.title.text, fmt: c.layout.yaxis.fmt }; };
const feed = (role, frame) => { frames[role] = frame; listeners.forEach((f) => f()); };
const tick = async (ms) => { clock += ms; timers.forEach((f) => f()); await flush(); };
(async () => {
  const out = {};
  vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), ctx);
  out.first = note.textContent;
  await flush();
  out.loaded = { note: note.textContent, value: last(0), actuator: last(1), asked: [...asked] };
  // live: a new frame extends its line; the same frame told again adds nothing; the socket dropping is a gap
  await tick(1000);
  const f1 = { connected: true, severity: "NONE", value: 5, tsNs: (clock + a.skew - 500) * 1e6 }; // stamped by the IOC, on the server's time
  feed("actuator", f1); feed("actuator", f1);
  await tick(1000);
  out.live = last(1);
  feed("actuator", null);
  await tick(2000);
  feed("actuator", { connected: true, severity: "NONE", value: 6, tsNs: (clock + a.skew - 60000) * 1e6 }); // the record's old stamp, after a reconnect
  await tick(1000);
  out.gap = last(1);
  out.calls = Plotly.calls.length;
  // another window: pressed, asked again, the x range follows
  buttons[0].fire("click");
  await flush();
  out.window = { pressed: buttons.map((b) => b.getAttribute("aria-pressed")), asked: asked[asked.length - 1], range: (last(0) || {}).range, note: note.textContent };
  out.watched = watched;
  out.listenerThrows = (() => { try { listeners.forEach((f) => f()); return false; } catch (_) { return true; } })();
  console.log(JSON.stringify(out));
})();
"""

_TRENDS = [
    [{"role": "value_display", "label": "value, display unit", "units": "", "type": "readback"}, {"role": "setpoint_display", "label": "setpoint, display unit", "units": "", "type": "readback"}],
    [{"role": "actuator", "label": "actuator", "units": "", "type": "readback"}, {"role": "target", "label": "last target", "units": "", "type": "readback"}],
]  # fmt: skip
_EGU = {"value_display": "urad", "setpoint_display": "urad", "actuator": "mm", "target": "V"}  # fmt: skip


def _stamp(ms: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S.") + f"{ms % 1000:03d}"  # fmt: skip


def _page(**over: Any) -> dict:
    import json

    server = T + 5000  # the server's clock runs 5 s ahead of this browser's
    history = {"to": server, "roles": {
        "value_display": {"samples": [[server - 30000, 1], [server - 20000, None], [server - 10000, 2]], "why": None},
        "setpoint_display": {"samples": [[server - 900000, 1.5]], "why": None},
        "actuator": {"samples": [[server - 4000, 4]], "why": None},
        "target": {"samples": [], "why": "nothing buffered yet"},
    }}  # fmt: skip
    again = {"to": server + 5000, "roles": {r: {"samples": [], "why": "nothing buffered yet"} for r in history["roles"]}}  # fmt: skip
    args = {"trends": _TRENDS, "egu": _EGU, "now": T, "skew": 5000, "width": 400, "answers": [{"status": 200, "body": history}, {"status": 200, "body": again}]}  # fmt: skip
    return run("trend.js", _PAGE, json.dumps(args | over))


@needs_node
def test_the_page_draws_the_history_then_extends_each_line_from_the_frames_it_is_sent() -> None:  # fmt: skip
    page, s = _page(), T + 5000
    assert page["first"] == "reading the history"
    loaded = page["loaded"]
    assert loaded["asked"] == ["/api/devices/lfs/trends?seconds=600"]
    assert loaded["note"] == "no earlier history for last target (nothing buffered yet): drawn from page load"  # fmt: skip
    value = loaded["value"]
    assert value["names"] == ["value, display unit", "setpoint, display unit"]
    # x is the server's clock, in the viewer's wall time, the window ending now and fixed; a null sample is a gap;
    # a value from before the window enters at its edge; with no live frame no line is held to now
    assert (value["kind"], value["fixed"], value["shape"]) == ("date", True, "hv")
    assert value["range"] == [_stamp(s - 600_000), _stamp(s)]
    assert value["x"][0] == [_stamp(s - 30000), _stamp(s - 20000), _stamp(s - 10000)] and value["y"][0] == [1, None, 2]  # fmt: skip
    assert value["x"][1] == [_stamp(s - 600_000)] and value["y"][1] == [1.5]
    assert (value["unit"], value["fmt"]) == ("urad", "urad")  # the records' shared EGU
    assert loaded["actuator"]["unit"] == ""  # mm and V: no shared unit
    assert loaded["actuator"]["y"] == [[4], [None]]
    # live: one new point (the frame told twice is one), held to now
    live = page["live"]
    assert live["y"][0] == [4, 5, 5] and live["x"][0][1] == _stamp(s + 500)
    assert live["range"] == [_stamp(s + 2000 - 600_000), _stamp(s + 2000)]
    # the socket dropped for 2 s: a gap of that width, then the value at its arrival, never at the record's old stamp
    gap = page["gap"]
    assert gap["y"][0] == [4, 5, None, 6, 6]
    assert gap["x"][0][2:4] == [_stamp(s + 2000), _stamp(s + 4000)]
    # the window choice: pressed, asked again, x follows; that answer had no history, and says so
    assert page["window"]["pressed"] == ["true", "false", "false"]
    assert page["window"]["asked"] == "/api/devices/lfs/trends?seconds=120"
    assert page["window"]["range"] == [_stamp(s + 5000 - 120_000), _stamp(s + 5000)]
    assert page["window"]["note"] == "no earlier history (nothing buffered yet): the plots start at page load"  # fmt: skip
    assert sorted(set(page["watched"])) == [0, 1] and len(page["watched"]) == 2


@needs_node
def test_vole_or_the_server_silent_the_plots_start_at_page_load_and_say_so() -> None:
    page = _page(answers=[{"status": 503, "body": {"detail": "The module did not answer: it answered 0"}}])  # fmt: skip
    assert page["loaded"]["note"] == "no earlier history (The module did not answer: it answered 0): the plots start at page load"  # fmt: skip
    assert page["loaded"]["actuator"]["y"] == [[None], [None]]
    assert page["live"]["y"][0] == [5, 5]  # the line starts with the first frame
    down = _page(answers=[])
    assert down["loaded"]["note"] == "no earlier history (the Control Room did not answer): the plots start at page load"  # fmt: skip
    assert down["live"]["y"][0] == [5, 5]


@needs_node
def test_a_trend_fault_never_reaches_the_rest_of_the_page() -> None:
    # Plotly throwing on every draw, or missing: the device page's listeners still run, and the page says what it can
    assert _page(plotlyThrows=True)["listenerThrows"] is False
    missing = _page(noPlot=True)
    assert missing["listenerThrows"] is False and missing["calls"] == 0
    assert missing["loaded"]["note"] == "the plot library did not load"
    assert (
        _page(width=0)["calls"] == 0
    )  # a hidden axis pane has no width: not drawn until shown
