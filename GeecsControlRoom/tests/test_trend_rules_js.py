"""trend_rules.js under node: each row one rule of the settables page's trends, then trend.js itself over a fake page (skipped without node)."""

from __future__ import annotations

from typing import Any

from tests._node import call_all, needs_node, run

T = 1_800_000_000_000  # a time, in ms
LIVE = {"connected": True, "severity": "NONE", "value": 2.5, "tsNs": (T - 400) * 1_000_000}  # fmt: skip
WANTED = [{"role": "actuator", "label": "actuator"}, {"role": "target", "label": "last target"}]  # fmt: skip
OK = {"ok": True, "now": T, "roles": {"actuator": [[T - 5, 1]], "target": [[T - 4, 2]]}, "state": {"actuator": "history", "target": "history"}, "why": {}}  # fmt: skip
NONE = {"ok": True, "now": T, "roles": {"actuator": [], "target": []}}


def _got(actuator: str, target: str, why: str = "Vole did not answer") -> dict:
    """An answer whose two lines are in the given states."""
    state = {"actuator": actuator, "target": target}
    roles = {r: [[T - 5, 1]] if s == "history" else [] for r, s in state.items()}
    return {"ok": True, "now": T, "roles": roles, "state": state, "why": {r: "not buffered until now" if s == "starting" else why for r, s in state.items() if s != "history"}}  # fmt: skip


ROWS: list[tuple[str, str, list[Any], Any]] = [
    # one of Hare's frames as a point
    ("point: the value at the record's own stamp", "point", [LIVE, T], [T - 400, 2.5]),
    ("point: a stamp after now is drawn now", "point", [LIVE | {"tsNs": (T + 900) * 1_000_000}, T], [T, 2.5]),
    ("point: no stamp is now", "point", [{"connected": True, "value": 3}, T], [T, 3]),
    ("point: no frame (the socket down, or nothing yet) is a gap", "point", [None, T], [T, None]),
    ("point: a disconnected record is a gap", "point", [LIVE | {"connected": False}, T], [T, None]),
    ("point: a frame that does not say it is connected is a gap (the page's own reading of a frame)", "point", [{"value": 3, "severity": "NO_ALARM"}, T], [T, None]),
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
    # the live buffer's bounds: by time (never drawn again), then by count (the page must know)
    ("trim: inside the longest window and under the cap, nothing goes", "trim", [[[900, 1], [950, 2]], 1000, 200, 5], None),
    ("trim: by count, the stamp of the newest point lost", "trim", [[[910, 1], [920, 2], [930, 3], [940, 4]], 1000, 200, 2], 920),
    ("trim: by time alone is no loss a window could show", "trim", [[[100, 1], [200, 2], [300, 3], [950, 4]], 1000, 200, 5], None),
    ("hole: the cap took live points newer than the history's end", "hole", [[[10, 1], [20, 2]], 25], True),
    ("hole: the cap took only what the history covers", "hole", [[[10, 1], [20, 2]], 20], False),
    ("hole: nothing lost", "hole", [[[10, 1]], None], False),
    ("hole: no history to be cut off from", "hole", [[], 25], False),
    # the history as the route gives it
    ("past: oldest first, the malformed dropped", "past", [[[30, 3], [10, 1], [20, None], [None, 5], "x", [40, "y"], [50]]], [[10, 1], [20, None], [30, 3]]),
    ("past: nothing is nothing", "past", [None], []),
    ("merge: the history, then the live points after its last sample", "merge", [[[10, 1], [20, 2]], [[20, 2], [30, 3]]], [[10, 1], [20, 2], [30, 3]]),
    ("merge: no history is the live points", "merge", [[], [[30, 3]]], [[30, 3]]),
    ("merge: the history is never cut to make room for live points", "merge", [[[t, t] for t in range(1, 6)], [[t, t] for t in range(6, 11)]], [[t, t] for t in range(1, 11)]),
    ("merge: live points lost after the history's end: a gap closes it, its last value never held across", "merge", [[[10, 1]], [[500, 5], [600, 6]], 400], [[10, 1], [10, None], [500, 5], [600, 6]]),
    ("merge: live points lost that the history covers: no gap", "merge", [[[10, 1], [450, 4]], [[500, 5]], 400], [[10, 1], [450, 4], [500, 5]]),
    ("kept: the answer's samples", "kept", [_got("history", "starting"), "actuator", [[1, 1]]], [[T - 5, 1]]),
    ("kept: a failed read never blanks a past already shown", "kept", [_got("failed", "starting"), "actuator", [[1, 1]]], [[1, 1]]),
    ("kept: a failed read with nothing earlier is nothing", "kept", [_got("failed", "starting"), "actuator", []], []),
    ("kept: a PV only now buffered has no past", "kept", [_got("history", "starting"), "target", [[1, 1]]], []),
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
    ("answer: the history per role, its state, and why a role has none", "answer", [True, {"to": T - 9, "now": T, "roles": {"actuator": {"samples": [[T - 5, 1]], "state": "history", "why": None}, "target": {"samples": [], "state": "failed", "why": "Vole did not answer"}, "step": {"samples": [], "state": "starting", "why": "not buffered until now"}}}, 200],
     {"ok": True, "now": T, "roles": {"actuator": [[T - 5, 1]], "target": [], "step": []}, "state": {"actuator": "history", "target": "failed", "step": "starting"}, "why": {"target": "Vole did not answer", "step": "not buffered until now"}}),
    ("answer: without the server's stamp, the window's end", "answer", [True, {"to": T, "roles": {}}, 200], {"ok": True, "now": T, "roles": {}, "state": {}, "why": {}}),
    ("answer: a state is never taken on trust: samples are history, no samples and no word is failed", "answer", [True, {"roles": {"a": {"samples": [], "state": "history"}, "b": {"samples": [[1, 1]], "state": "failed"}}}, 200],
     {"ok": True, "now": None, "roles": {"a": [], "b": [[1, 1]]}, "state": {"a": "failed", "b": "history"}, "why": {"a": "no samples"}}),
    ("answer: a refusal's words", "answer", [False, {"detail": "lfs serves no trend role"}, 404], {"ok": False, "why": "lfs serves no trend role"}),
    ("answer: a refusal without words", "answer", [False, None, 502], {"ok": False, "why": "answered 502"}),
    ("answer: a 200 that is no history", "answer", [True, [], 200], {"ok": False, "why": "no history in the answer"}),
    ("note: while an ask is out", "historyNote", [None, WANTED], "reading the history"),
    ("note: every line has its history: no words", "historyNote", [OK, WANTED], ""),
    ("note: the Control Room silent, nothing drawn before", "historyNote", [{"ok": False, "why": "the Control Room did not answer"}, WANTED], "no earlier history (the Control Room did not answer): drawn from page load"),
    ("note: the Control Room silent on a later read: the past drawn is said to be the earlier read's, never missing", "historyNote", [{"ok": False, "why": "the Control Room did not answer"}, WANTED, ["actuator", "target"]], "the history could not be read again (the Control Room did not answer): the past drawn is the earlier read's"),
    ("note: a later read failed with one line drawn from before", "historyNote", [{"ok": False, "why": "answered 503"}, WANTED, ["actuator"]], "the history for actuator could not be read again (answered 503): the past drawn is the earlier read's; no earlier history for last target (answered 503): drawn from page load"),
    ("note: Vole silent for every line", "historyNote", [_got("failed", "failed"), WANTED], "no earlier history (Vole did not answer): drawn from page load"),
    ("note: not buffered until now is said apart from a failure", "historyNote", [_got("starting", "starting"), WANTED], "Canopy's buffer has only now begun keeping these values: drawn from page load"),
    ("note: one line only now buffered", "historyNote", [_got("history", "starting"), WANTED], "Canopy's buffer has only now begun keeping last target: drawn from page load"),
    ("note: one line failed, one only now buffered", "historyNote", [_got("failed", "starting"), WANTED], "no earlier history for actuator (Vole did not answer): drawn from page load; Canopy's buffer has only now begun keeping last target: drawn from page load"),
    ("note: one line's read failed while its earlier past is drawn: never 'no earlier history' above it", "historyNote", [_got("failed", "history"), WANTED, ["actuator"]], "the history for actuator could not be read again (Vole did not answer): the past drawn is the earlier read's"),
    ("note: a line the answer does not name has no history", "historyNote", [NONE | {"state": {}, "why": {}}, WANTED], "no earlier history (no samples): drawn from page load"),
]  # fmt: skip


@needs_node
def test_every_row() -> None:
    got = call_all("trend_rules.js", [(fn, args) for _, fn, args, _ in ROWS])
    for (name, _, _, want), answer in zip(ROWS, got):
        assert answer == want, name


@needs_node
def test_the_traces_are_one_line_per_role_in_the_panels_order() -> None:
    lines = [{"label": "actuator", "points": [[110, 1], [150, 2]], "hold": True}, {"label": "last target", "points": [], "hold": False}]  # fmt: skip
    one, none = call_all("trend_rules.js", [("traces", [lines, 100, 200, 100])])[0]
    assert one == {"type": "scatter", "mode": "lines", "name": "actuator", "x": [110, 150, 200], "y": [1, 2, 2], "connectgaps": False, "line": {"width": 1.25, "shape": "linear"}}  # fmt: skip
    # a line with nothing yet keeps its place (and so its colour) with one null point
    assert (none["name"], none["x"], none["y"]) == ("last target", [None], [None])


@needs_node
def test_close_posts_are_joined_and_a_quiet_stretch_is_held_never_ramped() -> None:
    def drawn(points: list, start: int, end: int) -> tuple:
        (one,) = call_all("trend_rules.js", [("traces", [[{"label": "a", "points": points, "hold": False}], start, end, 1000])])[0]  # fmt: skip
        return one["x"], one["y"]

    # 10 s window: posts 1 s apart are joined; 5 s of quiet is held at the old value until the next post
    assert drawn([[1000, 1], [2000, 2], [7000, 3], [8000, 4]], 0, 10000) == ([1000, 2000, 7000, 7000, 8000], [1, 2, 2, 3, 4])  # fmt: skip
    # a record scanned every 2 s is joined whichever way its stamps jitter; exactly 2.5 s is still joined
    assert drawn([[1000, 1], [2998, 2], [5001, 3], [7501, 4]], 0, 10000) == ([1000, 2998, 5001, 7501], [1, 2, 3, 4])  # fmt: skip
    # 1 h window: quiet is a hundredth of it (36 s), so 30 s apart is joined and 40 s is held
    assert drawn([[1000, 1], [31000, 2], [71000, 3]], 0, 3600000) == ([1000, 31000, 71000, 71000], [1, 2, 2, 3])  # fmt: skip
    # the value the window opened with is held however near its edge the next post is: a step sliding out stays a step
    assert drawn([[-3600000, 0], [1000, 5]], 0, 10000) == ([0, 1000, 1000], [0, 0, 5])
    assert drawn([[0, 0], [100, 5]], 0, 10000) == ([0, 100, 100], [0, 0, 5])
    # a value carried to now past the quiet time is one flat segment, no point added
    (held,) = call_all("trend_rules.js", [("traces", [[{"label": "a", "points": [[1000, 5]], "hold": True}], 0, 10000, 1000])])[0]  # fmt: skip
    assert (held["x"], held["y"]) == ([1000, 10000], [5, 5])
    # a plot narrower than 100 columns: quiet is one column, so steady posts thinned to a column's ends stay joined
    steady = [[t * 1000, t % 7] for t in range(1, 3600)]
    (thin,) = call_all("trend_rules.js", [("traces", [[{"label": "a", "points": steady, "hold": False}], 0, 3600000, 60])])[0]  # fmt: skip
    assert len(thin["x"]) == len(set(thin["x"])) > 60
    # a gap is no quiet stretch: nothing is held into it or out of it
    assert drawn([[1000, 1], [2000, None], [9000, 2]], 0, 10000) == (
        [1000, 2000, 9000],
        [1, None, 2],
    )


# The real trend.js in a VM over a fake page: the device page's frames, Plotly, fetch and the redraw timer (one, fired once a tick whatever its period) are the
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
const buttons = [10, 30, 120, 600, 3600].map((s) => el({ dataset: { window: String(s) }, attrs: { "aria-pressed": String(s === 600) } }));
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
  if (a.showLater) { // an axis pane hidden at load: nothing asked or drawn until it is shown
    hosts.forEach((h) => { h.clientWidth = 400; });
    await tick(1000);
    out.later = { asked: [...asked], calls: Plotly.calls.length, note: note.textContent };
    return console.log(JSON.stringify(out));
  }
  if (a.fast) { // a window of seconds is redrawn on every 200 ms tick, a longer one once a second
    const draws = async (ms) => { const before = Plotly.calls.length; await tick(ms); return Plotly.calls.length - before; };
    await tick(1000);
    out.long = [await draws(200), await draws(200), await draws(600)]; // the page's 10 min window
    buttons[0].fire("click"); await flush();
    out.ten = [await draws(200), await draws(200)];
    buttons[1].fire("click"); await flush();
    out.thirty = [await draws(200), await draws(200)];
    buttons[2].fire("click"); await flush();
    await tick(1000);
    out.back = [await draws(200), await draws(800)];
    return console.log(JSON.stringify(out));
  }
  if (a.flood) { // more live frames than the buffer holds, all after the history's end
    const cap = window.GeecsTrendRules.CAP;
    for (let i = 0; i < cap + 50; i++) { clock += 1; feed("actuator", { connected: true, severity: "NO_ALARM", value: 100 + i, tsNs: (clock + a.skew) * 1e6 }); }
    await tick(1000); // inside the re-read's 30 s: drawn with the gap, not asked again yet
    out.holed = { y: last(1).y[0].slice(0, 3), asked: asked.length };
    await tick(31000);
    out.reread = { y: last(1).y[0].slice(0, 2), asked: asked.length, url: asked[asked.length - 1], note: note.textContent };
    return console.log(JSON.stringify(out));
  }
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
  buttons[2].fire("click");
  const pending = { note: note.textContent, y: (last(1) || { y: [] }).y[0] }; // before the answer: the note is the pending one, over the past already held
  await flush();
  out.window = { pending, pressed: buttons.map((b) => b.getAttribute("aria-pressed")), asked: asked[asked.length - 1], range: (last(0) || {}).range, note: note.textContent };
  // and once more, with the server now silent: the past stays drawn, and the note says whose it is
  buttons[4].fire("click");
  await flush();
  out.failed = { note: note.textContent, y: (last(1) || { y: [] }).y[0], asked: asked[asked.length - 1] };
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
    roles = {
        "value_display": {"samples": [[server - 30000, 1], [server - 20000, None], [server - 10000, 2]], "state": "history", "why": None},
        "setpoint_display": {"samples": [[server - 900000, 1.5]], "state": "history", "why": None},
        "actuator": {"samples": [[server - 4000, 4]], "state": "history", "why": None},
        "target": {"samples": [], "state": "starting", "why": "not buffered until now"},
    }  # fmt: skip
    # `to` was taken before a slow Vole answered: the page's clock is `now`, stamped when the answer was built
    history = {"to": server - 3000, "now": server, "roles": roles}
    again = {"to": server + 2000, "now": server + 5000, "roles": roles}
    args = {"trends": _TRENDS, "egu": _EGU, "now": T, "skew": 5000, "width": 400, "answers": [{"status": 200, "body": history}, {"status": 200, "body": again}]}  # fmt: skip
    return run("trend.js", _PAGE, json.dumps(args | over))


_STARTING = (
    "Canopy's buffer has only now begun keeping last target: drawn from page load"
)


@needs_node
def test_the_page_draws_the_history_then_extends_each_line_from_the_frames_it_is_sent() -> None:  # fmt: skip
    page, s = _page(), T + 5000
    assert page["first"] == "reading the history"
    loaded = page["loaded"]
    assert loaded["asked"] == ["/api/devices/lfs/trends?seconds=600"]
    assert loaded["note"] == _STARTING
    value = loaded["value"]
    assert value["names"] == ["value, display unit", "setpoint, display unit"]
    # x is the server's clock when it answered (`now`, never the earlier `to`), in the viewer's wall time, the window
    # ending now and fixed; a null sample is a gap; a value from before the window enters at its edge; with no live
    # frame no line is held to now
    assert (value["kind"], value["fixed"], value["shape"]) == ("date", True, "linear")
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
    assert sorted(set(page["watched"])) == [0, 1] and len(page["watched"]) == 2


@needs_node
def test_a_window_choice_is_pending_until_its_answer_and_a_failed_read_keeps_the_past() -> None:  # fmt: skip
    page, s = _page(), T + 5000
    window = page["window"]
    # on the click, before the answer: the note is the pending one (never the last window's), the past held stays drawn
    assert window["pending"]["note"] == "reading the history" and window["pending"]["y"][0] == 4  # fmt: skip
    assert window["pressed"] == ["false", "false", "true", "false", "false"]
    assert window["asked"] == "/api/devices/lfs/trends?seconds=120"
    assert window["range"] == [_stamp(s + 5000 - 120_000), _stamp(s + 5000)]
    assert window["note"] == _STARTING
    # the next read fails: the lines keep their past, and the note says it is the earlier read's, never that there is none
    failed = page["failed"]
    assert failed["asked"] == "/api/devices/lfs/trends?seconds=3600" and failed["y"][0] == 4  # fmt: skip
    assert failed["note"] == (
        "the history for value, display unit, setpoint, display unit, actuator could not be read again "
        "(the Control Room did not answer): the past drawn is the earlier read's; "
        "no earlier history for last target (the Control Room did not answer): drawn from page load"
    )


@needs_node
def test_vole_or_the_server_silent_the_plots_start_at_page_load_and_say_so() -> None:
    page = _page(answers=[{"status": 503, "body": {"detail": "The module did not answer: it answered 0"}}])  # fmt: skip
    assert page["loaded"]["note"] == "no earlier history (The module did not answer: it answered 0): drawn from page load"  # fmt: skip
    assert page["loaded"]["actuator"]["y"] == [[None], [None]]
    assert page["live"]["y"][0] == [5, 5]  # the line starts with the first frame
    down = _page(answers=[])
    assert down["loaded"]["note"] == "no earlier history (the Control Room did not answer): drawn from page load"  # fmt: skip
    assert down["live"]["y"][0] == [5, 5]


@needs_node
def test_a_hidden_axis_pane_reads_its_history_when_it_is_first_shown() -> None:
    page = _page(width=0, showLater=True)
    assert page["loaded"]["asked"] == [] and page["loaded"].get("value") is None
    assert page["loaded"]["note"] == "reading the history"
    later = page["later"]
    assert later["asked"] == ["/api/devices/lfs/trends?seconds=600"] and later["calls"] > 0  # fmt: skip
    assert later["note"] == _STARTING


@needs_node
def test_live_points_lost_to_the_cap_are_a_gap_until_the_history_is_read_again() -> None:  # fmt: skip
    page = _page(flood=True)
    # the buffer dropped live points newer than the history's end: the history's last value (4) is closed by a gap,
    # never held up to the oldest live point kept; the history is not asked again inside 30 s of the last read
    assert page["holed"]["y"][:2] == [4, None] and page["holed"]["asked"] == 1
    # then it is: the same window, and the line is the new answer's
    assert page["reread"]["asked"] == 2 and page["reread"]["url"] == "/api/devices/lfs/trends?seconds=600"  # fmt: skip
    assert page["reread"]["note"] == _STARTING


@needs_node
def test_a_window_of_seconds_is_redrawn_on_every_tick_and_a_longer_one_each_second() -> None:  # fmt: skip
    page = _page(fast=True)
    assert page["long"] == [
        0,
        0,
        2,
    ]  # 200 and 400 ms after a draw: nothing; at the second, both plots
    assert page["ten"] == [2, 2] and page["thirty"] == [
        2,
        2,
    ]  # both plots, every 200 ms
    assert page["back"] == [
        0,
        2,
    ]  # 2 min again: nothing at 200 ms, both plots at the second


@needs_node
def test_a_trend_fault_never_reaches_the_rest_of_the_page() -> None:
    # Plotly throwing on every draw, or missing: the device page's listeners still run, and the page says what it can
    assert _page(plotlyThrows=True)["listenerThrows"] is False
    missing = _page(noPlot=True)
    assert missing["listenerThrows"] is False and missing["calls"] == 0
    assert missing["loaded"]["note"] == "the plot library did not load"
