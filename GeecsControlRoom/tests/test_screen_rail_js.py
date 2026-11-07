"""A screen's write path under node: the real device_rail.js and device_writes.js over a fake screen (skipped without node).

The widgets are screen.js's shape (a `.w` wrapper carrying the binding and its
`confirm` note, the rail's control inside it, an outcome line); each widget's
page answers `for(el)` as screen.js's `pageOf` does, and each device has its
own device_writes. The fetch and the hazard dialog are the harness's.
"""

from __future__ import annotations

from typing import Any

from tests._node import needs_node, run

_SCREEN = r"""
const vm = require("vm"), fs = require("fs"), path = require("path");
const dir = path.dirname(process.argv[1]);
const req = (f) => require(path.join(dir, f));
const C = req("camera_rules.js"), D = req("device_rules.js"), S = req("screen_rules.js");
const flush = async () => { for (let i = 0; i < 10; i++) await new Promise((r) => setImmediate(r)); };
function el(tag, props = {}) {
  const on = {}, kids = props.kids || {};
  return Object.assign({ tagName: tag, dataset: {}, attrs: {}, hidden: false, textContent: "", value: "", disabled: false, title: "", options: [],
    setAttribute(k, v) { this.attrs[k] = String(v); }, getAttribute(k) { return this.attrs[k]; },
    addEventListener(t, f) { (on[t] = on[t] || []).push(f); }, fire(t, ev = {}) { (on[t] || []).forEach((f) => f(ev)); },
    querySelector(sel) { return kids[sel] || null; }, querySelectorAll() { return []; }, closest() { return null; },
    replaceChildren(...xs) { this.options = xs; }, focus() {}, blur() {} }, props);
}
const line = () => el("SPAN", { kids: { b: el("B"), ".dot": el("SPAN"), ".word": el("SPAN") } });
// a widget: its wrapper (binding, note, outcome line), its field, and the controls inside
function widget(data, controls, setpoint = false) {
  const out = el("SPAN"), w = el("DIV", { dataset: data, kids: { ".out": out } });
  const field = el("DIV", { kids: { ".sv": line(), ".rb": line(), ".differs": el("SPAN") } });
  for (const c of controls) c.closest = (sel) => (sel === ".w" ? w : sel === ".field" || (setpoint && sel === ".field.setpoint") ? field : null);
  return { w, out, controls };
}
const pairs = ["0", "1"].map((v) => el("BUTTON", { dataset: { role: "DO.Ch1", value: v, pair: "", enumLabel: "" }, textContent: v === "0" ? "Off" : "On" }));
const plc = widget({ device: "PLC", role: "DO.Ch1", confirm: "Shutter the laser first." }, pairs);
const narrowed = el("SELECT", { dataset: { role: "pos", choices: JSON.stringify(["A", "C"]) } });
const wheel = widget({ device: "Wheel", role: "pos" }, [narrowed]);
const unread = el("SELECT", { dataset: { role: "pos2" } });
const wheel2 = widget({ device: "Wheel", role: "pos2" }, [unread]);
const field = el("INPUT", { type: "number", step: "any", dataset: { role: "x:u_a:b", readback: "x:u_a:b" } });
const bare = widget({ pv: "x:u_a:b" }, [field], true);
const all = [...pairs, narrowed, unread, field];
const box = el("MAIN", { querySelectorAll: (sel) => (sel === ".field [data-role]" ? all : []) });
// the module's entries and Hare's values per device, as screen.js's pageOf answers them
const DEV = {
  PLC: { roles: { "DO.Ch1|": { role: "DO.Ch1", pv: "plc:do1", writePv: "plc:do1:SP", writable: true, read: "monitor", verdict: null } },
         frames: { "DO.Ch1": { value: 0, connected: true, severity: "NO_ALARM", alarmStatus: 0 } }, meta: { "DO.Ch1": { enumChoices: ["out", "in"] } } },
  Wheel: { roles: { "pos|": { role: "pos", pv: "w:pos", writePv: "w:pos:SP", writable: true, read: "monitor", verdict: null },
                    "pos2|": { role: "pos2", pv: "w:pos2", writePv: "w:pos2:SP", writable: true, read: "monitor", verdict: null } },
           frames: { pos: { value: 0, connected: true }, pos2: { value: 1, connected: true } }, meta: { pos: { enumChoices: ["A", "B", "C", "D"] }, pos2: {} } },
};
const device = (name) => ({ name, root: "", view: () => ({ ok: true, state: "live", roles: DEV[name].roles, source: "gateway" }),
  entry: (role) => DEV[name].roles[role + "|"], frame: (role) => DEV[name].frames[role], meta: (role) => DEV[name].meta[role] || {},
  word: () => "live", target: () => null });
const barePage = { name: null, root: "", view: () => ({ ok: true, roles: {} }), entry: () => undefined, target: () => null,
  frame: () => ({ value: 1.5, connected: true }), meta: () => ({}), word: () => "live" };
const dialog = { open: false, onclick: null, onclose: null, showModal() { this.open = true; }, close() { this.open = false; setTimeout(() => this.onclose && this.onclose()); } };
const ids = { operator: el("INPUT", { value: "ann" }), "write-out": el("DIV"), "hazard-dialog": dialog, "hazard-dialog-text": el("P"),
  "hazard-dialog-roles": el("P"), "hazard-dialog-note": el("P"), "hazard-dialog-title": el("H4"), "kit-state": el("SCRIPT", { textContent: "{}" }) };
const puts = [];
const fetch = async (url, init) => { puts.push({ url, body: JSON.parse(init.body) }); return { json: async () => ({ outcome: "accepted", journalId: puts.length }) }; };
const document = { getElementById: (id) => ids[id] || null, activeElement: null };
function Option(text, value) { return { text, textContent: text, value }; }
const window = { GeecsCameraRules: C, GeecsDeviceRules: D, GeecsScreenRules: S };
const ctx = vm.createContext({ window, document, fetch, console, Option, CSS: { escape: (s) => s }, setTimeout, Date });
for (const f of ["device_writes.js", "device_rail.js"]) vm.runInContext(fs.readFileSync(path.join(dir, f), "utf8"), ctx);
const writes = {};
const writesOf = (name) => (writes[name] = writes[name] || window.GeecsDeviceWrites.make(device(name), () => rail.fill()));
const page = { for: (c) => (c.closest(".w").dataset.pv ? barePage : device(c.closest(".w").dataset.device)), on() {}, view: () => null };
const rail = window.GeecsDeviceRail.make(page, box, { for: (c) => (c.closest(".w").dataset.pv ? { busy: () => false, write: async () => false } : writesOf(c.closest(".w").dataset.device)) });
(async () => {
  const out = {};
  out.pair = pairs.map((b) => [b.textContent, b.getAttribute("aria-pressed"), b.disabled]);
  pairs[1].fire("click");
  await flush();
  out.asked = { open: dialog.open, note: ids["hazard-dialog-note"].textContent, title: ids["hazard-dialog-title"].textContent, puts: puts.length };
  dialog.onclick({ target: { value: "write" } });
  await flush();
  out.written = { puts: puts.slice(), line: plc.out.textContent };
  pairs[1].fire("click");
  await flush();
  dialog.onclick({ target: { value: "cancel" } });
  await flush();
  out.cancelled = { puts: puts.length, line: plc.out.textContent };
  out.narrowed = { options: narrowed.options.map((o) => [o.text, o.value]), disabled: narrowed.disabled };
  narrowed.value = "2";
  narrowed.fire("change");
  await flush();
  out.chosen = { put: puts[puts.length - 1], open: dialog.open, line: wheel.out.textContent };
  // an unsent choice (no operator named) goes back to what the device reads, the select let go
  let blurred = 0;
  narrowed.blur = () => { blurred += 1; };
  ids.operator.value = "";
  narrowed.value = "2";
  narrowed.fire("change");
  await flush();
  out.unsent = { value: narrowed.value, blurred, puts: puts.length, line: wheel.out.textContent };
  out.unread = unread.disabled;
  out.bare = { disabled: field.disabled, value: field.value };
  console.log(JSON.stringify(out));
})();
"""


@needs_node
def test_a_screen_widget_writes_through_the_rail_and_its_note_asks_first() -> None:
    page: dict[str, Any] = run("device_rail.js", _SCREEN)
    # the pair: the enum's own words, the reading's state pressed, enabled on the module's word
    assert page["pair"] == [["out", "true", False], ["in", "false", False]]
    # the widget's confirm note asks before anything is sent (a note alone: no hazard role marked)
    assert page["asked"] == {
        "open": True,
        "note": "Shutter the laser first.",
        "title": "Write?",
        "puts": 0,
    }
    [put] = page["written"]["puts"]
    assert put["url"] == "/api/devices/PLC/put"
    assert put["body"] == {
        "role": "DO.Ch1",
        "value": 1,
        "operator": "ann",
        "confirm": True,
    }
    assert (
        page["written"]["line"] == "DO.Ch1 ← 1 · accepted"
    )  # on the widget's own line
    assert page["cancelled"] == {
        "puts": 1,
        "line": "unsent · not confirmed",
    }  # a cancel sends nothing
    # the select: the PV's choices narrowed to the family role's, by the index the gate takes
    assert page["narrowed"] == {"options": [["A", "0"], ["C", "2"]], "disabled": False}
    assert page["chosen"]["put"]["body"] == {
        "role": "pos",
        "value": 2,
        "operator": "ann",
    }
    assert not page["chosen"]["open"] and page["chosen"]["line"] == "pos ← 2 · accepted"
    assert page["unsent"] == {"value": "0", "blurred": 1, "puts": 2, "line": "unsent · name yourself in the header first"}  # fmt: skip
    assert (
        page["unread"] is True
    )  # no choices read yet: nothing to choose, the module would refuse
    assert page["bare"] == {
        "disabled": True,
        "value": "1.5",
    }  # a bare PV: shown, never written


# The real screen.js (with the real rail, writes, device source and rules) in a VM over a fake page: Hare, the
# poller and the DOM are the harness's, the module's device route answered through the poller's onView.
_PAGE = r"""
const vm = require("vm"), fs = require("fs"), path = require("path");
const dir = path.dirname(process.argv[1]);
const req = (f) => require(path.join(dir, f));
const C = req("camera_rules.js"), D = req("device_rules.js"), S = req("screen_rules.js");
const flush = async () => { for (let i = 0; i < 10; i++) { await new Promise((r) => setImmediate(r)); await new Promise((r) => setTimeout(r, 2)); } };
let clock = 1000;
function el(tag, props = {}) {
  const on = {}, kids = props.kids || {};
  return Object.assign({ tagName: tag, dataset: {}, attrs: {}, hidden: false, textContent: "", value: "", disabled: false, title: "", options: [],
    setAttribute(k, v) { this.attrs[k] = String(v); }, getAttribute(k) { return this.attrs[k]; },
    addEventListener(t, f) { (on[t] = on[t] || []).push(f); }, fire(t, ev = {}) { (on[t] || []).forEach((f) => f(ev)); },
    querySelector(sel) { return kids[sel] || null; }, querySelectorAll() { return []; }, closest() { return null; },
    replaceChildren(...xs) { this.options = xs; }, focus() {}, blur() {} }, props);
}
const line = () => el("SPAN", { kids: { b: el("B"), ".dot": el("SPAN"), ".word": el("SPAN") } });
// a DB device's setpoint shown at precision 0 (its record's PREC is 3), and a bare-PV readout whose units are Hare's alone
const spField = el("DIV", { kids: { ".sv": line(), ".rb": line(), ".differs": el("SPAN") } });
const sp = el("INPUT", { type: "number", step: "any", dataset: { role: "X", readback: "X" } });
const spW = el("DIV", { dataset: { device: "Dev", role: "X", deferred: "1", fits: "setpoint", precision: "0" }, kids: { ".out": el("SPAN") } });
sp.closest = (s) => (s === ".w" ? spW : s === ".field" || s === ".field.setpoint" ? spField : null);
const roField = el("DIV", { kids: { ".ws": line() } });
const ro = el("SPAN", { dataset: { ro: "x:press" }, parentElement: roField });
const egu = el("SMALL", { dataset: { egu: "x:press", type: "readback" } });
const roW = el("DIV", { dataset: { pv: "x:press" }, kids: { ".out": el("SPAN") } });
for (const e of [ro, egu]) e.closest = (s) => (s === ".w" ? roW : null);
ro.parentElement = roField;
const main = el("MAIN", { querySelectorAll: (s) => (s === ".field [data-role]" ? [sp] : s === "[data-ro]" ? [ro] : s === "[data-egu]" ? [egu] : []) });
const ids = { "kit-state": el("SCRIPT", { textContent: "{}" }), operator: el("INPUT", { value: "ann" }), "hare-state": el("SPAN"), confirm: el("DIALOG"),
  "write-out": el("DIV"), "hazard-dialog": el("DIALOG"), "hazard-dialog-text": el("P"), "hazard-dialog-roles": el("P"), "hazard-dialog-note": el("P"), "hazard-dialog-title": el("H4") };
const document = { getElementById: (id) => ids[id] || null, activeElement: null, body: { dataset: {} },
  querySelectorAll: (s) => (s === ".w" ? [spW, roW] : s === ".w[data-device], .ind[data-device]" ? [spW] : []),
  querySelector: (s) => (s === "main.screen" ? main : s === ".hare .dot" ? el("SPAN") : null) };
const META = { "dev:x": { precision: 3 }, "dev:x:SP": { precision: 3 }, "x:press": { units: "Torr", precision: 2 } };
const FRAMES = { "dev:x": { value: 4.4, connected: true, severity: "NO_ALARM", alarmStatus: 0 }, "dev:x:SP": { value: 4.4, connected: true }, "x:press": { value: 0.0012, connected: true } };
const watchers = {};
const hare = { watch(pvs, fn) { for (const pv of pvs) { (watchers[pv] = watchers[pv] || []).push(fn); fn(pv, FRAMES[pv] || null, META[pv] || {}); } },
  meta: (pv) => META[pv] || {}, onStatus(fn) { this.status = fn; } };
const tell = (pv) => (watchers[pv] || []).forEach((fn) => fn(pv, FRAMES[pv], META[pv]));
let onView = null;
const window = { GeecsCameraRules: C, GeecsDeviceRules: D, GeecsScreenRules: S, GeecsHare: { fromPage: () => hare },
  GeecsPoller: { makePoller: (o) => { onView = onView || o.onView; return { tick() {} }; } } };
const fetch = async (url, init) => { FRAMES["dev:x:SP"] = { value: JSON.parse(init.body).value, connected: true }; return { json: async () => ({ outcome: "accepted", journalId: 1 }) }; };
function Option(text, value) { return { text, textContent: text, value }; }
const ctx = vm.createContext({ window, document, fetch, console, Option, CSS: { escape: (s) => s }, setTimeout, setInterval: () => 0,
  requestAnimationFrame: (f) => setTimeout(f, 0), Date: { now: () => clock } });
for (const f of ["device_source.js", "device_writes.js", "device_rail.js"]) vm.runInContext(fs.readFileSync(path.join(dir, f), "utf8"), ctx);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), ctx);
(async () => {
  const out = {};
  hare.status("open");
  onView("Dev", { ok: true, value: D.deviceView(true, { state: "live", source: "gateway",
    roles: [{ role: "X", axis: null, pv: "dev:x", writePv: "dev:x:SP", type: "setpoint", writable: true, read: "monitor", verdict: null }] }) });
  await flush();
  out.shown = { field: sp.value, now: spField.querySelector(".rb").querySelector("b").textContent, ro: ro.textContent, units: egu.textContent };
  sp.value = "5"; sp.dataset.draft = "1";
  sp.fire("keydown", { key: "Enter", preventDefault() {} });
  await flush();
  tell("dev:x:SP"); clock += 1500; tell("dev:x"); await flush();
  out.settled = { set: spField.querySelector(".sv").querySelector("b").textContent, differs: !spField.querySelector(".differs").hidden, armed: sp.dataset.armed !== undefined };
  console.log(JSON.stringify(out));
})();
"""


@needs_node
def test_a_widgets_precision_shows_but_never_judges_and_a_readouts_units_are_hares() -> (
    None
):
    page: dict[str, Any] = run("screen.js", _PAGE)
    # shown at the widget's precision 0; a bare readout carries its units from Hare's metadata alone
    assert page["shown"] == {"field": "4", "now": "4", "ro": "0.0012", "units": "Torr"}
    # set 5, the device at 4.4: judged within the record's PREC 3 (0.001), never the widget's 0 digits (1)
    assert page["settled"] == {"set": "5", "differs": True, "armed": True}
