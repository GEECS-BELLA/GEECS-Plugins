"""GeecsKit.picker — the one drawer picker every surface's "Add …" drawer uses.

The real ``kit.js`` runs under node over a DOM just big enough for it: the
elements the picker is handed, clickable rows, and a disabled-aware click.
Each case drives it the way a person does (open, type, click, shift-click,
Enter) and reads back what Add delivered.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from geecs_web_theme import static_dir
from geecs_web_theme.testing import node_available

_DOM = r"""
const handlers = new Map();
function on(node, kind, fn) {
  const key = node.__id + ":" + kind;
  (handlers.get(key) || handlers.set(key, []).get(key)).push(fn);
}
function fire(node, kind, ev) {
  for (const fn of handlers.get(node.__id + ":" + kind) || []) fn(ev || {});
}
let ids = 0;
function node(tag) {
  const n = {
    __id: ++ids, tagName: tag, children: [], attrs: {}, disabled: false,
    textContent: "", value: "", className: "", type: "",
    setAttribute(k, v) { this.attrs[k] = String(v); },
    getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
    hasAttribute(k) { return k in this.attrs; },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener(kind, fn) { on(this, kind, fn); },
    focus() {},
    click() { if (!this.disabled) fire(this, "click", { shiftKey: false }); },
  };
  Object.defineProperty(n, "textContent", {
    get() { return this._text || ""; },
    set(v) { this._text = v; if (v === "") this.children = []; },
  });
  return n;
}
global.window = {};
global.document = {
  readyState: "complete",
  documentElement: node("html"),
  addEventListener() {},
  querySelector() { return null; },
  querySelectorAll() { return []; },
  createElement: (tag) => node(tag),
  getElementById() { return null; },
  contains() { return true; },
};
global.setTimeout = (fn) => fn();
"""

_DRIVER = r"""
const el = { drawer: node("aside"), opener: node("button"), search: node("input"),
             list: node("nav"), add: node("button"), all: node("button"), none: node("button") };
const added = [];
let ROWS = [
  { name: "A", pickable: true }, { name: "B", pickable: true },
  { name: "C", pickable: false, sub: "in the table" }, { name: "D", pickable: true },
  { name: "E", pickable: true },
];
let ROOM = Infinity, SHOWN_ONLY = false;
const notes = [];
const p = window.GeecsKit.picker(Object.assign({}, el, {
  onRender: (info) => notes.push(info.full),
  noun: ["thing", "things"],
  rows: (q) => ROWS.filter((r) => !q || r.name.toLowerCase().includes(q.toLowerCase())),
  room: () => ROOM,
  shownOnly: () => SHOWN_ONLY,
  onAdd: (names) => added.push(names.slice().sort()),
}));
const row = (name) => el.list.children.find((b) => b.getAttribute("data-name") === name);
const click = (name, shift) => { if (!row(name).disabled) fire(row(name), "click", { shiftKey: !!shift }); };
const type = (text) => { el.search.value = text; fire(el.search, "input"); };
const enter = () => fire(el.search, "keydown", { key: "Enter", preventDefault() {} });
const state = () => ({ added, full: notes[notes.length - 1], label: el.add.textContent, addDisabled: el.add.disabled,
                       pressed: el.list.children.filter((b) => b.getAttribute("aria-pressed") === "true").map((b) => b.getAttribute("data-name")) });
fire(el.opener, "click");
"""


def _run(steps: str) -> dict:
    if not node_available():  # pragma: no cover - CI and dev machines have it
        pytest.skip("node not available")
    kit = (static_dir() / "kit.js").read_text()
    script = _DOM + kit + _DRIVER + steps + "\nconsole.log(JSON.stringify(state()));"
    done = subprocess.run(
        ["node", "-"], input=script, text=True, capture_output=True, timeout=30
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_a_click_toggles_and_add_delivers_them_all() -> None:
    out = _run("click('A'); click('B'); click('B'); click('D'); el.add.click();")
    assert out["added"] == [["A", "D"]]


def test_the_add_button_counts_the_picks() -> None:
    out = _run("click('A'); click('D');")
    assert out["label"] == "Add 2 things" and out["addDisabled"] is False
    assert out["pressed"] == ["A", "D"]


def test_shift_click_takes_the_range_and_skips_the_unpickable() -> None:
    out = _run("click('A'); click('E', true); el.add.click();")
    assert out["added"] == [["A", "B", "D", "E"]]  # C is not pickable


def test_select_all_shown_and_clear() -> None:
    out = _run("type('a'); fire(el.all, 'click');")
    assert out["pressed"] == ["A"]
    out = _run("fire(el.all, 'click'); fire(el.none, 'click');")
    assert out["pressed"] == [] and out["addDisabled"] is True


def test_picks_survive_a_narrowing_search() -> None:
    out = _run("click('A'); type('d'); click('D'); type(''); el.add.click();")
    assert out["added"] == [["A", "D"]]


def test_a_pick_never_passes_the_room_left() -> None:
    """A bounded list refuses a batch past its bound whole; the picker stops at it."""
    out = _run(
        "ROOM = 2; click('A'); click('B'); click('D'); fire(el.all, 'click'); el.add.click();"
    )
    assert out["added"] == [["A", "B"]]
    out = _run("ROOM = 2; click('A'); click('E', true); el.add.click();")
    assert out["added"] == [["A", "B"]]


def test_enter_with_one_row_left_adds_it() -> None:
    """GPT review of #1128: Enter on a single match must add it, not click a disabled Add."""
    out = _run("type('d'); enter();")
    assert out["added"] == [["D"]]
    # with picks, Enter adds the picks
    out = _run("click('A'); click('B'); type('e'); enter();")
    assert out["added"] == [["A", "B"]]
    # several matches and nothing picked: Enter does nothing
    out = _run("enter();")
    assert out["added"] == []


def test_refresh_keeps_the_picks_when_the_rows_arrive_late() -> None:
    """GPT review of #1128: a drawer opened before its list loaded fills when it lands."""
    out = _run(
        "ROWS = []; fire(el.opener, 'click');"
        "ROWS = [{name: 'A', pickable: true}, {name: 'B', pickable: true}]; p.refresh();"
        "click('A'); p.refresh(); el.add.click();"
    )
    assert out["added"] == [["A"]]


def test_shown_only_drops_picks_that_left_the_rows() -> None:
    """Typed rows are not a catalog: a pick of a half-typed name must not survive."""
    out = _run(
        "SHOWN_ONLY = true; type('a'); click('A'); type('b'); click('B'); el.add.click();"
    )
    assert out["added"] == [["B"]]


def test_opening_starts_a_fresh_selection() -> None:
    out = _run("click('A'); fire(el.opener, 'click'); click('B'); el.add.click();")
    assert out["added"] == [["B"]]


def test_the_full_flag_follows_every_pick_and_clear() -> None:
    """Review of #1128: a "list is full" note must appear on the filling pick and go on Clear."""
    assert _run("ROOM = 2; click('A'); click('B');")["full"] is True
    assert (
        _run("ROOM = 2; click('A'); click('B'); fire(el.none, 'click');")["full"]
        is False
    )
    assert _run("ROOM = 2; click('A');")["full"] is False
