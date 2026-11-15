"""A capture's answer is installed only while it still answers the form: an edit or a later Capture drops it.

Under node, the page's ``setups.js`` with its readbacks held and released
out of order: a capture answered after the name and variables were edited
offers no Save (it would create the old name with the old variables); of two
overlapping captures the later one's answer is the one shown and saved,
whatever order they resolve in; and a Save sends the capture it was offered,
leaving Save off when the form was edited while it was on its way.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from geecs_web_theme.testing import node_available

_SETUPS = (
    Path(__file__).resolve().parents[1] / "geecs_configurator" / "static" / "setups.js"
)

_HARNESS = r"""
function el(tag) {
  const n = { tagName: tag.toUpperCase(), children: [], attrs: {}, on: {}, value: "", _t: "", disabled: false, dataset: {},
    setAttribute(k, v) { this.attrs[k] = String(v); },
    getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener(k, fn) { this.on[k] = fn; },
    get childNodes() { return this.children; },
    get textContent() { return this._t + this.children.map((c) => c.textContent).join(""); },
    set textContent(v) { this._t = String(v); this.children = []; } };
  return n;
}
const ids = ["cap-vars", "cap-save", "capture-open", "cap-settables", "cap-pick-hint", "cap-name", "cap-desc", "cap-pick", "cap-read", "cap-result"];
const nodes = {};
ids.forEach((i) => { nodes["#" + i] = el(i === "cap-save" || i === "cap-read" ? "button" : "input"); });
nodes["#cap-save"].disabled = true;
const body = el("body"); body.dataset = { root: "", id: "", experiment: "Bench" };
global.document = { body, createElement: el, querySelector: (s) => nodes[s] || null, querySelectorAll: () => [] };
global.window = { GeecsConfiguratorPage: { operator: () => "Ada", ready: () => true, dirty: () => false } };
global.location = { href: "" };
const sent = [], held = [];
global.fetch = (url, init) => {
  const method = (init && init.method) || "GET", payload = init && init.body ? JSON.parse(init.body) : undefined;
  sent.push([method, url, payload]);
  return new Promise((resolve) => held.push({ method, url, payload, answer: (status, out) => resolve({ status, text: () => Promise.resolve(JSON.stringify(out)) }) }));
};
const settle = () => new Promise((r) => setTimeout(r, 5));
require(process.argv[1]);
const type = (sel, text) => { nodes[sel].value = text; nodes[sel].on.input({ target: nodes[sel] }); };
const capture = () => nodes["#cap-read"].on.click();
const save = () => nodes["#cap-save"].on.click();
const draft = (p) => ({ id: p.name + ".yaml", document: { name: p.name, steps: p.variables.map((v) => ({ do: "set", device: v.split(":")[0], variable: v.split(":")[1], value: 1 })) }, unread: [] });
const puts = () => sent.filter((s) => s[0] === "PUT").map((s) => [s[1].split("?")[0], s[2].document.steps.map((x) => x.device + ":" + x.variable)]);
const pending = (method) => held.filter((h) => !h.done && h.method === method);
const release = (h, status, out) => { h.done = true; h.answer(status, out); };
(async () => {
  const out = {};
  // 1. edited while the capture was pending: the old answer installs nothing
  type("#cap-name", "first"); type("#cap-vars", "A:X");
  capture();
  const first = pending("POST")[0];
  type("#cap-name", "second"); type("#cap-vars", "B:Y");
  release(first, 200, draft(first.payload)); await settle();
  out.edited = { saveDisabled: nodes["#cap-save"].disabled, says: nodes["#cap-result"].textContent };
  save(); await settle();
  out.edited.puts = puts();
  // 2. two overlapping captures, answered newest first: the newer stays
  type("#cap-name", "third"); type("#cap-vars", "C:Z");
  capture(); const older = pending("POST")[0];
  type("#cap-vars", "D:W");
  capture(); const newer = pending("POST")[1];
  release(newer, 200, draft(newer.payload)); await settle();
  release(older, 200, draft(older.payload)); await settle();
  out.overlap = { saveDisabled: nodes["#cap-save"].disabled, says: nodes["#cap-result"].textContent };
  save(); await settle();
  const put = pending("PUT").slice(-1)[0];
  out.overlap.puts = puts();
  // 3. an edit while the save is on its way: the save sends the capture it was offered, and Save stays off
  type("#cap-name", "fourth");
  release(put, 502, { detail: "the push failed" }); await settle();
  out.saving = { saveDisabled: nodes["#cap-save"].disabled };
  // 3b. a save answered 200 after an edit opens the setup it created
  type("#cap-name", "sixth"); type("#cap-vars", "F:U");
  capture(); const six = pending("POST")[0];
  release(six, 200, draft(six.payload)); await settle();
  save(); const sixPut = pending("PUT").slice(-1)[0];
  type("#cap-name", "seventh");
  release(sixPut, 200, { outcome: "committed" }); await settle();
  out.saving.opened = location.href;
  // 4. an answer that still matches is installed as before
  type("#cap-name", "fifth"); type("#cap-vars", "E:V");
  capture(); const fine = pending("POST")[0];
  release(fine, 200, draft(fine.payload)); await settle();
  out.fresh = { saveDisabled: nodes["#cap-save"].disabled, says: nodes["#cap-result"].textContent };
  console.log(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


@pytest.fixture
def run() -> dict:
    if not node_available():
        pytest.skip("node is not installed")
    done = subprocess.run(["node", "-e", _HARNESS, str(_SETUPS)], capture_output=True, text=True, timeout=60)  # fmt: skip
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_a_capture_answered_after_an_edit_offers_no_save(run: dict) -> None:
    edited = run["edited"]
    assert edited["saveDisabled"] is True
    assert "Capture again" in edited["says"]
    assert edited["puts"] == []  # first.yaml with A:X is never created


def test_of_two_overlapping_captures_the_later_one_is_shown_and_saved(
    run: dict,
) -> None:
    overlap = run["overlap"]
    assert overlap["saveDisabled"] is False and "D:W" in overlap["says"] and "C:Z" not in overlap["says"]  # fmt: skip
    assert overlap["puts"] == [["/api/documents/setup/third.yaml", ["D:W"]]]


def test_an_edit_while_the_save_is_on_its_way_leaves_save_off(run: dict) -> None:
    assert run["saving"]["saveDisabled"] is True
    assert run["saving"]["opened"].endswith(
        "&kind=setup&id=sixth.yaml"
    )  # the capture it sent


def test_a_capture_that_still_answers_the_form_is_installed(run: dict) -> None:
    fresh = run["fresh"]
    assert fresh["saveDisabled"] is False and "E:V" in fresh["says"]
