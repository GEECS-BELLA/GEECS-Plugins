"""GeecsKit.toast — one short line that something happened, gone on its own.

The real ``kit.js`` runs under node over a DOM just big enough for it, with
timers held so a case can read the toast before and after its clock runs
out.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from geecs_web_theme import static_dir
from geecs_web_theme.testing import node_available

_DOM = r"""
function node(tag) {
  return {
    tagName: tag, children: [], attrs: {}, className: "", hidden: false, textContent: "",
    setAttribute(k, v) { this.attrs[k] = String(v); },
    getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener() {},
  };
}
const body = node("body");
global.window = {};
global.document = {
  readyState: "complete",
  documentElement: node("html"),
  body,
  addEventListener() {},
  querySelector: () => body.children.find((c) => c.className === "toast") || null,
  querySelectorAll() { return []; },
  createElement: (tag) => node(tag),
  getElementById() { return null; },
};
"""

_DRIVER = r"""
const timers = [];
global.setTimeout = (fn, ms) => { timers.push({ fn, ms }); return timers.length; };
global.clearTimeout = (id) => { if (timers[id - 1]) timers[id - 1].fn = () => {}; };
const run = () => timers.splice(0).forEach((t) => t.fn());
const toasts = () => body.children.filter((c) => c.className === "toast");
const state = () => ({ count: toasts().length, text: toasts()[0] && toasts()[0].textContent,
                       hidden: toasts()[0] && toasts()[0].hidden,
                       role: toasts()[0] && toasts()[0].getAttribute("role"),
                       ms: timers.map((t) => t.ms) });
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


def test_a_toast_shows_its_text_and_is_announced() -> None:
    out = _run('window.GeecsKit.toast("copied");')
    assert out["text"] == "copied" and out["hidden"] is False
    assert out["role"] == "status"


def test_a_toast_hides_when_its_clock_runs_out() -> None:
    out = _run('window.GeecsKit.toast("copied", 500); run();')
    assert out["hidden"] is True


def test_a_second_toast_replaces_the_first_and_restarts_the_clock() -> None:
    out = _run(
        'window.GeecsKit.toast("one"); window.GeecsKit.toast("two"); run();'
        ' window.GeecsKit.toast("three");'
    )
    assert out["count"] == 1 and out["text"] == "three" and out["hidden"] is False
