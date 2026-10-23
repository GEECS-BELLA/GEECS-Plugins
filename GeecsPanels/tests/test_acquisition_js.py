"""acquisition.js's behaviour, run in node against a stub DOM (no browser)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1] / "geecs_panels" / "static" / "acquisition.js"
)

# A stub DOM: just what acquisition.js touches. Elements are made on first
# lookup (a value is a string, as in a DOM); `type()` is a user edit (value +
# an input event), `snap()` a snapshot.
_HARNESS = r"""
const els = {};
const el = (id) => els[id] || (els[id] = {
  id, _v: "", get value() { return this._v; }, set value(v) { this._v = String(v); },
  hidden: false, disabled: false, textContent: "", on: {},
  addEventListener(ev, fn) { (this.on[ev] = this.on[ev] || []).push(fn); },
  fire(ev) { for (const fn of this.on[ev] || []) fn({ target: this }); },
  showModal() {}, close() {}, focus() { document.activeElement = this; },
  querySelectorAll() { return Object.values(els).filter((e) => e.id.startsWith("cam-")); },
});
globalThis.document = { activeElement: null, getElementById: el };
const puts = [], hooks = { snapshot: [], streamLost: [] };
globalThis.window = { GeecsCameraView: {
  root: { dataset: { writable: "true", unlockUrl: "/unlock" } }, hooks,
  write: {
    post: async () => {}, put: async (pv, v) => { puts.push([pv, v]); },
    putAll: async (pairs) => { for (const p of pairs) puts.push(p); }, sayError() {},
  },
}};
const acq = (x) => ({ min_x: x, min_y: 0, size_x: 1280, size_y: 1024, max_x: 1280, max_y: 1024 });
const snap = (x) => { for (const fn of hooks.snapshot) fn({
  lock: { state: "free", owner: "" }, image: { age: 0 }, readings: [], acquisition: acq(x),
}); };
const type = (id, v) => { el(id).focus(); el(id).value = String(v); el(id).fire("input"); };
const tick = () => new Promise((r) => setTimeout(r, 0));
require(process.argv[2]);
(async () => {
  const out = {};
  snap(0);
  type("cam-minx", 100);          // x edited ...
  type("cam-sizex", 600);         // ... then focus moves to width
  snap(0);                        // the next snapshot arrives
  out.drafted = [el("cam-minx").value, el("cam-sizex").value];
  out.revertShown = !el("cam-roi-revert").hidden;
  el("cam-roi-apply").fire("click"); el("roi-confirm").fire("click"); await tick();
  out.puts = puts.slice();
  document.activeElement = null;
  snap(100);                      // the camera's answer refills the fields
  out.afterApply = [el("cam-minx").value, !el("cam-roi-revert").hidden];
  type("cam-miny", 50); el("cam-roi-revert").fire("click");
  out.reverted = el("cam-miny").value;
  console.log(JSON.stringify(out));
})();
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_an_roi_draft_survives_snapshots_until_applied(tmp_path: Path) -> None:
    """Editing x, then width: the next snapshot must not reset x (GPT P2 on #1137)."""
    harness = tmp_path / "harness.js"
    harness.write_text(_HARNESS)
    run = subprocess.run(
        ["node", str(harness), str(_SCRIPT)], capture_output=True, text=True, timeout=30
    )
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)
    assert out["drafted"] == ["100", "600"]
    assert out["revertShown"] is True
    assert out["puts"] == [
        ["cam1:MinX", 100],
        ["cam1:MinY", 0],
        ["cam1:SizeX", 600],
        ["cam1:SizeY", 1024],
    ]
    assert out["afterApply"] == ["100", False]  # readbacks back in charge
    assert out["reverted"] == "0"  # Revert discards the draft
