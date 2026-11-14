"""Copy and delete in the browser: ``embed.js``'s ``copyFlow`` and ``deleteFlow`` under node, against this app's own answers.

What a host relies on: a delete asks the check first and opens no dialog for
a document another names (its referrers listed); the dialog's Delete waits
for the name typed and says the file stays in git's history; the DELETE
carries the etag served and the operator. A copy is a create at a new name
in the same folder, the stem-named field renamed, a recipe's ``output_name``
set to the new name.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from geecs_web_theme.testing import node_available

_EMBED = (
    Path(__file__).resolve().parents[1] / "geecs_configurator" / "static" / "embed.js"
)

_HARNESS = r"""
function el(tag) {
  const n = { tagName: tag.toUpperCase(), children: [], attrs: {}, on: {}, value: "", parentNode: null, _t: "", disabled: false,
    setAttribute(k, v) { this.attrs[k] = String(v); }, getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
    appendChild(c) { c.parentNode = this; this.children.push(c); return c; },
    removeChild(c) { this.children = this.children.filter((x) => x !== c); },
    addEventListener(k, fn) { this.on[k] = fn; },
    get textContent() { return this._t + this.children.map((c) => c.textContent).join(""); },
    set textContent(v) { this._t = String(v); this.children = []; } };
  return n;
}
global.document = { createElement: el, createTextNode: (t) => Object.assign(el("#text"), { _t: t }), body: el("body") };
const walk = (n, f, out = []) => { if (f(n)) out.push(n); n.children.forEach((c) => walk(c, f, out)); return out; };
const G = require(process.argv[1]);
const settle = () => new Promise((r) => setTimeout(r, 5));
const sent = [];
let check = { path: "p/high.yaml", deletable: true, referrers: [], reason: "" };
const call = (method, path, body) => { sent.push([method, path, body]);
  if (method === "POST") return Promise.resolve({ status: 200, body: check });
  if (method === "DELETE") return Promise.resolve({ status: 200, body: { outcome: "committed", path: "p/high.yaml", author: "Ada via configurator", commit: "abc1234567", deleted: true } });
  return Promise.resolve({ status: 200, body: { outcome: "committed", author: "Ada via configurator" } }); };
(async () => {
  const out = {}, host = el("div"), status = el("div"), deleted = [], copied = [];
  const ctx = (kind, docId, value) => ({ call, kind, query: () => "?experiment=Bench", docId: () => docId, value: () => value, etag: () => "e1", operator: () => "Ada", status });
  // refused: named by another, no dialog
  check = { path: "p/high.yaml", deletable: false, referrers: ["p/composite.yaml"], reason: "still named by p/composite.yaml" };
  await G.deleteFlow(Object.assign(ctx("setup", "high.yaml", {}), { host, deleted: (b) => deleted.push(b) }))();
  out.refused = [host.children.length, status.textContent.includes("p/composite.yaml"), sent.filter((s) => s[0] === "DELETE").length];
  // deletable: the dialog, its Delete held until the name is typed
  check = { path: "p/high.yaml", deletable: true, referrers: [], reason: "" };
  await G.deleteFlow(Object.assign(ctx("setup", "high.yaml", {}), { host, deleted: (b) => deleted.push(b) }))();
  const dlg = host.children[0];
  const go = walk(dlg, (n) => n.tagName === "BUTTON" && n.textContent === "Delete high")[0], typed = walk(dlg, (n) => n.tagName === "INPUT")[0];
  out.dialog = [dlg.tagName, go.disabled, dlg.textContent.includes("stays in git's history")];
  go.on.click(); await settle();
  out.heldUntilTyped = sent.filter((s) => s[0] === "DELETE").length;
  typed.value = "high"; typed.on.input(); out.enabled = !go.disabled;
  go.on.click(); await settle();
  out.deleteSent = sent.filter((s) => s[0] === "DELETE").map((s) => [s[1], s[2]]);
  out.deleted = [deleted.length, host.children.length, status.textContent.includes("stays in git's history")];
  // copy: a create at the new name, the stem-named field renamed, a recipe's output_name set
  await G.copyFlow(Object.assign(ctx("setup", "high.yaml", { name: "high", steps: [] }), { copied: (id) => copied.push(id) }))("low");
  await G.copyFlow(Object.assign(ctx("analysis_recipe", "HTU/UC_Top.yaml", { device: "UC_Top", output_name: "UC_Top" }), { copied: (id) => copied.push(id) }))("UC_Top_roi");
  out.copies = sent.filter((s) => s[0] === "PUT").map((s) => [s[1], s[2]]);
  out.copied = copied;
  console.log(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


@pytest.fixture(scope="module")
def ran() -> dict:
    if not node_available():  # pragma: no cover - CI and dev machines have it
        pytest.skip("node not available")
    done = subprocess.run(["node", "-e", _HARNESS, str(_EMBED)], text=True, capture_output=True, timeout=30)  # fmt: skip
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_a_document_another_names_opens_no_dialog_and_lists_who(ran: dict) -> None:
    assert ran["refused"] == [0, True, 0]


def test_the_dialog_holds_delete_until_the_name_is_typed_and_says_history_keeps_it(ran: dict) -> None:  # fmt: skip
    assert ran["dialog"] == ["DIALOG", True, True]
    assert ran["heldUntilTyped"] == 0 and ran["enabled"] is True


def test_the_delete_sends_the_served_etag_and_the_operator(ran: dict) -> None:
    assert ran["deleteSent"] == [["/api/documents/setup/high.yaml?experiment=Bench", {"etag": "e1", "operator": "Ada"}]]  # fmt: skip
    assert ran["deleted"] == [
        1,
        0,
        True,
    ]  # the host told, the dialog gone, the answer shown


def test_a_copy_is_a_create_at_the_new_name_renamed_as_its_kind_needs(
    ran: dict,
) -> None:
    (setup_url, setup), (recipe_url, recipe) = ran["copies"]
    assert setup_url == "/api/documents/setup/low.yaml?experiment=Bench" and setup["create"] is True and setup["etag"] == ""  # fmt: skip
    assert setup["document"]["name"] == "low"
    assert (
        recipe_url
        == "/api/documents/analysis_recipe/HTU/UC_Top_roi.yaml?experiment=Bench"
    )
    assert recipe["document"]["output_name"] == "UC_Top_roi" and recipe["document"]["device"] == "UC_Top"  # fmt: skip
    assert ran["copied"] == ["low.yaml", "HTU/UC_Top_roi.yaml"]
