"""The embedded editor's mount contract: ``embed.js`` and the theme's ``form.js`` under node, against this app.

The configurator's own answers (the kind's list, the schema, the document,
the catalogue) are read from the real app and handed to a stub ``fetch``,
so a reshaped route fails here; the host is a stub DOM. What is pinned is
the contract a host relies on: the drawer draws only ``fields``, the save
sends the whole document (the undrawn fields as stored), the etag it was
served and the host's operator, never saves without one, asks the dry run
first and stops on its refusal, and hands ``preview`` the document.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from geecs_web_theme import static_dir
from geecs_web_theme.testing import node_available

from geecs_configurator.save import SaveRequest

_EMBED = (
    Path(__file__).resolve().parents[1] / "geecs_configurator" / "static" / "embed.js"
)
API = "http://cfg.test"
Q = "?experiment=Bench"

_HARNESS = r"""
let ids = 0;
function el(tag) {
  const n = { __id: ++ids, tagName: tag.toUpperCase(), children: [], attrs: {}, on: {}, value: "", parentNode: null, _t: "", disabled: false,
    setAttribute(k, v) { this.attrs[k] = String(v); if (k === "value") this.value = String(v); },
    getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
    appendChild(c) { c.parentNode = this; this.children.push(c); if (this.tagName === "SELECT" && c.attrs.selected !== undefined) this.value = c.attrs.value; return c; },
    removeChild(c) { this.children = this.children.filter((x) => x !== c); },
    addEventListener(k, fn) { this.on[k] = fn; },
    get firstChild() { return this.children[0] || null; },
    get textContent() { return this._t + this.children.map((c) => c.textContent).join(""); },
    set textContent(v) { this._t = String(v); this.children = []; } };
  return n;
}
global.document = { createElement: el, createTextNode: (t) => Object.assign(el("#text"), { _t: t }), body: el("body") };
const walk = (n, f, out = []) => { if (f(n)) out.push(n); n.children.forEach((c) => walk(c, f, out)); return out; };
const button = (scope, text) => walk(scope, (n) => n.tagName === "BUTTON" && n.textContent === text)[0];
const settle = () => new Promise((r) => setTimeout(r, 5));
global.GeecsForm = require(process.argv[1]);
const G = require(process.argv[2]);
const C = JSON.parse(require("fs").readFileSync(0, "utf8"));
const seen = [];
let kinds = C.kinds;
global.fetch = (url, init) => {
  const method = (init && init.method) || "GET", body = init && init.body ? JSON.parse(init.body) : undefined;
  seen.push([method, url, body]);
  const key = method + " " + url;
  const [status, out] = key in C.answers ? C.answers[key] : url.endsWith("/api/kinds") ? [200, kinds] : [404, { detail: "unknown " + key }];
  const answer = { status, text: () => Promise.resolve(JSON.stringify(out)) };
  return url === hold.url ? hold.until.then(() => answer) : Promise.resolve(answer);
};
const hold = { url: null, until: null };
(async () => {
  const out = {}, previews = [], saved = [];
  let who = "";
  const opts = { api: C.api + "/", kind: "setup", experiment: "Bench", id: "low_power", fields: ["description"], operator: () => who,
                 preview: (d) => previews.push(d), onSaved: (b) => saved.push(b.outcome) };
  G.open(opts); await settle();
  const drawer = document.body.children[0];
  out.drawer = [drawer.attrs.class, drawer.attrs["data-open"]];
  out.drawn = walk(drawer, (n) => n.attrs["data-path"] !== undefined).map((n) => n.attrs["data-path"]);
  const desc = walk(drawer, (n) => n.attrs["data-path"] === "description")[0];
  const box = walk(desc, (n) => ["INPUT", "TEXTAREA"].includes(n.tagName))[0];
  box.value = "edited"; box.on.input({ target: box });
  button(drawer, "Save").on.click(); await settle();
  out.noOperator = [seen.filter((s) => s[0] !== "GET").length, drawer.textContent.includes("Type your name")];
  who = "Ada";
  button(drawer, "Save").on.click(); await settle();
  out.sent = seen.filter((s) => s[0] !== "GET");
  out.saved = saved;
  button(drawer, "Preview").on.click();
  out.preview = previews[0];
  // the same document again: the drawer as it was, nothing fetched
  const before = seen.length;
  G.open(opts); await settle();
  out.reopenFetched = seen.length - before;
  // a kind with a dry run that refuses: nothing is PUT
  kinds = kinds.map((k) => k.kind === "setup" ? Object.assign({}, k, { dry_run: "a check" }) : k);
  C.answers["POST " + C.api + "/api/dry-run/setup?experiment=Bench&id=low_power.yaml"] = [200, { kind: "setup", by: "scanner", ran: true, refusal: "no such device" }];
  const n = seen.length;
  G.open(Object.assign({}, opts, { fields: null })); await settle();
  button(document.body.children[0], "Save").on.click(); await settle();
  out.refused = [seen.slice(n).filter((s) => s[0] !== "GET").map((s) => s[0]), document.body.children[0].textContent.includes("refused your edit")];
  out.drawers = document.body.children.length;
  // a slow document overtaken by another opened meanwhile: the drawer shows the later one
  let release; hold.until = new Promise((r) => { release = r; });
  hold.url = C.api + "/api/documents/setup/low_power.yaml?experiment=Bench";
  G.open(Object.assign({}, opts, { id: "low_power.yaml", fields: ["name"] })); await settle();
  G.open(Object.assign({}, opts, { id: "far.yaml", fields: ["name"] })); await settle();
  release(); await settle();
  const nameBox = walk(walk(document.body.children[0], (n) => n.attrs["data-path"] === "name")[0], (n) => n.tagName === "INPUT")[0];
  out.race = nameBox.value;
  // a refusal's errors are drawn at their fields
  C.answers["POST " + C.api + "/api/dry-run/setup?experiment=Bench&id=far.yaml"] = [200, { kind: "setup", by: "portal", ran: true, refusal: "name: bad name", errors: [{ loc: ["name"], msg: "a bad name" }] }];
  button(document.body.children[0], "Save").on.click(); await settle();
  out.placed = walk(walk(document.body.children[0], (n) => n.attrs["data-path"] === "name")[0], () => true).some((n) => n._t === "a bad name");
  // a bare name is the kind's one document with that stem, in any folder; two are refused, named
  out.stem = G.byStem(["HTU/a.yaml", "HTU/b.yml"], "b");
  try { G.byStem(["HTU/a.yaml", "HTT/a.yaml"], "a"); out.twice = "accepted"; } catch (e) { out.twice = e.message; }
  // a preview that answers a promise is drawn in the drawer
  G.open(Object.assign({}, opts, { id: "far.yaml", fields: null, preview: () => Promise.resolve(el("img")), toolbar: Object.assign(el("span"), { _t: "host tools" }) })); await settle();
  button(document.body.children[0], "Preview").on.click(); await settle();
  out.previewed = walk(document.body.children[0], (n) => n.tagName === "IMG").length;
  out.toolbar = document.body.children[0].textContent.includes("host tools");
  // a page not on the kit: the drawer is drawn in a .kit wrapper of its own, with its own scrim
  delete require.cache[require.resolve(process.argv[2])];
  document.body = Object.assign(el("body"), { classList: { contains: () => false } });
  require(process.argv[2]).open(opts); await settle();
  const wrap = document.body.children[0];
  out.wrapped = [wrap.attrs.class, wrap.children.map((c) => c.attrs.class), wrap.children[1].attrs["data-scrim"]];
  console.log(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


def _answers(client: TestClient) -> dict:
    """The app's own answers for what the embed reads, keyed as the stub's fetch sees them."""
    out = {}
    for path in ("/api/documents/setup" + Q, "/api/schema/setup", "/api/documents/setup/low_power.yaml" + Q, "/api/documents/setup/far.yaml" + Q):  # fmt: skip
        r = client.get(path)
        assert r.status_code == 200, path
        out[f"GET {API}{path}"] = [200, r.json()]
    doc = client.get("/api/documents/setup/low_power.yaml" + Q).json()["document"]
    out[f"POST {API}/api/dry-run/setup{Q}&id=low_power.yaml"] = [200, client.post("/api/dry-run/setup" + Q + "&id=low_power.yaml", json={"document": doc}).json()]  # fmt: skip
    out[f"PUT {API}/api/documents/setup/low_power.yaml{Q}"] = [200, {"outcome": "committed", "author": "Ada via configurator", "etag": "e2", "tier": "operator", "changes": []}]  # fmt: skip
    return out


@pytest.fixture(scope="module")
def ran(tmp_path_factory: pytest.TempPathFactory) -> dict:
    if not node_available():  # pragma: no cover - CI and dev machines have it
        pytest.skip("node not available")
    from conftest import FILES

    from geecs_configurator.app import create_app
    from geecs_configurator.settings import Settings

    root = tmp_path_factory.mktemp("root")
    for rel, text in FILES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    with TestClient(create_app(Settings(configs_root=root, experiment="Bench"))) as client:  # fmt: skip
        stdin = {"api": API, "answers": _answers(client), "kinds": client.get("/api/kinds").json()}  # fmt: skip
        etag = client.get(f"/api/documents/setup/low_power.yaml{Q}").json()["etag"]
        stored = client.get(f"/api/documents/setup/low_power.yaml{Q}").json()["document"]  # fmt: skip
    done = subprocess.run(["node", "-e", _HARNESS, str(static_dir() / "form.js"), str(_EMBED)], input=json.dumps(stdin), text=True, capture_output=True, timeout=30)  # fmt: skip
    assert done.returncode == 0, done.stderr
    return {**json.loads(done.stdout), "etag": etag, "stored": stored}


def test_the_drawer_draws_only_the_fields_asked(ran: dict) -> None:
    assert ran["drawer"] == ["drawer", "true"]
    assert ran["drawn"] == ["description"]


def test_no_save_without_the_hosts_operator(ran: dict) -> None:
    assert ran["noOperator"] == [0, True]


def test_the_save_sends_the_whole_document_its_etag_and_the_operator(ran: dict) -> None:
    # the dry run first (this app has no consumer check for a setup: by "none", so straight on), then the PUT
    (dry, _, _), (method, url, body) = ran["sent"]
    assert dry == "POST" and (method, url) == (
        "PUT",
        f"{API}/api/documents/setup/low_power.yaml{Q}",
    )
    SaveRequest.model_validate(body)  # the save route's own model
    assert body["etag"] == ran["etag"] and body["operator"] == "Ada"
    assert body["document"] == {
        **ran["stored"],
        "description": "edited",
    }  # the undrawn steps kept as stored
    assert ran["saved"] == ["committed"]


def test_preview_gets_the_document(ran: dict) -> None:
    assert ran["preview"]["description"] == "edited" and ran["preview"]["steps"] == ran["stored"]["steps"]  # fmt: skip


def test_the_same_document_reopens_as_left_in_one_drawer(ran: dict) -> None:
    assert ran["reopenFetched"] == 0 and ran["drawers"] == 1


def test_a_dry_run_refusal_stops_the_save(ran: dict) -> None:
    assert ran["refused"] == [["POST"], True]


def test_a_slow_document_never_overwrites_the_one_opened_after_it(ran: dict) -> None:
    assert ran["race"] == "far"


def test_a_refusals_errors_are_drawn_at_their_fields(ran: dict) -> None:
    assert ran["placed"] is True


def test_a_bare_name_is_its_stem_in_any_folder_and_never_two(ran: dict) -> None:
    assert ran["stem"] == "HTU/b.yml" and "two documents are named a" in ran["twice"]


def test_a_preview_promise_is_drawn_beside_the_hosts_tools(ran: dict) -> None:
    assert ran["previewed"] == 1 and ran["toolbar"] is True


def test_off_the_kit_the_drawer_brings_its_own_kit_wrapper_and_scrim(ran: dict) -> None:
    assert ran["wrapped"] == ["kit", ["scrim", "drawer"], "geecs-configurator-scrim"]


def test_the_configurators_page_saves_through_the_embeds_one_flow() -> None:
    """One save flow: the page has no save of its own, and loads embed.js before itself."""
    static = Path(__file__).resolve().parents[1] / "geecs_configurator"
    page = (static / "static" / "configurator.js").read_text()
    assert "GeecsConfigurator.saveFlow(" in page and '"PUT"' not in page and "/api/dry-run/" not in page  # fmt: skip
    home = (static / "templates" / "home.html").read_text()
    assert home.index("/static/embed.js") < home.index("/static/configurator.js")
