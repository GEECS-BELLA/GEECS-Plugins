"""form.js — the schema-driven form every GEECS editor draws with.

The real ``form.js`` runs under node: its rules (the tree a form is drawn
from, the edits, the diff with its tier) are called on a schema shaped as
pydantic writes one (``$defs``, a discriminated union, an optional
sub-document, a map, a field stricter than its document), and ``mount`` is
driven through a DOM just big enough for it, the way a person edits.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from geecs_web_theme import static_dir
from geecs_web_theme.testing import js_colour_literals, node_available

_FORM = static_dir() / "form.js"

#: A setup-like document: steps a discriminated union, `captured` an optional
#: sub-document, `limits` a map, `exclude` a maintainer field of an operator document.
SCHEMA = {
    "title": "Doc",
    "type": "object",
    "tier": "operator",
    "$defs": {
        "Set": {
            "type": "object",
            "properties": {
                "do": {"const": "set", "type": "string"},
                "device": {"type": "string", "description": "The device to set."},
                "value": {"anyOf": [{"type": "string"}, {"type": "number"}]},
            },
            "required": ["do", "device", "value"],
        },
        "Wait": {
            "type": "object",
            "properties": {
                "do": {"const": "wait", "type": "string"},
                "seconds": {"type": "number", "minimum": 0},
            },
            "required": ["do", "seconds"],
        },
        "Capture": {
            "type": "object",
            "properties": {"by": {"type": "string"}},
            "required": ["by"],
        },
    },
    "properties": {
        "schema_version": {"type": "integer", "default": 1},
        "name": {"type": "string", "description": "The name."},
        "mode": {"enum": ["fast", "slow"], "type": "string", "default": "fast"},
        "armed": {"type": "boolean", "default": False},
        "steps": {
            "type": "array",
            "items": {
                "discriminator": {"propertyName": "do", "mapping": {}},
                "oneOf": [{"$ref": "#/$defs/Set"}, {"$ref": "#/$defs/Wait"}],
            },
        },
        "captured": {
            "anyOf": [{"$ref": "#/$defs/Capture"}, {"type": "null"}],
            "default": None,
        },
        "limits": {"type": "object", "additionalProperties": {"type": "number"}},
        "exclude": {
            "type": "array",
            "items": {"type": "string"},
            "tier": "maintainer",
        },
    },
    "required": ["name", "steps"],
}

DOC = {
    "name": "low",
    "mode": "fast",
    "steps": [
        {"do": "set", "device": "A", "value": 1.5},
        {"do": "wait", "seconds": 2},
    ],
    "captured": None,
    "limits": {"x": 1},
    "exclude": ["a", "b"],
}

_RULES = r"""
const F = require(process.argv[1]);
const [S, D] = JSON.parse(require("fs").readFileSync(0, "utf8"));
const t = F.tree(S, D);
const by = (n, k) => n.children.find((c) => c.key === k);
const steps = by(t, "steps");
const out = {
  fields: t.children.map((c) => [c.key, c.kind, c.required, c.tier, c.label]),
  help: by(t, "name").help, options: by(t, "mode").options,
  step0: [steps.items[0].choice, steps.items[0].children.map((c) => [c.key, c.kind, c.help])],
  step1: steps.items[1].children.map((c) => c.key),
  captured: [by(t, "captured").kind, by(t, "captured").unset, by(t, "captured").nullable],
  limits: by(t, "limits").items.map((c) => [c.key, c.kind]),
  added: F.addItem(D, ["steps"], S.properties.steps.items, S).steps[2],
  removed: F.removeAt(D, ["steps", 0]).steps,
  moved: F.moveItem(D, ["steps", 1], -1).steps.map((s) => s.do),
  movedPastEnd: F.moveItem(D, ["steps", 1], +1).steps.map((s) => s.do),
  chosen: F.choose(D, ["steps", 1], S.properties.steps.items, "set", S).steps[1],
  setNew: F.setAt(D, ["limits", "y"], 3).limits,
  unchanged: D.steps.length,
  parsed: [F.parse("number", "2.5"), F.parse("number", "two"), F.parse("number", ""), F.parse("scalar", "3"), F.parse("scalar", "on"), F.parse("boolean", "false"), F.parse("string", "")],
  defaults: F.defaultFor(S, S),
  diffs: {
    none: F.diff(D, JSON.parse(JSON.stringify(D)), S),
    keyOrder: F.diff(D, Object.assign({ exclude: D.exclude }, D), S),
    value: F.diff(D, F.setAt(D, ["steps", 0, "value"], 2), S),
    reorder: F.diff(D, F.moveItem(D, ["exclude", 1], -1), S),
    strict: F.diff(D, F.setAt(F.setAt(D, ["name", ], "x"), ["exclude", 2], "c"), S),
    variant: F.diff(D, F.choose(D, ["steps", 1], S.properties.steps.items, "set", S), S),
    added: F.diff(D, F.setAt(D, ["captured"], { by: "me" }), S),
  },
  errors: F.placeErrors(t, [{ loc: ["steps", 0, "set", "value"], msg: "bad" }, { loc: "name", msg: "short" }, { loc: [], msg: "whole" }]),
};
console.log(JSON.stringify(out));
"""


def _node(harness: str, stdin: str) -> dict:
    if not node_available():  # pragma: no cover - CI and dev machines have it
        pytest.skip("node not available")
    done = subprocess.run(["node", "-e", harness, str(_FORM)], input=stdin, text=True, capture_output=True, timeout=30)  # fmt: skip
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.fixture(scope="module")
def rules() -> dict:
    return _node(_RULES, json.dumps([SCHEMA, DOC]))


def test_the_script_carries_no_literal_colour() -> None:
    assert js_colour_literals(_FORM.read_text()) == []


def test_the_tree_follows_the_schema(rules: dict) -> None:
    fields = {k: rest for k, *rest in rules["fields"]}
    assert "schema_version" not in fields  # tools own it
    assert fields["name"] == ["string", True, None, "Name"]
    assert fields["mode"][0] == "enum" and rules["options"] == ["fast", "slow"]
    assert fields["armed"][0] == "boolean" and fields["steps"][:2] == ["array", True]
    assert fields["limits"][0] == "map" and rules["limits"] == [["x", "number"]]
    assert fields["exclude"] == [
        "array",
        False,
        "maintainer",
        "Exclude",
    ]  # the stricter field is marked
    assert rules["help"] == "The name."  # the description is the help


def test_a_discriminated_union_is_a_choice_then_the_variant(rules: dict) -> None:
    choice, children = rules["step0"]
    assert choice == {"prop": "do", "tags": ["set", "wait"], "tag": "set"}
    assert children == [
        ["device", "string", "The device to set."],
        ["value", "scalar", ""],
    ]  # $ref resolved
    assert rules["step1"] == ["seconds"]


def test_an_optional_sub_document_is_unset_until_set(rules: dict) -> None:
    assert rules["captured"] == ["object", True, True]


def test_edits_return_a_new_document(rules: dict) -> None:
    assert rules["added"] == {"do": "set", "device": "", "value": None}
    assert rules["removed"] == [{"do": "wait", "seconds": 2}]
    assert rules["moved"] == ["wait", "set"] and rules["movedPastEnd"] == [
        "set",
        "wait",
    ]
    assert rules["chosen"] == {"do": "set", "device": "", "value": None}
    assert rules["setNew"] == {"x": 1, "y": 3}
    assert rules["unchanged"] == 2  # the original is never mutated


def test_input_text_parses_to_the_kind(rules: dict) -> None:
    # a word in a number field stays a word (the service names the error); an empty optional is absent (dropped by JSON)
    assert rules["parsed"] == [2.5, "two", None, 3, "on", False, ""]


def test_defaults_fill_the_required_fields(rules: dict) -> None:
    assert rules["defaults"] == {"name": "", "steps": []}


def test_the_diff_and_its_tier(rules: dict) -> None:
    d = rules["diffs"]
    assert d["none"] == {"changes": [], "tier": None}
    assert d["keyOrder"] == {"changes": [], "tier": None}  # key order is not a change
    assert d["value"]["changes"] == [{"path": ["steps", 0, "value"], "op": "change", "before": 1.5, "after": 2, "tier": "operator"}]  # fmt: skip
    # a reordered list is a change at every index it moved, at the list's tier
    assert [(c["path"], c["tier"]) for c in d["reorder"]["changes"]] == [(["exclude", 0], "maintainer"), (["exclude", 1], "maintainer")]  # fmt: skip
    assert d["reorder"]["tier"] == "maintainer"
    # the strictest changed tier decides
    assert [c["tier"] for c in d["strict"]["changes"]] == ["operator", "maintainer"]
    assert d["strict"]["tier"] == "maintainer"
    assert [(c["path"], c["op"]) for c in d["variant"]["changes"]] == [(["steps", 1], "change")]  # fmt: skip
    assert [(c["path"], c["op"]) for c in d["added"]["changes"]] == [(["captured"], "change")]  # fmt: skip


def test_errors_land_on_the_form_paths(rules: dict) -> None:
    # pydantic's location names the variant's tag; the form's path does not
    assert rules["errors"] == {"steps.0.value": "bad", "name": "short", "": "whole"}


_DOM = r"""
let ids = 0;
function el(tag) {
  const n = { __id: ++ids, tagName: tag.toUpperCase(), children: [], attrs: {}, on: {}, value: "", parentNode: null, _t: "",
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
global.document = { createElement: el, createTextNode: (t) => Object.assign(el("#text"), { _t: t }) };
const walk = (n, f, out = []) => { if (f(n)) out.push(n); n.children.forEach((c) => walk(c, f, out)); return out; };
const at = (host, path) => walk(host, (n) => n.attrs["data-path"] === path)[0];
const control = (field) => walk(field, (n) => ["INPUT", "SELECT", "TEXTAREA"].includes(n.tagName))[0];
const button = (scope, text) => walk(scope, (n) => n.tagName === "BUTTON" && n.textContent === text)[0];
"""

_MOUNT = (
    _DOM
    + r"""
const F = require(process.argv[1]);
const [S, D] = JSON.parse(require("fs").readFileSync(0, "utf8"));
const host = el("div"), seen = [];
const form = F.mount(host, { schema: S, value: D, onChange: (d) => seen.push(d) });
const name = at(host, "name");
const out = { kitClasses: [...new Set(walk(host, (n) => n.attrs.class).map((n) => n.attrs.class))].sort() };
out.nameControl = [control(name).tagName, control(name).value, walk(name, (n) => n.attrs.class === "hint")[0].textContent];
out.tierMarks = walk(host, (n) => n.attrs.class === "tier").map((n) => [n.attrs["data-tier"], n.parentNode.parentNode.attrs["data-path"]]);
out.choice = control(at(host, "steps.0")).value;
control(name).value = "high"; control(name).on.input({ target: control(name) });
out.afterType = [form.value().name, at(host, "name").attrs["data-changed"]];
button(at(host, "steps"), "Add").on.click();
out.afterAdd = form.value().steps.length;
button(at(host, "steps.1"), "Up").on.click();
out.afterUp = form.value().steps.map((s) => s.do);
button(at(host, "steps.0"), "Remove").on.click();
out.afterRemove = form.value().steps.map((s) => s.do);
button(at(host, "captured"), "Set").on.click();
out.afterSet = form.value().captured;
out.placed = form.setErrors([{ loc: ["name"], msg: "too short" }]);
out.invalid = [control(at(host, "name")).attrs["aria-invalid"], walk(at(host, "name"), (n) => n.attrs.class === "err")[0].textContent];
out.calls = seen.length;
const ro = el("div"); F.mount(ro, { schema: S, value: D, readOnly: true });
out.readOnly = [walk(ro, (n) => n.tagName === "BUTTON").length, walk(ro, (n) => n.attrs.disabled !== undefined).length > 5];
console.log(JSON.stringify(out));
"""
)


@pytest.fixture(scope="module")
def mounted() -> dict:
    return _node(_MOUNT, json.dumps([SCHEMA, DOC]))


def test_mount_draws_kit_controls(mounted: dict) -> None:
    assert {"field", "fold", "panel", "group", "hint", "err", "tier", "req", "eyebrow"} <= set(mounted["kitClasses"])  # fmt: skip
    assert mounted["nameControl"] == ["INPUT", "low", "The name."]
    assert mounted["tierMarks"] == [["maintainer", "exclude"]]
    assert mounted["choice"] == "set"


def test_editing_hands_the_document_back(mounted: dict) -> None:
    assert mounted["afterType"] == ["high", "true"]  # the field says it changed
    assert mounted["afterAdd"] == 3
    assert mounted["afterUp"] == ["wait", "set", "set"]
    assert mounted["afterRemove"] == ["set", "set"]
    assert mounted["afterSet"] == {"by": ""}
    assert mounted["calls"] == 5


def test_errors_are_drawn_at_the_field(mounted: dict) -> None:
    assert mounted["placed"] == {"name": "too short"}
    assert mounted["invalid"] == ["true", "too short"]


def test_read_only_offers_no_edit(mounted: dict) -> None:
    buttons, disabled = mounted["readOnly"]
    assert buttons == 0 and disabled
