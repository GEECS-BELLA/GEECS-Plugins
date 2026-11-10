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


# ---------------------------------------------------------------------------
# The shapes behind review round 1 on #1311: an undiscriminated union, a
# union with a structured member, an open object, a patternProperties map,
# optional and nullable scalars, a maintainer field inside list items and
# inside one variant, an unknown tier.

SHAPES = {
    "type": "object",
    "tier": "operator",
    "$defs": {
        "Real": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {"enum": ["real"], "type": "string", "default": "real"},
                "target": {"type": "string"},
                "word": {"anyOf": [{"type": "string"}, {"type": "number"}]},
            },
            "required": ["target"],
        },
        "Pseudo": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {"enum": ["pseudo"], "type": "string"},
                "expr": {"type": "string"},
                "owner": {"type": "string", "tier": "maintainer"},
            },
            "required": ["kind", "expr"],
        },
        "Grant": {
            "type": "object",
            "properties": {
                "who": {"type": "string"},
                "scope": {"type": "string", "tier": "maintainer"},
            },  # fmt: skip
            "required": ["who"],
        },
        "Step": {
            "type": "object",
            "properties": {"step": {"type": "string"}},
            "additionalProperties": True,
        },
    },
    "properties": {
        "variables": {
            "type": "object",
            "additionalProperties": {
                "anyOf": [{"$ref": "#/$defs/Real"}, {"$ref": "#/$defs/Pseudo"}]
            },
        },  # fmt: skip
        "range": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "string"},
            ]
        },  # fmt: skip
        "steps": {"type": "array", "items": {"$ref": "#/$defs/Step"}},
        "roles": {
            "type": "object",
            "patternProperties": {"^[a-z]+$": {"type": "string"}},
        },  # fmt: skip
        "note": {"type": "string"},
        "trigger": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "count": {"type": "integer"},
        "grants": {"type": "array", "items": {"$ref": "#/$defs/Grant"}},
        "odd": {"type": "string", "tier": "wizard"},
        "rows": {
            "type": "array",
            "items": {
                "anyOf": [
                    {"type": "array", "items": {"type": "number"}},
                    {"type": "string"},
                ]
            },
        },  # fmt: skip
    },
}

SHAPES_DOC = {
    "variables": {
        "a": {"kind": "real", "target": "U_S1H:Current", "word": "4.0"},
        "b": {"kind": "pseudo", "expr": "a * 2", "owner": "ops"},
    },
    "range": [3, 6],
    "steps": [{"step": "roi", "x": 1, "corners": [1, 2], "label": "1"}],
    "roles": {"cam": "camera"},
    "trigger": None,
    "grants": [{"who": "op", "scope": "all"}],
    "rows": [[1], [2]],
}

_SHAPES = r"""
const F = require(process.argv[1]);
const [S, D] = JSON.parse(require("fs").readFileSync(0, "utf8"));
const t = F.tree(S, D);
const by = (n, k) => n.children.find((c) => String(c.key) === k);
const vars = by(t, "variables");
const step0 = by(t, "steps").items[0];
const errs = [
  { loc: ["variables", "a", "Real", "target"], msg: "missing" },
  { loc: ["variables", "b", "Pseudo", "bogus"], msg: "extra" },
  { loc: ["steps", 0, "corners", 1], msg: "deep" },
  { loc: [], msg: "whole" },
];
console.log(JSON.stringify({
  picked: vars.items.map((n) => [n.choice.prop, n.choice.tag, n.children.map((c) => c.key)]),
  range: by(t, "range").kind, roles: [by(t, "roles").kind, by(t, "roles").items.map((n) => n.kind)],
  stepKeys: step0.children.map((c) => [c.key, c.kind, c.extra]),
  placed: F.placeErrors(t, errs),
  refused: F.addItem(D, ["roles"], { type: "string" }, S, "cam") === D,
  addedKey: F.addItem(D, ["roles"], { type: "string" }, S, "dog").roles,
  unknown: [F.strictest("operator", "wizard"), F.strictest("wizard", "code"), F.diff(D, F.setAt(D, ["odd"], "x"), S).tier],
  tiers: {
    removeItem: F.diff(D, F.removeAt(D, ["grants", 0]), S).changes.map((c) => [c.path, c.op, c.tier]),
    addItem: F.diff(D, F.addItem(D, ["grants"], S.$defs.Grant, S), S).tier,
    switchVariant: F.diff(D, F.setAt(D, ["variables", "a"], { kind: "pseudo", expr: "1" }), S).changes.map((c) => [c.path, c.op, c.tier]),
    insideVariant: F.diff(D, F.setAt(D, ["variables", "a", "target"], "U_S2V:Current"), S).changes.map((c) => [c.path, c.tier]),
  },
}));
"""


@pytest.fixture(scope="module")
def shapes() -> dict:
    return _node(_SHAPES, json.dumps([SHAPES, SHAPES_DOC]))


def test_an_undiscriminated_union_is_picked_by_its_structure(shapes: dict) -> None:
    # no discriminator and no const: the enum value and the closed key set pick the variant
    assert shapes["picked"] == [[None, "Real", ["kind", "target", "word"]], [None, "Pseudo", ["kind", "expr", "owner"]]]  # fmt: skip


def test_a_union_with_a_structured_member_is_json_and_maps_and_open_objects_show_every_key(shapes: dict) -> None:  # fmt: skip
    assert shapes["range"] == "json"
    assert shapes["roles"] == ["map", ["string"]]  # one patternProperties is a map
    # an open object's undeclared keys are drawn, never hidden
    assert shapes["stepKeys"] == [["step", "string", False], ["x", "any", True], ["corners", "any", True], ["label", "any", True]]  # fmt: skip


def test_errors_walk_to_the_nearest_drawn_node(shapes: dict) -> None:
    assert shapes["placed"] == {
        "variables.a.target": "missing",  # the variant's class name skipped
        "variables.b": "extra",  # a key never drawn: its nearest drawn node
        "steps.0.corners": "deep",  # below a JSON value: the value
        "": "whole",
    }


def test_an_existing_map_key_is_refused(shapes: dict) -> None:
    assert shapes["refused"] is True
    assert shapes["addedKey"] == {"cam": "camera", "dog": ""}


def test_an_unknown_tier_ranks_strictest(shapes: dict) -> None:
    assert shapes["unknown"] == ["wizard", "wizard", "wizard"]


def test_a_whole_subtree_takes_the_strictest_tier_inside_it(shapes: dict) -> None:
    t = shapes["tiers"]
    assert t["removeItem"] == [[["grants", 0], "remove", "maintainer"]]
    assert t["addItem"] == "maintainer"
    assert t["switchVariant"] == [
        [["variables", "a"], "change", "maintainer"]
    ]  # Pseudo has a maintainer field
    assert t["insideVariant"] == [[["variables", "a", "target"], "operator"]]


_TOUCH = (
    _DOM
    + r"""
const F = require(process.argv[1]);
const [S, D] = JSON.parse(require("fs").readFileSync(0, "utf8"));
const fire = (c, text) => { c.value = text; (c.on.input || c.on.change)({ target: c }); };
const host = el("div"), form = F.mount(host, { schema: S, value: D });
const controls = () => walk(host, (n) => ["INPUT", "SELECT", "TEXTAREA"].includes(n.tagName) && n.attrs.disabled === undefined && n.attrs["aria-label"] !== "new key");
// every enabled control touched with the text it shows: nothing changes
controls().forEach((c) => fire(c, c.tagName === "TEXTAREA" ? c.textContent : c.value));
const out = { touched: controls().length, noop: [F.equal(form.value(), D), F.diff(D, form.value(), S).changes] };
// type, then erase: the stored state comes back (absent stays absent, null stays null)
["note", "trigger", "count"].forEach((p) => { const c = control(at(host, p)); fire(c, "7"); fire(c, ""); });
out.erased = [form.value(), F.diff(D, form.value(), S).changes];
// bad JSON: the field is invalid, the document keeps its last good value
const range = control(at(host, "range"));
fire(range, "[3, 6, ");
out.badJson = [form.invalid(), form.value().range, range.attrs["aria-invalid"]];
fire(range, "[3, 7]");
out.goodJson = [form.invalid(), form.value().range];
// errors draw: a banner at the top and on an item panel
form.setErrors([{ loc: [], msg: "whole" }, { loc: ["grants", 0], msg: "bad grant" }]);
out.banners = walk(host, (n) => n.attrs.class === "banner").map((n) => n.textContent);
// an existing key in "new key" is refused at the control
const keyIn = walk(at(host, "roles"), (n) => n.attrs["aria-label"] === "new key")[0];
keyIn.value = "cam"; button(at(host, "roles"), "Add").on.click();
out.keyRefused = [keyIn.attrs["aria-invalid"], Object.keys(form.value().roles)];
console.log(JSON.stringify(out));
"""
)


@pytest.fixture(scope="module")
def touched() -> dict:
    return _node(_TOUCH, json.dumps([SHAPES, SHAPES_DOC]))


def test_a_no_op_touch_of_every_control_changes_nothing(touched: dict) -> None:
    assert touched["touched"] > 10
    assert touched["noop"] == [
        True,
        [],
    ]  # the "4.0" word, [3, 6] and the step parameters survive


def test_an_emptied_optional_returns_to_the_stored_state(touched: dict) -> None:
    value, changes = touched["erased"]
    assert changes == [] and value["trigger"] is None and "note" not in value and "count" not in value  # fmt: skip


def test_json_that_does_not_parse_is_invalid_and_keeps_the_last_good_value(touched: dict) -> None:  # fmt: skip
    assert touched["badJson"] == [["range"], [3, 6], "true"]
    assert touched["goodJson"] == [[], [3, 7]]


def test_errors_show_as_banners_and_existing_keys_are_refused(touched: dict) -> None:
    assert touched["banners"] == ["whole", "bad grant"]
    assert touched["keyRefused"] == ["true", ["cam"]]


_ROUND2 = (
    _DOM
    + r"""
const F = require(process.argv[1]);
const [S, D] = JSON.parse(require("fs").readFileSync(0, "utf8"));
const U = S.properties.variables.additionalProperties;
const pickedTag = (doc, key) => F.tree(S, doc).children.find((c) => c.key === "variables").items.find((n) => n.key === key).choice.tag;
const toPseudo = F.choose(D, ["variables", "a"], U, "Pseudo", S), toReal = F.choose(D, ["variables", "b"], U, "Real", S);
const out = {
  toPseudo: [toPseudo.variables.a, pickedTag(toPseudo, "a")],
  toReal: [toReal.variables.b, pickedTag(toReal, "b")],
  tiers: {
    switchToPseudo: F.diff(D, toPseudo, S).tier, switchToReal: F.diff(D, toReal, S).tier,
    addReal: F.diff(D, F.setAt(D, ["variables", "c"], { kind: "real", target: "X" }), S).tier,
    addPseudo: F.diff(D, F.setAt(D, ["variables", "c"], { kind: "pseudo", expr: "1" }), S).tier,
  },
  otherAlone: F.placeErrors(F.tree(S, D), [{ loc: ["variables", "a", "Pseudo", "expr"], msg: "pseudo wants expr" }]),
  otherDropped: F.placeErrors(F.tree(S, D), [{ loc: ["variables", "a", "Pseudo", "expr"], msg: "pseudo wants expr" }, { loc: ["variables", "a", "Real", "target"], msg: "bad target" }]),
};
const host = el("div"), form = F.mount(host, { schema: S, value: D });
const fire = (c, text) => { c.value = text; c.textContent = text; c.on.input({ target: c }); };
fire(control(at(host, "range")), "[1, ");
button(at(host, "grants"), "Add").on.click();  // a redraw elsewhere
const range = control(at(host, "range"));
out.kept = [range.textContent, range.attrs["aria-invalid"], form.invalid()];
fire(control(at(host, "rows.0")), "[oops");
out.beforeRemove = form.invalid().sort();
button(at(host, "rows.0"), "Remove").on.click();  // rows.0 is gone: its unparsed text with it
out.afterRemove = [form.invalid(), form.value().rows];
console.log(JSON.stringify(out));
"""
)


@pytest.fixture(scope="module")
def round2() -> dict:
    return _node(_ROUND2, json.dumps([SHAPES, SHAPES_DOC]))


def test_a_variant_switch_fits_the_chosen_variant_never_a_hybrid(round2: dict) -> None:
    assert round2["toPseudo"] == [
        {"kind": "pseudo", "expr": ""},
        "Pseudo",
    ]  # kind: "real" not carried
    assert round2["toReal"] == [
        {"kind": "real", "target": ""},
        "Real",
    ]  # expr, owner not carried


def test_a_switch_takes_the_two_variants_tiers_an_add_the_fitted_ones(
    round2: dict,
) -> None:
    assert round2["tiers"] == {"switchToPseudo": "maintainer", "switchToReal": "maintainer", "addReal": "operator", "addPseudo": "maintainer"}  # fmt: skip


def test_an_error_for_an_undrawn_variant_goes_to_the_union_or_is_dropped(
    round2: dict,
) -> None:
    assert round2["otherAlone"] == {"variables.a": "pseudo wants expr"}
    assert round2["otherDropped"] == {"variables.a.target": "bad target"}


def test_unparsed_text_survives_a_redraw_and_goes_with_its_field(round2: dict) -> None:
    assert round2["kept"] == ["[1, ", "true", ["range"]]
    assert round2["beforeRemove"] == ["range", "rows.0"]
    assert round2["afterRemove"] == [["range"], [[2]]]
