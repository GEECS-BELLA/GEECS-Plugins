"""form.js over every published file-kind schema with real documents: a no-op touch changes nothing, every error shows.

For each file kind, documents (GEECS-Schemas' own validating fixtures, the
synthetic documents below, and the kind's default) are mounted under node,
every enabled control is touched with the text it shows, and the document
must come back equal with an empty diff. Real pydantic error locations for
broken drafts go through ``setErrors`` and every message must be drawn. The
script's tier list must be GEECS-Schemas' own.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from geecs_schemas import KIND_STORAGE, SCHEMA_REGISTRY, TIERS, FileStorage
from geecs_web_theme import static_dir
from geecs_web_theme.testing import node_available
from pydantic import ValidationError

pytestmark = pytest.mark.skipif(not node_available(), reason="node is not installed")

_FIXTURES = Path(__file__).resolve().parents[2] / "GEECS-Schemas/tests/fixtures"
#: GEECS-Schemas' fixture folders, by the kind their files are.
_FIXTURE_KINDS = {"optimizer_configs": "optimizer_config", "setups": "setup", "groups": "analysis_group"}  # fmt: skip

SYNTHETIC = {
    "setup": {
        "name": "high_power",
        "description": "Pumps up, attenuator out.",
        "steps": [
            {"do": "set", "device": "U_AmpPump", "variable": "Current", "value": 38.5},
            {"do": "set", "device": "U_Att", "variable": "Position", "value": "4.0"},
            {"do": "wait", "seconds": 5},
            {
                "do": "check",
                "device": "U_AmpPump",
                "variable": "Current",
                "expected": 38.5,
                "tolerance": 0.2,
            },  # fmt: skip
            {"do": "apply", "setup": "probe_high"},
        ],
        "captured": {
            "by": "Operator",
            "at": "2026-11-07T14:02:00",
            "readbacks": [{"device": "U_AmpPump", "variable": "Current"}],
        },  # fmt: skip
    },
    "archive_policy": {
        "include_setpoints": True,
        "exclude": ["*:connected"],
        "default_sampling_period": 1.0,
    },  # fmt: skip
    "scan_variables": {
        "variables": {
            "jet_z": {"target": "U_ESP_JetXYZ:Position.Axis 3", "kind": "motor"},
            "gas_pressure": {"target": "U_HP_Daq:AnalogOutput.Channel 1"},
            "e_beam_angle_x": {
                "kind": "pseudo",
                "mode": "relative",
                "targets": [
                    {"target": "U_S3H:Current", "forward": "composite_var * 1"}
                ],
            },
        }
    },
    "native_devices": {
        "devices": {
            "UC_ExampleCam": {"kind": "aravis_camera", "trigger_source": "software"},
            "U_ESP_JetXYZ": {"kind": "epics_motor", "axes": ["x", "y", "z"]},
        }
    },
    "trigger_profile": {
        "name": "laser_off",
        "states": {
            "SCAN": [
                {"device": "U_DG645", "variable": "Amplitude.Ch AB", "value": "4.0"}
            ]
        },  # fmt: skip
    },
    "himg_job": {"kind": "himg_compact", "device": "CamA"},
}

#: Broken drafts whose pydantic locations must all be drawn.
BROKEN = {
    "setup": {
        "name": "x",
        "steps": [
            {"do": "set", "device": "A", "variable": "V", "value": [1]},
            {"do": "wait", "seconds": -1, "bogus": 1},
        ],
    },  # fmt: skip
    "archive_policy": {
        "include_setpoints": "maybe",
        "sampling_overrides": [{"pattern": 3}],
    },
    "scan_variables": {
        "variables": {"a": {"kind": "motor"}, "b": {"kind": "pseudo", "targets": "x"}}
    },  # fmt: skip
    "native_devices": {
        "devices": {"UC": {"kind": "aravis_camera", "trigger_source": "nope"}}
    },
    "trigger_profile": {"states": {"SCAN": [{"device": ""}]}},
}

_DOM = r"""
let ids = 0;
function el(tag) {
  const n = { tagName: tag.toUpperCase(), children: [], attrs: {}, on: {}, value: "", parentNode: null, _t: "",
    setAttribute(k, v) { this.attrs[k] = String(v); },
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
const F = require(process.argv[1]);
const cases = JSON.parse(require("fs").readFileSync(0, "utf8"));
console.log(JSON.stringify(cases.map(([S, D, errs]) => {
  if (D === null) D = F.defaultFor(S, S);
  const host = el("div"), form = F.mount(host, { schema: S, value: D });
  if (errs) {
    form.setErrors(errs);
    // each message, by the data-path of the drawn node that shows it ("" for the document's banner)
    const where = (n) => { for (let p = n.parentNode; p; p = p.parentNode) if (p.attrs && p.attrs["data-path"] !== undefined) return p.attrs["data-path"]; return ""; };
    const shown = walk(host, (n) => n.attrs.class === "err" || n.attrs.class === "banner").filter((n) => n.textContent);
    const at = {};
    errs.forEach((e) => { const n = shown.find((x) => x.textContent.indexOf(e.msg) >= 0); if (n) at[e.msg] = where(n); });
    return at;
  }
  const controls = walk(host, (n) => ["INPUT", "SELECT", "TEXTAREA"].includes(n.tagName) && n.attrs.disabled === undefined && n.attrs["aria-label"] !== "new key");
  controls.forEach((c) => { c.value = c.tagName === "TEXTAREA" ? c.textContent : c.value; (c.on.input || c.on.change)({ target: c }); });
  return [controls.length, F.equal(form.value(), D), F.diff(D, form.value(), S).changes.map((c) => c.path.join("."))];
})));
"""


def _node(cases: list) -> list:
    done = subprocess.run(["node", "-e", _DOM, str(static_dir() / "form.js")], input=json.dumps(cases), text=True, capture_output=True, timeout=60)  # fmt: skip
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _documents() -> list[tuple[str, str, dict | None]]:
    docs: list[tuple[str, str, dict | None]] = []
    for folder, kind in _FIXTURE_KINDS.items():
        for path in sorted(_FIXTURES.rglob(f"{folder}/*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            try:
                SCHEMA_REGISTRY[kind].model_validate(raw)
            except ValidationError:
                continue
            docs.append((kind, path.name, raw))
    for kind, raw in SYNTHETIC.items():
        SCHEMA_REGISTRY[kind].model_validate(
            raw
        )  # the synthetic documents are real ones
        docs.append((kind, "synthetic", raw))
    files = [k for k, s in KIND_STORAGE.items() if isinstance(s, FileStorage)]
    return docs + [(k, "default", None) for k in files]


def test_a_no_op_touch_of_every_control_changes_no_document() -> None:
    docs = _documents()
    got = _node(
        [[SCHEMA_REGISTRY[k].model_json_schema(), raw, None] for k, _, raw in docs]
    )
    bad = {f"{k}:{name}": r for (k, name, _), r in zip(docs, got) if not (r[1] and r[2] == [])}  # fmt: skip
    assert bad == {}
    assert sum(r[0] for r in got) > 100  # the touch reached the controls
    assert {k for k, name, _ in docs if name != "default"} >= {"setup", "scan_variables", "optimizer_config", "native_devices"}  # fmt: skip


def _drawn_at(
    kind: str, raw: dict, extra: list[dict] | None = None
) -> tuple[list[dict], dict]:
    try:
        SCHEMA_REGISTRY[kind].model_validate(raw)
        errs = []
    except ValidationError as exc:
        errs = [{"loc": list(e["loc"]), "msg": f"{e['msg']} #{i}"} for i, e in enumerate(exc.errors())]  # fmt: skip
    errs += extra or []
    return errs, _node([[SCHEMA_REGISTRY[kind].model_json_schema(), raw, errs]])[0]


def test_every_pydantic_error_is_drawn_unless_it_names_an_undrawn_variant() -> None:
    total = 0
    for kind, raw in BROKEN.items():
        errs, at = _drawn_at(kind, raw)
        missing = [".".join(map(str, e["loc"])) for e in errs if e["msg"] not in at]
        # dropped only when the variable's drawn variant (ScanVariable) has errors of its own
        assert [m for m in missing if ".PseudoScanVariable." not in m] == [], kind
        total += len(errs)
    assert total >= 10


def test_errors_land_at_their_drawn_node() -> None:
    errs, at = _drawn_at(
        "setup", BROKEN["setup"], [{"loc": [], "msg": "the whole document"}]
    )
    by_loc = {".".join(map(str, e["loc"])): at.get(e["msg"]) for e in errs}
    assert (
        by_loc["steps.0.set.value.str"] == "steps.0.value"
    )  # a union-of-scalars member's suffix
    assert by_loc["steps.1.wait.seconds"] == "steps.1.seconds"  # a variant tag skipped
    assert (
        by_loc["steps.1.wait.bogus"] == "steps.1.bogus"
    )  # an undeclared key is drawn, so its error is there
    assert by_loc[""] == ""  # the document's own banner
    # a list item's own error (a model validator's location) shows on the item's panel
    errs, at = _drawn_at("archive_policy", BROKEN["archive_policy"], [{"loc": ["sampling_overrides", 0], "msg": "the override as a whole"}])  # fmt: skip
    by_loc = {".".join(map(str, e["loc"])): at.get(e["msg"]) for e in errs}
    assert by_loc["sampling_overrides.0"] == "sampling_overrides.0"
    assert by_loc["sampling_overrides.0.match"] == "sampling_overrides.0.match"
    # an undrawn variant's class name: alone (the drawn variant has none) it goes to the union's banner
    pseudo = {"variables": {"a": {"kind": "pseudo", "mode": "relative", "targets": [{"target": "U_S3H:Current", "forward": "a"}]}}}  # fmt: skip
    _, at = _drawn_at("scan_variables", pseudo, [{"loc": ["variables", "a", "ScanVariable", "target"], "msg": "a motor needs a target"}])  # fmt: skip
    assert at == {"a motor needs a target": "variables.a"}


_SWITCH = r"""
const F = require(process.argv[1]);
const [S, D] = JSON.parse(require("fs").readFileSync(0, "utf8"));
const U = S.properties.variables.additionalProperties;
const tag = (doc) => F.tree(S, doc).children.find((c) => c.key === "variables").items[0].choice.tag;
const there = F.choose(D, ["variables", "v"], U, "PseudoScanVariable", S), back = F.choose(there, ["variables", "v"], U, "ScanVariable", S);
console.log(JSON.stringify([[there.variables.v, tag(there)], [back.variables.v, tag(back)]]));
"""


def test_a_scan_variable_switches_variant_and_back_without_a_hybrid() -> None:
    schema = SCHEMA_REGISTRY["scan_variables"].model_json_schema()
    doc = {"variables": {"v": {"target": "U_S1H:Current", "kind": "motor"}}}
    done = subprocess.run(["node", "-e", _SWITCH, str(static_dir() / "form.js")], input=json.dumps([schema, doc]), text=True, capture_output=True, timeout=30)  # fmt: skip
    (pseudo, pseudo_tag), (motor, motor_tag) = json.loads(done.stdout)
    assert (pseudo["kind"], pseudo_tag, "target" in pseudo) == (
        "pseudo",
        "PseudoScanVariable",
        False,
    )
    assert motor_tag == "ScanVariable" and motor.get("kind") != "pseudo"
    assert not {"targets", "mode"} & set(
        motor
    )  # nothing of the pseudo variant carried back


def test_the_script_knows_the_schemas_tiers() -> None:
    done = subprocess.run(["node", "-e", "console.log(JSON.stringify(require(process.argv[1]).TIERS))", str(static_dir() / "form.js")], text=True, capture_output=True, timeout=30)  # fmt: skip
    assert json.loads(done.stdout) == list(TIERS)
