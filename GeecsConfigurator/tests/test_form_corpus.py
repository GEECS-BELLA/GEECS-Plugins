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
_FIXTURE_KINDS = {"optimizer_configs": "optimizer_config", "actions": "action_plan_library", "groups": "analysis_group"}  # fmt: skip

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
    const shown = walk(host, (n) => n.attrs.class === "err" || n.attrs.class === "banner").map((n) => n.textContent).join(" | ");
    return errs.filter((e) => shown.indexOf(e.msg) < 0).map((e) => e.loc.join("."));
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


def test_every_pydantic_error_location_is_drawn() -> None:
    cases, kinds = [], []
    for kind, raw in BROKEN.items():
        with pytest.raises(ValidationError) as exc:
            SCHEMA_REGISTRY[kind].model_validate(raw)
        errs = [{"loc": list(e["loc"]), "msg": f"{e['msg']} #{i}"} for i, e in enumerate(exc.value.errors())]  # fmt: skip
        cases.append([SCHEMA_REGISTRY[kind].model_json_schema(), raw, errs])
        kinds.append((kind, len(errs)))
    missing = _node(cases)
    assert dict(zip([k for k, _ in kinds], missing)) == {k: [] for k, _ in kinds}
    assert sum(n for _, n in kinds) >= 10


def test_the_script_knows_the_schemas_tiers() -> None:
    done = subprocess.run(["node", "-e", "console.log(JSON.stringify(require(process.argv[1]).TIERS))", str(static_dir() / "form.js")], text=True, capture_output=True, timeout=30)  # fmt: skip
    assert json.loads(done.stdout) == list(TIERS)
