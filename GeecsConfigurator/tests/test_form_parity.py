"""The theme's form.js over every file kind's real schema, and its diff against the service's.

The page draws with form.js and marks the tier live; the service's diff is
the one a save will trust. Both must say the same thing on the same draft.
"""

from __future__ import annotations

import json
import subprocess

import pytest
from geecs_schemas import KIND_STORAGE, SCHEMA_REGISTRY, FileStorage
from geecs_web_theme import static_dir
from geecs_web_theme.testing import node_available

from geecs_configurator.store import diff_documents

pytestmark = pytest.mark.skipif(not node_available(), reason="node is not installed")

_HARNESS = r"""
const F = require(process.argv[1]);
const cases = JSON.parse(require("fs").readFileSync(0, "utf8"));
console.log(JSON.stringify(cases.map(([s, b, a]) => (b === null ? F.tree(s, F.defaultFor(s, s)).children.length : F.diff(b, a, s)))));
"""

STEPS = [{"do": "set", "device": "A", "variable": "V", "value": 1.5}, {"do": "wait", "seconds": 2}]  # fmt: skip
SCAN_VARS = {
    "variables": {
        "jet_z": {"target": "U_ESP_JetXYZ:Position.Axis 3", "kind": "motor"},
        "pseudo": {
            "kind": "pseudo",
            "targets": [{"target": "U_S3H:Current", "forward": "pseudo * 1"}],
        },  # fmt: skip
    }
}
CASES = [
    ("setup", {"name": "x", "steps": STEPS}, {"name": "x", "steps": STEPS[::-1]}),
    (
        "setup",
        {"name": "x", "steps": STEPS},
        {
            "name": "y",
            "steps": [
                {**STEPS[0], "value": "on"},
                {"do": "check", "device": "A", "variable": "V", "expected": 1},
            ],
        },
    ),  # fmt: skip
    (
        "setup",
        {"name": "x", "steps": STEPS},
        {
            "name": "x",
            "steps": STEPS,
            "captured": {"by": "me", "at": "2026-11-10T00:00:00"},
        },
    ),  # fmt: skip
    (
        "archive_policy",
        {"include_setpoints": True},
        {"include_setpoints": False, "exclude": ["a"], "default_sampling_period": 2.0},
    ),  # fmt: skip
    ("archive_policy", {"exclude": ["a", "b"]}, {"exclude": ["b", "a"]}),
    # type-strict: true is not 1, nested or not
    ("archive_policy", {"include_setpoints": True}, {"include_setpoints": 1}),
    ("archive_policy", {"include_status": False}, {"include_status": 0}),
    (
        "setup",
        {"name": "x", "steps": STEPS},
        {"name": "x", "steps": [STEPS[0], {"do": "wait", "seconds": True}]},
    ),  # fmt: skip
    # an undiscriminated union (ScanVariable | PseudoScanVariable): both walk into the variant
    (
        "scan_variables",
        SCAN_VARS,
        {
            "variables": {
                **SCAN_VARS["variables"],
                "jet_z": {**SCAN_VARS["variables"]["jet_z"], "description": "the jet"},
            }
        },
    ),  # fmt: skip
    (
        "scan_variables",
        SCAN_VARS,
        {"variables": {"jet_z": SCAN_VARS["variables"]["jet_z"]}},
    ),  # fmt: skip
]


def _node(cases: list) -> list:
    done = subprocess.run(["node", "-e", _HARNESS, str(static_dir() / "form.js")], input=json.dumps(cases), text=True, capture_output=True, timeout=30)  # fmt: skip
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_form_js_draws_every_file_kind() -> None:
    files = [k for k, s in KIND_STORAGE.items() if isinstance(s, FileStorage)]
    counts = _node(
        [[SCHEMA_REGISTRY[k].model_json_schema(), None, None] for k in files]
    )
    assert all(n > 0 for n in counts), dict(zip(files, counts))


def test_the_page_and_the_service_diff_alike() -> None:
    js = _node([[SCHEMA_REGISTRY[k].model_json_schema(), b, a] for k, b, a in CASES])
    for (kind, b, a), page in zip(CASES, js):
        service = diff_documents(b, a, SCHEMA_REGISTRY[kind].model_json_schema())
        assert [(c["path"], c["op"], c["tier"]) for c in page["changes"]] == [(c.path, c.op, c.tier) for c in service], kind  # fmt: skip
    assert (
        js[3]["tier"] == "maintainer" and js[4]["tier"] == "maintainer"
    )  # a reordered exclude list is the maintainer's
    assert [c["path"] for c in js[5]["changes"]] == [
        ["include_setpoints"]
    ]  # true -> 1 is a change
    assert [c["path"] for c in js[8]["changes"]] == [
        ["variables", "jet_z", "description"]
    ]
