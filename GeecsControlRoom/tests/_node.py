"""Run a page script under node: its exported rules called with JSON arguments, or a harness of its own."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from geecs_web_theme.testing import node_available

STATIC = Path(__file__).resolve().parents[1] / "geecs_control_room/static"
needs_node = pytest.mark.skipif(not node_available(), reason="node is not installed")

_CALL_ALL = """
const R = require(process.argv[1]);
const calls = JSON.parse(require("fs").readFileSync(0, "utf8"));
console.log(JSON.stringify(calls.map(([fn, args]) => R[fn](...args))));
"""


def run(script: str, harness: str, stdin: str = "") -> Any:
    """Run ``harness`` with ``process.argv[1]`` the static ``script``; its stdout, as JSON."""
    out = subprocess.run(
        ["node", "-e", harness, str(STATIC / script)],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout
    return json.loads(out)


def call_all(script: str, calls: list[tuple[str, list[Any]]]) -> list[Any]:
    """Each ``(export, args)`` called once, in one node process; the answers in order."""
    return run(script, _CALL_ALL, json.dumps(calls))
