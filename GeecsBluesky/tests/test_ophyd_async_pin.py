"""The declared ophyd-async is the release this package is tested and deployed on (#1142 review).

A release the declaration admits but the suite never ran can break the
worker: 0.21 replaces ``MotorMoveLogic``, which the native motor extends,
and ``geecs_bluesky.namespace`` imports that module whatever the
experiment's devices.  So the declaration and the lock move together.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


def test_the_declared_ophyd_async_is_exactly_the_locked_release() -> None:
    declared = tomllib.loads((PACKAGE / "pyproject.toml").read_text())["tool"][
        "poetry"
    ]["dependencies"]["ophyd-async"]
    lock = tomllib.loads((PACKAGE / "poetry.lock").read_text())
    (locked,) = [p["version"] for p in lock["package"] if p["name"] == "ophyd-async"]
    assert declared == f"=={locked}"
