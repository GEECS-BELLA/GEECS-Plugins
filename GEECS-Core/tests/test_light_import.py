"""The light-import rule: the stdlib-only modules load neither pydantic nor MySQL."""

from __future__ import annotations

import subprocess
import sys

LIGHT_MODULES = (
    "geecs_core.transport",
    "geecs_core.pv_naming",
    "geecs_core.exceptions",
    "geecs_core.configs_repo",
)
HEAVY_MODULES = ("pydantic", "mysql", "mysql.connector", "geecs_core.db")


def test_light_modules_import_without_pydantic_or_mysql() -> None:
    """A fresh interpreter importing the light modules leaves the heavy ones unloaded."""
    code = (
        "import sys\n"
        + "".join(f"import {name}\n" for name in LIGHT_MODULES)
        + f"print(','.join(m for m in {HEAVY_MODULES!r} if m in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""
