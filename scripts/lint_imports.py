#!/usr/bin/env python3
"""Check every package's imports against ``.importlinter``, from sources alone.

import-linter builds its graph with grimp, which parses source files and
never executes them, so a package's dependencies need not be installed:
each import package only has to be findable. This puts every top-level
package directory (``<Package>/``, the parent of its import package) on
``sys.path`` and then runs ``lint-imports`` on the root ``.importlinter``.
One command therefore works from the root environment, a bare worktree,
pre-commit's hook venv and CI, and always analyses *this* checkout's
sources, never whatever the active environment happens to have installed.

Usage::

    python scripts/lint_imports.py            # exit 1 on a broken contract
    python scripts/lint_imports.py --verbose  # per-contract detail
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    """Put every package directory on ``sys.path`` and run import-linter."""
    args = sys.argv[1:] if argv is None else argv
    for pyproject in sorted(REPO_ROOT.glob("*/pyproject.toml")):
        sys.path.insert(0, str(pyproject.parent))
    from importlinter.cli import lint_imports

    return lint_imports(
        config_filename=str(REPO_ROOT / ".importlinter"),
        no_cache=True,
        verbose="--verbose" in args,
    )


if __name__ == "__main__":
    sys.exit(main())
