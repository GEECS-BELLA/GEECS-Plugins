#!/usr/bin/env python3
"""Draft a package's CHANGELOG entry from the PRs merged since its last bump.

A package is bumped when it is deployed or tagged, not per PR (root
``CLAUDE.md`` § "Release & Versioning"). This prints the draft block for
that bump: every merged PR that touched ``<Package>/`` since the date the
``version =`` line last changed, one bullet per PR title. Edit, then paste:
    scripts/release_notes.py GEECS-Core [--since 2026-10-01] [--version 0.14.0]
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(*cmd: str) -> str:
    """Run a command at the repo root and return its stdout."""
    return subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def last_bump_date(package: str) -> str:
    """Return the date (YYYY-MM-DD) the package's ``version =`` line last changed."""
    pyproject = f"{package}/pyproject.toml"
    out = _run("git", "log", "-G^version = ", "--format=%cs", "-1", "--", pyproject)
    if not out.strip():
        raise SystemExit(f"no version history for {pyproject}")
    return out.strip()


def merged_prs(since: str) -> list[dict]:
    """Return merged PRs since ``since`` as gh JSON rows (number, title, files)."""
    args = "pr list --state merged --json number,title,files --limit 1000".split()
    return json.loads(_run("gh", *args, "--search", f"merged:>={since}"))


def draft(package: str, prs: list[dict], version: str, today: str) -> str:
    """Render the draft block: the PRs whose files touch ``<package>/``, oldest first."""
    prefix = package.rstrip("/") + "/"
    hits = sorted(
        (
            pr
            for pr in prs
            if any(f["path"].startswith(prefix) for f in pr.get("files") or [])
        ),
        key=lambda pr: pr["number"],
    )
    lines = [f"## [{version}] - {today}", "", "### Changed", ""]
    bullets = [f"- {pr['title']} (#{pr['number']})" for pr in hits]
    lines += bullets or ["- (no merged PRs touched this package)"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Print the draft CHANGELOG block for one package."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("package", help="package directory, e.g. GEECS-Core")
    p.add_argument("--since", help="YYYY-MM-DD; default: the last version bump's date")
    p.add_argument("--version", default="X.Y.Z", help="the new version for the heading")
    args = p.parse_args(argv)
    if not (REPO_ROOT / args.package / "pyproject.toml").is_file():
        print(f"not a package directory: {args.package}", file=sys.stderr)
        return 2
    since = args.since or last_bump_date(args.package)
    today = datetime.date.today().isoformat()
    print(draft(args.package, merged_prs(since), args.version, today), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
