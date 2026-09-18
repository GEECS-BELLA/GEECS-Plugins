#!/usr/bin/env python3
"""Work out which CI test legs a change actually needs.

Why this exists
---------------
The unit-tests workflow used to run every package's suite on every push,
which cost ~10 minutes whether the diff touched one leaf package or the
whole tree. This script narrows that to the legs a change can plausibly
break: the packages it touched, plus everything that depends on them.

The dependency graph is **derived from the packages' own
``pyproject.toml`` path dependencies**, never hand-maintained here. A new
intra-repo dependency therefore widens CI the moment it is declared, with
no second place to remember to edit.

Selection is deliberately conservative:

* a touched package pulls in its whole *reverse* closure (dependents, and
  their dependents), because that is the real blast radius — changing
  ``GEECS-Data-Utils`` legitimately needs 10 of the 11 legs;
* anything that could change how CI itself runs (``.github/``,
  ``scripts/``, the root lock/pyproject, the pre-commit config) selects
  **every** leg;
* a path the classifier does not recognise selects every leg, so a new
  top-level directory fails safe rather than silently skipping tests.

Note the contrast with ``scripts/check.sh``, which runs only the suites of
*directly touched* packages. That asymmetry is intentional: check.sh is a
fast local pre-flight, while CI is the backstop and must cover dependents.
The backstop behind *this* script is the workflow's full run on every push
to master — so a mis-selection on a PR surfaces minutes after merge rather
than never.

Usage
-----
    ./scripts/ci_select.py --base origin/master        # diff HEAD against base
    ./scripts/ci_select.py --files -                   # read paths on stdin
    ./scripts/ci_select.py --all                       # every leg
    ./scripts/ci_select.py --base origin/master --explain

Writes a human-readable plan to stderr and, with ``--github``, appends
``legs=<json>`` / ``any=<bool>`` to ``$GITHUB_OUTPUT`` for the matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Packages whose suites run from the ROOT poetry env, all in one CI leg
# named "root" (which also runs the repo-level tests/ directory). Keep in
# sync with scripts/check.sh's ROOT_ENV_PKGS.
ROOT_ENV_PKGS = frozenset(
    {"ImageAnalysis", "ScanAnalysis", "GEECS-Data-Utils", "GEECS-Schemas"}
)
ROOT_LEG = "root"

# Packages that install and run from their own directory — one CI leg each.
# Keep in sync with scripts/check.sh's OWN_ENV_PKGS.
OWN_ENV_PKGS = (
    "GEECS-Core",
    "GEECS-DataPortal",
    "GEECS-LogTriage",
    "GEECS-MCP",
    "GeecsBluesky",
    "GeecsCAGateway",
    "GeecsLogbook",
    "GeecsPvaGateway",
    "GeecsScanner",
    "GeecsWebTheme",
)

# Packages that are real dependencies but ship no CI suite of their own.
# They select their dependents' legs and nothing else.
NO_SUITE_PKGS = frozenset({"LogMaker4GoogleDocs"})

# Paths that can change how CI itself runs, or that sit outside any
# package: select everything.
INFRA_PREFIXES = (".github/", "scripts/")
INFRA_FILES = frozenset({"pyproject.toml", "poetry.lock", ".pre-commit-config.yaml"})

# Documentation, planning and agent context: no suite can observe these.
IGNORED_PREFIXES = ("docs/", "Planning/", ".claude/", "extras/", "deploy/")
IGNORED_SUFFIXES = (".md",)
IGNORED_FILES = frozenset({".gitignore", "LICENSE", "AGENTS.md", "mkdocs.yml"})

# Undeclared coupling the pyproject graph cannot express: GeecsWebTheme's
# tests walk the OTHER packages' templates and stylesheets to enforce the
# no-literal-colour and .kit-scoping rules, so a template edited anywhere
# must run the theme leg. This is a reverse edge (theme depends on nobody),
# which is exactly why it has to be spelled out. Mirrors the same rule in
# scripts/check.sh.
THEME_GUARDED_GLOBS = (
    "GEECS-DataPortal/geecs_portal/templates/",
    "GeecsLogbook/geecs_logbook/static/",
    "GeecsLogbook/geecs_logbook/templates/",
    "ScanAnalysis/scan_analysis/config_editor/static/",
    "ScanAnalysis/scan_analysis/config_editor/templates/",
)


def discover_graph() -> dict[str, set[str]]:
    """Map each package to the intra-repo packages it depends on.

    Reads every ``<Package>/pyproject.toml`` and resolves Poetry path
    dependencies (``{ path = "../Other" }``) across all dependency groups
    and extras, so an optional dependency still widens CI.
    """
    graph: dict[str, set[str]] = {}
    for pyproject in sorted(REPO_ROOT.glob("*/pyproject.toml")):
        pkg = pyproject.parent.name
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:  # pragma: no cover
            print(f"ci_select: cannot read {pyproject}: {exc}", file=sys.stderr)
            graph.setdefault(pkg, set())
            continue
        graph[pkg] = _path_deps(data)
    return graph


def _path_deps(data: dict) -> set[str]:
    """Collect sibling-directory names from every dependency table."""
    poetry = data.get("tool", {}).get("poetry", {})
    tables: list[dict] = [poetry.get("dependencies", {})]
    for group in poetry.get("group", {}).values():
        tables.append(group.get("dependencies", {}))

    found: set[str] = set()
    for table in tables:
        if not isinstance(table, dict):
            continue
        for spec in table.values():
            if isinstance(spec, dict):
                path = spec.get("path")
                if isinstance(path, str) and path.startswith("../"):
                    found.add(path[len("../") :].strip("/"))
    return found


def dependents_closure(graph: dict[str, set[str]], pkg: str) -> set[str]:
    """Return *pkg* plus every package that transitively depends on it."""
    out = {pkg}
    frontier = [pkg]
    while frontier:
        node = frontier.pop()
        for candidate, deps in graph.items():
            if node in deps and candidate not in out:
                out.add(candidate)
                frontier.append(candidate)
    return out


def all_legs() -> list[str]:
    """Every CI leg, in a stable order."""
    return [ROOT_LEG, *OWN_ENV_PKGS]


def legs_for_packages(packages: set[str]) -> set[str]:
    """Map a set of affected packages onto CI legs."""
    legs: set[str] = set()
    for pkg in packages:
        if pkg in ROOT_ENV_PKGS or pkg in NO_SUITE_PKGS:
            legs.add(ROOT_LEG)
        if pkg in OWN_ENV_PKGS:
            legs.add(pkg)
    return legs


def classify(
    changed: list[str], graph: dict[str, set[str]]
) -> tuple[set[str], list[str]]:
    """Turn changed paths into the legs CI must run, plus the reasons why."""
    known = set(graph) | ROOT_ENV_PKGS | set(OWN_ENV_PKGS) | NO_SUITE_PKGS
    legs: set[str] = set()
    reasons: list[str] = []
    touched: set[str] = set()

    for path in changed:
        if not path:
            continue

        if path in INFRA_FILES or path.startswith(INFRA_PREFIXES):
            reasons.append(f"{path}: CI infrastructure — every leg")
            return set(all_legs()), reasons

        if (
            path in IGNORED_FILES
            or path.startswith(IGNORED_PREFIXES)
            or path.endswith(IGNORED_SUFFIXES)
        ):
            continue

        if path.startswith(THEME_GUARDED_GLOBS):
            legs.add("GeecsWebTheme")
            reasons.append(f"{path}: template/stylesheet — theme guard leg")

        top = path.split("/", 1)[0]

        if top == "tests":
            legs.add(ROOT_LEG)
            reasons.append(f"{path}: repo-level tests — root leg")
            continue

        if top in known:
            touched.add(top)
            continue

        # An unrecognised top-level path: fail safe rather than skip.
        reasons.append(f"{path}: unrecognised path — every leg (fail-safe)")
        return set(all_legs()), reasons

    for pkg in sorted(touched):
        affected = dependents_closure(graph, pkg)
        legs |= legs_for_packages(affected)
        others = sorted(affected - {pkg})
        detail = f" → dependents: {', '.join(others)}" if others else " (leaf)"
        reasons.append(f"{pkg} changed{detail}")

    return legs, reasons


def changed_files(base: str | None, files: str | None) -> list[str]:
    """Collect the changed-path list from a git ref or from stdin."""
    if files is not None:
        source = sys.stdin.read() if files == "-" else Path(files).read_text()
        return [line.strip() for line in source.splitlines() if line.strip()]

    merge_base = subprocess.run(
        ["git", "merge-base", "HEAD", base],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--name-only", merge_base],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line.strip() for line in diff.splitlines() if line.strip()]


def main() -> int:
    """Entry point: print the selected legs, one per line, on stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base", help="git ref to diff HEAD against")
    source.add_argument("--files", help="file with changed paths, or '-' for stdin")
    source.add_argument("--all", action="store_true", help="select every leg")
    parser.add_argument("--explain", action="store_true", help="print why each leg ran")
    parser.add_argument("--github", action="store_true", help="write $GITHUB_OUTPUT")
    args = parser.parse_args()

    graph = discover_graph()

    if args.all:
        legs, reasons = set(all_legs()), ["--all requested"]
    else:
        legs, reasons = classify(changed_files(args.base, args.files), graph)

    ordered = [leg for leg in all_legs() if leg in legs]

    print("== ci_select plan", file=sys.stderr)
    if args.explain:
        for reason in reasons:
            print(f"   {reason}", file=sys.stderr)
    print(
        f"   legs: {' '.join(ordered) if ordered else '(none — nothing testable changed)'}"
        f"  [{len(ordered)}/{len(all_legs())}]",
        file=sys.stderr,
    )

    if args.github and (out := os.environ.get("GITHUB_OUTPUT")):
        with open(out, "a", encoding="utf-8") as handle:
            handle.write(f"legs={json.dumps(ordered)}\n")
            handle.write(f"any={'true' if ordered else 'false'}\n")

    for leg in ordered:
        print(leg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
