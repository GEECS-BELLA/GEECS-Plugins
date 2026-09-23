"""Tests for ``scripts/ci_select.py`` — the CI test-leg selector.

The selector decides which package suites CI runs for a given diff. Its
failure mode is silent and expensive: under-selecting means a change ships
with its dependents untested and CI still goes green. These tests therefore
concentrate on the *narrowing* direction — that a foundational change really
does pull its dependents in, and that unrecognised input fails safe.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    """Import ``scripts/ci_select.py`` by path (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location(
        "ci_select", REPO_ROOT / "scripts" / "ci_select.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ci_select"] = module
    spec.loader.exec_module(module)
    return module


ci_select = _load_module()


# --- graph derivation --------------------------------------------------------


def test_graph_is_read_from_the_real_pyprojects() -> None:
    """The dependency graph reflects what the packages actually declare.

    Spot-checks edges that exist in the tree today. If these break, either
    a dependency was genuinely removed (update the test) or the TOML
    parsing regressed (fix the parser) — both worth a failure.
    """
    graph = ci_select.discover_graph()

    assert "GEECS-Core" in graph["GeecsBluesky"]
    assert "GEECS-Data-Utils" in graph["ImageAnalysis"]
    assert "GeecsBluesky" in graph["GeecsScanner"]
    # Leaf/foundational packages depend on nothing in-repo.
    assert graph["GEECS-Data-Utils"] == set()
    assert graph["GeecsWebTheme"] == set()


def test_optional_extra_dependencies_still_count() -> None:
    """An ``optional = true`` path dep widens CI like any other.

    GeecsBluesky's GEECS-Analysis edge is behind the ``optimize`` extra; a
    parser that skipped optional deps would under-select on an
    GEECS-Analysis change, which is exactly the silent failure this guards.
    """
    graph = ci_select.discover_graph()
    assert "GEECS-Analysis" in graph["GeecsBluesky"]
    assert "ImageAnalysis" not in graph["GeecsBluesky"]
    legs, _ = ci_select.classify(["GEECS-Analysis/geecs_analysis/run.py"], graph)
    assert "GeecsBluesky" in legs


# --- the narrowing direction: dependents must be pulled in -------------------


def test_foundational_change_pulls_in_dependents() -> None:
    """A GEECS-Data-Utils edit runs the suites that import it.

    This is the test that bites: a naive selector that ran only the
    touched package's own leg would return {"root"} here and pass CI
    while leaving eight dependent suites unrun.
    """
    graph = ci_select.discover_graph()
    legs, _ = ci_select.classify(["GEECS-Data-Utils/geecs_data_utils/x.py"], graph)

    for dependent in ("GeecsBluesky", "GeecsScanner", "GEECS-DataPortal", "GEECS-MCP"):
        assert dependent in legs, (
            f"{dependent} depends on GEECS-Data-Utils but was not selected"
        )
    # ImageAnalysis/ScanAnalysis are dependents too; they live in the root leg.
    assert ci_select.ROOT_LEG in legs


def test_transitive_dependents_are_reached() -> None:
    """GEECS-Core → GeecsBluesky → GeecsScanner is followed all the way.

    GeecsScanner does not depend on GEECS-Core through ImageAnalysis or any
    single hop it declares first; only the transitive walk reaches it.
    """
    graph = ci_select.discover_graph()
    legs, _ = ci_select.classify(["GEECS-Core/geecs_core/transport/x.py"], graph)

    assert "GeecsBluesky" in legs  # direct dependent
    assert "GeecsScanner" in legs  # via GeecsBluesky
    assert "GEECS-MCP" in legs  # via GeecsBluesky


def test_leaf_package_selects_only_itself() -> None:
    """Nothing imports GeecsPvaGateway, so nothing else needs to run."""
    graph = ci_select.discover_graph()
    legs, _ = ci_select.classify(["GeecsPvaGateway/geecs_pva_gateway/server.py"], graph)

    assert legs == {"GeecsPvaGateway"}


# --- fail-safe behaviour -----------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/unit-tests.yml",
        "scripts/ci_select.py",
        "scripts/check.sh",
        "pyproject.toml",
        "poetry.lock",
        ".pre-commit-config.yaml",
    ],
)
def test_infrastructure_changes_select_every_leg(path: str) -> None:
    """A change to CI itself cannot be trusted to narrow CI."""
    graph = ci_select.discover_graph()
    legs, _ = ci_select.classify([path], graph)

    assert legs == set(ci_select.all_legs())


def test_unrecognised_top_level_path_fails_safe() -> None:
    """A new top-level directory runs everything rather than nothing.

    Without this, adding a package and forgetting to register it would
    make CI silently skip it — a green check that means nothing.
    """
    graph = ci_select.discover_graph()
    legs, reasons = ci_select.classify(["SomeBrandNewPackage/mod.py"], graph)

    assert legs == set(ci_select.all_legs())
    assert any("fail-safe" in reason for reason in reasons)


def test_every_package_directory_is_classifiable() -> None:
    """No package in the tree lands in the fail-safe branch.

    Catches the reverse drift: a package added to the repo but never added
    to this script would silently widen every PR to the full matrix.
    """
    graph = ci_select.discover_graph()
    known = (
        set(graph)
        | ci_select.ROOT_ENV_PKGS
        | set(ci_select.OWN_ENV_PKGS)
        | ci_select.NO_SUITE_PKGS
    )
    for pyproject in REPO_ROOT.glob("*/pyproject.toml"):
        assert pyproject.parent.name in known, (
            f"{pyproject.parent.name} has a pyproject.toml but ci_select does not "
            "know it — add it to ROOT_ENV_PKGS, OWN_ENV_PKGS or NO_SUITE_PKGS"
        )


# --- narrowing that is safe --------------------------------------------------


def test_docs_only_change_selects_nothing() -> None:
    """The case this whole change exists for."""
    graph = ci_select.discover_graph()
    legs, _ = ci_select.classify(
        ["docs/platform/site_profile.md", "CLAUDE.md", "Planning/notes.md"], graph
    )

    assert legs == set()


# --- paths that look ignorable but are pinned by root tests/ -----------------


@pytest.mark.parametrize(
    "path",
    [
        # tests/test_render_units_sh.py, tests/test_bootstrap_host_sh.py
        "deploy/render_units.sh",
        "deploy/bootstrap_host.sh",
        "deploy/site.env.example",
        # tests/test_skill_frontmatter.py
        ".claude/skills/land/SKILL.md",
        # GEECS-Schemas' published contract artifacts. test_schema_export.py
        # says a docs reorg dropping one must fail CI, not green-skip (#730).
        "docs/geecs_schemas/scan_request.schema.json",
        "docs/geecs_schemas/schema_reference.md",
    ],
)
def test_root_tested_paths_select_the_root_leg(path: str) -> None:
    """These live outside any package but root tests/ validates them.

    Each would otherwise be swallowed by the docs/.claude/deploy ignore
    rules, silently skipping the test written to make it loud.
    """
    graph = ci_select.discover_graph()
    legs, _ = ci_select.classify([path], graph)

    assert ci_select.ROOT_LEG in legs, (
        f"{path} is pinned by root tests/ but was not selected"
    )


def test_package_unit_file_selects_both_its_package_and_root() -> None:
    """A systemd unit is validated by a ROOT test, not its package's suite.

    tests/test_render_units_sh.py walks the per-package unit templates, so
    editing one to reintroduce a hard-coded lab path must run the root leg
    as well as the package's own.
    """
    graph = ci_select.discover_graph()
    legs, _ = ci_select.classify(["GeecsScanner/deploy/geecs-scanner.service"], graph)

    assert ci_select.ROOT_LEG in legs
    assert "GeecsScanner" in legs


def test_template_edit_runs_the_theme_guard() -> None:
    """GeecsWebTheme's tests walk other packages' templates.

    The theme depends on nobody, so the pyproject graph cannot express
    this reverse edge; it is spelled out in the script and pinned here.
    """
    graph = ci_select.discover_graph()
    legs, _ = ci_select.classify(
        ["GEECS-DataPortal/geecs_portal/templates/run.html"], graph
    )

    assert "GeecsWebTheme" in legs
    assert "GEECS-DataPortal" in legs


def test_every_guarded_surface_selects_the_theme_leg() -> None:
    """Every surface the theme guard walks must select the theme leg.

    Read from GeecsWebTheme's own ``_SURFACES`` list, so the two cannot
    drift. This is the test that bites: an earlier draft matched a
    hand-written list of guarded directories and missed all three
    GeecsScanner surfaces, which would have let a literal colour land
    there with the guard never running.

    Parsed rather than imported: the root env has no GeecsWebTheme.
    """
    import ast

    guard = REPO_ROOT / "GeecsWebTheme" / "tests" / "test_no_literal_colours.py"
    tree = ast.parse(guard.read_text(encoding="utf-8"))
    surfaces: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "_SURFACES" for t in node.targets
        ):
            surfaces = [
                n.value
                for n in ast.walk(node.value)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
            ]
    assert surfaces, "could not parse _SURFACES out of the theme guard"

    graph = ci_select.discover_graph()
    for surface in surfaces:
        if surface.startswith("GeecsWebTheme/"):
            continue  # its own package already selects the leg
        legs, _ = ci_select.classify([surface], graph)
        assert "GeecsWebTheme" in legs, (
            f"{surface} is walked by the theme guard but does not select the theme leg"
        )


def test_root_tests_select_the_root_leg() -> None:
    """Repo-level tests/ belongs to the root-env leg."""
    graph = ci_select.discover_graph()
    legs, _ = ci_select.classify(["tests/test_ci_select.py"], graph)

    assert legs == {ci_select.ROOT_LEG}


def test_the_diff_disables_rename_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """``git diff`` must run with ``--no-renames``.

    With rename detection on (git's default), ``--name-only`` prints only a
    rename's DESTINATION. A module moved from GeecsBluesky to GeecsScanner
    would then select only GeecsScanner's leg, and GeecsBluesky's tests —
    which may import what just left — would not run. Verified against real
    git: `git mv PkgA/mod.py PkgB/mod.py` lists only `PkgB/mod.py` without
    the flag, both paths with it.
    """
    calls: list[list[str]] = []

    class _Result:
        stdout = ""

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _Result()

    monkeypatch.setattr(ci_select.subprocess, "run", _fake_run)
    ci_select.changed_files("origin/master", None)

    diff_calls = [c for c in calls if "diff" in c]
    assert diff_calls, "no git diff was issued"
    for cmd in diff_calls:
        assert "--no-renames" in cmd, (
            "git diff must use --no-renames, or a cross-package move selects "
            "only the destination package's leg"
        )


def test_all_legs_are_unique_and_ordered() -> None:
    """The matrix list has no duplicates (a duplicate would run a leg twice)."""
    legs = ci_select.all_legs()
    assert len(legs) == len(set(legs))
    assert legs[0] == ci_select.ROOT_LEG
