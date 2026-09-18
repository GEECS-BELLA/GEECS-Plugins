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

    GeecsBluesky's ImageAnalysis edge is behind the ``optimize`` extra; a
    parser that skipped optional deps would under-select on an
    ImageAnalysis change, which is exactly the silent failure this guards.
    """
    graph = ci_select.discover_graph()
    assert "ImageAnalysis" in graph["GeecsBluesky"]


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


def test_theme_guarded_globs_point_at_real_directories() -> None:
    """The guarded template paths still exist.

    A moved template directory would silently stop selecting the theme
    leg, so the literal prefixes are pinned to the tree.
    """
    for prefix in ci_select.THEME_GUARDED_GLOBS:
        assert (REPO_ROOT / prefix).is_dir(), f"{prefix} no longer exists"


def test_root_tests_select_the_root_leg() -> None:
    """Repo-level tests/ belongs to the root-env leg."""
    graph = ci_select.discover_graph()
    legs, _ = ci_select.classify(["tests/test_ci_select.py"], graph)

    assert legs == {ci_select.ROOT_LEG}


def test_all_legs_are_unique_and_ordered() -> None:
    """The matrix list has no duplicates (a duplicate would run a leg twice)."""
    legs = ci_select.all_legs()
    assert len(legs) == len(set(legs))
    assert legs[0] == ci_select.ROOT_LEG
