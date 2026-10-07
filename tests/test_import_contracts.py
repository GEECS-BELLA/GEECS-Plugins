"""Pin ``.importlinter`` to the packages on disk.

import-linter analyses only the ``root_packages`` it is told about, so a new
package left off the list is invisible to every contract — a drift hole in
the check that exists to stop drift. These tests tie the list, and the
layers table that gives each package its place in the graph, to the import
packages under every ``<Package>/pyproject.toml``.
"""

from __future__ import annotations

import configparser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    assert cfg.read(REPO_ROOT / ".importlinter"), ".importlinter is missing"
    return cfg


def _import_packages_on_disk() -> set[str]:
    """The import package (a directory with ``__init__.py``) of each package."""
    return {
        init.parent.name
        for pyproject in REPO_ROOT.glob("*/pyproject.toml")
        for init in pyproject.parent.glob("*/__init__.py")
        if init.parent.name != "tests"
    }


def test_root_packages_match_the_packages_on_disk() -> None:
    """Every package is a root of the graph, and nothing deleted lingers."""
    roots = set(_config()["importlinter"]["root_packages"].split())
    on_disk = _import_packages_on_disk()
    assert roots == on_disk, {
        "unlisted on disk": sorted(on_disk - roots),
        "listed, not on disk": sorted(roots - on_disk),
    }


def test_every_root_package_has_a_layer() -> None:
    """The layers contract places each package; an unplaced one has no rules."""
    cfg = _config()
    roots = set(cfg["importlinter"]["root_packages"].split())
    layered = {
        name.strip()
        for row in cfg["importlinter:contract:package-layers"]["layers"].splitlines()
        for name in row.split("|")
        if name.strip()
    }
    assert layered == roots, {
        "root without a layer": sorted(roots - layered),
        "layered but not a root": sorted(layered - roots),
    }
