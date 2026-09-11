"""The deploy tree's contracts: what a restart installs, and from where.

``launch.bat`` is copied to every camera server once at bootstrap, so the
things it names — the intra-repo packages it reinstalls and the fleet
requirements it installs offline from the share's wheel cache — are pinned
here, where a change is reviewed instead of discovered as a crash loop
(the 0.4.4 fleet's launcher predated GEECS-Core; ``DEPLOYMENT.md``).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"
PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

#: The intra-repo packages pull-on-restart reinstalls, in dependency order.
REINSTALLED = (
    "GEECS-Schemas",
    "GEECS-Core",
    "GEECS-Data-Utils",
    "GeecsCAGateway",
    "GeecsPvaGateway",
)


def _pins() -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in (DEPLOY / "requirements-fleet.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([0-9][0-9A-Za-z.]*)", line)
        assert match, f"fleet requirement must be an exact pin: {line!r}"
        pins[match.group(1).lower()] = match.group(2)
    return pins


def test_fleet_requirements_are_exact_pins_of_declared_dependencies() -> None:
    """Every fleet pin is a dependency the package declares (and vice versa for new ones)."""
    pins = _pins()
    assert pins, "requirements-fleet.txt lists nothing"
    from packaging.specifiers import SpecifierSet

    declared = {
        name.lower(): spec
        for name, spec in tomllib.loads(PYPROJECT.read_text())["tool"]["poetry"][
            "dependencies"
        ].items()
    }
    undeclared = set(pins) - set(declared)
    assert not undeclared, f"fleet pins not in pyproject dependencies: {undeclared}"
    for name, version in pins.items():
        spec = declared[name]
        spec = spec["version"] if isinstance(spec, dict) else spec
        assert version in SpecifierSet(
            spec.replace("^", "~=") if spec.startswith("^") else spec
        ), f"{name}=={version} violates pyproject's {spec!r}"
    assert "h5py" in pins  # the file plugin's container (0.7.0)


def test_launcher_reinstalls_every_intra_repo_package_and_the_fleet_pins() -> None:
    """The reinstall line names all five packages; the wheel step precedes it."""
    text = (DEPLOY / "launch.bat").read_text()
    reinstall = next(
        line for line in text.splitlines() if "--no-deps --no-build-isolation" in line
    )
    for package in REINSTALLED:
        assert f'"%GEECS_PVA_SOURCE%\\{package}"' in reinstall, package
    wheels = text.index("--no-index --find-links")
    assert wheels < text.index("--no-deps --no-build-isolation")
    assert "requirements-fleet.txt" in text
    # The cache path is resolved OUTSIDE the parenthesized block (cmd expands
    # %VAR% inside a block when the block is parsed, not when the line runs):
    # the assignment from the for-loop, not merely the clearing `set`.
    assert text.index('set "GEECS_PVA_WHEELS=%%~fI') < text.index(
        'if not "%GEECS_PVA_SOURCE%"=="" ('
    )
    # The pin file is the closure: no dependency resolution on either side.
    assert "--no-index --no-deps --find-links" in text
    assert "--no-deps" in (DEPLOY / "stage_wheels.sh").read_text()


def test_stage_script_reads_the_same_requirements() -> None:
    text = (DEPLOY / "stage_wheels.sh").read_text()
    assert "requirements-fleet.txt" in text
    assert "--platform win_amd64" in text and "--python-version 3.11" in text
    assert "--no-deps" in text  # dependencies were frozen at bootstrap
    assert "pva-wheels" in text and "pva-wheels" in (DEPLOY / "launch.bat").read_text()
