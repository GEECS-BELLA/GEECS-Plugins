"""Tests for ``scripts/quality_scoreboard.py`` — the debt-marker ratchet.

Counting is pinned on source snippets; check/update on a synthetic git
repo so the real tree's counts never decide the outcome.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# Spelled in pieces (an explicit +, which ruff format never folds) so this
# file never counts toward the ratchet itself.
NOQA = "#" + " noqa"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "quality_scoreboard", REPO_ROOT / "scripts" / "quality_scoreboard.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["quality_scoreboard"] = module
    spec.loader.exec_module(module)
    return module


qs = _load_module()


def test_counts_stamps_noqa_any_and_long_functions() -> None:
    body = "\n".join(f"    x = {i}" for i in range(85))
    source = (
        "from typing import Any\n"
        f"def big(a: Any, b: dict[str, Any]) -> 'list[Any]':  {NOQA}: C901, PLR0915\n"
        f"{body}\n"
        f"import os  {NOQA}: F401\n"
        f"y = 1  {NOQA}\n"
        "z: typing.Any = None\n"
        "Alias = dict[str, Any]\n"  # outside an annotation: not counted
        "def small(): return 1\n"
    )
    counts = qs.count_source(source)
    assert counts["stamps"] == 1
    assert counts["noqa"] == 2
    assert counts["Any"] == 4
    assert counts[">80"] == 1


def test_file_level_forms_count() -> None:
    counts = qs.count_source(f"{NOQA.replace('noqa', 'ruff: noqa')}: C901\n")
    assert counts["stamps"] == 1
    assert qs.count_source(f"{NOQA.replace('noqa', 'flake8: noqa')}\n")["noqa"] == 1


def test_syntax_error_keeps_the_line_counts() -> None:
    counts = qs.count_source(f"def (:  {NOQA}: PLR0912\n")
    assert counts["stamps"] == 1 and counts["Any"] == 0


PYPROJECT = (
    '[tool.pyright]\ninclude = ["Pkg/pkg"]\n'
    '[tool.ruff.lint]\nextend-select = ["C901", "PLR0912", "PLR0915"]\n'
    "[tool.ruff.lint.mccabe]\nmax-complexity = {cx}\n"
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A git repo with one package, a pre-commit exclude and a baseline."""
    (tmp_path / ".pre-commit-config.yaml").write_text(
        "exclude: |\n  (?x)^(\n    extras/\n  )\n\nrepos: []\n"
    )
    (tmp_path / "pyproject.toml").write_text(PYPROJECT.format(cx=10))
    (tmp_path / "Pkg" / "pkg").mkdir(parents=True)
    (tmp_path / "Pkg" / "pyproject.toml").write_text("")
    (tmp_path / "Pkg" / "pkg" / "a.py").write_text(
        f"def f():  {NOQA}: C901\n    pass\n"
    )
    (tmp_path / "extras").mkdir()
    (tmp_path / "extras" / "old.py").write_text(f"x = 1  {NOQA}: C901\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "quality_baseline.json").write_text(
        '{"stamps": 1, "noqa": 0, "max_complexity": 10, '
        '"max_branches": 12, "max_statements": 50, "rules_off": 0, '
        '"rogue_configs": 0, "excludes": []}\n'
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    return tmp_path


def _add(repo: Path, name: str, text: str) -> None:
    (repo / "Pkg" / "pkg" / name).write_text(text)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)


def test_scoreboard_excludes_and_groups(repo: Path) -> None:
    board = qs.scoreboard(repo)
    assert set(board) == {"Pkg", "(ruff config)"}  # extras/ excluded
    assert board["Pkg"]["stamps"] == 1
    assert qs.pyright_packages(repo) == {"Pkg"}


def test_check_passes_at_the_baseline(repo: Path) -> None:
    assert qs.main(["--root", str(repo), "--check"]) == 0


def test_check_fails_on_a_new_stamp(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _add(repo, "b.py", f"def g():  {NOQA}: PLR0912\n    pass\n")
    assert qs.main(["--root", str(repo), "--check"]) == 1
    assert "exceeds the baseline 1" in capsys.readouterr().out


def test_lower_count_asks_and_update_rewrites(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _add(repo, "a.py", "def f():\n    pass\n")
    assert qs.main(["--root", str(repo), "--check"]) == 0
    assert "lower the baseline" in capsys.readouterr().out
    assert qs.main(["--root", str(repo), "--check", "--update"]) == 0
    baseline = json.loads((repo / "scripts" / "quality_baseline.json").read_text())
    assert baseline["stamps"] == 0 and baseline["rules_off"] == 0


LINT = "[tool.ruff.lint]\n"
PREFIXES = ["C901", "PLR0912", "PLR0915", "C90", "PLR09", "PLR", "C", "ALL"]
ROOT = PYPROJECT.format(cx=10)
ESCAPES = {
    "max-complexity": ("pyproject.toml", PYPROJECT.format(cx=15)),
    "max-statements": (
        "pyproject.toml",
        ROOT + "[tool.ruff.lint.pylint]\nmax-statements = 80\n",
    ),
    "max-branches": (
        "pyproject.toml",
        ROOT + "[tool.ruff.lint.pylint]\nmax-branches = 20\n",
    ),
    "deselected": ("pyproject.toml", ROOT.replace(', "PLR0915"]', "]")),
    "select-replaced": (
        "pyproject.toml",
        ROOT.replace("extend-select", "select").replace('"C901", ', ""),
    ),
    "legacy-top-level-ignore": (
        "pyproject.toml",
        ROOT.replace(
            "[tool.ruff.lint]\n", '[tool.ruff]\nignore = ["C90"]\n[tool.ruff.lint]\n'
        ),
    ),
    "lint-exclude": (
        "pyproject.toml",
        ROOT.replace(LINT, LINT + 'exclude = ["Pkg/**"]\n'),
    ),
    "extend-exclude": (
        "pyproject.toml",
        ROOT.replace(
            "[tool.ruff.lint]\n",
            '[tool.ruff]\nextend-exclude = ["Pkg"]\n[tool.ruff.lint]\n',
        ),
    ),
    "package-ruff-table": ("Pkg/pyproject.toml", '[tool.ruff.lint]\nselect = ["E"]\n'),
    "root-extends-elsewhere": (
        "pyproject.toml",
        ROOT.replace(
            "[tool.ruff.lint]\n",
            '[tool.ruff]\nextend = "lint.toml"\n[tool.ruff.lint]\n',
        ),
    ),
    "child-extends-elsewhere": ("Pkg/ruff.toml", 'extend = "lint.toml"\n'),
    "include": (
        "pyproject.toml",
        ROOT.replace(
            "[tool.ruff.lint]\n", '[tool.ruff]\ninclude = ["*.pyi"]\n[tool.ruff.lint]\n'
        ),
    ),
    "extend-include": (
        "Pkg/ruff.toml",
        'extend = "../pyproject.toml"\nextend-include = ["*.pyi"]\n',
    ),
    "package-ruff-toml": ("Pkg/ruff.toml", "line-length = 100\n"),
    "package-dot-ruff-toml": ("Pkg/.ruff.toml", "line-length = 100\n"),
    "extending-child-ignores": (
        "Pkg/ruff.toml",
        'extend = "../pyproject.toml"\n[lint]\nignore = ["PLR"]\n',
    ),
}
for code in PREFIXES:
    for key in ("ignore", "extend-ignore"):
        ESCAPES[f"{key}-{code}"] = (
            "pyproject.toml",
            ROOT.replace(LINT, LINT + f'{key} = ["{code}"]\n'),
        )
    for key in ("per-file-ignores", "extend-per-file-ignores"):
        ESCAPES[f"{key}-{code}"] = (
            "pyproject.toml",
            ROOT + f'[tool.ruff.lint.{key}]\n"**/*.py" = ["{code}"]\n',
        )


@pytest.mark.parametrize("name", sorted(ESCAPES))
def test_config_escapes_are_ratcheted(repo: Path, name: str) -> None:
    path, text = ESCAPES[name]
    (repo / path).write_text(text)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    assert qs.main(["--root", str(repo), "--check"]) == 1


def test_a_swapped_exclude_fails(repo: Path) -> None:
    pyproject = repo / "pyproject.toml"
    pyproject.write_text(ROOT.replace(LINT, LINT + 'exclude = ["extras/**"]\n'))
    assert qs.main(["--root", str(repo), "--update"]) == 0
    pyproject.write_text(ROOT.replace(LINT, LINT + 'exclude = ["**"]\n'))
    assert qs.main(["--root", str(repo), "--check"]) == 1
    pyproject.write_text(ROOT)
    assert qs.main(["--root", str(repo), "--check"]) == 0  # fewer is fine


def test_extending_child_without_escapes_passes(repo: Path) -> None:
    (repo / "Pkg" / "ruff.toml").write_text(
        'extend = "../pyproject.toml"\nline-length = 100\n'
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    assert qs.main(["--root", str(repo), "--check"]) == 0


def test_file_deleted_from_disk_is_skipped(repo: Path) -> None:
    (repo / "Pkg" / "pkg" / "a.py").unlink()
    assert "Pkg" not in qs.scoreboard(repo)


def test_real_precommit_exclude_parses() -> None:
    exclude = qs.precommit_exclude(REPO_ROOT)
    assert exclude.match("extras/anything.py")
    assert not exclude.match("GEECS-Core/geecs_core/__init__.py")
