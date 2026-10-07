"""Tests for ``scripts/doc_audit.py`` — the documentation drift audit.

The audit's prose checks are advisory everywhere except in a
``STRICT_PROSE`` package, where they are hard outside ``tests/``. These
tests pin that escalation (both directions: a strict package fails
``--strict``, a loose one does not, a strict package's tests stay
advisory) and the one-issue-number rule of the ``narrative`` check, on a
synthetic two-package repo so the real tree's prose never decides the
outcome.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    """Import ``scripts/doc_audit.py`` by path (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location(
        "doc_audit", REPO_ROOT / "scripts" / "doc_audit.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["doc_audit"] = module
    spec.loader.exec_module(module)
    return module


doc_audit = _load_module()

ESSAY = '"""An essay.\n\n' + "\n".join(f"Line {i}." for i in range(30)) + '\n"""\n'


def _package(root: Path, name: str, module_doc: str, test_doc: str = "") -> None:
    """A minimal package: pyproject, one import package, one test file."""
    pkg = root / name
    (pkg / name.lower()).mkdir(parents=True)
    (pkg / "pyproject.toml").write_text(
        f'[tool.poetry]\nname = "{name.lower()}"\nversion = "0.1.0"\n'
    )
    (pkg / name.lower() / "__init__.py").write_text(module_doc)
    if test_doc:
        (pkg / "tests").mkdir()
        (pkg / "tests" / "test_x.py").write_text(test_doc)


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    """``Tidy`` (long module docstring, long test docstring) and ``Loose``."""
    _package(tmp_path, "Tidy", ESSAY, ESSAY)
    _package(tmp_path, "Loose", ESSAY)
    return tmp_path


def _severities(root: Path, only: set[str]) -> dict[str, str]:
    repo = doc_audit.Repo(root)
    return {f.path: doc_audit.severity_of(f) for f in doc_audit.run(repo, only)}


# --- STRICT_PROSE escalation ---------------------------------------------------


def test_strict_prose_escalates_outside_tests(repo_root: Path, monkeypatch) -> None:
    """In a strict package, long-doc is hard in code and advisory in tests."""
    monkeypatch.setattr(doc_audit, "STRICT_PROSE", {"Tidy"})
    sev = _severities(repo_root, {"long-doc"})
    assert sev == {
        "Tidy/tidy/__init__.py": "hard",
        "Tidy/tests/test_x.py": "advisory",
        "Loose/loose/__init__.py": "advisory",
    }


def test_without_the_set_every_prose_finding_is_advisory(
    repo_root: Path, monkeypatch
) -> None:
    monkeypatch.setattr(doc_audit, "STRICT_PROSE", set())
    assert set(_severities(repo_root, {"long-doc"}).values()) == {"advisory"}


@pytest.mark.parametrize(("strict", "expected"), [({"Tidy"}, 1), (set(), 0)])
def test_strict_exit_code_follows_the_set(
    repo_root: Path, monkeypatch, capsys, strict: set[str], expected: int
) -> None:
    """``--strict`` exits 1 only when a strict package has a prose finding."""
    monkeypatch.setattr(doc_audit, "STRICT_PROSE", strict)
    rc = doc_audit.main(["--root", str(repo_root), "--only", "long-doc", "--strict"])
    out = capsys.readouterr().out
    assert rc == expected
    assert ("Tidy (prose strict)" in out) is bool(strict)


def test_strict_prose_names_packages_on_disk() -> None:
    """A renamed or deleted package must leave the set, not silently un-strict it."""
    on_disk = {p.name for p in doc_audit.Repo(REPO_ROOT).packages}
    assert doc_audit.STRICT_PROSE <= on_disk, doc_audit.STRICT_PROSE - on_disk


# --- narrative: one bare issue number per docstring ---------------------------


@pytest.mark.parametrize(
    ("doc", "hits"),
    [
        ('"""Pins the rule from #123."""\n', []),
        ('"""Pins #123.\n\nRestates #123 later."""\n', []),  # one issue, twice
        ('"""See #123.\n\nAnd #456 too."""\n', ["#123", "#456"]),
        ('"""Ruled on 2026-01-02, see #123."""\n', ["2026-01-02"]),
        # Present-tense phrases and quoted examples are not history.
        ('"""The lock used to serialize puts; M3 is a mirror."""\n', []),
        ('"""Readback at the time of the trigger, as ISO ``2026-09-24``."""\n', []),
        ('"""A snapshot at the time a shot lands."""\n', []),
        ('"""It used to be a list, back when M6 cutover ran."""\n', ["used to be"]),
        ('"""The defaults file in force at the time."""\n', ["at the time"]),
    ],
)
def test_narrative_rule_of_thumb(tmp_path: Path, doc: str, hits: list[str]) -> None:
    """One issue may pin a rule; stories are flagged; present tense is not."""
    _package(tmp_path, "Pkg", doc)
    findings = doc_audit.run(doc_audit.Repo(tmp_path), {"narrative"})
    assert [f.message.split(":")[0].strip("`") for f in findings] == hits
