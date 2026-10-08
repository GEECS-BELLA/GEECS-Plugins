"""Pin ``scripts/release_notes.py`` — the release-time CHANGELOG draft.

The script filters merged PRs (gh JSON rows) to those touching one package
directory and renders one bullet per PR title; these tests drive the pure
``draft`` function with fake rows, so no network or gh is needed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "release_notes", REPO_ROOT / "scripts" / "release_notes.py"
)
assert _spec and _spec.loader
release_notes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release_notes)


def _pr(number: int, title: str, *paths: str) -> dict:
    return {"number": number, "title": title, "files": [{"path": p} for p in paths]}


def test_draft_keeps_only_the_packages_prs_oldest_first() -> None:
    prs = [
        _pr(12, "Core: second", "GEECS-Core/geecs_core/x.py"),
        _pr(10, "Portal only", "GEECS-DataPortal/app.py"),
        _pr(11, "Core: first", "README.md", "GEECS-Core/CHANGELOG.md"),
    ]
    text = release_notes.draft("GEECS-Core", prs, "0.14.0", "2026-10-07")
    assert text.splitlines()[0] == "## [0.14.0] - 2026-10-07"
    assert "- Core: first (#11)\n- Core: second (#12)\n" in text
    assert "Portal only" not in text


def test_draft_does_not_match_a_package_name_prefix() -> None:
    prs = [_pr(5, "Schemas", "GEECS-Schemas-Extra/x.py")]
    assert "(#5)" not in release_notes.draft(
        "GEECS-Schemas", prs, "1.0.0", "2026-10-07"
    )


def test_draft_with_no_hits_says_so() -> None:
    assert "no merged PRs" in release_notes.draft(
        "GEECS-Core", [], "X.Y.Z", "2026-10-07"
    )
