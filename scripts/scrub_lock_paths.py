#!/usr/bin/env python3
"""Normalize absolute path-dependency URLs in ``poetry.lock`` files (#753).

Poetry serializes a path dependency that appears inside an *extras* string as
the absolute ``file://`` URL of the checkout where ``poetry lock`` ran::

    optimize = ["imageanalysis @ file:///Users/<user>/.../GEECS-Plugins/ImageAnalysis", ...]

That is a user home path committed to a public repo, and a dangling path when
the lock ran inside a session worktree. Poetry has no knob for a portable
form (``[package.source] url`` is made lock-relative; ``[package.extras]`` is
not — design report on #753). The URL is inert at install time: Poetry
matches the extra's dependency to the locked ``[[package]]`` *by name*, and
that package carries its own lock-relative ``source.url`` — so any prefix
installs identically, and this hook pins one: ``file:///GEECS-Plugins/<Pkg>``.

Runs as an auto-fixing pre-commit hook on every ``poetry.lock``: rewrites
in place and exits 1 when it changed a file — the ruff-format contract, so
``scripts/commit.sh`` re-stages the fix and the commit succeeds first try.
``--check`` reports without writing (CI / tests).

It never strips a URL to a bare name and never emits a relative ``file:../``
form: a bare name in an extras string resolves to **PyPI** on the next
consumer install (``geecs-data-utils`` is a live PyPI name), and ``file:../``
silently installs nothing. The last path component must be a top-level
package directory of this repo (one holding a ``pyproject.toml``); anything
else is left alone for the ``poetry-lock-no-worktree-paths`` reject hook.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path, PurePosixPath

CANONICAL_PREFIX = "file:///GEECS-Plugins/"

# A file:// URL up to the closing quote / whitespace / list delimiter that
# ends it inside a TOML string or array. Poetry writes forward slashes on
# every platform (``file:///C:/Users/...`` on Windows).
_FILE_URL_RE = re.compile(r"file://(?P<path>/[^\"'\s,\]]+)")


def package_dirs(repo_root: Path) -> frozenset[str]:
    """Return the names of the top-level package directories of the repo.

    Parameters
    ----------
    repo_root : Path
        The monorepo root (the directory holding the root ``pyproject.toml``).

    Returns
    -------
    frozenset[str]
        Directory names directly under ``repo_root`` that contain a
        ``pyproject.toml`` — the only legal last components of a path-dep URL.
    """
    return frozenset(
        p.name
        for p in repo_root.iterdir()
        if p.is_dir() and (p / "pyproject.toml").is_file()
    )


def scrub_text(text: str, packages: frozenset[str]) -> tuple[str, list[str]]:
    """Rewrite every path-dep ``file://`` URL in ``text`` to the canonical prefix.

    Parameters
    ----------
    text : str
        The lock file contents.
    packages : frozenset[str]
        Top-level package directory names (see :func:`package_dirs`).

    Returns
    -------
    tuple[str, list[str]]
        The rewritten text and one ``"<old> -> <new>"`` line per substitution
        (empty when nothing changed).
    """
    changes: list[str] = []

    def _sub(match: re.Match[str]) -> str:
        path = match.group("path")
        name = PurePosixPath(path).name
        if name not in packages:
            return match.group(0)  # not a repo package — leave it for the reject hook
        new = f"{CANONICAL_PREFIX}{name}"
        if match.group(0) == new:
            return new
        changes.append(f"{match.group(0)} -> {new}")
        return new

    return _FILE_URL_RE.sub(_sub, text), changes


def scrub_file(lock: Path, packages: frozenset[str], *, check: bool) -> bool:
    """Scrub one lock file in place (or report, with ``check``).

    Parameters
    ----------
    lock : Path
        The ``poetry.lock`` to process.
    packages : frozenset[str]
        Top-level package directory names (see :func:`package_dirs`).
    check : bool
        When true, report what would change and leave the file untouched.

    Returns
    -------
    bool
        True when the file was (or, under ``check``, would be) modified.
    """
    original = lock.read_text(encoding="utf-8")
    rewritten, changes = scrub_text(original, packages)
    if not changes:
        return False
    verb = "would rewrite" if check else "rewrote"
    print(f"{lock}: {verb} {len(changes)} path-dependency URL(s)", file=sys.stderr)
    for line in changes:
        print(f"  {line}", file=sys.stderr)
    if not check:
        lock.write_text(rewritten, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    """Entry point: exit 1 when any lock was (or would be) modified.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments (``None`` reads ``sys.argv``).

    Returns
    -------
    int
        Process exit status — 1 after a rewrite (so pre-commit fails the run
        and the fixed file gets re-staged), 0 when every lock was clean.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "locks", nargs="+", type=Path, help="poetry.lock files to scrub"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="monorepo root whose top-level package dirs are the legal URL targets",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report only; exit 1 if a rewrite is needed",
    )
    args = parser.parse_args(argv)
    packages = package_dirs(args.repo_root)
    if not packages:
        print(
            f"{args.repo_root}: no top-level package directories found", file=sys.stderr
        )
        return 2
    modified = [scrub_file(lock, packages, check=args.check) for lock in args.locks]
    return 1 if any(modified) else 0


if __name__ == "__main__":
    raise SystemExit(main())
