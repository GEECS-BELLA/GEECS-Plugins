#!/usr/bin/env python3
"""Audit the monorepo's documentation for drift, in code and in Markdown.

Why this exists
---------------
Documentation here has two layers that drift independently. Every package
carries the same small set of documents (``README.md``, ``CLAUDE.md``,
``CHANGELOG.md``, ``pyproject.toml``) and the root ``CLAUDE.md`` plus
``mkdocs.yml`` describe the set as a whole; nothing ties them together, so
a package is added without a row in the repository map, a version is
bumped without a changelog entry, a page moves and a link keeps the old
path. Underneath, docstrings and comments name modules, functions, files
and *retired things*, and after a fast development stretch they keep
naming them long after the rename or deletion, and grow into design essays
with dates, issue numbers and rulings that belong in the changelog.

This script finds both kinds of drift. It is **read-only** and
**stdlib-only** (Python >= 3.11 for ``tomllib``); it never imports the
packages (cross-references resolve against an AST index of the repo), so
it runs from any checkout with no environment set up, in a few seconds.

What it checks
--------------
**Hard findings** (``--strict`` exits 1 when any exist):

Per package, from its top-level files:

* ``required-docs``: ``README.md``, ``CLAUDE.md``, ``CHANGELOG.md`` exist
  (``LogMaker4GoogleDocs`` is exempt from the README: it is being replaced,
  not documented, per the root ``CLAUDE.md``).
* ``changelog-version``: the newest ``## [x.y.z]`` heading equals
  ``version`` in ``pyproject.toml``.
* ``changelog-format``: Keep a Changelog is cited and every version
  heading starts ``## [x.y.z] - YYYY-MM-DD`` (a trailing annotation is
  fine; ``— current`` and ``- unreleased`` are not).

Across the repo, from the root files and ``docs/``:

* ``repo-map``: the root ``CLAUDE.md`` repository-map table lists every
  package and nothing that no longer exists.
* ``changelog-list``: the root "Release & Versioning" list of packages
  with a changelog matches the packages on disk.
* ``agents-shim``: ``AGENTS.md`` points at ``CLAUDE.md`` and stays short.
* ``nav-target``: every ``.md`` in ``mkdocs.yml``'s ``nav:`` exists.
* ``orphan-page``: every ``.md`` under ``docs/`` is in ``nav:`` or linked
  from a page that is (``docs/sites/`` is HTML by design and exempt, and so
  is a page ``mkdocs.yml`` lists under ``exclude_docs:``).
* ``broken-link``: every relative Markdown link resolves.

In docstrings, ``#:`` attribute comments and Markdown (``CHANGELOG.md``
excluded: it is where history is supposed to live):

* ``dangling-ref``: a Sphinx role (``:func:``, ``:class:``, ``:mod:``,
  ``:meth:``, ``:data:``, ``:attr:``, ``:exc:``) or a backticked dotted
  name rooted in a repo import package (``geecs_bluesky.plans.registry``)
  names something the AST index cannot find. A bare role target
  (``:meth:`probe```) passes if that name is defined anywhere.
* ``dangling-path``: a backticked file path (by extension, or ``a/b``
  rooted in a top-level directory) exists nowhere: it is resolved against
  the citing file's directory, its package root, the repo root and every
  package; a bare filename found anywhere in the repo is accepted. A
  pytest node id is checked by its file part; placeholders (``<expt>``,
  ``NNN``) are skipped, as is the gitignored ``.claude/worktrees/``.
* ``stale-term``: a name from ``STALE_TERMS`` (retired packages, classes
  and phrases) appears outside a line that is itself talking about the
  retirement. **Extend the list in the PR that deletes a thing.**

**Advisory findings** (counted per package; listed with ``--advisory``;
never fail ``--strict``):

* ``narrative``: a date, a ``#NNN`` issue reference, a person's name, a
  "ruling", a phase number or a history word in a docstring, ``#:``
  comment or ``CLAUDE.md``. The repo's rule of thumb: one bare issue
  number may pin a non-obvious rule; dates, names and stories belong in
  the changelog.
* ``long-doc``: a module docstring over 25 lines, a def/class docstring
  over 60, or a ``#:`` comment block over 6.
* ``boilerplate``: template openings ("This module provides ...") and
  hand-written Functions/Constants tables that mkdocstrings renders anyway.

The prose checks skip the packages in ``SKIP_PROSE`` (the ones the owner
ruled not worth tidying: legacy, experimental, or already template-style)
and the root directories in ``SKIP_ROOT_PROSE`` (``Planning/``, design
scaffolding that describes code as it will be; ``apps_script/``, LogMaker's
Google side); ``--all`` audits them anyway. The structural checks always cover every
package, because the repository map and the changelog list are about the
set, not about any one package.

Usage
-----
    ./scripts/doc_audit.py                    # Markdown report; exit 0
    ./scripts/doc_audit.py --strict           # exit 1 on any hard finding
    ./scripts/doc_audit.py -p GeecsBluesky    # one package (repeatable)
    ./scripts/doc_audit.py --only dangling-ref,stale-term
    ./scripts/doc_audit.py --advisory         # list advisory hits too
    ./scripts/doc_audit.py --json report.json # machine-readable copy

Adding a check: write a function ``(Repo) -> Iterable[Finding]``, register
it in ``CHECKS`` under a kebab-case name with its severity, and describe it
above. Keep every check pure (no writes) and cheap (no imports of the
packages themselves).
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import re
import sys
import tomllib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path

# --------------------------------------------------------------------------- #
# Policy tables: the parts a maintainer edits
# --------------------------------------------------------------------------- #

#: Packages exempt from one required doc, with the reason kept beside it.
README_EXEMPT = {
    "LogMaker4GoogleDocs",  # being replaced by GeecsLogbook, per the root CLAUDE.md
}

#: Packages the prose checks leave alone unless ``--all`` is given. Ruled by
#: the owner on 2026-09-30: legacy (LogMaker), an experiment (MCP), and the
#: two analysis packages whose template-style docstrings are not worth a pass.
SKIP_PROSE = {"LogMaker4GoogleDocs", "GEECS-MCP", "GEECS-Analysis", "ImageAnalysis"}

#: Root-level directories the prose checks leave alone, with the reason.
#: ``--all`` audits them anyway.
SKIP_ROOT_PROSE = {
    # Design scaffolding, deleted in the PR that lands the work it describes
    # (Planning/README.md): it cites code as it will be, not as it is.
    "Planning",
    # The Google-side half of LogMaker4GoogleDocs, which SKIP_PROSE covers.
    "apps_script",
}

#: Top-level non-package directories a backtick path may legitimately cite.
CITABLE_DIRS = {"docs", "scripts", "deploy", ".claude", ".github", "Planning"}

#: Retired names: (term regex, what to write instead, package scope or None).
#: A line that talks *about* the retirement (see ``RETIREMENT_WORDS``) is never
#: flagged, so the root CLAUDE.md's "Deleted legacy packages" section stays
#: legal. Extend this list in the PR that deletes a thing.
STALE_TERMS: list[tuple[str, str, tuple[str, ...] | None]] = [
    (r"\bGEECS-PythonAPI\b", "geecs_core.client.GeecsDevice", None),
    (r"\bGEECS-Scanner-GUI\b", "GeecsScanner + GeecsBluesky", None),
    (r"\bGEECS-Console\b", "GeecsScanner, the web scanner console", None),
    (r"\bPySide6\b", "nothing; no Qt surface remains", None),
    (
        r"\bthe console\b(?![ -]s(?:cript|tream))",
        "the scanner (the PySide6 console is gone)",
        None,
    ),
    (r"\bLiveWatch\b", "nothing; the portal runs analysis on request", None),
    (r"\bGeecsSession\b", "GeecsNamespace / the qs_client manager", None),
    (r"\bBlueskyScanner\b", "the queueserver worker + GeecsScanner", None),
    (
        r"\bfunnel\b",
        "nothing; plans are stock",
        ("GeecsBluesky", "GeecsScanner", "GEECS-MCP"),
    ),
    (
        r"\bScanRequest\b",
        "Preset / the stock plan verbs",
        ("GeecsBluesky", "GeecsScanner", "GEECS-MCP"),
    ),
    (
        r"stock\s+`?bluesky\.plans`?\s+verbs",
        "the worker's registered plans (plan_names.py)",
        None,
    ),
]

#: A line containing one of these is describing a retirement, not using the
#: retired thing, so stale-term skips it.
RETIREMENT_WORDS = re.compile(
    r"\b(deleted|retired|removed|replaced|successor|legacy|superseded|no longer|"
    r"never adopt|final state|deprecat)\w*|tag `",
    re.I,
)

#: Narrative markers: things that belong in a changelog, not a docstring.
NARRATIVE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"
    r"|(?<![\w/])#\d{3,4}\b"
    r"|\bSam\b|\bowner'?s\b|\bruling\b|\bphase\s*\d\b|\bM\d\b"
    r"|\b(incident|historically|used to|back when|originally|at the time)\b",
    re.I,
)

#: Template prose mkdocstrings already renders, or that says nothing.
BOILERPLATE = re.compile(
    r"^\s*This module (provides|contains|implements|defines)\b"
    r"|^\s*(Functions|Constants|Classes)\s*\n\s*-{3,}",
    re.M,
)

LONG_MODULE_DOC = 25
LONG_DEF_DOC = 60
LONG_ATTR_COMMENT = 6

SPHINX_ROLE = re.compile(
    r":(?:func|class|mod|meth|data|attr|exc|obj):`(?P<target>[^`]+)`"
)
DOTTED_NAME = re.compile(
    r"`(?P<target>~?[a-z_][a-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+(?:\(\))?)`"
)
BACKTICK = re.compile(r"`([^`\n]+)`")
PLACEHOLDER = re.compile(r"[<>{}*?@$\[\]]|NNN|YYYY|\bMM\b|\.\.\.")
PATH_EXTS = (
    ".py",
    ".md",
    ".sh",
    ".toml",
    ".css",
    ".html",
    ".js",
    ".yml",
    ".yaml",
    ".json",
    ".ini",
    ".service",
    ".txt",
    ".cfg",
)
SKIP_PARTS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "site",
    ".claude",
    "extras",
    "third_party_sdks",
    "__pycache__",
    ".ipynb_checkpoints",
    "worktrees",
}

VERSION_HEADING = re.compile(r"^## \[(?P<ver>[^\]]+)\](?P<rest>.*)$")
VERSION_HEADING_STRICT = re.compile(
    r"^## \[\d+\.\d+\.\d+\] [-—] \d{4}-\d{2}-\d{2}(\s|$)"
)
MAP_ROW = re.compile(r"^\|\s*`(?P<name>[^`/]+)/`\s*\|")
NAV_MD = re.compile(r"(?::|-)\s*([\w./ -]+\.md)\s*$")
MD_LINK = re.compile(r"(?<!!)\[[^\]]*\]\((?P<target>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
IMG_LINK = re.compile(r"!\[[^\]]*\]\((?P<target>[^)\s]+)\)")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Finding:
    """One thing the audit wants a human to look at."""

    check: str
    package: str
    path: str
    message: str
    line: int | None = None

    def render(self) -> str:
        """Format as ``path:line  [check]  message`` for the terminal."""
        loc = f"{self.path}:{self.line}" if self.line else self.path
        return f"{loc}  [{self.check}]  {self.message}"


@dataclass(frozen=True)
class Package:
    """A top-level directory with a ``pyproject.toml``."""

    name: str
    root: Path

    @property
    def version(self) -> str | None:
        """The ``version`` declared in ``pyproject.toml``, if any."""
        data = tomllib.loads((self.root / "pyproject.toml").read_text())
        poetry = data.get("tool", {}).get("poetry", {})
        return poetry.get("version") or data.get("project", {}).get("version")

    @property
    def import_packages(self) -> list[Path]:
        """Directories directly under the package root with an ``__init__.py``."""
        return sorted(
            p.parent
            for p in self.root.glob("*/__init__.py")
            if p.parent.name != "tests"
        )


@dataclass
class DocText:
    """A run of prose to audit: a docstring, a ``#:`` block, or a Markdown file."""

    path: Path
    line: int
    text: str
    kind: str  # "module", "def", "attr", "markdown"


@dataclass
class Repo:
    """The checkout under audit, with lazily built indexes."""

    root: Path
    selected: set[str] | None = None
    include_skipped: bool = False
    _packages: list[Package] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Discover every top-level directory with a ``pyproject.toml``."""
        self._packages = [
            Package(p.parent.name, p.parent)
            for p in sorted(self.root.glob("*/pyproject.toml"))
            if not p.parent.name.startswith(".")
        ]

    @property
    def packages(self) -> list[Package]:
        """Every package on disk (selection never narrows the indexes)."""
        return self._packages

    @property
    def audited(self) -> list[Package]:
        """The packages the structural checks and the table cover."""
        return [
            p for p in self._packages if not self.selected or p.name in self.selected
        ]

    def rel(self, path: Path) -> str:
        """A repo-relative POSIX path for reports."""
        return path.relative_to(self.root).as_posix()

    def package_of(self, path: Path) -> str:
        """The package name a path lives in, or ``docs`` or ``root``."""
        parts = path.relative_to(self.root).parts
        if parts and parts[0] in {p.name for p in self._packages}:
            return parts[0]
        return "docs" if parts and parts[0] == "docs" else "root"

    def prose_in_scope(self, path: Path) -> bool:
        """Whether the prose checks read this file."""
        pkg = self.package_of(path)
        if self.selected:
            return pkg in self.selected
        if self.include_skipped:
            return True
        parts = path.relative_to(self.root).parts
        if pkg == "root" and parts and parts[0] in SKIP_ROOT_PROSE:
            return False
        return pkg not in SKIP_PROSE

    def _skip(self, path: Path) -> bool:
        return any(part in SKIP_PARTS for part in path.relative_to(self.root).parts)

    def markdown_files(self) -> Iterator[Path]:
        """Every ``.md`` in packages, ``docs/`` and the root, minus vendored trees."""
        for path in sorted(self.root.rglob("*.md")):
            if not self._skip(path) and self.prose_in_scope(path):
                yield path

    def python_files(self) -> Iterator[Path]:
        """Every ``.py`` in a prose-audited package, minus vendored trees."""
        for pkg in self._packages:
            if not self.prose_in_scope(pkg.root / "x"):
                continue
            for path in sorted(pkg.root.rglob("*.py")):
                if not self._skip(path):
                    yield path

    # -- indexes ----------------------------------------------------------- #

    @cached_property
    def symbol_index(self) -> tuple[set[str], set[str]]:
        """``(dotted, bare)``: every importable dotted name, and every bare name.

        Built from the AST of every import package in every package, so a
        GeecsScanner docstring may cite ``geecs_bluesky.qs_client``.
        Re-exports (``from x import y`` in an ``__init__``) count as members
        of the importing module, so ``geecs_schemas.Preset`` resolves.
        """
        dotted: set[str] = set()
        bare: set[str] = set()
        for pkg in self._packages:
            for top in pkg.import_packages:
                for path in top.rglob("*.py"):
                    if self._skip(path):
                        continue
                    mod = ".".join(path.relative_to(top.parent).with_suffix("").parts)
                    mod = mod.removesuffix(".__init__")
                    dotted.add(mod)
                    bare.add(mod.rsplit(".", 1)[-1])
                    parts = mod.split(".")
                    for cut in range(1, len(parts)):
                        dotted.add(".".join(parts[:cut]))  # incl. namespace packages
                    try:
                        tree = ast.parse(path.read_text(errors="replace"))
                    except SyntaxError:
                        continue
                    for node in tree.body:
                        for name in _names_defined(node):
                            dotted.add(f"{mod}.{name}")
                            bare.add(name)
                        if isinstance(node, ast.ClassDef):
                            for sub in node.body:
                                for name in _names_defined(sub):
                                    dotted.add(f"{mod}.{node.name}.{name}")
                                    bare.add(name)
        return dotted, bare

    @cached_property
    def file_names(self) -> set[str]:
        """Every basename in the repo, for bare-filename path citations."""
        return {p.name for p in self.root.rglob("*") if not self._skip(p)}

    @cached_property
    def citable_heads(self) -> set[str]:
        """Top-level names a path citation may start with."""
        heads = {p.name for p in self._packages} | {
            d for d in CITABLE_DIRS if (self.root / d).is_dir()
        }
        for pkg in self._packages:
            heads.update(t.name for t in pkg.import_packages)
        return heads


def _names_defined(node: ast.AST) -> Iterator[str]:
    """Names a statement defines: defs, classes, assignments, imports."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        yield node.name
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                yield target.id
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        yield node.target.id
    elif isinstance(node, ast.ImportFrom):
        for alias in node.names:
            if alias.name != "*":
                yield alias.asname or alias.name
    elif isinstance(node, ast.Import):
        for alias in node.names:
            yield (alias.asname or alias.name).split(".")[0]


# --------------------------------------------------------------------------- #
# Prose extraction
# --------------------------------------------------------------------------- #


def iter_doc_texts(repo: Repo) -> Iterator[DocText]:
    """Every docstring, ``#:`` block and Markdown file the prose checks read."""
    for path in repo.markdown_files():
        if path.name == "CHANGELOG.md":
            continue
        yield DocText(path, 1, path.read_text(errors="replace"), "markdown")
    for path in repo.python_files():
        source = path.read_text(errors="replace")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        doc = _docstring_node(tree)
        if doc is not None:
            yield DocText(
                path, doc.lineno, ast.get_docstring(tree, clean=False) or "", "module"
            )
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = _docstring_node(node)
                if doc is not None:
                    yield DocText(
                        path,
                        doc.lineno,
                        ast.get_docstring(node, clean=False) or "",
                        "def",
                    )
        block: list[str] = []
        start = 0
        for lineno, line in enumerate(source.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#:"):
                if not block:
                    start = lineno
                block.append(stripped[2:].strip())
            elif block:
                yield DocText(path, start, "\n".join(block), "attr")
                block = []
        if block:
            yield DocText(path, start, "\n".join(block), "attr")


def _docstring_node(node: ast.AST) -> ast.Expr | None:
    body = getattr(node, "body", None)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        if isinstance(body[0].value.value, str):
            return body[0]
    return None


def _prose_lines(text: str) -> Iterator[tuple[int, str]]:
    """Lines of a doc text outside fenced code blocks, as 0-based offsets."""
    in_fence = False
    for offset, line in enumerate(text.splitlines()):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            yield offset, line


# --------------------------------------------------------------------------- #
# Per-package structure checks
# --------------------------------------------------------------------------- #


def check_required_docs(repo: Repo) -> Iterator[Finding]:
    """Each package ships README, CLAUDE and CHANGELOG."""
    for pkg in repo.audited:
        for name in ("README.md", "CLAUDE.md", "CHANGELOG.md"):
            if name == "README.md" and pkg.name in README_EXEMPT:
                continue
            if not (pkg.root / name).is_file():
                yield Finding("required-docs", pkg.name, pkg.name, f"missing {name}")


def check_changelog_version(repo: Repo) -> Iterator[Finding]:
    """The newest changelog heading equals the pyproject version."""
    for pkg in repo.audited:
        changelog = pkg.root / "CHANGELOG.md"
        if not changelog.is_file():
            continue
        declared = pkg.version
        newest: tuple[int, str] | None = None
        for lineno, line in enumerate(changelog.read_text().splitlines(), 1):
            m = VERSION_HEADING.match(line)
            if m and m.group("ver").lower() != "unreleased":
                newest = (lineno, m.group("ver"))
                break
        path = repo.rel(changelog)
        if newest is None:
            yield Finding(
                "changelog-version", pkg.name, path, "no `## [x.y.z]` heading found"
            )
        elif declared is None:
            yield Finding(
                "changelog-version",
                pkg.name,
                pkg.name + "/pyproject.toml",
                "no version declared",
            )
        elif newest[1] != declared:
            yield Finding(
                "changelog-version",
                pkg.name,
                path,
                f"newest entry is {newest[1]} but pyproject.toml says {declared}",
                newest[0],
            )


def check_changelog_format(repo: Repo) -> Iterator[Finding]:
    """Keep a Changelog is named and version headings carry a date."""
    for pkg in repo.audited:
        changelog = pkg.root / "CHANGELOG.md"
        if not changelog.is_file():
            continue
        text = changelog.read_text()
        path = repo.rel(changelog)
        if "keepachangelog.com" not in text:
            yield Finding(
                "changelog-format", pkg.name, path, "does not cite Keep a Changelog"
            )
        for lineno, line in enumerate(text.splitlines(), 1):
            m = VERSION_HEADING.match(line)
            if not m or m.group("ver").lower() == "unreleased":
                continue
            if not VERSION_HEADING_STRICT.match(line):
                yield Finding(
                    "changelog-format",
                    pkg.name,
                    path,
                    f"heading is not `## [x.y.z] - YYYY-MM-DD`: {line.strip()!r}",
                    lineno,
                )


# --------------------------------------------------------------------------- #
# Root CLAUDE.md and docs-site checks
# --------------------------------------------------------------------------- #


def _root_claude(repo: Repo) -> str | None:
    path = repo.root / "CLAUDE.md"
    return path.read_text() if path.is_file() else None


def check_repo_map(repo: Repo) -> Iterator[Finding]:
    """The repository-map table and the packages on disk agree."""
    text = _root_claude(repo)
    if text is None:
        yield Finding("repo-map", "root", "CLAUDE.md", "root CLAUDE.md is missing")
        return
    listed: dict[str, int] = {}
    in_map = False
    for lineno, line in enumerate(text.splitlines(), 1):
        if re.match(r"^\|\s*Package\s*\|", line):
            in_map = True
            continue
        if in_map:
            if not line.startswith("|"):
                break
            m = MAP_ROW.match(line)
            if m:
                listed[m.group("name")] = lineno
    if not listed:
        yield Finding("repo-map", "root", "CLAUDE.md", "no `| Package |` table found")
        return
    on_disk = {p.name for p in repo.packages}
    for name in sorted(on_disk - listed.keys()):
        yield Finding(
            "repo-map",
            "root",
            "CLAUDE.md",
            f"package `{name}/` has no row in the repository map",
        )
    for name, lineno in sorted(listed.items()):
        if name not in on_disk:
            yield Finding(
                "repo-map",
                "root",
                "CLAUDE.md",
                f"map row `{name}/` names no package on disk",
                lineno,
            )


def check_changelog_list(repo: Repo) -> Iterator[Finding]:
    """The root CLAUDE.md list of packages with a changelog is complete."""
    text = _root_claude(repo)
    if text is None:
        return
    m = re.search(
        r"Every package has a `CHANGELOG.md`.*?format:\s*(?P<list>(?:`[^`]+/`[,.\s]*)+)",
        text,
        re.S,
    )
    if not m:
        yield Finding(
            "changelog-list",
            "root",
            "CLAUDE.md",
            "no `Every package has a CHANGELOG.md` list found",
        )
        return
    lineno = text[: m.start()].count("\n") + 1
    listed = set(re.findall(r"`([^`/]+)/`", m.group("list")))
    with_changelog = {
        p.name for p in repo.packages if (p.root / "CHANGELOG.md").is_file()
    }
    for name in sorted(with_changelog - listed):
        yield Finding(
            "changelog-list",
            "root",
            "CLAUDE.md",
            f"`{name}/` has a CHANGELOG.md but is not listed",
            lineno,
        )
    for name in sorted(listed - with_changelog):
        yield Finding(
            "changelog-list",
            "root",
            "CLAUDE.md",
            f"`{name}/` is listed but has no CHANGELOG.md",
            lineno,
        )


def check_agents_shim(repo: Repo) -> Iterator[Finding]:
    """AGENTS.md stays a pointer to CLAUDE.md, never a second policy."""
    path = repo.root / "AGENTS.md"
    if not path.is_file():
        yield Finding(
            "agents-shim", "root", "AGENTS.md", "missing (Codex compatibility shim)"
        )
        return
    text = path.read_text()
    if "CLAUDE.md" not in text:
        yield Finding("agents-shim", "root", "AGENTS.md", "does not point at CLAUDE.md")
    body = [ln for ln in text.splitlines() if ln.strip()]
    if len(body) > 20:
        yield Finding(
            "agents-shim",
            "root",
            "AGENTS.md",
            f"{len(body)} non-blank lines; the shim should stay a pointer",
        )


def _nav_targets(repo: Repo) -> list[tuple[int, str]]:
    """Every ``.md`` path named in ``mkdocs.yml`` after ``nav:``."""
    mkdocs = repo.root / "mkdocs.yml"
    if not mkdocs.is_file():
        return []
    out: list[tuple[int, str]] = []
    in_nav = False
    for lineno, line in enumerate(mkdocs.read_text().splitlines(), 1):
        if line.startswith("nav:"):
            in_nav = True
            continue
        if in_nav and line and not line.startswith((" ", "-", "#")):
            break
        if in_nav:
            m = NAV_MD.search(line)
            if m:
                out.append((lineno, m.group(1)))
    return out


def check_nav_targets(repo: Repo) -> Iterator[Finding]:
    """Every nav entry points at a page under docs/."""
    docs = repo.root / "docs"
    for lineno, target in _nav_targets(repo):
        if not (docs / target).is_file():
            yield Finding(
                "nav-target",
                "docs",
                "mkdocs.yml",
                f"nav entry `{target}` does not exist under docs/",
                lineno,
            )


def _links_in(path: Path) -> Iterator[tuple[int, str]]:
    """Every relative link target (page or image) in a Markdown file."""
    for offset, line in _prose_lines(path.read_text(errors="replace")):
        for pattern in (MD_LINK, IMG_LINK):
            for m in pattern.finditer(line):
                target = m.group("target")
                if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
                    continue  # absolute URL, mailto, or in-page anchor
                if PLACEHOLDER.search(target):
                    continue  # `/entry/{id}`-style documentation of a route
                yield offset + 1, target.split("#", 1)[0]


def _resolve(base: Path, target: str) -> Path:
    return (base.parent / target).resolve() if target else base


def _excluded_docs(repo: Repo) -> set[Path]:
    """Pages listed under ``exclude_docs:`` in ``mkdocs.yml``: kept out on purpose."""
    mkdocs = repo.root / "mkdocs.yml"
    out: set[Path] = set()
    if not mkdocs.is_file():
        return out
    in_block = False
    for line in mkdocs.read_text().splitlines():
        if line.startswith("exclude_docs:"):
            in_block = True
            continue
        if in_block:
            if line.startswith((" ", "\t")) and line.strip():
                out.add((repo.root / "docs" / line.strip()).resolve())
            elif line.strip():
                break
    return out


def check_orphan_pages(repo: Repo) -> Iterator[Finding]:
    """Every docs page is in nav or linked from a page that is."""
    docs = repo.root / "docs"
    if not docs.is_dir() or (repo.selected and "docs" not in repo.selected):
        return
    excluded = _excluded_docs(repo)
    reachable = {(docs / t).resolve() for _, t in _nav_targets(repo)}
    frontier = list(reachable)
    while frontier:
        page = frontier.pop()
        if not page.is_file():
            continue
        for _, target in _links_in(page):
            hit = _resolve(page, target)
            if hit.suffix == ".md" and hit.is_file() and hit not in reachable:
                reachable.add(hit)
                frontier.append(hit)
    for page in sorted(docs.rglob("*.md")):
        if "sites" in page.relative_to(docs).parts or page.name == "CLAUDE.md":
            continue
        if page.resolve() in excluded:
            continue
        if page.resolve() not in reachable:
            yield Finding(
                "orphan-page",
                "docs",
                repo.rel(page),
                "not in nav and not linked from any nav page",
            )


def check_broken_links(repo: Repo) -> Iterator[Finding]:
    """Relative Markdown links resolve to something on disk."""
    for page in repo.markdown_files():
        for lineno, target in _links_in(page):
            if target and not _resolve(page, target).exists():
                yield Finding(
                    "broken-link",
                    repo.package_of(page),
                    repo.rel(page),
                    f"link target `{target}` does not exist",
                    lineno,
                )


# --------------------------------------------------------------------------- #
# Prose checks: references, paths, stale terms, advisory
# --------------------------------------------------------------------------- #


def check_dangling_refs(repo: Repo) -> Iterator[Finding]:
    """Sphinx roles and dotted names resolve against the AST index."""
    dotted, bare = repo.symbol_index
    roots = {t.name for pkg in repo.packages for t in pkg.import_packages}
    builtin_names = set(dir(builtins))
    local_names: dict[Path, set[str]] = {}
    for doc in iter_doc_texts(repo):
        if doc.path.suffix == ".py" and doc.path not in local_names:
            local_names[doc.path] = set(
                re.findall(
                    r"[A-Za-z_][A-Za-z0-9_]*", doc.path.read_text(errors="replace")
                )
            )
        for offset, line in _prose_lines(doc.text):
            targets = [m.group("target") for m in SPHINX_ROLE.finditer(line)]
            targets += [m.group("target") for m in DOTTED_NAME.finditer(line)]
            for raw in dict.fromkeys(targets):
                target = raw.lstrip("~").removesuffix("()")
                head = target.split(".")[0]
                if "." not in target:
                    # A bare role target: a parameter, a local, a sibling module or
                    # anything defined anywhere passes; only a name nobody has fails.
                    ok = target in bare or target in dotted or target in builtin_names
                    ok = ok or target in local_names.get(doc.path, set())
                elif head in roots:
                    ok = target in dotted or _is_member_of_known(target, dotted)
                else:
                    continue  # stdlib or third-party dotted name: not ours to check
                if not ok:
                    yield Finding(
                        "dangling-ref",
                        repo.package_of(doc.path),
                        repo.rel(doc.path),
                        f"`{target}` is not defined anywhere in the repo",
                        doc.line + offset,
                    )


def _is_member_of_known(target: str, dotted: set[str]) -> bool:
    """``mod.Class.attr`` where ``mod.Class`` exists: a member the AST cannot see."""
    parent = target.rsplit(".", 1)[0]
    return "." in parent and parent in dotted


def check_dangling_paths(repo: Repo) -> Iterator[Finding]:
    """Backticked file paths in prose exist somewhere they could mean."""
    heads = repo.citable_heads
    pkg_roots = [p.root for p in repo.packages]
    for doc in iter_doc_texts(repo):
        pkg_root = next((r for r in pkg_roots if doc.path.is_relative_to(r)), repo.root)
        import_dirs = [p.parent for p in pkg_root.glob("*/__init__.py")]
        for offset, line in _prose_lines(doc.text):
            for token in BACKTICK.findall(line):
                token = token.strip().rstrip(",.:;").split("::", 1)[0]
                if not token or " " in token or PLACEHOLDER.search(token):
                    continue
                if (
                    token.startswith(("/", "~", "file:", "http", "./", "../"))
                    or ".claude/worktrees" in token
                ):
                    continue
                if "/" not in token or token.startswith("."):
                    continue  # a bare filename is usually a runtime or configs-repo file
                head, _, _ = token.partition("/")
                has_ext = token.endswith(PATH_EXTS)
                if not (has_ext or head in heads):
                    continue
                bases = [doc.path.parent, pkg_root, repo.root, *pkg_roots, *import_dirs]
                bases += [
                    repo.root / n
                    for n in heads
                    if n in line and (repo.root / n).is_dir()
                ]
                if any(
                    (b / token).exists() or (b / (token + ".py")).exists()
                    for b in bases
                ):
                    continue
                if head not in heads and not any((b / head).is_dir() for b in bases):
                    continue  # the directory exists nowhere either: another repo's layout
                yield Finding(
                    "dangling-path",
                    repo.package_of(doc.path),
                    repo.rel(doc.path),
                    f"cites `{token}`, which exists nowhere",
                    doc.line + offset,
                )


def check_stale_terms(repo: Repo) -> Iterator[Finding]:
    """Retired names do not appear except in prose about their retirement."""
    for doc in iter_doc_texts(repo):
        pkg = repo.package_of(doc.path)
        lines = doc.text.splitlines()
        for offset, line in _prose_lines(doc.text):
            context = "\n".join(lines[max(0, offset - 2) : offset + 3])
            if RETIREMENT_WORDS.search(context):
                continue
            for pattern, instead, scope in STALE_TERMS:
                if scope and pkg not in scope:
                    continue
                m = re.search(pattern, line)
                if m:
                    yield Finding(
                        "stale-term",
                        pkg,
                        repo.rel(doc.path),
                        f"`{m.group(0)}` is retired; write: {instead}",
                        doc.line + offset,
                    )


def check_narrative(repo: Repo) -> Iterator[Finding]:
    """Dates, issue numbers, names and rulings in docstrings and CLAUDE.md."""
    for doc in iter_doc_texts(repo):
        if doc.kind == "markdown" and doc.path.name != "CLAUDE.md":
            continue  # READMEs and docs pages may tell a story; CLAUDE.md is doctrine
        for offset, line in _prose_lines(doc.text):
            m = NARRATIVE.search(line)
            if m:
                yield Finding(
                    "narrative",
                    repo.package_of(doc.path),
                    repo.rel(doc.path),
                    f"`{m.group(0)}`: history belongs in the changelog",
                    doc.line + offset,
                )


def check_long_docs(repo: Repo) -> Iterator[Finding]:
    """Module docstrings, def docstrings and ``#:`` blocks that became essays."""
    limits = {"module": LONG_MODULE_DOC, "def": LONG_DEF_DOC, "attr": LONG_ATTR_COMMENT}
    for doc in iter_doc_texts(repo):
        limit = limits.get(doc.kind)
        if limit is None:
            continue
        n = len(doc.text.strip().splitlines())
        if n > limit:
            yield Finding(
                "long-doc",
                repo.package_of(doc.path),
                repo.rel(doc.path),
                f"{doc.kind} docstring is {n} lines (limit {limit})",
                doc.line,
            )


def check_boilerplate(repo: Repo) -> Iterator[Finding]:
    """Template openings and hand-written member tables."""
    for doc in iter_doc_texts(repo):
        if doc.kind != "module":
            continue
        m = BOILERPLATE.search(doc.text)
        if m:
            first = m.group(0).strip().splitlines()[0]
            yield Finding(
                "boilerplate",
                repo.package_of(doc.path),
                repo.rel(doc.path),
                f"template prose: {first!r}",
                doc.line,
            )


# --------------------------------------------------------------------------- #
# Registry and report
# --------------------------------------------------------------------------- #

Check = Callable[[Repo], Iterable[Finding]]

#: name -> (check, severity). Order is report order.
CHECKS: dict[str, tuple[Check, str]] = {
    "required-docs": (check_required_docs, "hard"),
    "changelog-version": (check_changelog_version, "hard"),
    "changelog-format": (check_changelog_format, "hard"),
    "repo-map": (check_repo_map, "hard"),
    "changelog-list": (check_changelog_list, "hard"),
    "agents-shim": (check_agents_shim, "hard"),
    "nav-target": (check_nav_targets, "hard"),
    "orphan-page": (check_orphan_pages, "hard"),
    "broken-link": (check_broken_links, "hard"),
    "dangling-ref": (check_dangling_refs, "hard"),
    "dangling-path": (check_dangling_paths, "hard"),
    "stale-term": (check_stale_terms, "hard"),
    "narrative": (check_narrative, "advisory"),
    "long-doc": (check_long_docs, "advisory"),
    "boilerplate": (check_boilerplate, "advisory"),
}


def severity(check: str) -> str:
    """``hard`` or ``advisory`` for a check name."""
    return CHECKS[check][1]


def run(repo: Repo, only: set[str] | None = None) -> list[Finding]:
    """Run the selected checks and return every finding, sorted."""
    findings: list[Finding] = []
    for name, (check, _) in CHECKS.items():
        if only and name not in only:
            continue
        findings.extend(check(repo))
    return sorted(
        set(findings),
        key=lambda f: (severity(f.check) != "hard", f.check, f.path, f.line or 0),
    )


def package_stats(repo: Repo) -> dict[str, dict[str, int | str]]:
    """Per-package size figures for the report table."""
    stats: dict[str, dict[str, int | str]] = {}
    for pkg in repo.audited:
        py_files = [p for p in pkg.root.rglob("*.py") if not repo._skip(p)]
        lines = doc_lines = 0
        for path in py_files:
            source = path.read_text(errors="replace")
            lines += source.count("\n") + 1
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ):
                    d = _docstring_node(node)
                    if d is not None:
                        doc_lines += (d.end_lineno or d.lineno) - d.lineno + 1
        md_lines = sum(
            p.read_text(errors="replace").count("\n") + 1
            for p in pkg.root.rglob("*.md")
            if not repo._skip(p)
        )
        stats[pkg.name] = {
            ".py": len(py_files),
            "lines": lines,
            "doc share": f"{100 * doc_lines // lines}%" if lines else "-",
            ".md lines": md_lines,
        }
    docs = repo.root / "docs"
    if docs.is_dir() and (not repo.selected or "docs" in repo.selected):
        md_lines = sum(
            p.read_text(errors="replace").count("\n") + 1 for p in docs.rglob("*.md")
        )
        stats["docs"] = {".py": 0, "lines": 0, "doc share": "-", ".md lines": md_lines}
    return stats


def render_report(repo: Repo, findings: list[Finding], show_advisory: bool) -> str:
    """The Markdown report: a per-package table, then the findings."""
    stats = package_stats(repo)
    names = list(CHECKS)
    counts: dict[str, dict[str, int]] = {
        pkg: dict.fromkeys(names, 0) for pkg in [*stats, "root"]
    }
    for f in findings:
        counts.setdefault(f.package, dict.fromkeys(names, 0))[f.check] += 1
    active = [n for n in names if any(c[n] for c in counts.values())]
    out = ["# Documentation audit", ""]
    header = ["Package", ".py", "lines", "doc share", ".md lines", *active]
    out.append("| " + " | ".join(header) + " |")
    out.append("|---|" + "---:|" * (len(header) - 1))
    for pkg in [*stats, "root"]:
        if pkg == "root" and not any(counts["root"].values()):
            continue
        s = stats.get(pkg, {".py": "", "lines": "", "doc share": "", ".md lines": ""})
        skipped = (
            " (prose skipped)"
            if pkg in SKIP_PROSE and not repo.include_skipped and not repo.selected
            else ""
        )
        row = [
            pkg + skipped,
            s[".py"],
            s["lines"],
            s["doc share"],
            s[".md lines"],
            *(counts[pkg][n] or "" for n in active),
        ]
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    hard = [f for f in findings if severity(f.check) == "hard"]
    advisory = [f for f in findings if severity(f.check) == "advisory"]
    out += ["", f"**{len(hard)} hard finding(s), {len(advisory)} advisory.**", ""]
    if hard:
        out += ["## Hard findings", ""]
        out += [f"- {f.render()}" for f in hard]
    if advisory and show_advisory:
        out += ["", "## Advisory", ""]
        out += [f"- {f.render()}" for f in advisory]
    elif advisory:
        out += ["", "_Advisory findings are counted above; `--advisory` lists them._"]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the audit, print the report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument(
        "-p",
        "--package",
        action="append",
        help="audit only this package (repeatable; `docs` is one)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="run the prose checks on the SKIP_PROSE packages too",
    )
    parser.add_argument("--only", help="comma-separated check names (default: all)")
    parser.add_argument(
        "--strict", action="store_true", help="exit 1 if any hard finding exists"
    )
    parser.add_argument(
        "--advisory",
        action="store_true",
        help="list advisory findings, not just count them",
    )
    parser.add_argument(
        "--json",
        metavar="FILE",
        help="also write the findings as JSON (`-` for stdout only)",
    )
    parser.add_argument(
        "--list-checks", action="store_true", help="print check names and exit"
    )
    args = parser.parse_args(argv)

    if args.list_checks:
        for name, (fn, sev) in CHECKS.items():
            print(f"{name:20s} {sev:9s} {(fn.__doc__ or '').strip().splitlines()[0]}")
        return 0

    only = set(args.only.split(",")) if args.only else None
    unknown = (only or set()) - CHECKS.keys()
    if unknown:
        parser.error(f"unknown check(s): {', '.join(sorted(unknown))}")

    repo = Repo(
        args.root.resolve(), set(args.package) if args.package else None, args.all
    )
    known = {p.name for p in repo.packages} | {"docs"}
    if repo.selected and not repo.selected <= known:
        parser.error(f"unknown package(s): {', '.join(sorted(repo.selected - known))}")

    findings = run(repo, only)
    payload = {
        "hard": sum(severity(f.check) == "hard" for f in findings),
        "advisory": sum(severity(f.check) == "advisory" for f in findings),
        "findings": [asdict(f) | {"severity": severity(f.check)} for f in findings],
    }
    if args.json == "-":
        print(json.dumps(payload, indent=2))
    else:
        print(render_report(repo, findings, args.advisory), end="")
        if args.json:
            Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
    return 1 if args.strict and payload["hard"] else 0


if __name__ == "__main__":
    sys.exit(main())
