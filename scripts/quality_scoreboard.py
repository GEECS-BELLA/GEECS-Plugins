"""Quality scoreboard and ratchet: debt markers may only go down.

Prints, per package and in total, over tracked ``.py`` files (``git
ls-files``, minus the ``.pre-commit-config.yaml`` top-level exclude):

- ``stamps``: lines carrying a complexity-baseline noqa (C901, PLR0912,
  PLR0915), struck by ``ruff check --add-noqa`` when the rules landed;
- ``noqa``: every other noqa line;
- ``Any``: ``Any`` inside annotations (AST: parameters, returns, annotated
  assignments, quoted annotations by regex). Aliases built outside an
  annotation (``Json = dict[str, Any]``) and ``cast(Any, x)`` are missed;
- ``>80``: functions longer than 80 lines (``def`` line to last line);
- ``pyright``: whether the package is in ``[tool.pyright] include``.

Lines are matched as text, so a noqa spelled inside a string literal counts.

``--check`` exits 1 when the stamp or noqa total EXCEEDS
``scripts/quality_baseline.json``; a lower total asks for the baseline to
be lowered, which ``--update`` does. The root ``pyproject.toml`` counts too:
each ``per-file-ignores`` glob naming a complexity code is a stamp (the
``(pyproject)`` row); the three thresholds and ``rules_off`` (complexity
codes not selected, or ignored) are ratcheted the same way. The
doc_audit advisory count is not a column: the audit takes ~15 s and
``scripts/check.sh`` already prints it.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE = REPO_ROOT / "scripts" / "quality_baseline.json"
RATCHETED = (
    "stamps",
    "noqa",
    "max_complexity",
    "max_branches",
    "max_statements",
    "rules_off",
)
COMPLEXITY_CODES = ("C901", "PLR0912", "PLR0915")
COLUMNS = ("stamps", "noqa", "Any", ">80")
# Also the file-level forms (a comment reading ``ruff: noqa: C901`` or ``flake8: noqa``).
NOQA = re.compile(
    r"#\s*(?:(?:ruff|flake8)\s*:\s*)?noqa\b(?P<codes>:[^#\n]*)?", re.IGNORECASE
)
STAMP = re.compile(r"\b(C901|PLR0912|PLR0915)\b")
ANY_WORD = re.compile(r"\bAny\b")
LONG_FUNCTION = 80


def precommit_exclude(root: Path) -> re.Pattern[str]:
    """The top-level ``exclude:`` block of ``.pre-commit-config.yaml``."""
    lines = (root / ".pre-commit-config.yaml").read_text().splitlines()
    start = lines.index("exclude: |") + 1
    block = []
    for line in lines[start:]:
        if line and not line.startswith(" "):
            break
        block.append(line)
    return re.compile("\n".join(block), re.VERBOSE)


def tracked_files(root: Path) -> list[str]:
    """Tracked ``.py`` paths, minus the pre-commit exclude."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    exclude = precommit_exclude(root)
    return [p for p in out.split("\0") if p and not exclude.match(p)]


def package_of(root: Path, path: str) -> str:
    """The top-level package directory, or ``(repo)`` for scripts/tests/docs."""
    top = path.split("/", 1)[0]
    return (
        top if "/" in path and (root / top / "pyproject.toml").is_file() else "(repo)"
    )


def count_any(tree: ast.AST) -> int:
    """``Any`` occurrences inside annotations."""
    annotations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.arg | ast.AnnAssign) and node.annotation is not None:
            annotations.append(node.annotation)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.returns:
            annotations.append(node.returns)
    total = 0
    for annotation in annotations:
        for sub in ast.walk(annotation):
            if (
                isinstance(sub, ast.Name)
                and sub.id == "Any"
                or isinstance(sub, ast.Attribute)
                and sub.attr == "Any"
            ):
                total += 1
            elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                total += len(ANY_WORD.findall(sub.value))
    return total


def count_source(source: str) -> Counter[str]:
    """The scoreboard counts for one file's source text."""
    counts: Counter[str] = Counter()
    for line in source.splitlines():
        match = NOQA.search(line)
        if match:
            stamped = match["codes"] and STAMP.search(match["codes"])
            counts["stamps" if stamped else "noqa"] += 1
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return counts
    counts["Any"] += count_any(tree)
    counts[">80"] += sum(
        1
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
        and n.end_lineno is not None
        and n.end_lineno - n.lineno + 1 > LONG_FUNCTION
    )
    return counts


def scoreboard(root: Path) -> dict[str, Counter[str]]:
    """Counts per package."""
    board: dict[str, Counter[str]] = defaultdict(Counter)
    for path in tracked_files(root):
        if not (root / path).is_file():  # deleted from disk, not yet `git rm`ed
            continue
        source = (root / path).read_text(encoding="utf-8", errors="replace")
        board[package_of(root, path)] += count_source(source)
    ignores = pyproject_tool(root, "ruff").get("lint", {}).get("per-file-ignores", {})
    board["(pyproject)"]["stamps"] += sum(
        1 for codes in ignores.values() if STAMP.search(" ".join(codes))
    )
    return dict(board)


def pyproject_tool(root: Path, tool: str) -> dict:  # free-form TOML table
    """The root ``pyproject.toml``'s ``[tool.<tool>]`` table."""
    with open(root / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh).get("tool", {}).get(tool, {})


def pyright_packages(root: Path) -> set[str]:
    """Top-level directories named in the root ``[tool.pyright] include``."""
    include = pyproject_tool(root, "pyright").get("include", [])
    return {entry.split("/", 1)[0] for entry in include}


def render(board: dict[str, Counter[str]], pyright: set[str]) -> str:
    """The scoreboard as a fixed-width table."""
    rows = [("package", *COLUMNS, "pyright")]
    total: Counter[str] = Counter()
    for pkg in sorted(board):
        total += board[pkg]
        rows.append(
            (
                pkg,
                *(str(board[pkg][c]) for c in COLUMNS),
                "yes" if pkg in pyright else "",
            )
        )
    packages = board.keys() - {"(repo)", "(pyproject)"}
    rows.append(
        (
            "TOTAL",
            *(str(total[c]) for c in COLUMNS),
            f"{len(pyright & packages)}/{len(packages)}",
        )
    )
    width = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(
            c.ljust(w) if i == 0 else c.rjust(w)
            for i, (c, w) in enumerate(zip(r, width, strict=True))
        )
        for r in rows
    )


def ratchet(totals: dict[str, int], baseline: Path, update: bool) -> int:
    """Compare the ratcheted totals to the baseline; 1 when any exceeds it."""
    base = json.loads(baseline.read_text()) if baseline.exists() else {}
    status = 0
    for key in RATCHETED:
        allowed = base.get(key)
        if allowed is None or totals[key] > allowed:
            print(
                f"ratchet: {key} = {totals[key]} exceeds the baseline {'(none)' if allowed is None else allowed} — "
                "split the function / fix the finding instead of adding a noqa"
            )
            status = 1
        elif totals[key] < allowed:
            print(
                f"ratchet: {key} = {totals[key]} < {allowed}: lower the baseline "
                "(scripts/quality_scoreboard.py --update)"
            )
    if update:
        baseline.write_text(
            json.dumps({k: totals[k] for k in RATCHETED}, indent=2) + "\n"
        )
        print(f"ratchet: baseline rewritten ({baseline.name})")
        return 0
    return status


def main(argv: list[str] | None = None) -> int:
    """Print the scoreboard, or check/update the ratchet."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if a ratcheted count exceeds the baseline",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite the baseline to the current counts",
    )
    args = parser.parse_args(argv)
    board = scoreboard(args.root)
    totals = {k: sum(c[k] for c in board.values()) for k in ("stamps", "noqa")}
    lint = pyproject_tool(args.root, "ruff").get("lint", {})
    totals["max_complexity"] = lint.get("mccabe", {}).get("max-complexity", 10)
    pylint = lint.get("pylint", {})  # ruff's defaults for PLR0912 / PLR0915
    totals["max_branches"] = pylint.get("max-branches", 12)
    totals["max_statements"] = pylint.get("max-statements", 50)
    selected = set(lint.get("select", [])) | set(lint.get("extend-select", []))
    ignored = set(lint.get("ignore", [])) | set(lint.get("extend-ignore", []))
    totals["rules_off"] = sum(
        c not in selected or c in ignored for c in COMPLEXITY_CODES
    )
    if args.check or args.update:
        status = ratchet(
            totals, args.root / BASELINE.relative_to(REPO_ROOT), args.update
        )
        print(
            "ratchet: "
            + ", ".join(f"{k} {v}" for k, v in totals.items())
            + (" FAILED" if status else " ok")
        )
        return status
    print(render(board, pyright_packages(args.root)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
