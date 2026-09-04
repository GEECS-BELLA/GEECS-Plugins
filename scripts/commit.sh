#!/usr/bin/env bash
#
# commit.sh — commit with pre-commit's auto-fixes applied *first*.
#
# Why this exists
# ---------------
# pre-commit's auto-fixing hooks (ruff-format, ruff --fix, trailing-whitespace,
# end-of-file-fixer) rewrite files during the commit-time hook run. When they
# do, pre-commit *fails the commit* ("files were modified by this hook") so you
# re-stage the fixes and try again. On a merge commit it goes further: pre-commit
# stashes unstaged changes, a hook rewrites a staged file, and restoring the
# stash conflicts — the commit aborts, often silently if you only skim the tail
# of the output. This bites hardest in many-file / merge / agent-driven commits.
#
# What it does
# ------------
# Runs pre-commit on the staged files FIRST (applying auto-fixes), re-stages
# exactly those files, then commits. The commit-time hook run is then a clean
# no-op — the fixers are idempotent — so the commit succeeds on the first try,
# with the check-* hooks (ast, yaml, merge-conflict, no-commit-to-branch, …)
# still enforced as a final gate. All arguments pass through to `git commit`.
#
# Usage
# -----
#   git add <files>
#   scripts/commit.sh -m "your message"      # any `git commit` args work
#
# It does NOT stage anything you did not `git add` yourself — it only re-stages
# the files that were already staged (capturing the auto-fixes to them).
set -euo pipefail

# Run from the repo root, wherever the script was invoked from. This pins
# `poetry` below to the ROOT env (a package cwd would resolve that package's
# env instead), and makes the root-relative paths that `git diff --cached
# --name-status` emits line up with the `git add` in the re-staging loop —
# which silently re-staged nothing when invoked from a subdirectory.
# Consequence: any pathspec args you pass through to `git commit` are
# interpreted relative to the repo root, not your shell's cwd.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# The files staged for this commit, recorded NUL-delimited in a temp file
# as <status>\0<path>\0 pairs (space-safe; bash variables cannot hold NUL, so
# a file — not a var). --no-renames keeps every record a pair: a detected
# rename would otherwise emit three fields (R<score>, src, dst), and
# splitting it into D src + A dst is exactly the shape the loop below wants.
staged="$(mktemp)"
existing="$(mktemp)"
trap 'rm -f "$staged" "$existing"' EXIT
git diff --cached --name-status --no-renames -z >"$staged"
if [ ! -s "$staged" ]; then
    echo "commit.sh: nothing staged — 'git add' your changes first." >&2
    exit 1
fi

# Prefer the root Poetry env's pre-commit over one on PATH (same runner
# preference as scripts/check.sh). The config's `language: system` hooks
# (jupyter-notebook-clear-output) resolve their executable from the invoking
# environment: the root env has jupyter, a system pre-commit typically does
# not. The env must be probed for its OWN bin/pre-commit — `poetry run
# pre-commit --version` would lie, because `poetry run` falls through to the
# caller's PATH for binaries the env lacks. Machines without a usable root
# env fall back to a PATH pre-commit as before.
ROOT_ENV=""
JUPYTER_OK=1
if command -v poetry >/dev/null 2>&1; then
    ROOT_ENV="$(poetry env info --path 2>/dev/null || true)"
fi
if [ -n "$ROOT_ENV" ] && [ -x "$ROOT_ENV/bin/pre-commit" ]; then
    pc() { poetry run pre-commit "$@"; }
    # `poetry run` PATH = env bin + caller PATH, so either may supply jupyter.
    if [ ! -x "$ROOT_ENV/bin/jupyter" ] && ! command -v jupyter >/dev/null 2>&1; then
        JUPYTER_OK=0
    fi
elif command -v pre-commit >/dev/null 2>&1; then
    pc() { pre-commit "$@"; }
    command -v jupyter >/dev/null 2>&1 || JUPYTER_OK=0
else
    echo "commit.sh: no pre-commit found (neither the root poetry env nor PATH) — run 'poetry install' at the repo root" >&2
    exit 1
fi
# Whichever runner won: if it cannot find jupyter, the notebook-clearing hook
# dies with "Executable `jupyter` not found" on every staged .ipynb — a
# phantom failure unrelated to the change. Skip that one hook loudly instead
# (the exported SKIP also covers the commit-time hook run below); CI's
# pre-commit workflow still enforces it.
if [ "$JUPYTER_OK" -eq 0 ]; then
    export SKIP="${SKIP:+$SKIP,}jupyter-notebook-clear-output"
    echo "commit.sh: effective pre-commit cannot find 'jupyter' — skipping jupyter-notebook-clear-output (CI still runs it)" >&2
fi

# Apply auto-fixes to the staged set (pre-commit defaults to the staged files).
# It exits non-zero when it rewrites a file — expected here, so don't abort.
pc run || true

# Re-stage exactly the originally-staged files so the fixes are included —
# except staged *deletions* (index status D), which have nothing to re-stage:
# no hook can have rewritten a file the commit removes, and the deletion stays
# staged as-is. Filtering on the index status rather than on-disk existence
# matters for `git rm --cached` (untracking a file that stays on disk, now
# matched by .gitignore): the path exists, so an existence test would pass it
# to `git add`, which refuses ignored paths ("Use -f if you really want to add
# them") and, under set -e, aborted the script before `git commit` (#761).
# Worse, when the untracked file was NOT ignored, `git add` silently re-added
# it: the commit went through with the user's staged untracking reverted.
# The existence test stays as a second guard: a staged add/modify whose file
# was since removed from the worktree would otherwise be turned into a
# deletion by `git add` — something the user never staged.
while IFS= read -r -d '' status && IFS= read -r -d '' path; do
    case "$status" in D*) continue ;; esac
    if [ -e "$path" ] || [ -L "$path" ]; then
        printf '%s\0' "$path"
    fi
done <"$staged" >"$existing"

# Skip when everything staged was a deletion (empty input would make GNU
# xargs still run `git add --` once, and BSD xargs skip it — moot it).
# `xargs -0 ... < file` is portable across BSD (macOS) and GNU xargs.
if [ -s "$existing" ]; then
    xargs -0 git add -- <"$existing"
fi

# Commit. The commit-time hook run is now clean, so this passes on the first try.
git commit "$@"
