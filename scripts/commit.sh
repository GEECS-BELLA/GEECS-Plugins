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

# The files staged for this commit, recorded NUL-delimited in a temp file
# (space-safe; bash variables cannot hold NUL, so a file — not a var).
staged="$(mktemp)"
trap 'rm -f "$staged"' EXIT
git diff --cached --name-only -z >"$staged"
if [ ! -s "$staged" ]; then
    echo "commit.sh: nothing staged — 'git add' your changes first." >&2
    exit 1
fi

# Prefer a pre-commit on PATH; fall back to the Poetry environment.
if command -v pre-commit >/dev/null 2>&1; then
    pc() { pre-commit "$@"; }
else
    pc() { poetry run pre-commit "$@"; }
fi

# Apply auto-fixes to the staged set (pre-commit defaults to the staged files).
# It exits non-zero when it rewrites a file — expected here, so don't abort.
pc run || true

# Re-stage exactly the originally-staged files so the fixes are included.
# `xargs -0 ... < file` is portable across BSD (macOS) and GNU xargs.
xargs -0 git add -- <"$staged"

# Commit. The commit-time hook run is now clean, so this passes on the first try.
git commit "$@"
