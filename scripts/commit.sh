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
existing="$(mktemp)"
trap 'rm -f "$staged" "$existing"' EXIT
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

# Re-stage exactly the originally-staged files so the fixes are included —
# but only those that still exist on disk. A staged *deletion* is in neither
# the index nor the worktree, so `git add` (even with -A) rejects its pathspec
# as fatal; and there is nothing to re-stage anyway, since no hook can have
# rewritten a file that does not exist. The deletion stays staged as-is.
while IFS= read -r -d '' path; do
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
