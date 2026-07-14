---
name: land
description: >
  Land a change as a PR the GEECS-Plugins way: branch off the right base,
  scope check, version bump + CHANGELOG, tests run the way CI runs them,
  commit.sh, PR body with the hardware-verification section, CI watch,
  merge, roll-forward. Use when the user says "land this", "open a PR",
  "commit and push this", "prep this branch for review", or when a change
  in the working tree is finished and needs to become a merged PR.
---

# /land — land a change as a PR, the GEECS-Plugins way

Guides a change from working tree to merged PR following this repo's
conventions. The ritual below is permanent; anything tied to the current
branch layout lives in `CONTRIBUTING.md` and is read at run time.

## Arguments

`$ARGUMENTS` — optional: a short description of the change, and/or an
explicit target base branch. If empty, infer both from the working tree
and the topology in `CONTRIBUTING.md`.

## Pick the base branch

The branch topology (which work targets which base, until the M6 cutover)
lives in **CONTRIBUTING.md § "Branch topology"** — that is the single
canonical copy; read it now and pick the base matching the content of the
change. Never target `master` directly — it lags the active branches
until M6.

**Roll-forward rule (until M6):** the greenfield GUI branch is stacked on
the engine branch. After merging a PR into the engine branch, merge it
forward into greenfield (`git merge origin/<engine-branch> --no-edit
--no-verify` — `--no-verify` because pre-commit's auto-fixers abort merge
commits) and push.

## Steps

1. **Branch**: create `<area>/<short-name>` off the correct base (per
   CONTRIBUTING.md) in a worktree under `.claude/worktrees/<feature-name>/`.
   Never commit to the base branches or master directly.
2. **Scope check**: one concern per PR. If the diff mixes concerns, split
   it. Flag judgment-call additions explicitly in the PR body so they are
   cheap to veto (owner preference).
3. **Version + CHANGELOG** for every package whose code changed:
   `poetry version patch|minor` from inside the package dir (patch = bug
   fix, minor = feature/behavior change), plus a Keep-a-Changelog entry
   under the new version. Docs-only changes to a package still get a
   patch bump by repo convention (see #536/#537 precedent).
4. **Tests**: `./scripts/check.sh` runs the affected suites the way CI
   does (`--all` before opening the PR; see `/check` for tiers and
   `/env-doctor` for env fixes). New behavior gets a pinning test;
   report exact counts, never "tests pass".
5. **Commit** with `./scripts/commit.sh -m "..."` after `git add` (plain
   `git commit` fights the auto-fixing pre-commit hooks). Subject shape:
   `Package X.Y.Z: what changed (#issue)`.
6. **PR**: base per CONTRIBUTING.md. Body must include: what + why, a
   per-concern LOC breakdown when bundling, test counts, and — for
   anything touching scan execution or devices — a **hardware
   verification** section: either the live results or an explicit
   "OWED: <what to verify, expected numbers>" so the code-complete vs
   hardware-verified distinction is never implicit. OWED entries are
   picked up later by `/lab-day` — write them so a future session can
   run them without this PR's context.
7. **Merge**: wait for CI (`gh pr checks <n> --watch`), merge with
   `--merge --delete-branch`. Then do the roll-forward merge if the base
   was the engine branch (rule above). Remove the worktree after merge
   unless it hosts an open stacked branch — and always confirm with the
   user before removing any worktree.
8. **Stacked PRs**: base on the parent PR's branch and say "merge #N
   first" in the body — but beware: GitHub auto-closes (unreopenable) a
   PR whose base branch is deleted. Prefer merging the parent first and
   retargeting before it merges, or re-file (precedent: #538 → #539).

## Hard rules (violations have shipped incidents)

- Analysis-side code never creates `scans/ScanNNN/` folders (root
  CLAUDE.md invariant — read it before touching anything path-shaped).
- Public repo: no lab account names, hostnames, or user home paths in
  committed files; internal `192.168.6.x` IPs are accepted practice.
- Behavior changes touching the gateway's externally observable surface
  update `PV_CONTRACT.md` + its pinned test in the same PR; event-schema
  changes update `EVENT_SCHEMA.md` (additive column conventions do not
  bump the schema version; field/semantics changes do).
