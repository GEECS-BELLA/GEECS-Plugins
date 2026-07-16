---
name: land
description: >
  Land a change as a PR the GEECS-Plugins way: branch off the right base,
  scope check, version bump + CHANGELOG, tests run the way CI runs them,
  commit.sh, PR body with the hardware-verification section, adversarial
  review by a fresh-context subagent (correctness + redundancy +
  placement) with dispositioned findings, CI watch, merge, roll-forward.
  Use when the user says "land this", "open a PR",
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
change (`dev` for vision-world work; `master` only for analysis/legacy
work with no dev-only imports).

**Roll-forward rule (until M6):** after merging an analysis/legacy PR
into `master`, merge `master` forward into `dev` (`git merge
origin/master --no-edit --no-verify` — `--no-verify` because pre-commit's
auto-fixers abort merge commits) and push. Nothing merges `dev` →
`master` until the M6 cutover.

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
   hardware-verified distinction is never implicit.
7. **Adversarial review** (runs during the CI wait; see the section
   below): spawn a fresh-context reviewer subagent on the PR's diff,
   post its surviving findings as a PR comment, and disposition every
   finding — fix it, or waive it with a stated reason — before merging.
   No PR merges with an undispositioned finding.
8. **Merge**: wait for CI (`gh pr checks <n> --watch`), merge with
   `--merge --delete-branch`. Then do the roll-forward merge if the base
   was the engine branch (rule above). Remove the worktree after merge
   unless it hosts an open stacked branch — and always confirm with the
   user before removing any worktree.
9. **Stacked PRs**: base on the parent PR's branch and say "merge #N
   first" in the body — but beware: GitHub auto-closes (unreopenable) a
   PR whose base branch is deleted. Prefer merging the parent first and
   retargeting before it merges, or re-file (precedent: #538 → #539).

## The adversarial review step

The author must not be the only reader of a diff before it lands. The
reviewer is a **separate subagent with fresh context**: give it the PR
number (or the diff), the branch, and the brief below — never your own
rationale, summaries, or the PR body's framing beyond what any outside
reviewer would see. Its job is to refute the change, not to appreciate
it.

The reviewer brief (pass verbatim, plus the PR/diff reference):

> Review this diff adversarially through three lenses, using the whole
> repository as context (root and package `CLAUDE.md` files, the
> dependency graph, existing modules and tests):
>
> 1. **Correctness** — try to construct concrete failure scenarios
>    (inputs/state → wrong behavior). Check edge cases the tests skip,
>    invariant violations (scan-folder creation, public-repo hygiene,
>    contract files traveling with behavior), and whether new tests
>    would actually fail if the code were wrong.
> 2. **Redundancy** — does some or all of this already exist? Search
>    the repo for existing implementations, helpers, or patterns that
>    overlap what the diff adds (the repo has grown duplicate config
>    readers, worker classes, and prompt dialogs before — assume
>    duplication until a search says otherwise, and cite the files you
>    checked).
> 3. **Placement** — is there a more natural home for each new piece,
>    given the package dependency graph and each package's stated
>    boundaries (e.g. pure logic belongs below GUI packages; schema
>    knowledge in its one module; browser/console kit boundaries)?
>    Flag code that a future feature would have to move.
>
> Report only findings that survive your own attempt to refute them,
> ranked by severity, each with the concrete scenario or the existing
> file it duplicates / the better home. Say "no surviving findings"
> if that is the honest result — do not pad.

Disposition, in a PR comment before merge: each finding is either
**fixed** (commit referenced) or **waived** with a one-line reason.
Waivers are cheap-to-veto flags for the owner, same as judgment-call
additions. If a finding invalidates the approach, stop and surface it
instead of merging.

## Hard rules (violations have shipped incidents)

- Analysis-side code never creates `scans/ScanNNN/` folders (root
  CLAUDE.md invariant — read it before touching anything path-shaped).
- Public repo: no lab account names, hostnames, or user home paths in
  committed files; internal `192.168.6.x` IPs are accepted practice.
- Behavior changes touching the gateway's externally observable surface
  update `PV_CONTRACT.md` + its pinned test in the same PR; event-schema
  changes update `EVENT_SCHEMA.md` (additive column conventions do not
  bump the schema version; field/semantics changes do).
