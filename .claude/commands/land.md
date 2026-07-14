# /land — land a change as a PR, the GEECS-Plugins way

Guides a change from working tree to merged PR following this repo's
conventions. The **ritual is permanent**; the branch topology is data —
update only the "Current branch topology" section when it changes.

## Arguments

`$ARGUMENTS` — optional: a short description of the change, and/or an
explicit target base branch. If empty, infer both from the working tree
and the topology table below.

## Current branch topology (UPDATE AT M6 CUTOVER)

| Work | Base branch |
|---|---|
| Engine/backend: `GeecsBluesky`, `GeecsCAGateway`, `GEECS-Schemas`, `GEECS-Data-Utils`, `GEECS-LogTriage`, `Planning/`, repo tooling | `feat/vision-v1` |
| Console + G1 world: `GEECS-Console`, `GEECS-Scanner-GUI`, `docs/` overhaul | `feat/greenfield-epics-bluesky-gui` |
| Everything else / unclear | `feat/vision-v1` (then ask) |

`feat/greenfield-epics-bluesky-gui` is stacked on `feat/vision-v1`.
**After merging into vision-v1, roll forward**: merge `feat/vision-v1`
into greenfield (`git merge origin/feat/vision-v1 --no-edit --no-verify`
— `--no-verify` because pre-commit's auto-fixers abort merge commits) and
push. Never target `master` directly — it lags both branches until M6.

## Steps

1. **Branch**: create `<area>/<short-name>` off the correct base (see
   table) in a worktree under `.claude/worktrees/<feature-name>/`. Never
   commit to the base branches or master directly.
2. **Scope check**: one concern per PR. If the diff mixes concerns, split
   it. Flag judgment-call additions explicitly in the PR body so they are
   cheap to veto (owner preference).
3. **Version + CHANGELOG** for every package whose code changed:
   `poetry version patch|minor` from inside the package dir (patch = bug
   fix, minor = feature/behavior change), plus a Keep-a-Changelog entry
   under the new version. Docs-only changes to a package still get a
   patch bump by repo convention (see #536/#537 precedent).
4. **Tests**: run the affected package suites the way CI does (see
   `/env-doctor` for which env runs what). New behavior gets a pinning
   test; report exact counts, never "tests pass".
5. **Commit** with `./scripts/commit.sh -m "..."` after `git add` (plain
   `git commit` fights the auto-fixing pre-commit hooks). Subject shape:
   `Package X.Y.Z: what changed (#issue)`.
6. **PR**: base per the table. Body must include: what + why, a
   per-concern LOC breakdown when bundling, test counts, and — for
   anything touching scan execution or devices — a **hardware
   verification** section: either the live results or an explicit
   "owed: <what to verify, expected numbers>" so the code-complete vs
   hardware-verified distinction is never implicit.
7. **Merge**: wait for CI (`gh pr checks <n> --watch`), merge with
   `--merge --delete-branch`. Then do the roll-forward merge if the base
   was vision-v1 (step above). Remove the worktree after merge unless it
   hosts an open stacked branch.
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
