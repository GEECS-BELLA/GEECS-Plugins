# Contributing to GEECS-Plugins

This is the monorepo for BELLA beamline data acquisition, analysis, and
logging tooling. Each top-level directory is an independent Python package
with its own `pyproject.toml`, managed by **Poetry**. This page is the
human-facing contract; the same rules (plus deep architectural context)
live in the root and per-package `CLAUDE.md` files, which are the canonical
instructions for AI-assisted development — if you work with Claude/Codex,
those files are loaded automatically, and repo-checked skills under
`.claude/skills/` (e.g. `/land`, `/check`, `/triage`, `/scan-audit`,
`/env-doctor`, `/get-started`)
encode the recurring workflows. New to the repo entirely? Start with
[Getting started](docs/tutorials/getting_started.md) (published on the
docs site), or launch Claude Code from the repo root and type
`/get-started`.

## Setup

- Python **3.11** (`>=3.11,<3.12`) and Poetry. `poetry install` at the repo
  root builds the main dev environment; each package can also be installed
  standalone from its own directory.
- Some packages need extras for their full test suite:
  `GeecsBluesky` → `poetry install --extras "ca tiled"`;
  `GEECS-Console` → set `QT_QPA_PLATFORM=offscreen` for tests.
- Install pre-commit hooks once: `poetry run pre-commit install`.

## Branch topology (post-M6: one mainline)

This section is the **single canonical copy** of the branch layout — the
PR template and the `/land` skill point here rather than repeating it.
The M6 cutover (2026-08-20, PR #631) collapsed the two-line layout to
one mainline:

- **`master` is the mainline and the default target for every PR** —
  engine, console, gateways, schemas, analysis, docs, tooling.
- **Merges into `master` are ALWAYS performed by the human maintainer.**
  Agents prepare the PR — branch, commit, adversarial review, CI watch —
  and then hand the merge to the maintainer; they never click merge on a
  master-targeted PR. (Bulk integration merges whose constituent PRs
  were each already reviewed do not get a fresh adversarial re-review —
  say so in the PR body.)
- `dev` is **retired** — frozen at the cutover, kept only so pre-cutover
  PRs based on it don't auto-close. Do not target it or branch from it.
- The final legacy-scanner state (GEECS-Scanner-GUI, GEECS-PythonAPI) is
  preserved at the tag **`legacy-scanner-final`** — anyone still on the
  legacy line checks out the tag, never a branch.

**Personal branches for new developers.** A developer new to the repo
gets a long-lived personal integration branch off `master`, named
`users/<name>`. Their feature branches PR into that personal branch —
the `/get-started` skill sets this up, and `/land` targets it — so they
can merge their own work at their own pace. Promotion from
`users/<name>` into `master` is a separate PR that **only the
maintainer merges** (the general master-merge rule above). To keep the
personal branch from going stale, the agent merges `master` forward
*into* `users/<name>` periodically — that direction is routine
maintenance, not a mainline merge.

(Grep hits for the old branch names — Planning/ notes, CHANGELOGs — are
historical record, not instruction: leave them.)

## Planning/ is development scratch, not documentation

`Planning/` holds design notes *while the work they describe is live* —
open questions, deferred items, strategy that code and CLAUDE.md files
don't yet record. When a plan is executed (or abandoned), delete its
directory in the PR that finishes the work; anything still load-bearing
moves to the owning package's `CLAUDE.md` or the docs site first.
(Post-M6 the folder lives on the mainline like everything else — the old
"purged before reaching master" rule died with the two-branch layout;
delete-when-executed is the whole discipline. Audited 2026-07-13: five
executed/superseded plans deleted; the survivors each hold live
deferred-work or strategy content. Re-audited 2026-08-21
post-queueserver-migration: ten executed plan files deleted
(acquisition_modes, cutover 00+02, external_assets) with load-bearing
content extracted to package docs; survivors each hold live deferred
work.)

## Every PR that changes a package

1. `poetry version patch|minor` inside the package (patch = bug fix,
   minor = feature/behavior change; `1.0.0` is reserved).
2. Add a [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) entry to
   the package's `CHANGELOG.md` under the new version.
3. One concern per PR. When bundling is unavoidable, give a per-concern
   breakdown in the PR body.
4. Run `./scripts/check.sh` (it mirrors the CI env/marker mapping;
   `--all` before opening the PR). State exact test results
   ("477 passed"), and for anything touching scan
   execution or devices, fill in the **hardware verification** section of
   the PR template — either live results or an explicit "owed:" note.
   Code-complete and hardware-verified are different states here, and PRs
   are expected to say which they are.
5. **Adversarial review before merge** — this one applies to *all* PRs,
   including tooling/docs-only ones that change no package. A review by
   someone (or, for AI-assisted work, a fresh-context agent) who did not
   write the diff, covering three lenses: correctness (concrete failure
   scenarios), redundancy (does this already exist somewhere in the
   repo?), and placement (is there a more natural home, given the
   dependency graph and package boundaries?). The review report is
   posted on the PR either way — "no surviving findings" is itself the
   record — and each finding is dispositioned — fixed (and confirmed by
   the reviewer), or waived with a stated reason — before merge. The
   reviewer brief lives in `.claude/skills/land/SKILL.md`.

## Committing

Use `./scripts/commit.sh -m "..."` after `git add` — the pre-commit
auto-fixers (ruff, ruff-format) rewrite files mid-commit and abort a plain
`git commit`; the helper applies fixes, re-stages, and commits in one shot.
Style: NumPy docstrings, type hints on public functions, Pydantic v2
(`model_validate`/`model_dump`, never `.dict()`/`.parse_obj()`).

## Rules with incident history (do not relearn these live)

- **Analysis code never creates `scans/ScanNNN/` folders.** Only the
  scanner side (`claim_scan_number` in GeecsBluesky) brings scan folders
  into existence. Auto-creating an "apparently missing" folder has
  orphaned real data in production. Pinned by tests; details in the root
  `CLAUDE.md` ("Cross-package invariants").
- **This repo is public.** No lab account names, hostnames, or user home
  paths in committed files (generic placeholders instead); internal
  `192.168.6.x` addresses are accepted practice.
- **Contract files travel with behavior**: gateway-visible changes update
  `GeecsCAGateway/PV_CONTRACT.md` + its pinned test in the same PR;
  event-data changes update `GeecsBluesky/EVENT_SCHEMA.md`.
- The legacy packages (`GEECS-PythonAPI`, `GEECS-Scanner-GUI`) are deleted
  (2026-08-20); their final state is preserved at the tag
  `legacy-scanner-final`. Successors: `geecs_core.client.GeecsDevice` and
  GEECS-Console.

## Tests

CI (`.github/workflows/unit-tests.yml`) runs: root `tests/`, ImageAnalysis,
ScanAnalysis, GEECS-Data-Utils, GEECS-Schemas from the **root env** and
GeecsBluesky from its **own env** (Ubuntu); on the greenfield branch a
second job runs the GEECS-Console suite from its own env on **Windows**
(control-room machines run Windows). The GeecsCAGateway and
GEECS-LogTriage suites are not in CI — run them locally when touching
those packages. Everything is hermetic — no lab network, no hardware.
`integration`-marked tests need the lab and are deselected by default;
never run the top-level hardware scripts without lab access and operator
coordination.

## Where to learn the architecture

Start with the root `CLAUDE.md` (repository map, dependency graph,
invariants), then the `CLAUDE.md` of the package you're touching. The
published docs site (`docs/`, MkDocs) is the user-facing counterpart.
