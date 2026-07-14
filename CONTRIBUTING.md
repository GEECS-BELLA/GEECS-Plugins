# Contributing to GEECS-Plugins

This is the monorepo for BELLA beamline data acquisition, analysis, and
logging tooling. Each top-level directory is an independent Python package
with its own `pyproject.toml`, managed by **Poetry**. This page is the
human-facing contract; the same rules (plus deep architectural context)
live in the root and per-package `CLAUDE.md` files, which are the canonical
instructions for AI-assisted development — if you work with Claude/Codex,
those files are loaded automatically, and repo-checked skills under
`.claude/skills/` (e.g. `/land`, `/check`, `/triage`, `/scan-audit`,
`/env-doctor`)
encode the recurring workflows.

## Setup

- Python **3.11** (`>=3.11,<3.12`) and Poetry. `poetry install` at the repo
  root builds the main dev environment; each package can also be installed
  standalone from its own directory.
- Some packages need extras for their full test suite:
  `GeecsBluesky` → `poetry install --extras "ca tiled"`;
  `GEECS-Console` → set `QT_QPA_PLATFORM=offscreen` for tests.
- Install pre-commit hooks once: `poetry run pre-commit install`.

## Branch topology (until the M6 cutover)

This section is the **single canonical copy** of the branch layout — the
PR template and the `/land` skill point here rather than repeating it, so
at M6 only this section (and the pointers' one-line reminders) needs
editing. Two lines (collapsed from three on 2026-07-13 — `feat/vision-v1`
was retired into `dev`, and `feat/greenfield-epics-bluesky-gui` was
renamed `dev`):

- `dev` — the vision world, and the default target for development:
  GeecsBluesky, GeecsCAGateway, GEECS-Schemas, GEECS-Console, the scan
  browser, Planning docs, repo tooling, the docs site.
- `master` — the legacy-scanner line, kept deployable through the M6 gap:
  the pure-legacy GEECS-Scanner-GUI plus **living analysis development**
  (ImageAnalysis, ScanAnalysis, and their data-utils needs). Target
  `master` for analysis work *unless it imports something that only
  exists on `dev`* (e.g. `geecs_data_utils.tiled_catalog`) — then target
  `dev`.

`master` is merged forward into `dev` periodically, so analysis work
flows into the vision world automatically; nothing merges the other way
until the M6 cutover (tag `master` first, then merge `dev` in). (At M6
this section collapses to "everything targets `master`" — also prune the
branch names from `pick_base()` in `scripts/check.sh` then; harmless if
forgotten, it skips deleted branches and falls back to the default
branch. Grep hits for the *old* branch names — Planning/ notes,
CHANGELOGs — are historical record, not instruction: leave them.)

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
- `GEECS-PythonAPI` is being refactored elsewhere — no new features there.

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
