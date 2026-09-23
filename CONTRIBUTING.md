# Contributing to GEECS-Plugins

This is the monorepo for BELLA beamline data acquisition, analysis, and
logging tooling. Each top-level directory is an independent Python package
with its own `pyproject.toml`, managed by **Poetry**. This page is the
human-facing contract; the same rules (plus deep architectural context)
live in the root and per-package `CLAUDE.md` files, which are the canonical
instructions for AI-assisted development — if you work with Claude/Codex,
those files are loaded automatically, and repo-checked skills under
`.claude/skills/` (e.g. `/land`, `/check`, `/triage`, `/scan-audit`,
`/env-doctor`, `/get-started`, `/lab-status`, `/fleet-status`)
encode the recurring workflows. New to the repo entirely? Start with
[Getting started](docs/tutorials/getting_started.md) (published on the
docs site), or launch Claude Code from the repo root and type
`/get-started`.

## Setup

- Python **3.11** (`>=3.11,<3.12`) and Poetry. `poetry install` at the repo
  root builds the main dev environment; each package can also be installed
  standalone from its own directory.
- Some packages need extras for their full test suite:
  `GeecsBluesky` → `poetry install --extras "ca tiled"`.
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
moves to the owning package's `CLAUDE.md` or the docs site first. The
folder's own `README.md` states the discipline and survives every audit
(git does not track empty directories, so pruning everything would take
the convention with it).

**Docstrings state the rule, not its provenance.** A docstring says what
the invariant *is*, in a sentence that stands on its own — never
`see Planning/x.md §4.4`. A reader chasing that pointer must either spend
a read on a several-hundred-line design doc or risk missing the
constraint, and they pay that tax on every future read. The three homes
for "why": the invariant goes in the docstring, the derivation stays in
git history (deleting a plan does not destroy it, and a `git log`
reference cannot dangle the way a path can), and a rule that binds more
than one package goes in a `CLAUDE.md`.

Audits (post-M6 the folder lives on the mainline like everything else —
the old "purged before reaching master" rule died with the two-branch
layout; delete-when-executed is the whole discipline):

- **2026-07-13** — five executed/superseded plans deleted; the survivors
  each hold live deferred-work or strategy content.
- **2026-08-21**, post-queueserver-migration — ten executed plan files
  deleted (acquisition_modes, cutover 00+02, external_assets) with
  load-bearing content extracted to package docs.
- **2026-09-16**, post-native-Bluesky — all 32 remaining files deleted
  across seven directories: the arcs they planned are on `master`
  (native_bluesky, data_portal), self-declared superseded
  (data_capture), or stale against deleted code (device_read_path,
  schema_refactor, cutover_strategy). Load-bearing content moved first —
  the DB-metadata join and `.DESC` rules to `GEECS-Core/CLAUDE.md`, the
  reader-side HDF5-over-SMB rules to `GEECS-Data-Utils/CLAUDE.md`, the
  two genuinely open items to issues #929 and #930 — and ~140 `Planning/`
  provenance citations were stripped out of docstrings rather than
  repointed, which is what the docstring rule above now forbids
- **2026-09-22**, post-non-scalar-over-PVA — the last `Planning/` file
  deleted (`data_capture/02_nonscalar_over_pva.md`, written after the
  2026-09-16 prune and finished by #950). `Planning/` is now its `README.md`
  alone. Every durable rule it carried was already where it is enforced
  rather than described: the declaration table and both exclusion traps
  are `geecs_core.db.device_streams` itself, the byte-exact waveform wire
  format is the decoder's docstring in `geecs_data_utils.io.arrays`, the
  three-kinds-of-stack table and the un-padding rule are
  `GEECS-Data-Utils/CLAUDE.md`, the DB-sourced capture gate is
  `GeecsBluesky/CLAUDE.md`, and what the DB's `set` / `defaultvalue` /
  `get` columns actually mean is `GEECS-Core/DESIGN.md`. The owner ruled
  the one remaining candidate — the MagSpec resampling note — not worth
  keeping, since analysis handles it (2026-09-22). The folder keeps its
  `README.md` so the convention survives the emptying — git does not track
  empty directories, so pruning everything would take the discipline with
  it. CHANGELOG citations were left dangling on purpose: like the dead
  branch names, they are historical record, not instruction.

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
`poetry.lock` files ride the same path: Poetry writes the locking
checkout's absolute path into path-dependency extras, and the
`scrub-lock-paths` hook normalizes it to `file:///GEECS-Plugins/…` — never
hand-edit those URLs to bare names or `file:../` (`file:../` installs
nothing; the bare form is the one whose PyPI fallthrough #753 observed in
its PEP 621 experiment — root `CLAUDE.md` § Agent & Worktree Policy has
the precise statement).
Style: NumPy docstrings, type hints on public functions, Pydantic v2
(`model_validate`/`model_dump`, never `.dict()`/`.parse_obj()`).

## Rules with incident history (do not relearn these live)

- **Analysis code never creates `scans/ScanNNN/` folders.** Only the
  scanner side (`claim_scan_number` in GeecsBluesky) brings scan folders
  into existence. Auto-creating an "apparently missing" folder has
  orphaned real data in production. Pinned by tests; details in the root
  `CLAUDE.md` ("Cross-package invariants").
- **Facility values have one home** — `config.ini` client-side,
  `/etc/geecs/site.env` host-side; committed files carry lab addresses,
  experiment names, account paths, and timezones only as examples or
  placeholders (`docs/platform/site_profile.md`).
- **This repo is public.** No lab account names, hostnames, or user home
  paths in committed files (generic placeholders instead); internal
  `192.168.6.x` addresses are accepted practice.
- **Contract files travel with behavior**: gateway-visible changes update
  `GeecsCAGateway/PV_CONTRACT.md` + its pinned test in the same PR;
  event-data changes update `GeecsBluesky/EVENT_SCHEMA.md`.
- The legacy packages (`GEECS-PythonAPI`, `GEECS-Scanner-GUI`) are deleted
  (2026-08-20); their final state is preserved at the tag
  `legacy-scanner-final`. Successors: `geecs_core.client.GeecsDevice` and
  GeecsScanner + GeecsBluesky. The PySide6 `GEECS-Console` that sat between
  them was deleted 2026-09-14 (final state at the tag `geecs-console-v0.32.1-final`); the web
  scanner (`GeecsScanner`) is the operator front end.

## Tests

CI (`.github/workflows/unit-tests.yml`, one Ubuntu job) runs: root
`tests/`, ImageAnalysis, ScanAnalysis, GEECS-Data-Utils and GEECS-Schemas
from the **root env**, and GeecsWebTheme, GeecsLogbook, GeecsScanner,
GEECS-DataPortal, GEECS-Core, GeecsCAGateway, GeecsBluesky (pure unit
tests, `qs-client` extra included), GEECS-MCP and GeecsPvaGateway each
from its **own env**. Not in CI — run locally via `scripts/check.sh` when
touching it: GEECS-LogTriage. Everything is hermetic — no lab network, no
hardware.
`integration`-marked tests need the lab and are deselected by default;
never run the top-level hardware scripts without lab access and operator
coordination.

## Where to learn the architecture

Start with the root `CLAUDE.md` (repository map, dependency graph,
invariants), then the `CLAUDE.md` of the package you're touching. The
published docs site (`docs/`, MkDocs) is the user-facing counterpart.
