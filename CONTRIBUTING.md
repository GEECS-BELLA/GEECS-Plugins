# Contributing to GEECS-Plugins

This is the monorepo for BELLA beamline data acquisition, analysis, and
logging tooling. Each top-level directory is an independent Python package
with its own `pyproject.toml`, managed by **Poetry**. This page is the
human-facing contract and the one home of the team's working norms
(review, PRs, GitHub conduct, what to delete) for every GEECS-BELLA repo:
the private repos (geecs-epics, the `ioc-*` repos) point here rather than
restating them, and state only what differs there (geecs-epics#124 makes
it so). The root and per-package `CLAUDE.md` files hold each repo's own
rules and the deep architectural context; they are the canonical
instructions for AI-assisted development, loaded automatically when you
work with Claude/Codex, and the root `CLAUDE.md` imports this page so the
norms load with them. Repo-checked skills under `.claude/skills/` (e.g.
`/land`, `/check`, `/triage`, `/scan-audit`, `/env-doctor`,
`/get-started`, `/lab-status`, `/fleet-status`, `/package-tidy`) encode
the recurring workflows. New to the repo entirely? Start with
[Getting started](docs/tutorials/getting_started.md) (published on the
docs site), or launch Claude Code from the repo root and type
`/get-started`.

## Setup

- Python **3.11** (`>=3.11,<3.12`; GeecsCanopy, GEECS-Core and GEECS-Schemas
  allow `<3.13` for Canopy's 3.12 image) and Poetry. `poetry install` at the repo
  root builds the main dev environment; each package can also be installed
  standalone from its own directory.
- Some packages need extras for their full test suite:
  `GeecsBluesky` → `poetry install --extras "ca tiled"`.
- Install pre-commit hooks once: `poetry run pre-commit install`.
  Optional, faster: [prek](https://github.com/j178/prek), a drop-in
  reimplementation that reads the same `.pre-commit-config.yaml`
  (with uv: `uvx prek run --all-files`, or `uv tool install prek`
  then `prek install` in place of the line above). Trialled 2026-11-03 with prek 0.5.10: every hook passes unchanged,
  the local ones (`scrub-lock-paths`, `poetry-lock-no-worktree-paths`,
  `import-contracts`) included; a warm all-files run took 3.8 s against
  pre-commit's 4.7 s. pre-commit stays the reference — CI and
  `scripts/commit.sh` run it — so a hook that behaves differently under
  prek is a prek bug, not a config change.

- The private facility repo's map (`geecs-epics/docs/index.md`: which page
  owns which host, device or incident fact) can load into your sessions
  here. Project-scoped: an untracked `CLAUDE.local.md` beside this file
  with the line `@../geecs-epics/docs/index.md` (the path to your clone;
  a worktree under `.claude/worktrees/` has its own file, so there use an
  absolute path); Claude Code asks once per checkout, in an interactive
  session, and loads nothing until approved — a headless session cannot approve it, and a
  missing target is silent (observed, not documented). Machine-wide
  instead: the same line in `~/.claude/CLAUDE.md` loads with no prompt in
  every session on the machine, this public repo's included, so remember
  what it carries (desktop Cowork sessions skip user-scope imports outside
  their working directory). Either way, verify once with `/context all`.

## Branch topology (post-M6: one mainline)

This section is the **single canonical copy** of the branch layout — the
PR template and the `/land` skill point here rather than repeating it.
The M6 cutover (2026-08-20, PR #631) collapsed the two-line layout to
one mainline:

- **`master` is the mainline and the default target for every PR** —
  engine, console, gateways, schemas, analysis, docs, tooling.
- **Merges into `master` are ALWAYS performed by the human maintainer.**
  Agents prepare the PR — branch, commit, review (step 4's tier), CI watch —
  and then hand the merge to the maintainer; they never click merge on a
  master-targeted PR. (Bulk integration merges whose constituent PRs
  were each already reviewed do not get a fresh re-review — say so in
  the PR body, per § "Every PR that changes a package" step 4.)
- **The one bot exception:** Renovate (`renovate.json`) merges its own
  GitHub Actions and pre-commit hook-pin PRs (ruff excepted) into
  `master` once every check is green, without a review comment — a
  category the maintainer approved on 2026-11-03. Runtime dependencies,
  lock files and container images are never automerged; they are his
  merge like any other PR.
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
- **2026-11-03**, consolidation day — five of seven files deleted:
  analysis_refactor.md (its own condition met: the old analysis cores
  went with ImageAnalysis in #1039), geecs_canopy.md (Canopy adopted,
  pushes 1–3 on master and deployed), epics_estate.md (the web estate
  it planned was superseded by Canopy; tag `web-estate-final`),
  device_panels.md (superseded by `device_pages.md`) and
  geecs_plotter.md (the arc moved into GEECS-DataPortal). The rules
  other files cited by section number (the bridge's Appendix A, the
  Canopy adoption conditions, the run lock, the recipe vocabulary) were
  inlined into `GeecsCanopy/CLAUDE.md`, `GeecsControlRoom/CLAUDE.md`,
  `GEECS-Analysis/CLAUDE.md` and the docstrings that keep them; the open
  items were listed in the PR body rather than filed. Kept:
  `operator_surfaces.md` (§11 decisions open) and `device_pages.md` (arc
  continues).
- **2026-11-04**, raw-frames arc (the camera page's Raw view, slices 0,
  R1–R3 on `feat/raw-frames`) — `raw_frames.md` deleted in its last slice,
  ahead of the promotion PR. Its contract and rules moved to
  `GeecsControlRoom/CLAUDE.md` § "The raw frame" (the source and the
  transform rule, the byte contract, the read discipline, the per-mode
  levels, the rules with rows), the raw roles and why `read: never` /
  `on_request` to `GEECS-Schemas/CLAUDE.md`, the bandwidth, gzip and image
  rebuild notes to `GeecsControlRoom/README.md` § Deploy, and the operator's
  view to `docs/web_services/control_room.md` § "Raw view". The brief's
  § Status (the red team's dispositions) is in #1271's body; the hardware
  proof is owed after the deploy.
- **2026-11-13**, device-pages arc (GeecsControlRoom's device pages,
  slices 1–7 on `master`: the picker and the camera, trace and settables
  pages, the home's device directory, GeecsPanels deleted, the vocabulary
  merge) — `device_pages.md` deleted in slice 7, and
  `operator_surfaces.md` §6 folded into `GeecsControlRoom/CLAUDE.md`
  § "The device pages" (its number kept as a one-line stub: the file's
  sections are cited by number). Most rules had moved in their own PRs
  (§ "The settables page", "The setpoint control", "The trace", the raw
  frame and the JPEG; the panel fields in GEECS-Schemas' `epics/family.py`
  docstrings; the Plotly rule in `GeecsWebTheme/CLAUDE.md`); this PR
  moved the rest: the kind table, the picker, the resolution panes,
  `OWN_ROLES`, what a page never computes and the restart rule to
  § "The device pages", and the screen-schema deploy order to
  `GeecsControlRoom/README.md` § Deploy. The brief's § Status is in this
  PR's body. Kept: `operator_surfaces.md` (§11 open) and `configurator.md`
  (arc live).
- **2026-11-13**, shot-synchronization arc (#1278, on `feat/shot-sync`) —
  `shot_synchronization.md` deleted ahead of the promotion PR (#1296). The
  model was already where it is enforced: the strict slot, fly mode and
  the witness in `GeecsBluesky/CLAUDE.md` § "Acquisition", the shot table
  and its flags in `GeecsBluesky/EVENT_SCHEMA.md`, the native stream path
  in `NATIVE_DEVICES.md`, the join's rules in the
  `geecs_data_utils.shot_table` docstrings (the clock-step audit's limits
  among them). What gated keeps alive until it is retired, and what goes
  when the last LabVIEW device migrates, moved to `GeecsBluesky/CLAUDE.md`
  ("Gated is frozen"). The open questions are on #1278.

## Every PR that changes a package

1. **No version bump, no CHANGELOG entry** — a package is bumped when it
   is deployed or tagged, not per PR (root `CLAUDE.md` § "Release &
   Versioning"). The release that bumps it runs `poetry version
   patch|minor` (patch = bug fix, minor = feature/behavior change; `1.0.0`
   is reserved) and drafts its [Keep a
   Changelog](https://keepachangelog.com/en/1.1.0/) entry from the merged
   PR titles with `scripts/release_notes.py <Package>`. Until then a PR is
   identified by its merge commit. So write PR titles that read well as a
   changelog bullet.
2. One concern per PR. When bundling is unavoidable, give a per-concern
   breakdown in the PR body.
3. Run `./scripts/check.sh` (it mirrors the CI env/marker mapping;
   `--all` before opening the PR). State exact test results
   ("477 passed"), and for anything touching scan
   execution or devices, fill in the **hardware verification** section of
   the PR template — either live results or an explicit "owed:" note.
   Code-complete and hardware-verified are different states here, and PRs
   are expected to say which they are.
4. **Review before merge, sized to the risk** — every PR gets one,
   tooling/docs-only ones included; how deep depends on what it touches:
   - **Adversarial review** for code touching scan execution, devices,
     writes or contracts (PV/event/HTTP/schema contracts, gates) —
     wherever the PR is based, a personal branch included: this tier wins
     whenever a diff matches it.
   - **One correctness-only round** for docs, screenshots, `Planning/`-only
     changes, and personal-branch (`users/<name>`) constituents that
     touch none of the areas above.
   - **Everything else** (analysis code, web surfaces, `scripts/`, CI and
     hooks): adversarial — the default when no lighter tier fits.
   - **Bulk promotions** (an integration or personal branch → `master`)
     still follow `/land` — scope check, test counts, the hardware
     section — and **state the exception** in the body: which rounds were
     skipped because the constituent PRs were each reviewed, with a
     pointer to each one's review record.

   Either tier's report is posted on the PR and every finding
   dispositioned, as below; a correctness round is the reviewer brief's
   first lens alone. The adversarial review is a review by
   someone (or, for AI-assisted work, a fresh-context agent) who did not
   write the diff, covering three lenses: correctness (concrete failure
   scenarios), redundancy (does this already exist somewhere in the
   repo?), and placement (is there a more natural home, given the
   dependency graph and package boundaries?). The review report is
   posted on the PR either way — "no surviving findings" is itself the
   record — and each finding is dispositioned — fixed (and confirmed by
   the reviewer), or waived with a stated reason — before merge. The
   reviewer brief lives in `.claude/skills/land/SKILL.md`.
5. **A gating command's own exit code is the gate.** A pipe reports only
   its last command's status, so never gate on `cmd | tail && next`:
   use `cmd > file 2>&1; rc=$?; tail file; [ $rc -eq 0 ] && next`, or
   `set -o pipefail` as `scripts/check.sh` does. (#737 merged past a red
   check because `tail` returned 0; #842 pushed a failing test the same
   way.)
6. **A modernization removes the old implementation in the same PR** —
   superseded models, re-export shims, frozen GUIs, one-shot converters
   after their run — and the body reports adds against deletes per
   concern. Ask first only when the old surface has a live external
   consumer; otherwise delete. (The owner's rule on #803, 2026-09-08.)

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

## GitHub conduct

- **No session links, no "generated with" footers.** This repo is public:
  PR bodies, comments, review dispositions and commit messages carry no
  `claude.ai` session URLs and no generated-with line; a review report is
  pasted in full, never linked. A plain `Co-Authored-By:` trailer is fine.
- **No unprompted follow-up issues.** An issue is filed for a defect, a
  blocker, or something the maintainer named as owed. Nice-to-haves and
  reviewer "gaps" go in the PR body with a one-line "reopen on need":
  open issues cost the maintainer attention whether or not they matter.
- **An arc's decisions live on its tracking issue.** The issue body is the
  current plan; at the end of each slice one concise comment records what
  was planned, what review changed, what the lab found, what changed in
  the plan and why, the facts the next slice needs, and the dead ends.
  geecs-epics#16 is the model; #1084 carries the first such record here.

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
- **Legacy configs are disposable.** The configs repo's legacy YAML never
  constrains the code: when a config kind is regenerated in its new-schema
  form, the deployed catalog is regenerated and committed there — directly
  to the configs repo's main branch, the one place agents commit without
  a PR (the owner's rule, 2026-09-03) — and the legacy readers in this
  repo (converters, resolver fallbacks, dialect tests) are deleted in a
  normal PR. #779 was a config problem wearing an
  engine problem's clothes: a legacy `scan_devices.yaml` pair opted every
  axis out of the readback-confirming motor path that already existed.
- **A fact found wrong is fixed in its home, in the same session.** The
  correction is a PR to the file that owns the fact, with the reason in
  the PR body — never a note in agent memory, a side file or a workaround.
  The 2026-10-30 move of the worker-side services to the services box
  has been wrong in four public files since then (the fleet map, the
  gateway's `CLAUDE.md` and `DEPLOYMENT.md`, `deploy/site.env.example`;
  corrected in a companion PR) because the correction lived only in
  memory and one private runbook.
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
`tests/`, ScanAnalysis, GEECS-Data-Utils and GEECS-Schemas
from the **root env**, and GeecsWebTheme, GeecsLogbook, GeecsScanner,
GeecsControlRoom, GEECS-DataPortal, GEECS-Core, GeecsCAGateway, GeecsBluesky (pure unit
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
