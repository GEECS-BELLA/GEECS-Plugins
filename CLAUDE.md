# GEECS-Plugins — Developer Context for Claude

This is the monorepo for BELLA beamline data acquisition, analysis, and logging
tooling. Each subdirectory is an independent Python package with its own
`pyproject.toml` managed by **Poetry**.

## Repository Map

| Package | Description |
|---|---|
| `ScanAnalysis/` | Post-scan analysis framework: task queue, YAML config system, scan analyzers |
| `ImageAnalysis/` | Per-image analysis: pipelines, offline analyzers, config models |
| `GEECS-Console/` | Greenfield PySide6 operator console (Bluesky/gateway architecture): scan submission, live health/device panels, config editors, Tiled scan browser |
| `GEECS-Data-Utils/` | Scan path navigation, scalar loading, binning, Parquet database |
| `GEECS-Schemas/` | Pydantic-only config vocabulary: versioned schemas for every scanner config kind (scan request, save set, scan variables, trigger profile, action plans, derived channels) + legacy-YAML converters + the docgen Markdown reference generator. Depends on pydantic alone — importable from anywhere |
| `GeecsBluesky/` | Bluesky RunEngine backend: BlueskyScanner + headless GeecsSession, CA-backed ophyd-async devices (via GeecsCAGateway's PVs), Tiled integration |
| `GEECS-Core/` | The GEECS access **library**: UDP/TCP wire protocol (`transport/`), experiment DB (`db/GeecsDb`), PV naming contract, the one `GeecsError` tree, and the `FakeGeecsServer` test double — extracted from GeecsCAGateway 2026-08-20; see its `DESIGN.md` for the layering rules — plus the thin synchronous `GeecsDevice` client (`client/`), the successor to GEECS-PythonAPI's device objects |
| `GeecsCAGateway/` | The caproto CA gateway serving GEECS devices as PVs (readback + `:SP`) for Phoebus/Archiver/ophyd-async, built on GEECS-Core — see its `PV_CONTRACT.md` (client API contract), `DEPLOYMENT.md`, and `DESIGN.md` |
| `GeecsPvaGateway/` | The PVA peer of GeecsCAGateway: distributed pvAccess server on each Windows camera server, exposing that host's GEECS camera images as NTNDArray PVs (gated subscriptions, latest-wins). Images stay off the central CA gateway by design |
| `LogMaker4GoogleDocs/` | Google Docs/Drive API wrapper for automated experiment logs |

Each subpackage has its own `CLAUDE.md` with deep architectural detail.

The published mkdocs site lives under `docs/` and also has its own
`CLAUDE.md` covering documentation conventions — content organisation,
build commands, the headless-screenshot workflow for GUI pages, and
notebook hygiene constraints. Read it whenever you touch anything under
`docs/`.

## Agent & Worktree Policy

`CLAUDE.md` files are the canonical agent/developer instructions for this
repository. `AGENTS.md` exists only as a Codex compatibility shim that points
Codex to the root and package-level `CLAUDE.md` files. Do not duplicate policy
between `AGENTS.md` and `CLAUDE.md`; update the relevant `CLAUDE.md` instead.
`CONTRIBUTING.md` is the human-facing distillation of the same contract —
keep the three in sync when the rules change.

Recurring workflows are encoded as repo-checked skills under
`.claude/skills/<name>/SKILL.md`: `/land` (the PR ritual), `/check`
(lint + tests the way CI runs them, via `scripts/check.sh`), `/triage`
(scan-log error triage), `/scan-audit` (scan timing/cadence analysis),
`/env-doctor` (per-package Poetry env fixups). Each skill's frontmatter
`description` carries its trigger symptoms so sessions pull the skill in
on their own — keep those descriptions current when a skill changes.
Prefer invoking/updating a skill over re-deriving its workflow in a
session. Facts owned elsewhere (e.g. the branch topology, which lives in
`CONTRIBUTING.md`) are referenced from skills, never copied into them.

Worktrees should live **inside** the main checkout at `.claude/worktrees/`,
under stable names that describe the intended feature or fix — for example
`.claude/worktrees/pulse-duration-jitter/`,
`.claude/worktrees/interlock-suggestions/`, or
`.claude/worktrees/docs-apps-tab/`. The `.claude/worktrees/` path is
`.gitignore`d so worktree contents never pollute the main clone's git status
or staging area.

This is a deliberate reversal of an earlier policy that put worktrees as
siblings of the main clone (`GEECS-Plugins-feature-name/`). In practice the
sibling layout required agents to `cd` outside the project root constantly,
which triggers permission prompts on macOS and Linux sandboxes for every
command — enough friction to be a real drag on iteration. Living under
`.claude/worktrees/` keeps every command rooted inside the project tree the
agent already has permission to operate on.

Do not create worktrees in the repository root itself, inside subpackages
(e.g. `ImageAnalysis/.claude/worktrees/`), or in random tmp locations. The
canonical location is `<repo-root>/.claude/worktrees/<feature-name>/`.

**Always start Claude sessions with the repo root as the working directory.**
When Claude Code is configured to spawn a session worktree, it places the
worktree relative to wherever the session was launched from. Launching from
`<repo-root>/ImageAnalysis/` produces a worktree at
`ImageAnalysis/.claude/worktrees/<id>/`, not at `<repo-root>/.claude/worktrees/`.
The `**/.claude/worktrees/` pattern in `.gitignore` is a safety net for this
case, but the right fix is to launch from the repo root.

Remove worktrees after their PR is merged unless they are intentionally
long-lived for a distinct development stream.

`geecs-plugins-bluesky` (a sibling directory of this checkout) is **no longer
a worktree** — it has been promoted to its own standalone clone with an
independent `.git`, sharing only the `GEECS-BELLA/GEECS-Plugins` origin. Treat
it as a separate clone, not a linked worktree of this checkout: changes flow
between the two only through git (push/pull/PR), and each has its own local
Claude context (sessions and memory are keyed by directory path). It keeps its
own nested worktrees under `.claude/worktrees/`.

## Python & Tooling

- **Python:** `>=3.11, <3.12` across all packages — the integrated monorepo
  environment is Python 3.11 (the root project requires it). The sole exception
  is `LogMaker4GoogleDocs`, a standalone Google API wrapper with no GEECS deps,
  which keeps a looser `>=3.9` floor
- **Package manager:** Poetry — `poetry install` at the repo root installs the
  main dev environment. Each subpackage can also be installed standalone.
- **Linting:** `ruff` (replaces flake8/isort) + `pydocstyle` (numpy convention)
- **Pre-commit hooks:** ruff, ruff-format, pydocstyle, check-yaml, check-json,
  check-ast — run automatically on commit. The auto-fixing hooks rewrite files
  during the commit, which aborts that commit ("files were modified by this
  hook") so you re-stage and retry — and on *merge* commits triggers a
  stash/restore conflict that can silently abort. Use **`scripts/commit.sh -m
  "..."`** (after `git add`): it applies the auto-fixes first, re-stages them,
  then commits, so the commit succeeds on the first try. Any `git commit` args
  pass through.
- **Docs:** MkDocs (root `pyproject.toml`) — `mkdocs serve` from repo root

## Code Style Conventions

- **Docstrings:** NumPy convention (see `pydocstyle convention = "numpy"` in
  root `pyproject.toml`)
- **Type hints:** Required on all public methods/functions
- **Imports:** `ruff` enforces ordering — don't fight it, let the hook fix it
- **No `Any` without comment** — free-form dicts should be Pydantic models
  wherever feasible
- **Pydantic v2** throughout — use `model_validate()`, `model_dump()`,
  `model_fields`; avoid v1 patterns like `.dict()` or `.parse_obj()`

## Package Dependency Graph

Arrows read **"depends on / imports"** — `X → Y` means X imports Y. Verified
against each package's `[tool.poetry.dependencies]` (intra-repo path deps).

```
GEECS-Data-Utils     →  (no intra-repo deps — foundational data layer)
LogMaker4GoogleDocs  →  (no intra-repo deps — pure Google API wrapper)
GEECS-Schemas        →  (no intra-repo deps — pydantic-only config vocabulary)

ImageAnalysis        →  GEECS-Data-Utils
GEECS-Core           →  (no intra-repo deps — the GEECS access library:
                        transport, DB, PV naming, exceptions, fake server)
GeecsCAGateway       →  GEECS-Core (the access library it serves over CA),
                        GEECS-Schemas (schema-only vocabulary for optional
                        derived-channel overlays)
GeecsPvaGateway      →  GEECS-Core (transport, DB, pv_naming),
                        GeecsCAGateway (config helpers, e.g.
                        effective_vartype), GEECS-Data-Utils (IMAQ decode)
                        — the distributed PVA image server on the camera
                        servers
GeecsBluesky         →  GEECS-Data-Utils, GEECS-Core, GEECS-Schemas
                        (+ ImageAnalysis, optional via the `analysis` extra —
                        post-run image analysis over archived Tiled runs;
                        + ScanAnalysis/ImageAnalysis/xopt, optional via the
                        `optimize` extra — the relocated Xopt/evaluator
                        stack in geecs_bluesky.optimization)
ScanAnalysis         →  GEECS-Data-Utils, ImageAnalysis, LogMaker4GoogleDocs
GEECS-Console        →  GeecsBluesky, GEECS-Schemas, GEECS-Data-Utils,
                        GEECS-Core (GeecsDb for completions/health)
                        (its `optimization` extra installs the heavy deps
                        for geecs_bluesky.optimization — xopt/ScanAnalysis —
                        no geecs-scanner-gui dependency remains)
```

`GEECS-Core` is the GEECS access library: the UDP/TCP wire protocol, the
experiment DB, the PV naming contract, and the exception tree. The gateways
build their servers on it; GeecsBluesky and GEECS-Console import its library
parts (`GeecsDb`, `pv_naming`, wire-level exceptions) and consume the CA
gateway purely as a *service* (the PVs, via stock ophyd-async EPICS
signals) — nothing imports the gateway's server code except GeecsPvaGateway
(config helpers).

`GEECS-Data-Utils` is the foundational layer — everything depends on it and it
depends on nothing else in the repo. `GEECS-Console` sits at the top of the
DAQ side. `ScanAnalysis` and `ImageAnalysis` are the most actively
developed analysis packages. `LogMaker4GoogleDocs` is optional everywhere — missing it
causes silent skips, not errors.

## How Packages Are Used Together (Typical Analysis Flow)

1. **GEECS-Console** (via **GeecsBluesky**'s scan engine) runs a scan →
   writes per-shot data files to a date-structured folder on the data server
   (the legacy master-line GEECS-Scanner-GUI writes the same layout)
2. **GEECS-Data-Utils** `ScanPaths` / `ScanData` resolves the folder, loads
   scalar summary data from s-files or TDMS
3. **ImageAnalysis** `StandardAnalyzer` / `BeamAnalyzer` / etc. processes
   per-shot image files → `ImageAnalyzerResult`
4. **ScanAnalysis** `Array2DScanAnalyzer` or `Array1DScanAnalyzer` wraps an
   `ImageAnalyzer`, aggregates per-shot results, renders summary plots
5. **LogMaker4GoogleDocs** uploads summary figures to Google Drive and inserts
   them into the experiment Google Doc (triggered by `gdoc_slot` config)

## GEECS Data Folder Convention

```
{base_path}/{experiment}/Y{YYYY}/{MM-Month}/{YY_MMDD}/
  ├── scans/
  │   └── Scan{NNN}/
  │       ├── Scan{NNN}.tdms
  │       ├── ScanDataScan{NNN}.txt    (scanner-written scalar summary)
  │       ├── ScanInfoScan{NNN}.ini    (scan metadata)
  │       ├── <device>/...             (raw per-shot data)
  │       └── analysis_status/         (ScanAnalysis task queue YAML files)
  └── analysis/
      ├── s{NNN}.txt                   (watched s-file copy)
      └── Scan{NNN}/...                (analysis output tree)
```

`base_path` is typically a network drive (Windows: `Z:/data`, Linux/Mac: mounted
equivalent). Resolved by `GeecsPathsConfig` from `~/.config/geecs_python_api/config.ini`.

## Deleted legacy packages (2026-08-20)

`GEECS-PythonAPI` (legacy TCP device layer) and `GEECS-Scanner-GUI` (legacy
PyQt5 DAQ front-end) were **deleted from `dev`** at the end of the geecs-core
arc. Their successors: `geecs_core.client.GeecsDevice` for device
get/set/subscribe, `GEECS-Console` + `GeecsBluesky` for scans, and
`geecs_bluesky.optimization` for the Xopt stack. Both packages remain on
`master` for the legacy production line until the M6 cutover; never adopt
them in new code there either. The `~/.config/geecs_python_api/config.ini`
path they named is a permanent fleet contract and deliberately keeps its
name (see `GEECS-Core/DESIGN.md`).

## Release & Versioning

Each package is versioned independently using **semantic versioning**:

| Digit | When to bump | Example |
|-------|-------------|---------|
| `0.0.x` patch | Bug fix, no behaviour or API change | `0.7.1 → 0.7.2` |
| `0.x.0` minor | New feature or meaningful behaviour change (backwards-compatible) | `0.7.1 → 0.8.0` |
| `1.0.0` major | Stable production API, deployed across multiple experiments | reserved |

**On every PR that changes code in a package:**

1. Run `poetry version patch|minor|major` from inside the package directory —
   this edits `pyproject.toml` in place
2. Add an entry to the package's `CHANGELOG.md` under the new version number
3. Commit `pyproject.toml` and `CHANGELOG.md` together with the code changes

```bash
cd GEECS-Console && poetry version minor   # 0.20.2 → 0.21.0
cd GEECS-Core    && poetry version patch   # 0.2.0 → 0.2.1
```

Every package has a `CHANGELOG.md` following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format:
`GEECS-Data-Utils/`, `ScanAnalysis/`, `ImageAnalysis/`,
`LogMaker4GoogleDocs/`, `GeecsBluesky/`, `GEECS-Core/`, `GeecsCAGateway/`,
`GeecsPvaGateway/`, `GEECS-Schemas/`, `GEECS-Console/`.

Git tags (`geecs-scanner-v0.8.0` style) are cut at **milestones** — a state
deployed across experiments or one we may need to reproduce (e.g. the
access-layer landing, 2026-07-06) — not on every merge. The per-package
`CHANGELOG.md` + `pyproject.toml` versions are the routine record.

## Cross-package invariants

These are load-bearing rules that hold across multiple packages. Violating them
has caused real production incidents; the consequences aren't abstract.

### Analysis code is a consumer of scan folders, never a producer

Only the **scanner side** (BlueskyScanner — concretely `claim_scan_number`
in GeecsBluesky's `plans/run_wrapper.py`)
brings new `scans/ScanNNN/` folders into existence. Everything else — all of
ScanAnalysis, ImageAnalysis, LogMaker4GoogleDocs, every offline analyzer —
must treat the scan folder as preexisting and refuse to auto-create it.

Concretely, this means analysis-side code must **not**:

- Call `ScanPaths(read_mode=False)` — the create-if-missing path is reserved
  for scanner-side callers
- Use `Path.mkdir(parents=True, ...)` on any path that traverses up through
  `scans/ScanNNN/`. Output subdirectories inside an existing scan folder
  should use `mkdir(exist_ok=True)` only (no `parents=True`)
- Recover from a missing scan folder by creating it — log loudly and skip
  or raise, so the absence is surfaced rather than papered over. Do not try
  to force a `failed` / `no_data` task status into `scans/ScanNNN/analysis_status/`
  when the scan folder itself is absent; that status location lives inside the
  folder analysis code must not create.

**Why this matters:** silently creating a scan folder that *appears* missing —
when really it's just briefly invisible due to an SMB visibility blip, a
permissions glitch, or a snapshot/AV operation on the share — plants an empty
directory entry at the scan path. When the transient resolves, the underlying
data has been orphaned: there is now a different `ScanNNN/` at that path, and
the recovery operation that would have restored the original contents either
silently fails or overwrites the wrong target. We've shipped this failure mode
in production. Don't reintroduce it.

The rule is pinned by tests:

- `ScanAnalysis/tests/test_task_queue.py::TestScanFolderCreationInvariant`
- `ImageAnalysis/tests/analyzers/test_line_stitcher.py::TestLineStitcherScanFolderInvariant`
- `ImageAnalysis/tests/analyzers/test_magspec_calib.py::TestScanFolderInvariant`
- `ImageAnalysis/tests/processing/test_array1d_background.py`
- `GEECS-Data-Utils/tests/test_scan_paths_create_invariant.py`

Each package's CLAUDE.md restates this rule with package-specific guidance for
adding new analyzers/writers.

## Known debt we have deliberately deferred

Items below are *known* and *intentionally not being fixed right now*. If you
look at them and think "this is bad, let me fix it" — please don't. The
deferral is deliberate; the rationale is below. If you encounter a feature
request whose natural scope overlaps one of these, that's the right time to
revisit. Speculative cleanup is not.

- **`extras/` is a legacy dump pending pruning** — ~26 MB of unrelated
  one-off projects and third-party snapshots. Never build there, never cite
  its contents as reference implementations, and don't prune it
  opportunistically: what's still load-bearing needs the owner's judgment.

- **`LogMaker4GoogleDocs` needs a refactor.** It works in production
  (Google Doc log uploads) and is optional everywhere, but don't extend it
  or use it as a style reference until that refactor happens.

If you find yourself adding to this list, consider whether you're capturing
real institutional knowledge or accumulating procrastination. Both are
possible.
