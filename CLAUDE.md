# GEECS-Plugins — Developer Context for Claude

This is the monorepo for BELLA beamline data acquisition, analysis, and logging
tooling. Each subdirectory is an independent Python package with its own
`pyproject.toml` managed by **Poetry**.

## Repository Map

| Package | Description |
|---|---|
| `ScanAnalysis/` | Post-scan analysis framework: task queue, YAML config system, scan analyzers |
| `ImageAnalysis/` | Per-image analysis: pipelines, offline analyzers, config models |
| `GEECS-Scanner-GUI/` | PyQt5 DAQ front-end: scans, save elements, optimization (Xopt) |
| `GEECS-Data-Utils/` | Scan path navigation, scalar loading, binning, Parquet database |
| `GEECS-Schemas/` | Pydantic-only config vocabulary: versioned schemas for every scanner config kind (scan request, save set, scan variables, trigger profile, action plans, derived channels) + legacy-YAML converters + the docgen Markdown reference generator. Depends on pydantic alone — importable from anywhere |
| `GeecsBluesky/` | Bluesky RunEngine backend: BlueskyScanner + headless GeecsSession, CA-backed ophyd-async devices (via GeecsCAGateway), Tiled integration |
| `GeecsCAGateway/` | The GEECS access layer: UDP/TCP wire protocol, experiment DB, PV naming, and the caproto CA gateway serving GEECS devices as PVs (readback + `:SP`) for Phoebus/Archiver/ophyd-async — see its `PV_CONTRACT.md` (client API contract), `DEPLOYMENT.md`, and `DESIGN.md` |
| `LogMaker4GoogleDocs/` | Google Docs/Drive API wrapper for automated experiment logs |
| `GEECS-PythonAPI/` | Low-level device TCP layer — **under refactoring, do not touch** |

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

Recurring workflows are encoded as repo-checked skills in
`.claude/commands/`: `/land` (branch topology + the PR ritual), `/triage`
(scan-log error triage), `/scan-audit` (scan timing/cadence analysis),
`/env-doctor` (per-package Poetry env fixups). Prefer invoking/updating a
skill over re-deriving its workflow in a session.

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
GeecsCAGateway       →  GEECS-Schemas (schema-only vocabulary for optional
                        derived-channel overlays; otherwise the GEECS access layer:
                        wire protocol, DB, PV naming, CA server)
GeecsBluesky         →  GEECS-Data-Utils, GeecsCAGateway, GEECS-Schemas
                        (+ ImageAnalysis, optional via the `analysis` extra —
                        post-run image analysis over archived Tiled runs)
GEECS-PythonAPI      →  GEECS-Data-Utils
ScanAnalysis         →  GEECS-Data-Utils, ImageAnalysis, LogMaker4GoogleDocs
GEECS-Scanner-GUI    →  GEECS-PythonAPI, ImageAnalysis, ScanAnalysis,
                        GEECS-Data-Utils, GeecsBluesky
```

`GeecsCAGateway` is the self-contained GEECS access layer: the UDP/TCP wire
protocol, the experiment DB, the PV naming contract, and the caproto CA
server. GeecsBluesky imports its *library* parts (`GeecsDb`, `pv_naming`,
wire-level exceptions) and consumes its *service* (the PVs, via stock
ophyd-async EPICS signals) — it never imports the server.

`GEECS-Data-Utils` is the foundational layer — everything depends on it and it
depends on nothing else in the repo. `GEECS-Scanner-GUI` sits at the top and
pulls in everything. `ScanAnalysis` and `ImageAnalysis` are the most actively
developed packages. `LogMaker4GoogleDocs` is optional everywhere — missing it
causes silent skips, not errors.

Two declaration quirks worth knowing (both verified against the pyprojects):

- **`GEECS-PythonAPI` declares a dependency on `ImageAnalysis` but never imports
  it** — a stale/unused entry in `GEECS-PythonAPI/pyproject.toml`. It's an
  architecturally backwards edge (the low-level device layer pointing at the
  high-level analysis package) and is a candidate for removal whenever the
  python-api refactor next touches its dependencies. It is intentionally omitted
  from the graph above because no code relies on it.
- **`ScanAnalysis`'s dependency on `GEECS-PythonAPI` is currently commented out**
  in its `pyproject.toml`, so ScanAnalysis does not depend on python-api. (An
  earlier version of this graph claimed it did.)

## How Packages Are Used Together (Typical Analysis Flow)

1. **GEECS-Scanner-GUI** runs a scan → writes per-shot data files to a
   date-structured folder on the data server
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

## GEECS-PythonAPI — Handle With Care

This package provides TCP device connections and the experiment database query
layer. It is **being refactored** — do not add new features or restructure it.
Other packages use it primarily for:
- `ScanDevice` — subscribe to a device variable stream
- Database dict lookup — enumerate all devices in an experiment
- `config.ini` — shared config file all packages read from

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
cd GEECS-Scanner-GUI && poetry version minor   # 0.7.1 → 0.8.0
cd GEECS-PythonAPI   && poetry version patch   # 0.3.0 → 0.3.1
```

Every package has a `CHANGELOG.md` following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format:
`GEECS-Scanner-GUI/`, `GEECS-PythonAPI/`, `GEECS-Data-Utils/`,
`ScanAnalysis/`, `ImageAnalysis/`, `LogMaker4GoogleDocs/`,
`GeecsBluesky/`, `GeecsCAGateway/`, `GEECS-Schemas/`.

Git tags (`geecs-scanner-v0.8.0` style) are cut at **milestones** — a state
deployed across experiments or one we may need to reproduce (e.g. the
access-layer landing, 2026-07-06) — not on every merge. The per-package
`CHANGELOG.md` + `pyproject.toml` versions are the routine record.

## Cross-package invariants

These are load-bearing rules that hold across multiple packages. Violating them
has caused real production incidents; the consequences aren't abstract.

### Analysis code is a consumer of scan folders, never a producer

Only the **scanner side** (GEECS-Scanner-GUI's `ScanDataManager`, BlueskyScanner)
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

- **`GEECS-Scanner-GUI/geecs_scanner/app/app_controller.py` is a thin
  pass-through layer.** It was extracted from the main window during the bold
  refactor (PR landed May 2026) and ended up as a side struct that adds an
  indirection without removing complexity. We considered reverting; the cost
  of churn outweighs the benefit at this point. Leave it; it's harmless. Do
  not invest in extending it.

- **`GEECS-Scanner-GUI/geecs_scanner/engine/data_logger.py` and
  `device_manager.py` lack behavioral tests for several internal paths.** The
  code works in production; a future Bluesky-backed scan path may obsolete
  significant parts of these modules. Adding deep tests now risks pinning
  internals we don't intend to keep. If you need to *change* DataLogger or
  DeviceManager, write tests for the specific behavior you're modifying as
  part of that change.

- **`ScanConfig` lives in `geecs_data_utils` rather than
  `geecs_scanner.engine.models`.** This is logically backwards (engine config
  in the data-utils package) but the migration touches many files across two
  packages and has no forcing function. Defer until a feature naturally takes
  you into that area.

- **`GEECS-Scanner-GUI/geecs_scanner/app/geecs_scanner.py` is ~2200 lines and
  mixes seven concerns** (save-element list management, scan variable
  handling, presets, shot calculation, scan submission, optimization config,
  toolbar/menu wiring). Adding a new scan mode currently touches six places.
  Documented organizational debt; the fix is concern-cluster extraction (or a
  base class for editor windows). Wait for a specific feature ("add a new
  scan mode," "add a new editor") to drive the refactor with bounded scope.

- **`GEECS-PythonAPI` is being refactored elsewhere.** Don't add features here
  or restructure it. Other packages use it through `ScanDevice` and the
  database dict lookup; treat that as the public surface.

If you find yourself adding to this list, consider whether you're capturing
real institutional knowledge or accumulating procrastination. Both are
possible.
