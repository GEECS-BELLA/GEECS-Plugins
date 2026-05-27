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
| `GeecsBluesky/` | Bluesky RunEngine backend: BlueskyScanner, ophyd-async GEECS devices, Tiled integration |
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

Worktrees should live outside the repository checkout, as siblings of the main
clone, and should use stable names that describe the intended feature or fix.
For example, keep the main checkout at `GEECS-Plugins/` and create worktrees
such as `GEECS-Plugins-pulse-duration-jitter/`,
`GEECS-Plugins-interlock-suggestions/`, or
`GEECS-Plugins-bluesky-detectors/` next to it.

Do not create worktrees inside the repository root, inside subpackages, or
under tool-generated paths such as `.claude/worktrees/`. Remove worktrees after
their PR is merged unless they are intentionally long-lived for a distinct
development stream.

## Python & Tooling

- **Python:** `>=3.10, <3.12` across all packages (Scanner GUI is `<3.11`)
- **Package manager:** Poetry — `poetry install` at the repo root installs the
  main dev environment. Each subpackage can also be installed standalone.
- **Linting:** `ruff` (replaces flake8/isort) + `pydocstyle` (numpy convention)
- **Pre-commit hooks:** ruff, ruff-format, pydocstyle, check-yaml, check-json,
  check-ast — run automatically on commit
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

```
GEECS-PythonAPI  ←─── GEECS-Scanner-GUI
                 ←─── GEECS-Data-Utils
                 ←─── ScanAnalysis (optional)

GEECS-Data-Utils ←─── ScanAnalysis
                 ←─── ImageAnalysis (optional)

ImageAnalysis    ←─── ScanAnalysis
                 ←─── GEECS-Scanner-GUI (optimization evaluators)

ScanAnalysis     ←─── LogMaker4GoogleDocs (optional, for gdoc upload)
LogMaker4GoogleDocs  (no GEECS deps — pure Google API wrapper)
```

`ScanAnalysis` and `ImageAnalysis` are the most actively developed packages.
`LogMaker4GoogleDocs` is optional everywhere — missing it causes silent skips,
not errors.

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
`ScanAnalysis/`, `ImageAnalysis/`, `LogMaker4GoogleDocs/`.

Git tags on merge to master: `geecs-scanner-v0.8.0`, `geecs-python-api-v0.3.1`, etc.

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
