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
{base_path}/{experiment}/Y{YYYY}/{MM-Month}/{YY_MMDD}/scans/Scan{NNN}/
  └── Scan{NNN}.tdms
  └── ScanData_scan.txt          (scalar summary / s-file)
  └── scan_info.ini              (scan metadata)
  └── analysis/                  (created by ScanAnalysis)
      └── analysis_status/       (task queue YAML files)
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

## Architectural Roadmap

A full refactor roadmap lives in `ROADMAP.md` in separate branch. Read it before
making major structural changes. Key points:

- The goal is a headless-capable scan engine with explicit state, a typed event
  stream, and a single `DeviceCommandExecutor` owning all device interactions
- `ScanManager` can already run headlessly — the GUI adds display and dialogs only
- `device.set()` calls are currently scattered; consolidation is Block 6 of the
  roadmap and requires earlier blocks to be in place first
- Breaking changes are acceptable in the name of better organisation
