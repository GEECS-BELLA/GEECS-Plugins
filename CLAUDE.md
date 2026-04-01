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

## Active Branches / PRs to Know About

- `gdoc-autoupload` / PR #295 — GDoc image insertion via `gdoc_slot` config
  (merged or near-merged; see `ScanAnalysis/CLAUDE.md` for details)
- `config-manager-gui` — PyQt5 GUI for creating/editing YAML configs (not yet merged)
- `gdoc-hyperlinks` — Planned next PR: upload all analyzer outputs as hyperlinks
  instead of table cells
