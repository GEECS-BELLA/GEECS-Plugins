# Scan Config Editor GUI

A PyQt5 GUI for viewing, modifying, and creating the unified-diagnostic
configuration YAMLs that drive ScanAnalysis (and LiveWatch) post-PR-E.

## What it edits

Everything under `scan_analysis_configs/`:

```
scan_analysis_configs/
├── analyzers/<facility>/<stem>.yaml   ← unified diagnostic (DiagnosticAnalysisConfig)
└── groups/<facility>/<stem>.yaml      ← analyzer group (AnalysisGroupConfig)
```

Each analyzer YAML bundles the per-device `image:` (CameraConfig or
Line1DConfig — the discriminator is the `type:` field) and `scan:`
(ScanRuntimeConfig) sections plus the `image_analyzer:` class path.
Each group YAML lists a set of analyzer refs (with optional
per-group `enabled` / `priority` overrides).

## Features

- Browse `analyzers/` and `groups/` trees, organised by facility namespace
  (HTU / HTT / PW / UNCLASSIFIED).
- Create new analyzer and group YAMLs from minimal templates.
- Rename files (the on-disk stem is the canonical analyzer ID).
- Edit the unified diagnostic shape: `name`, `image_analyzer`,
  `image:` (typed CameraConfig / Line1DConfig with full sub-section
  editors), and `scan:` (priority, mode, save flags, background_source, …).
- Edit groups: name, description, upload-to-scanlog flag, plus a list
  of analyzer refs with per-ref `enabled` / `priority` overrides.
- Optional YAML preview pane (Tools → Toggle YAML Preview).
- **Analysis Preview** (Tools → Analysis Preview…) — run a single-image
  analysis test against the current diagnostic without leaving the GUI.

## How to run

From the `ScanAnalysis/` directory:

```bash
python -m ConfigFileGUI.main
```

## CLI options

| Flag | Description | Default |
|------|-------------|---------|
| `--log-level` | Console log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) | `INFO` |
| `--scan-config-dir` | Path to the `scan_analysis_configs/` directory | None (open via File menu) |

### Example

```bash
python -m ConfigFileGUI.main --scan-config-dir ../../GEECS-Plugins-Configs/scan_analysis_configs --log-level DEBUG
```

## Requirements

- Python 3.10+
- PyQt5
- All ScanAnalysis dependencies (installed automatically via poetry)
