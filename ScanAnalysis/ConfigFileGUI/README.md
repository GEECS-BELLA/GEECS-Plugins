# Config File Editor GUI

A PyQt5 GUI for viewing, modifying, and creating device configuration YAML files
used by the GEECS image analysis system.

> **Note:** This tool is under active development. Features are being added
> incrementally — see [the plan](../../../plans/Config_File_GUI_Plan.md) for the
> full roadmap.

## What It Does

- Browse and open device configuration YAML files from the
  `image_analysis_configs/` directory
- View and edit 2D camera configs (`CameraConfig`) and 1D line configs
  (`Line1DConfig`)
- Create new configuration files with sensible defaults
- Validate configurations against the Pydantic schema before saving

## How to Run

From the `ScanAnalysis/` directory:

```bash
python -m ConfigFileGUI.main
```

Or directly:

```bash
python ConfigFileGUI/main.py
```

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--log-level` | Console log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) | `INFO` |
| `--config-dir` | Path to the `image_analysis_configs/` directory | `None` (select via GUI) |

### Example

```bash
python -m ConfigFileGUI.main --config-dir ../../GEECS-Plugins-Configs/image_analysis_configs --log-level DEBUG
```
