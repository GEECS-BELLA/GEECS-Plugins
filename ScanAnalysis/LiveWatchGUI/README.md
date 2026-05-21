# LiveWatch GUI

PyQt5 GUI for configuring and launching LiveWatch scan analysis.

Replaces the notebook-based workflow (`live_watch.ipynb`) with a graphical
interface that lets you configure and start/stop the `LiveTaskRunner` without
editing code.

## Features

- **Experiment selection** — auto-populated from scan analysis config files
- **Date & scan number** — pick the analysis date and starting scan
- **GDoc upload toggle** — enable/disable Google Doc log uploads
- **Runtime options** — max items per cycle, dry run, rerun completed/failed
- **Live log display** — filterable, colour-coded log output
- **Thread-safe** — analysis runs in a background thread; GUI stays responsive

## Quick Start

```bash
# From the ScanAnalysis directory, install with GUI extras:
cd GEECS-Plugins/ScanAnalysis
poetry install

# Or install from the LiveWatchGUI directory:
cd LiveWatchGUI
poetry install

# Launch the GUI:
python -m LiveWatchGUI.main

# Or use the script entry point (after poetry install):
live-watch-gui
```

## Requirements

- Python 3.10+
- PyQt5 (Windows)
- All ScanAnalysis dependencies (installed automatically via poetry)
