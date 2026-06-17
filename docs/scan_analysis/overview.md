# Scan Analysis

The Scan Analysis package coordinates analysis across complete experimental
scans. Rather than analysing individual shots in isolation, it iterates a
configured analyzer across every shot in a scan, bins results by the scanned
parameter, renders summary figures, and appends derived scalars back to the
s-file. Display figures can optionally be uploaded to a Google Doc e-log via
`LogMaker4GoogleDocs`.

The fastest way to see it in action is the
[Analysis tutorial](../tutorials/analysis.md), which walks the canonical
ConfigFileGUI → group → LiveWatch loop end to end.

---

## Two modes of use

### Interactive / offline

Instantiate an analyzer and call `run_analysis(scan_tag)` with the tag of
the scan you want to process. This is the typical starting point for
developing or debugging an analyzer, and for reprocessing historical data.

```python
from geecs_data_utils import ScanTag
from scan_analysis.analyzers.common import Array2DScanAnalyzer

tag = ScanTag(year=2026, month=5, day=8, number=42, experiment="Undulator")
analyzer = Array2DScanAnalyzer(device_name="UC_TopView")
display_files = analyzer.run_analysis(tag)
```

The base class handles scan-folder location, s-file loading, and binning;
the analyzer's `_run_analysis_core()` does the work that's specific to the
diagnostic. See [Basic Usage (2D)](examples/basic_usage.ipynb) for a full
walkthrough.

### Live (automated) via LiveWatch

A `LiveTaskRunner` watches a data directory for new scans and dispatches an
analyzer group as each scan completes. Multiple runners can co-operate over
the same data directory — a heartbeat-based task queue ensures each scan is
claimed and processed exactly once.

```python
from geecs_data_utils import ScanTag
from scan_analysis.live_task_runner import LiveTaskRunner

today = ScanTag(year=2026, month=5, day=27, number=0, experiment="Undulator")
runner = LiveTaskRunner(
    analyzer_group="HTU/baseline",   # path-key into scan_analysis_configs/groups/
    date_tag=today,
    gdoc_enabled=False,
)
runner.run()
```

In practice nobody calls this directly — the **[LiveWatch GUI](../tutorials/analysis.md)**
wraps it with a friendly interface. The Python API exists so headless
runners and tests can drive it the same way.

---

## Config-driven workflow

The configuration model post-PR-E is two-tier:

* **Per-diagnostic configs** under `scan_analysis_configs/analyzers/`. One
  YAML per camera or 1D signal. Each is a
  [`DiagnosticAnalysisConfig`](../image_analysis/overview.md#how-a-diagnostic-is-described)
  bundling the ImageAnalysis-owned `image:` block, the ScanAnalysis-owned
  `scan:` block, and a `name` + `image_analyzer` class path.

* **Per-group configs** under `scan_analysis_configs/groups/`. One YAML
  per analyzer group — a named collection of analyzer refs that get run
  together. Refs are either bare strings (use the analyzer's own
  `scan.priority`) or dicts with per-group overrides (`enabled: false`,
  `priority: 5`).

Group YAMLs look like:

```yaml
name: HTU_baseline
description: standard HTU shift analysis
upload_to_scan-log: true
analyzers:
  - Amp4Input
  - Amp4Output
  - UC_TopView
  - {ref: GaiaMode, priority: 5}      # bumped vs the analyzer's own default
  - {ref: Amp3Input, enabled: false}  # temporarily disabled here, not deleted
```

`LiveTaskRunner` loads a group by path-key (`"HTU/baseline"`), resolves
each ref to its diagnostic config, instantiates the right `ImageAnalyzer`,
wraps it in the appropriate `ScanAnalyzer` (`Array2DScanAnalyzer` for
camera configs, `Array1DScanAnalyzer` for line configs), and dispatches
them per-scan according to their priorities.

Authoring these YAMLs by hand is fine; the
**[ConfigFileGUI](../tutorials/analysis.md)** is the friendlier path.

---

## Outputs

Each analyzer produces:

- **Display files** — summary figures (typically `.png`) that visualise
  the scan. Returned from `run_analysis()` for interactive use; recorded
  in the task-queue status file for live runs; optionally uploaded to the
  experiment's Google Doc when `gdoc_enabled=True`.
- **Derived scalars** appended back to the s-file as new columns.

See [GDoc Upload](examples/gdoc_upload.ipynb) for the e-log integration
details.

---

## Package layout

```
scan_analysis/
├── base.py                   # ScanAnalyzer abstract base
├── live_task_runner.py       # LiveTaskRunner — watches + dispatches
├── task_queue.py             # Heartbeat-based queue; claim/release/status YAML
├── gdoc_upload.py            # Optional LogMaker4GoogleDocs integration
├── config/
│   ├── diagnostic_models.py      # AnalyzerRef, AnalysisGroupConfig,
│   │                             #   ResolvedDiagnosticConfig, ScanRuntimeConfig
│   ├── diagnostic_factory.py     # create_diagnostic_analyzer(resolved)
│   └── analysis_group_loader.py  # discover_analyzers/groups + load_analysis_group
└── analyzers/
    ├── common/
    │   ├── array2d_scan_analysis.py   # Wraps an ImageAnalyzer for 2D shots
    │   ├── array1d_scan_analysis.py   # Same for 1D
    │   ├── single_device_scan_analyzer.py
    │   └── scatter_plotter_analysis.py
    └── Undulator/                 # Experiment-specific specialised analyzers
```

The common pattern: `LiveTaskRunner` reads a group YAML →
`load_analysis_group` → resolves refs → `create_diagnostic_analyzer` builds
each → `Array2DScanAnalyzer` (or 1D) wraps the underlying `ImageAnalyzer`
→ `run_analysis(scan_tag)` does the work.

---

## Writing a custom analyzer

For most camera and 1D workflows, the generic `Array2DScanAnalyzer` /
`Array1DScanAnalyzer` wrappers are enough — point them at an
`ImageAnalyzer` (custom or built-in) and configuration alone gets you the
behaviour you want. When the per-scan shape is genuinely different (e.g.
specialised stitching, multi-device correlation), subclass `ScanAnalyzer`
directly:

```python
from pathlib import Path
from typing import Optional, Union

from scan_analysis.base import ScanAnalyzer


class MyCustomAnalyzer(ScanAnalyzer):
    def _run_analysis_core(self) -> Optional[list[Union[Path, str]]]:
        # self.scan_data / self.auxiliary_data are already populated.
        # Do the analysis, save figures, return their paths.
        return [Path("path/to/summary_figure.png")]

    def cleanup(self) -> None:
        # Release per-scan memory so the task runner can move on.
        super().cleanup()
```

`cleanup()` is required (the base class raises `NotImplementedError`
intentionally) — implement it even if there's nothing to release, so the
runner doesn't accumulate state. See the [API Reference](api/base.md) for
the full surface area.

---

## Examples

| Notebook | What it covers |
|---|---|
| [Basic Usage (2D)](examples/basic_usage.ipynb) | Run an `Array2DScanAnalyzer` on a scan, interactively |
| [Basic Usage (1D)](examples/basic_usage_1D.ipynb) | The same flow for a 1D signal |
| [Live Watch](examples/live_watch.ipynb) | Drive `LiveTaskRunner` headlessly from a script |
| [GDoc Upload](examples/gdoc_upload.ipynb) | Wire summary figures into a Google Doc e-log |
| [Scatter Plot Analysis](examples/scatter_plot_analysis.ipynb) | Generic two-axis scatter analyzer over multiple devices |

## See also

- The [Analysis tutorial](../tutorials/analysis.md) — the no-Python
  ConfigFileGUI → group → LiveWatch path.
- [Image Analysis overview](../image_analysis/overview.md) — the
  per-shot processing layer that diagnostic configs configure.
- [Data Utils overview](../geecs_data_utils/overview.md) — `ScanTag` and
  `ScanPaths`, the path-resolution primitives this package uses.
- [API Reference](api/base.md) — the `ScanAnalyzer` base class surface.
