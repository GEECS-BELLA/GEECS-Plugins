# Scan Analysis

The Scan Analysis package coordinates analysis across complete experimental
scans. Rather than analysing individual shots in isolation, it iterates a
configured analyzer across every shot in a scan, bins results by the scanned
parameter, renders summary figures, and appends derived scalars back to the
s-file. Run analysis explicitly from the data portal or Python.

The fastest way to see it in action is the
[Analysis tutorial](../tutorials/analysis.md), which walks the canonical
config editor → preview → explicit run loop end to end.

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

### From the data portal

Select a completed scan and open its **Analysis** tab to run a configured
diagnostic. Use **edit configs** to edit and preview an unsaved document
before saving. The editor remains part of the portal.

Automatic watching and Google Docs uploads have been retired. The existing
queue/status API remains available for explicit MCP runs; a future automatic
analysis service has not yet been designed.

---

## Config-driven workflow

The configuration model post-PR-E is two-tier:

* **Per-diagnostic configs** under `scan_analysis_configs/analyzers/`. One
  YAML per camera or 1D signal. Each is a
  [`AnalysisDiagnostic`](../image_analysis/overview.md#how-a-diagnostic-is-described)
  bundling the typed `analyzer:` spec (`kind` + its parameters), the
  ImageAnalysis-owned `image:` block and the ScanAnalysis-owned `scan:` block.

* **Per-group configs** under `scan_analysis_configs/groups/`. One YAML
  per analyzer group — a named collection of analyzer refs that get run
  together. Refs are either bare strings (use the analyzer's own
  `scan.priority`) or dicts with per-group overrides (`enabled: false`,
  `priority: 5`).

Group YAMLs look like:

```yaml
name: HTU_baseline
description: standard HTU shift analysis
analyzers:
  - Amp4Input
  - Amp4Output
  - UC_TopView
  - {ref: GaiaMode, priority: 5}      # bumped vs the analyzer's own default
  - {ref: Amp3Input, enabled: false}  # temporarily disabled here, not deleted
```

`load_analysis_group` loads a group by path-key (`"HTU/baseline"`), resolves
each ref to its diagnostic config, builds a `ScanAnalyzer` for each, and
dispatches them per-scan according to their priorities. A recipe the
`geecs_analysis` core can run (beam, line, standard and trace kinds with
ported processing steps and no scan-context background) becomes a
`CoreScanAnalyzer`; anything else instantiates the right `ImageAnalyzer`
and wraps it in the legacy `Array2DScanAnalyzer` (camera configs) or
`Array1DScanAnalyzer` (line configs). Both routes write the same files.

Authoring these YAMLs by hand is fine;
the **[config editor](../tutorials/analysis.md)** is the friendlier path.

---

## Outputs

Each analyzer produces:

- **Display files** — summary figures (typically `.png`) that visualise
  the scan. Returned from `run_analysis()` for interactive use; recorded
  in the task-queue status file when running through the queue.
- **Derived scalars** appended back to the s-file as new columns.

Legacy `scan.gdoc_slot` and group `upload_to_scanlog` fields are accepted
but ignored. They are hidden in the editor.

---

## Package layout

```
scan_analysis/
├── base.py                   # ScanAnalyzer abstract base
├── task_queue.py             # Heartbeat-based queue; claim/release/status YAML
├── config/
│   ├── diagnostic_factory.py     # create_scan_analyzer(diag, ...)
│   └── analysis_group_loader.py  # discover_analyzers/groups + load_analysis_group,
│                                 #   ResolvedDiagnosticConfig (models: geecs_schemas.analysis)
└── analyzers/
    ├── common/
    │   ├── array2D_scan_analysis.py   # Wraps an ImageAnalyzer for 2D shots
    │   ├── array1d_scan_analysis.py   # Same for 1D
    │   ├── single_device_scan_analyzer.py
    │   └── scatter_plotter_analysis.py
    └── Undulator/                 # Experiment-specific specialised analyzers
```

The Python group workflow: read a group YAML →
`load_analysis_group` → resolves refs →
`create_scan_analyzer(r.diagnostic, id=r.id, priority=r.priority)` builds
each → a `CoreScanAnalyzer` on the analysis core, or `Array2DScanAnalyzer`
(or 1D) wrapping the underlying `ImageAnalyzer` for recipes the core does
not run yet → `run_analysis(scan_tag)` does the work.

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
| [Scatter Plot Analysis](examples/scatter_plot_analysis.ipynb) | Generic two-axis scatter analyzer over multiple devices |

## See also

- The [Analysis tutorial](../tutorials/analysis.md) — the no-Python
  config editor → preview → explicit run path.
- [Image Analysis overview](../image_analysis/overview.md) — the
  per-shot processing layer that diagnostic configs configure.
- [Data Utils overview](../geecs_data_utils/overview.md) — `ScanTag` and
  `ScanPaths`, the path-resolution primitives this package uses.
- [API Reference](api/base.md) — the `ScanAnalyzer` base class surface.
