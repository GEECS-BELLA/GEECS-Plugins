# Scan Analysis

The Scan Analysis package coordinates analysis across complete experimental scans. Rather than analyzing individual shots in isolation, it provides the scaffolding to iterate an analyzer across all shots in a scan, bin results by the scanned parameter, render summary figures, and append results back to the s-file. Results can optionally be uploaded directly to a Google Doc e-log.

The best way to get started is the [examples](#examples) below.

---

## Two Modes of Use

### Interactive / Offline

Instantiate an analyzer and call `run_analysis()` on a specific scan folder. This is the typical starting point for developing and testing a new analyzer, or for reprocessing historical data.

```python
from scan_analysis.analyzers.Undulator.screen_imager import UndulatorScreenImager

analyzer = UndulatorScreenImager(scan_tag)
display_files = analyzer.run_analysis()
```

See [Basic Usage](examples/basic_usage.ipynb) for a full walkthrough.

### Live (Automated)

A `LiveTaskRunner` watches a data directory for new scans and automatically dispatches registered analyzers as scans complete. Multiple runners can operate concurrently — a heartbeat-based task queue ensures each scan is claimed and processed exactly once.

```python
from scan_analysis.live_task_runner import LiveTaskRunner

runner = LiveTaskRunner(experiment="Undulator", config_path="config.yaml")
runner.run()
```

See [Live Watch](examples/live_watch.ipynb) for setup details.

---

## Config-Driven Workflow

For production deployments, analyzers are defined in a YAML config file rather than instantiated in code. The factory reads the config, imports the appropriate analyzer class, and wires up all parameters automatically. This is the recommended approach for running analysis alongside live experiments.

```yaml
analyzers:
  - analyzer: scan_analysis.analyzers.Undulator.screen_imager.UndulatorScreenImager
    gdoc_slot: 0   # upload summary figure to row 1, col 0 of the e-log
```

See [Config-Based Scan Analysis](examples/config_based_scan_analysis.ipynb) for the full config format and options.

---

## Outputs

Every analyzer produces **display files** — summary figures (typically `.png`) that visualize the scan results. These are:

- Returned directly from `run_analysis()` for interactive use
- Stored in the task queue status file for the live runner
- Optionally uploaded to a Google Doc e-log (controlled by `gdoc_slot` in the config)

See [GDoc Upload](examples/gdoc_upload.ipynb) for e-log integration setup.

---

## Writing a Custom Analyzer

Analyzers inherit from `ScanAnalyzer` (or `SingleDeviceScanAnalyzer` for single-device workflows) and implement one method:

```python
from scan_analysis.base import ScanAnalyzer

class MyAnalyzer(ScanAnalyzer):
    def _run_analysis_core(self):
        # load data, run analysis, save figures
        return [Path("path/to/summary_figure.png")]
```

The base class handles scan folder location, s-file loading, and the overall execution flow. See the [API Reference](api/base.md) for the full interface.

---

## Examples

| Notebook | What it covers |
|---|---|
| [Basic Usage (2D)](examples/basic_usage.ipynb) | Run an `Array2DAnalysis` analyzer on a scan |
| [Basic Usage (1D)](examples/basic_usage_1D.ipynb) | Run an `Array1DAnalysis` analyzer on a scan |
| [Config-Based Workflow](examples/config_based_scan_analysis.ipynb) | YAML config, factory instantiation, and batch runs |
| [Live Watch](examples/live_watch.ipynb) | Set up a `LiveTaskRunner` for automated processing |
| [GDoc Upload](examples/gdoc_upload.ipynb) | Integrate with Google Doc e-logs |
| [Variational Analysis](examples/variational_analysis.ipynb) | Custom variational analysis patterns |
