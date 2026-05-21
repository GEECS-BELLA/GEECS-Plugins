# Extending the Scanner

This page covers the three most common ways to extend the scanner without changing engine code: writing a custom **evaluator** (for optimization scans), a custom **scan analyzer** (for post-scan processing), and a custom **action sequence** (for setup/closeout logic too complex for declarative YAML).

If you want to extend the engine itself — add a new scan mode, change the lifecycle, modify how events are emitted — read the [Architecture page](architecture.md) first.

## Writing a custom evaluator

An **evaluator** turns the scalar log of a scan step into a single objective value (and optionally a dictionary of observables) that an optimizer can use. Evaluators are the bridge between "the scan ran and produced data" and "Xopt knows whether the result was good or bad."

The base class is `BaseEvaluator` in `geecs_scanner/optimization/base_evaluator.py`. There's also `MultiDeviceScanEvaluator`, which is the right starting point when the objective comes from camera scalars (like beam size or counts), and `ScalarLogEvaluator`, which is the right starting point when the objective comes from variables already in the s-file.

The minimum interface is two methods:

```python
from geecs_scanner.optimization.evaluators.multi_device_scan_evaluator import (
    MultiDeviceScanEvaluator,
)


class BeamSizeEvaluator(MultiDeviceScanEvaluator):
    """Minimize beam size: (x_fwhm * calibration)² + (y_fwhm * calibration)²."""

    def __init__(self, calibration: float = 24.4e-3, **kwargs):
        super().__init__(**kwargs)
        self.calibration = calibration
        self.objective_tag = "BeamSize"

    def compute_objective(self, scalar_results: dict, bin_number: int) -> float:
        x = self.get_scalar(self.primary_device, "x_fwhm", scalar_results)
        y = self.get_scalar(self.primary_device, "y_fwhm", scalar_results)
        return (x * self.calibration) ** 2 + (y * self.calibration) ** 2

    def compute_observables(self, scalar_results: dict, bin_number: int) -> dict:
        x = self.get_scalar(self.primary_device, "x_fwhm", scalar_results)
        y = self.get_scalar(self.primary_device, "y_fwhm", scalar_results)
        x_cal = x * self.calibration
        y_cal = y * self.calibration
        return {
            "x_fwhm_px": x,
            "y_fwhm_px": y,
            "x_fwhm_units": x_cal,
            "y_fwhm_units": y_cal,
            "size_quadrature_units2": x_cal**2 + y_cal**2,
        }
```

`compute_objective` returns the scalar that the optimizer minimizes. `compute_observables` (optional) returns extra context that gets logged alongside each evaluation — handy for after-the-fact analysis.

`scalar_results` is the dictionary the multi-device evaluator builds for one bin (one set of variable values). Keys are `(device_name, variable)` tuples; values are the scalar from analyzing the bin's images. `bin_number` is the index of the current evaluation within the optimization run.

The `self.primary_device` and `self.get_scalar(...)` helpers come from `MultiDeviceScanEvaluator` and shield you from the dictionary structure.

### Per-shot vs per-bin evaluation

By default `MultiDeviceScanEvaluator` averages per-shot scalars across each bin and calls `compute_objective` once per bin. If you need access to individual shots — for example to compute a noise estimate or a median — override `compute_objective_from_shots` instead:

```python
def compute_objective_from_shots(self, scalar_results_list: list[dict], bin_number: int) -> float:
    values = [self.get_scalar(self.primary_device, "fwhm", r) for r in scalar_results_list]
    return float(np.median(values))   # robust to single-shot outliers
```

Set `analysis_mode: per_shot` in your evaluator's analyzer config to enable this path. The default `per_bin` mode is faster and is the right choice for most objectives.

### Registering an evaluator in a scan

Evaluators are referenced by import path from a YAML config. Look at `geecs_scanner/optimization/example_configs/` for working examples and at the [Optimization Example notebook](examples/optimization/optimization_example.ipynb) for the end-to-end flow.

## Writing a custom scan analyzer

A **scan analyzer** runs after a scan completes (or live as scans complete) and produces summary figures, derived scalars, and optionally Google Doc updates. They live in the [Scan Analysis package](../scan_analysis/overview.md), not in the scanner itself, but they're a primary extension point for users.

The base class is `ScanAnalyzer` in `scan_analysis/base/`. The minimum interface is one method:

```python
from pathlib import Path
from scan_analysis.base import ScanAnalyzer


class MyAnalyzer(ScanAnalyzer):
    def _run_analysis_core(self) -> list[Path]:
        # 1. Load the scan's data (self.scan_paths is set up by the base class)
        df = self.scan_paths.load_sfile()

        # 2. Run your analysis
        summary = df.groupby("scan_var")["U_ModeImager.x_fwhm"].mean()

        # 3. Render a figure
        fig_path = self.scan_paths.get_folder() / "analysis" / "fwhm_vs_scan.png"
        fig_path.parent.mkdir(exist_ok=True)
        # ... matplotlib code that writes to fig_path ...

        # 4. Return a list of display files
        return [fig_path]
```

The base class handles scan folder location, s-file loading, and the overall execution flow. You implement only the analysis. Display files (typically `.png`) get returned and either displayed interactively, stored in the analysis status YAML, or uploaded to a Google Doc — depending on which mode you're running in.

There are two specialized base classes for common cases. `Array2DScanAnalyzer` wraps an `ImageAnalyzer` (from the [Image Analysis package](../image_analysis/overview.md)) and runs it across every shot in every bin, producing both per-bin summary plots and an updated s-file. `Array1DScanAnalyzer` is the equivalent for 1D data. If your analysis fits one of those patterns, inherit from the specialized class — you'll write much less code.

For the full pattern with config-driven instantiation, see the [Config-Based Scan Analysis notebook](../scan_analysis/examples/config_based_scan_analysis.ipynb).

## Writing a custom action

Most setup and closeout logic fits into the declarative `set` / `get` / `wait` / `execute` step types described in [Save Elements](save_elements.md#action-sequences). When it doesn't, the `run` step type imports a Python class:

```yaml
setup_action:
  steps:
    - action: run
      file_name: my_alignment.py
      class_name: PreScanAlignment
```

The file lives in your experiment's action library directory. The class needs a `run()` method:

```python
# my_alignment.py
import logging

logger = logging.getLogger(__name__)


class PreScanAlignment:
    """Iteratively maximize signal on the alignment camera."""

    def __init__(self, action_manager):
        self.action_manager = action_manager

    def run(self):
        logger.info("Starting pre-scan alignment")

        # action_manager.action_control gives access to the device layer
        # so you can read/set devices through the same retry/escalation policy
        # the engine uses.

        # ... iterative alignment logic ...

        logger.info("Pre-scan alignment complete")
```

The constructor is called with the active `ActionManager` so you can reach the device control layer through `self.action_manager.action_control`. Calls through that path use `DeviceCommandExecutor` — meaning failures get the same retry policy and escalation dialog as the engine.

Use `run` steps sparingly. They escape the validated YAML schema and trade a small amount of declarative clarity for arbitrary Python. The right time to reach for them is when an algorithm requires conditional logic (loop until X, branch on Y) that doesn't fit a linear sequence.

## Loading the result of your extension into a script

Once your evaluator or analyzer has run and produced data, you may want to load the result into a notebook for further exploration. The standard idiom:

```python
from geecs_data_utils import ScanPaths, ScanTag

tag = ScanTag(year=2026, month=5, day=8, number=42)
paths = ScanPaths(tag=tag, experiment="MyExperiment", read_mode=True)

# The s-file (with any analyzer-appended columns):
df = paths.load_sfile()

# Display files written by your analyzer:
analysis_dir = paths.get_folder() / "analysis"
display_files = sorted(analysis_dir.glob("*.png"))
```

If your analyzer writes structured output (CSVs, npy arrays, JSON) into `analysis/`, anyone who knows the scan number can load it back via `paths.get_folder() / "analysis"`. Stick to that convention rather than inventing new output paths — it makes downstream tooling much simpler.

## Where the test patterns are

If you're going to ship a custom evaluator or analyzer, mirror the existing test pattern:

- For evaluators: `GEECS-Scanner-GUI/tests/optimization/test_concrete_evaluators.py` builds synthetic `ImageAnalyzerResult` objects and exercises `compute_objective` / `compute_objective_from_shots` without touching real scan data. This is the right pattern — your tests should not require a network drive or a live database.
- For scan analyzers: `ScanAnalysis/tests/` has fixtures that build minimal scan folders on `tmp_path` and run analyzers against them. The same approach works for new analyzers.

A working test before merging is the difference between an extension that lasts and one that bit-rots within a release. The test surface is small; the cost of writing one is low.
