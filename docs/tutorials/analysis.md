# Configure and run analysis

A **recipe** says how to analyze one device's data: which files to read, an
ordered list of processing steps, one measure that turns each processed frame
into scalars, and the summary figures to draw for the scan. This page walks
the usual loop in the Data Portal: open a recorded shot, build or edit a
recipe while previewing it, then run it on the whole scan. Every step has a
Python equivalent; the
[analysis without the portal](../analysis/examples/analysis_without_the_portal.ipynb)
notebook does the same loop on a real scan.

The portal needs its `analysis` extra, a configured configs repository, and
readable scan data. Analysis reads scans; it never creates a scan folder.

For what each step, measure, summary and scalar means, with worked figures,
see the [recipe reference](../sites/analysis_recipes/index.html).

## 1. Open a recorded shot

In the portal, pick a day and a scan and open the camera or trace you want to
analyze. **edit configs** opens the config editor with that shot as its
preview input. See the [Data Portal guide](../web_services/data_portal.md) for
finding your way around, and [Getting started](getting_started.md) for the
path settings.

## 2. Build the recipe while previewing it

Recipes live under `scan_analysis_configs/analyzers/` in the configs
repository, one YAML file per recipe. A camera recipe looks like this:

```yaml
schema_version: 3
device: UC_ALineEBeam3          # the data folder under scans/ScanNNN/
input:
  kind: camera                  # or line, for traces
  format: device_hdf5           # the camera writes one HDF5 stack per scan
steps:                          # run in order; each sees the previous output
- step: background_constant
  level: 100
- step: roi
  bounds: [[0, 950], [20, 1000]]  # (y, x), half-open pixel indices
- step: zero_below
  level: 0
measure:
  kind: beam                    # 18 scalars: centroids, widths, peaks
summaries:
- kind: image_grid              # one panel per bin
- kind: average
```

The editor builds this form for you. Each step, measure and summary has a
**reference ↗** link to its card in the recipe reference, and a measure lists
the scalars it can write and what each means. Change a field, then **preview**
to run the **unsaved** recipe on the selected shot; preview writes nothing.
The editor reports validation errors before saving. A stale-file conflict on
save means someone else changed the file; reload and reconcile.

Things to check in the preview:

- **Is the measure describing the beam?** A saturated spot at the frame edge
  or a bright background can dominate a projection. `x_peak_location` at the
  frame edge, or a width close to the frame size, are the usual signs. Crop
  with `roi` and raise the background level until the overlays sit on the
  beam.
- **Is the input format right?** Cameras that save a stack per scan need
  `input.format: device_hdf5`; without it the run looks for per-shot files and
  reports "No file found for shot N".
- **Does the trace need joining?** A spectrometer split across cameras uses
  `input.siblings` to join them into one trace before the steps run.

## 3. Run it on the scan

Save the recipe, return to the scan's **Analysis** tab, pick the recipe, and
run it. The run uses the saved file. It writes figures and per-shot products
under the day's `analysis/ScanNNN/` folder and adds the scalars to the
analysis s-file as columns prefixed with the recipe's `output_name` (the
device name unless you set one). `scalar_suffix` tags the scalar columns
only. Raw data is never modified.

The `scan` section controls how the run goes over the scan:

- `average_frames_first: false` (default) measures every shot, then averages
  the scalars per bin. `true` averages each bin's raw frames first and
  measures once per bin. For a nonlinear quantity, such as a width on a noisy
  image, the two give different answers; pick the one that matches what you
  want to know.
- `save: false` skips the figures and products; the s-file columns are still
  written.
- `priority` orders recipes within a group; `workers` reads and measures in
  parallel.

The same run from Python:

```python
from image_analysis.config import load_diagnostic
from scan_analysis.config import create_scan_analyzer

recipe = load_diagnostic("UC_ALineEBeam3")  # by file stem
create_scan_analyzer(recipe).run_analysis(scan_tag)  # writes outputs
```

## Groups: several recipes at once

A group document under `scan_analysis_configs/groups/` names recipes by file
stem, for running a standard set from Python or the MCP tools:

```yaml
name: example_baseline
analyzers:
  - camera_beam
  - {ref: spectrum_line, priority: 5}
  - {ref: spare_camera, enabled: false}
```

```python
from scan_analysis.config import create_scan_analyzer, load_analysis_group

group = load_analysis_group("example_baseline", config_dir=config_dir)
for resolved in group.analyzers:
    analyzer = create_scan_analyzer(
        resolved.diagnostic, id=resolved.id, priority=resolved.priority
    )
    display_files = analyzer.run_analysis(scan_tag)
```

The [Scan Analysis overview](../scan_analysis/overview.md) explains the
factory, output contract, and custom analyzers.

## Retired features

LiveWatch and Google Docs uploads are removed. Existing `scan.gdoc_slot`
and group `upload_to_scanlog` fields validate but have no effect; the
editor hides them while preserving existing values. Analysis runs only when
someone asks for it, from the portal, Python, or the MCP tools.
