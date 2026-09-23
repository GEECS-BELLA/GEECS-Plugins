# Configure and run analysis

Use the data portal to edit a diagnostic, preview it on a recorded shot,
and explicitly run it on a completed scan. This workflow needs the portal's
`analysis` extra, a configured analysis-configs repository, and readable scan
data. The operator's scan must already exist; analysis never creates it.

## 1. Open a recorded scan

Open the data portal, select a day and scan, and inspect a camera image.
The portal resolves the data share and configs from the site's configuration.
See [Getting started](getting_started.md) for the path settings.

Select the processing diagnostic for that camera. Use **edit configs** to
open the editor with that scan's current shot as its preview input.

## 2. Edit and preview a diagnostic

Diagnostic YAML documents live under `scan_analysis_configs/analyzers/` in
the configs repository. Each document specifies:

- `name`: the device/channel used to locate input data.
- `analyzer`: the analysis kind and its parameters, such as `beam` or `line`.
- `image`: camera or line processing, including ROI, background, and filters.
- `scan`: execution mode, output saving, and renderer options.

Edit the fields, then use **preview** to inspect the **unsaved** document on
the selected shot. Preview does not write scan outputs or save the config.
The editor reports validation errors before saving. Save when the preview
and parameters are correct. A stale-file conflict means someone else changed
the config; reload and reconcile the changes before trying again.

Supported beam/line previews and image processing use `geecs-analysis`.
Recipes not yet ported use the existing ImageAnalysis implementation.
Vendor analyzers still require their corresponding host libraries.

## 3. Run analysis explicitly

Return to the scan's **Analysis** tab, select the diagnostic, and run it.
The run uses the saved configuration. Inspect the returned status and saved
figures, then compare the derived scalar columns with the expected result.

`scan.mode: per_shot` analyzes each shot before aggregating results.
`scan.mode: per_bin` averages raw frames within each bin before analysis.
These modes can give different answers for nonlinear measurements; choose
the mode that matches the diagnostic's intended measurement.

Saved outputs live in the day's `analysis/ScanNNN/` tree. Derived scalars
are added to the analysis s-file. `output_name` controls the output prefix;
`metric_suffix` affects scalar keys only. Raw acquisitions remain inputs.

## Groups and Python runs

Groups remain useful for explicit Python and MCP runs. A group document
under `scan_analysis_configs/groups/` names diagnostics by filename stem:

```yaml
name: example_baseline
analyzers:
  - camera_beam
  - {ref: spectrum_line, priority: 5}
  - {ref: spare_camera, enabled: false}
```

Given your existing `scan_tag` and configs root, load and run the group:

```python
from scan_analysis.config import load_analysis_group, create_scan_analyzer

# scan_tag identifies an existing scan; config_dir is the configs root.
group = load_analysis_group("example_baseline", config_dir=config_dir)
for resolved in group.analyzers:
    analyzer = create_scan_analyzer(
        resolved.diagnostic, id=resolved.id, priority=resolved.priority
    )
    display_files = analyzer.run_analysis(scan_tag)
```

Use the actual group filename stem in place of `example_baseline`.
The [Scan Analysis overview](../scan_analysis/overview.md) explains the
factory, queue, output contract, and custom analyzers.

## Retired features

LiveWatch and Google Docs uploads are removed. Existing `scan.gdoc_slot`
and group `upload_to_scanlog` fields validate but have no effect; the
editor hides them while preserving existing values. The queue/status files
remain for explicit MCP runs. Automatic post-scan analysis will need a
separate service design and is not part of this workflow.
