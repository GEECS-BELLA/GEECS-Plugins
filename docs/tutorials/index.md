# Tutorials

The suite's two halves — acquisition and analysis — each get their own
end-to-end tutorial. Pick the one that matches what you're trying to do.

<div class="grid cards" markdown>

-   :material-camera-iris:{ .lg .middle } **Acquisition tutorial**

    ---

    Configure an experiment, build a save element, and run your first
    NOSCAN and 1D scan with the Scanner GUI. Lands the data on disk.

    [:octicons-arrow-right-24: Acquisition tutorial](acquisition.md)

-   :material-chart-areaspline:{ .lg .middle } **Analysis tutorial**

    ---

    Build a per-camera analyzer config in ConfigFileGUI, add it to a
    group, run that group on a real scan via LiveWatch. Lands summary
    figures and (optionally) e-log uploads.

    [:octicons-arrow-right-24: Analysis tutorial](analysis.md)

</div>

These two paths are independent — you don't need to do one to do the other.
But the natural workflow is **acquisition produces data that the analysis
tutorial then processes**, so they're laid out alongside each other rather
than tucked away in their respective package tabs.

## Which to start with

| If you're… | Start here |
|---|---|
| New to GEECS, running scans on the beamline | [Acquisition](acquisition.md) |
| New to GEECS, processing data | [Analysis](analysis.md) |
| Already comfortable acquiring, learning analysis | [Analysis](analysis.md) |
| Already comfortable analysing, learning the scanner | [Acquisition](acquisition.md) |
| Joining the team, not sure | [Acquisition](acquisition.md) first, then [Analysis](analysis.md) |

After either tutorial, the per-package documentation
([GEECS Console](../geecs_console/overview.md),
[Image Analysis](../image_analysis/overview.md),
[Scan Analysis](../scan_analysis/overview.md)) covers the underlying
concepts and APIs in depth.
