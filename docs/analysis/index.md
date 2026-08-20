# Analysis

Everything for turning acquired scan data into results — per-image
processing, per-scan orchestration, and the path/loading layer they both
build on. If you have a scan folder on the data server and want to make
sense of what's in it, start here.

<div class="grid cards" markdown>

-   :material-image-filter-center-focus:{ .lg .middle } **Image Analysis**

    ---

    Per-shot image processing: YAML-described pipelines (background,
    masking, filtering, geometric transforms, thresholding) and
    specialised analyzers for beam profile, FROG, magspec, HASO
    wavefront, and 1D traces.

    [:octicons-arrow-right-24: Overview](../image_analysis/overview.md) ·
    [Analyzer index](../image_analysis/analyzer_index.md)

-   :material-chart-areaspline:{ .lg .middle } **Scan Analysis**

    ---

    Orchestrates analysis across a complete scan — shot binning, per-bin
    processing, summary-figure rendering, s-file appending — interactively
    or as a `LiveTaskRunner` that processes scans as they complete.

    [:octicons-arrow-right-24: Overview](../scan_analysis/overview.md)

-   :material-database-search:{ .lg .middle } **Data Utils**

    ---

    The foundational data layer: resolve `(experiment, date, scan_number)`
    to an on-disk path, load s-files, and use the common data structures
    the rest of the suite is built on. Usually a dependency, sometimes a
    direct import for ad-hoc exploration.

    [:octicons-arrow-right-24: Overview](../geecs_data_utils/overview.md)

</div>

New to the analysis side? The cross-package
[Analysis tutorial](../tutorials/analysis.md) walks the end-to-end path:
configure analyzers, run them over a scan, and read the results back.
