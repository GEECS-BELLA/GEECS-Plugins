# Analysis

Everything for turning acquired scan data into results — per-image
processing, per-scan orchestration, and the path/loading layer they both
build on. If you have a scan folder on the data server and want to make
sense of what's in it, start here.

<div class="grid cards" markdown>

-   :material-book-open-variant:{ .lg .middle } **Recipe reference**

    ---

    Every processing step, measure and summary a recipe can use, with
    worked figures, the parameters of each, and what every scalar written
    to the s-file means. The config editor's **reference ↗** links land
    here.

    [:octicons-arrow-right-24: Recipe reference](../sites/analysis_recipes/index.html) ·
    [Analysis without the portal](examples/analysis_without_the_portal.ipynb)

-   :material-image-filter-center-focus:{ .lg .middle } **Image Analysis**

    ---

    Per-shot image processing: YAML-described pipelines (background,
    masking, filtering, geometric transforms, thresholding) and
    specialised analyzers for beam profile, FROG, magspec and 1D
    traces. The HASO wavefront runs on the analysis core through
    WaveKit — see [HASO wavefront](haso.md).

    [:octicons-arrow-right-24: Overview](../image_analysis/overview.md) ·
    [Analyzer index](../image_analysis/analyzer_index.md)

-   :material-chart-areaspline:{ .lg .middle } **Scan Analysis**

    ---

    Orchestrates analysis across a complete scan — shot binning, per-bin
    processing, summary-figure rendering, s-file appending — whenever it is
    asked to, from the Data Portal, Python, or the MCP tools.

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
build a recipe in the portal's config editor, preview it on a shot, run it
over a scan. The
[analysis without the portal](examples/analysis_without_the_portal.ipynb)
notebook does the same from Python on a real scan, and shows how to try a
variant of a recipe for a one-off analysis.
