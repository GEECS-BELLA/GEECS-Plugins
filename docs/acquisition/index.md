# Acquisition

Running scans on the beamline — choosing what gets recorded, driving the
scan, and capturing per-shot data and images into both the classic GEECS
scan folder and a structured Tiled run.

<div class="grid cards" markdown>

-   :material-camera-iris:{ .lg .middle } **GEECS Console**

    ---

    The operator application: compose save sets, run scans and
    Xopt-driven optimizations through the Bluesky-backed engine, monitor
    live, and set devices. The same scan requests it submits can be run
    headlessly from your own scripts.

    [:octicons-arrow-right-24: Overview](../geecs_console/overview.md) ·
    [Running scans](../geecs_console/running_scans.md) ·
    [Save sets](../geecs_console/save_sets.md)

-   :material-magnify:{ .lg .middle } **Scan Browser**

    ---

    The quick-look client for recorded scans: day → scan →
    plot / table / telemetry-drift, straight from the Tiled catalog.
    Standalone — analysts never need the console.

    [:octicons-arrow-right-24: Scan Browser](../geecs_console/scan_browser.md) ·
    [Understanding the data](../geecs_console/scan_data.md)

</div>

Wondering where your data went or what a column means? Start at
[Scan Data](../geecs_console/scan_data.md). Something misbehaving? Start
at [Troubleshooting](../geecs_console/troubleshooting.md).
