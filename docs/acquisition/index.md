# Acquisition

Running scans on the beamline — choosing what gets recorded, driving the
scan, and capturing per-shot data and images into both the classic GEECS
scan folder and a structured Tiled run.

<div class="grid cards" markdown>

-   :material-camera-iris:{ .lg .middle } **GEECS Scanner**

    ---

    The operator front end, in a browser: pick a preset or compose a
    scan, submit it to the Bluesky queueserver, watch it live, move a
    device, run an action plan. The same presets it submits can be
    queued headlessly from your own scripts.

    [:octicons-arrow-right-24: Overview](../geecs_scanner/overview.md) ·
    [Running a scan](../geecs_scanner/overview.md#running-a-scan)

-   :material-magnify:{ .lg .middle } **Reading it back**

    ---

    Recorded scans are browsed in the Data Portal (day → scan →
    metadata, scalar plots, images, from any browser) and read from
    Python through the `ScanCatalog` layer in Data Utils.

    [:octicons-arrow-right-24: Data Utils](../geecs_data_utils/overview.md) ·
    [Where the data lands](../geecs_scanner/overview.md#where-the-data-lands)

</div>

Something misbehaving? Start at the scanner page's
[health chips](../geecs_scanner/overview.md#when-something-is-wrong), then
the [fleet map](../platform/fleet_map.md) for the service behind the chip.
