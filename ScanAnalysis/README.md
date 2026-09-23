# ScanAnalysis

Run configured camera and 1D analysis across completed scans, bin results,
write derived scalars, and render summary figures. Python 3.11 is required.

The operator entry point is the **Data Portal Analysis tab**. Its config
editor uses this package's `ConfigStore` and editor router, including
previews of unsaved diagnostic documents. Install the portal's `analysis`
extra to enable these features.

For Python use, install with `poetry install` from this directory (or the
repository root), then load a diagnostic/group with `scan_analysis.config`
and call the resulting analyzer's `run_analysis(scan_tag)`. See the
[analysis tutorial](../docs/tutorials/analysis.md) and
[developer context](CLAUDE.md) for configuration and output contracts.

LiveWatchGUI, LiveTaskRunner, and Google Docs uploads are retired. The
explicit task queue and its status files remain for MCP compatibility;
automatic post-scan analysis awaits a separate service design. Existing
`gdoc_slot` and `upload_to_scanlog` config fields are accepted but ignored.
Analysis never creates raw scan folders.
