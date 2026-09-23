# Changelog

## [0.12.0] - 2026-09-23

### Added

- `compat.v2_render`: v2 renderer-option translation for single, image-grid and
  waterfall figures. Preserves the legacy colormap-mode/limit rules, the
  index-wise waterfall stack on the first trace's x grid with midpoint cell
  edges, sort-implied even spacing, real zero positions and configured labels.
  No pyplot state and no file writes.

## [0.11.0] - 2026-09-23

### Added

- Object-API image grids with independent panel coordinates, typed overlays,
  explicit titles and one shared color scale/colorbar. Global finite-data
  autoscaling owns mutable normalizers; mixed uniform/nonuniform panels use
  the same palette. Invalid layouts and conflicting/reversed limits raise
  RenderError without file writes or pyplot state.

## [0.10.0] - 2026-09-23

### Added

- Pure post-analysis v2 result averages for noscan and bin summaries. Preserve
  ordinary-mean versus nanmean behavior, stored trace dtype and coordinate
  averaging, scalar-key policy and projection means without re-running measures.
  Empty/mixed-shape inputs skip the average; aggregates shed per-shot identity.

## [0.9.0] - 2026-09-23

### Added

- Streaming v2 execution over caller-supplied shot groups and loaders, retaining
  native-dtype average-before-analysis behavior, separate loaded/full membership,
  explicit failures, provenance and bounded raw-buffer lifetime. Declared member
  order makes reductions reproducible. No scan route or output sink changes.

## [0.8.0] - 2026-09-23

### Added

- Pure crosshair masking and fixed-canvas image rotation with legacy rasterization,
  interpolation and operation order. The v2 adapter supports camera fiducials
  and rotation; flips and distortion correction remain explicitly unported.
- Lazy OpenCV use for rotated masks, matching the existing analyzer algorithm.

## [0.7.0] - 2026-09-23

### Added

- Explicit source-host opt-in for v2 camera file-background requests and loaded
  frame bindings during v2 execution. Default compilation still rejects these
  recipes before live acquisition; no readers or fallback are added to the core.

## [0.6.0] - 2026-09-23

### Added

- Named in-memory Frame inputs for pure pipelines and analysis, with preflight
  validation and immutable binding snapshots. Background subtraction preserves
  ownership and supports explicit coordinate or sample alignment without I/O,
  broadcasting or resampling.

## [0.5.0] - 2026-09-23

### Added

- Pure circular masking and uniform trace interpolation, with preserved axes,
  units and provenance. The v2 adapter preserves local mask centers, operation
  order, zero padding and trace storage precision for existing beam/line recipes.

## [0.4.2] - 2026-09-23

### Changed
- Document portal processing/preview adoption and the remaining scan-runner
  migration boundary. No core runtime change.

## [0.4.1] - 2026-09-23

### Fixed
- Accept explicitly disabled identity transforms in v2 camera pipelines, as
  used by live beam diagnostics. Active geometric transforms remain unsupported.

## [0.4.0] - 2026-09-23

### Added
- Matplotlib object-API single-frame and same-grid waterfall rendering without
  pyplot or output writes, with calibrated coordinates and stable overlay ids.
- Numpy-free FigureSpec with grouped matplotlib kwargs and typed preview errors.
- Nonuniform image grids render with pcolormesh; ambiguous/nonmonotonic grids
  and mismatched waterfall coordinates are explicitly rejected.

## [0.3.0] - 2026-09-23

### Added
- Write-free v2 compilation/execution for the supported beam/line and
  preprocessing-only recipes, retaining input scaling, storage rounding,
  scalar names, camera ROI conventions and float64 trace result values.
- Explicit UnsupportedRecipe failures for unported active features.
- The migration harness can capture either legacy or new-core outputs from
  identical raw inputs and readers.

## [0.2.0] - 2026-09-23

### Added
- Beam, line and preprocessing-only measures, numpy-free scalar discovery and
  write-free `analyze(Frame, Analysis)` execution.
- Typed measurements with owned scalar maps, projections, centroid overlays
  and explicit notes for nonfinite results.
- Existing line/beam/slopes algorithms ported with scalar conventions retained;
  beam x/y coordinates come from Frame axes and RMS mutates only private scratch.

## [0.1.0] - 2026-09-23

### Added
- Pure, ordered and repeatable processing pipelines over coordinate-aware Frames.
- Numpy-free Pydantic specs and a builtin step registry.
- Constant background, coordinate/index ROI, median, Gaussian, clipping and
  zero-below steps with legacy numerical comparisons.
