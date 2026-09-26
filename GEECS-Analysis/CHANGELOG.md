# Changelog

## [0.15.0] - 2026-09-26

### Added

- `compat.v2_run.run_units(..., workers=N)`: a bounded, ordered `spawn`
  process pool for the scan loop (#1003). `workers <= 1` is the serial loop
  unchanged, no pool built. Above that each worker receives the recipe, the
  bound inputs and the loader once (pickled into its initializer), reads and
  analyzes its own groups, and the outcomes are yielded in declared group
  order through a window of at most `2 × workers` in flight — so a host's
  accumulation sees the serial sequence and computes the same numbers.
  Worker log records are forwarded to the parent's loggers (`QueueHandler`
  / `QueueListener`), closing the iterator shuts the pool down, and per-unit
  failures stay outcomes. Each worker also exits the moment the pool's owner
  dies (a daemon thread on the parent's sentinel), so a SIGTERM/SIGKILLed
  host that never reaches its `finally` leaves no worker holding a stack. A loader that is also a context manager is entered
  once per run and once per worker (a source keeping one stack handle).
- `compat.v2_average.RunningAverage`: the legacy summary average folded one
  measurement at a time — float64 sum (+ per-element count in bin mode) for
  camera frames, projections and markers, so memory is one frame however
  many are folded; scalars kept and reduced at the end; traces retained and
  reduced at storage dtype as before. The sequential fold is numpy's own
  order for a first-axis stack reduction, so the result equals
  `np.mean` / `np.nanmean` over the stack **bit for bit** for frames of more
  than one element — a stack of 1×1 frames reduces along a contiguous axis,
  pairwise (pinned against the
  stack, and by the unchanged differential tests against the legacy
  `ImageAnalyzerResult.average`). `average_results` is now built on it.
- `Measurement` pickles (a worker returns one to its parent): the scalar
  view travels as a plain dict and is restored read-only; notes are not
  re-annotated. Pinned in `tests/test_pickling.py` with the compiled
  recipes of both formats.

## [0.14.0] - 2026-09-24

### Added

- `geecs_analysis.recipe.recipe_schema()`: the recipe's JSON Schema with
  `steps` and `measure` bound to the registry's discriminated unions (one
  variant per registered step and measure, parameters typed), every step,
  measure and summary variant tagged `x-ndim` with the frame shapes it
  processes — the vocabulary the config editor's form lists.
- A `description=` on every step and measure parameter; the form shows it
  as the field's help. Pinned: `test_recipe_schema_binds_the_registry_vocabulary`.

## [0.13.0] - 2026-09-24

### Added

- `geecs_analysis.recipe`: `compile_recipe` binds a v3 `AnalysisRecipe` to
  the registry (unknown steps/measures/parameters, undeclared or unused
  frame bindings and dimensionality mismatches raise `RecipeError`) and
  compiles it to the same in-memory recipe the v2 adapter produces, so
  both formats run through one evaluator; `compile_document`, `figure_of`,
  `summaries_of` and `is_line` read either format.
- `geecs_analysis.summaries`: the frozen summary kinds, one file each,
  registered through `registry.summary` (option model, layout function,
  what it consumes, its filename marker): `image_grid`, `waterfall` (the
  legacy index-wise geometry, moved here) and `average`.
- `compat.convert.to_v3`: converts a v2 diagnostic the core serves into a
  recipe, built on the adapter's compile output and checked by
  recompiling; what the v3 shape does not carry is reported in its notes.

### Changed

- `render.specs.FigureSpec` is the schema's `FigureStyle` made immutable
  (one field list); `compat.v2_render` translates `RendererOptions` into a
  `FigureSpec` (`figure_v2`) and the fixed v2 summary pair (`summaries_v2`),
  and its `single_v2` / `image_grid_v2` / `waterfall_v2` draw through the
  kinds. A trace's colorbar keeps its signal label unless the document sets
  one. `FileBackground.fallback_level` may be `None` (a failed read is then
  an error).

## [0.12.1] - 2026-09-24

### Fixed

- Image colorbars (`single` / `draw_frame` and `image_grid`'s shared one) span
  the image axes as drawn instead of the layout slot: a fixed-aspect image no
  longer sits beside a colorbar taller than itself (operator figure review).
  The layout's pad, width and `extend` rules are kept; a recipe that passes a
  placement keyword (`location`, `orientation`, `shrink`, `anchor`,
  `panchor`) keeps matplotlib's placement untouched.

## [0.12.0] - 2026-09-23

### Added

- `compat.v2_render`: v2 renderer-option translation for single, image-grid and
  waterfall figures. Preserves the legacy colormap-mode/limit rules, the
  index-wise waterfall stack on the first trace's x grid with midpoint cell
  edges, sort-implied even spacing, real zero positions and configured labels;
  the image grid carries the legacy ``Scan parameter:`` title. No pyplot state
  and no file writes.

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
