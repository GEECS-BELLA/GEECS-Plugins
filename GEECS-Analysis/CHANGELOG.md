# Changelog

## [0.21.0] - 2026-09-29

### Added

- The `haso` measure (`measures/haso.py`): HASO wavefront reconstruction
  through a host-supplied WaveKit engine (service `haso`), a rewrite for
  the v3 recipe of the deleted v2 `haso` analyzer keeping its function.
  `HasoSpec`: `sensor_config` (a file name the host resolves), `mask`
  (numpy slice bounds on the slopes grid; unset keeps the sensor's pupil),
  `filters` (the legacy defaults: tilt x/y, curvature, astigmatism 0/45
  removed), `wavelength_nm` 800, `start_subpupil` (87, 64), `zonal_prefs`
  (100, 500, 1e-6). The processed frame is rounded and clipped to the
  sensor's uint16 pixels before the engine sees it (a numpy background
  subtraction equals the SDK's, verified); the processed zonal phase is
  the measurement frame, raw phase / intensity / slopes x, y / pupil the
  extras, `phase_rms` and `phase_pv` inside the pupil the scalars.
- `@measure(shot_store=...)` / `MeasureDefinition.shot_store`: a measure
  may name the per-scan store a scan host writes every single-shot frame
  and extras to (the `haso` measure's `wavefront`), beside `sidecar` for
  1D extras.

## [0.20.0] - 2026-09-28

### Added

- Scan backgrounds compile (#1003 item 5): a v2 `scan.background_source`
  (`scan_number` → that scan's mean, `from_current_scan` → this scan's
  median or percentile) and a v3 `from_scan` frame input become
  `ScanBackground` requests on the compiled recipe, with the background
  section compiled to `background_frame` on the computed frame (plus the
  additional constant), as the legacy wrapper rewrote it. The host computes
  the frame; the core never reads a scan. `autodetect` stays unported.
  `to_v3` converts them to `from_scan` inputs.

## [0.19.0] - 2026-09-28

### Changed

- `compile_v2` compiles `trace` recipes with an active ROI (unblocks
  148Spectro, #1003). A shot whose ROI selects no samples fails explicitly
  ("ROI selects no samples"), as a `line` recipe's already did; the legacy
  preprocessing-only analyzer returned an empty trace. Otherwise the output
  equals the legacy analyzer's (pinned across in-range, clipped, outside and
  single-point bounds at both storage dtypes).

## [0.18.0] - 2026-09-28

### Added

- The `ict` measure: ICT charge (`charge_pC`) and pulse time
  (`ICT Signal Peak_us`) from an oscilloscope trace, the legacy
  `ICT1DAnalyzer` (#1003). `algorithms.ict` is ImageAnalysis'
  `apply_ict_analysis` ported unchanged; a differential test holds it to
  the original bit for bit over 36 traces, including pulses near either
  end. `compile_v2` compiles the v2 `ict` kind to it and `to_v3` converts
  those diagnostics.

### Changed (from the legacy route)

- The charge filters the stored (float32-rounded) trace at float64; the
  legacy analyzer handed scipy the float32 array, so its low-pass ran in
  float32. On a real BCave ICT scan (26_0924 Scan011, 31 shots of
  0.4–2.7 pC) charges agree to a median 1e-7 and at most 7e-6 relative;
  pulse times are identical.
- A trace the algorithm cannot analyze gives NaN scalars with a note; the
  legacy analyzer wrote 0 pC, indistinguishable from no charge.
- Products store the processed trace at the recipe's `storage_dtype`
  (float32 by default); the legacy analyzer saved its raw input at float64.

## [0.17.0] - 2026-09-27

### Added

- `compile_v2` compiles the v2 `line_stitcher` kind to the `line` measure;
  the scan host joins the sibling traces. `to_v3` carries the siblings into
  `input.siblings` (checked on the converted recipe) and notes the dropped
  `output_label`.

## [0.16.0] - 2026-09-26

### Added

- The `frog` measure: GRENOUILLE/FROG pulse retrieval (#1003 item 6). It
  calls a retriever the host binds — Kane's 32-bit FROG.dll through
  ImageAnalysis' `FrogDllRetrieval`, natively or under Wine — and packages
  the result as the legacy `GrenouilleAnalyzer` did: `temporal_fwhm`,
  `spectral_fwhm`, `frog_error`, `frog_iterations`, `tw_per_joule`, the
  retrieved trace as the frame with its two projections, and the
  temporal/spectral lineouts as extras. `compile_v2` now compiles the v2
  `frog_retrieval` kind to it (so `to_v3` converts those diagnostics); a
  differential test holds it to the legacy analyzer on the same fake DLL.
- Measure **services**: `@measure(..., service=name)` declares a
  host-supplied collaborator the measure calls (the core may not start a
  program or read config). `bind_inputs(..., measure=)` binds it from
  `inputs`, `pipeline.apply_measure` hands it over, and it travels to pool
  workers with the inputs. `@measure(..., sidecar=name)` names the per-shot
  table a scan host writes from the measurement's extras.
- `Measurement.extras`: named auxiliary frames beside the main frame,
  neither drawn nor averaged; pickled with the measurement.

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
