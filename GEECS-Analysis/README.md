# GEECS Analysis

The new analysis core under development on the analysis integration branch.
This first slice provides pure, ordered processing for coordinate-aware 1D
traces and 2D images. Existing consumers still use ImageAnalysis/ScanAnalysis.

```python
from geecs_analysis.specs import Pipeline
from geecs_analysis.pipeline import apply_pipeline
from geecs_data_utils.frames import Frame

pipeline = Pipeline.model_validate({"steps": [
    {"step": "background_constant", "level": 90},
    {"step": "roi", "bounds": [[1, 600], [1, 600]], "units": "index"},
    {"step": "zero_below", "level": 0},
]})
processed = apply_pipeline(Frame.from_array(image), pipeline)
```

Steps may repeat and run in the supplied order. Specs reject unknown fields,
nonfinite parameters and invalid kernels. Importing `geecs_analysis.specs`
requires only Pydantic and the standard library.

ROI bounds follow numpy dimension order: `(y, x)` for an image, `(x,)` for a
trace. Index bounds are nonnegative half-open slices; omitted limits extend to
the edge. Axis bounds select inclusive physical coordinates, preserving sample
order (including descending or nonuniform axes). Empty selections raise.
Repeated crops retain the original coordinates and shot identity.

`clip_below` floors samples at the level; `zero_below` sets samples strictly
below the level to zero. These are distinct operations for nonzero thresholds.
Median and Gaussian filters use SciPy's reflect boundary mode on sample indices.
Frames may contain NaN/Inf samples; each numerical operation retains its
NumPy/SciPy semantics (for example, `zero_below` maps NaN to zero, as the
legacy image threshold does). There is no general invalid-data cleanup.

Reading inputs and resolving backgrounds belong to data-utils. These processing
steps neither read nor write files and have no scan, GUI, or device dependencies.
See [the migration plan](../Planning/analysis_refactor.md) for the next layers.

## Measurements

```python
from geecs_analysis.specs import Analysis
from geecs_analysis.run import analyze

recipe = Analysis(steps=pipeline.steps, measure={"kind": "beam"})
result = analyze(Frame.from_array(image), recipe)
# result.scalars: 18 bare beam keys; result.frame: processed frame
# result.overlays: projection_x, projection_y, com (when finite)
# result.notes: explicit names of any nonfinite scalars
```

`line` emits six trace statistics; `none` keeps only the processed frame.
`recipe.measure.emitted_scalars()` discovers keys without numerical imports.
Beam `enabled_stats` and `compute_slopes` preserve the v2 selection contract.

Algorithms retain legacy conventions for this migration: line intensity is a
sample sum (not quadrature); centroid and widths are computed in index space
then converted to axis units using interpolation and local spacing. Descending
axes therefore retain signed widths. Diagonal beam statistics and optional
slopes stay in local index space. Scientific changes belong in a separately
validated change. Measurement runs own their scratch arrays and never mutate
caller input, including when legacy RMS clips negative values internally.

## Existing v2 documents

```python
from geecs_analysis.compat.v2 import compile_v2, analyze_v2

compiled = compile_v2(document)  # already-validated AnalysisDiagnostic
result = analyze_v2(raw_array, compiled)  # HxW camera or Nx2 trace
```

Compilation is numpy-free and reads no files. It snapshots supported settings;
subsequent editor mutations do not change a compiled run. Unsupported active
features raise `UnsupportedRecipe`; this module does not invoke a fallback.
Readers remain separate and supply native-dtype arrays so legacy trace scaling
precision is preserved. Trace processing is currently restricted to float64,
with float32/float64 storage rounding before measurement. Preprocessing-only
`trace` recipes with active ROI stay unsupported: legacy may return an empty
array, which Frame intentionally cannot represent. Existing consumer
routes have not switched yet.

The v2 boundary explicitly preserves old camera ROI fallback/origin semantics
and the old float64 line result's clipped negative values, without mutating
inputs. Direct `Analysis` recipes retain the new coordinate-preserving,
immutable-frame semantics. No schema file is rewritten.

The migration harness accepts `capture --backend core` (legacy is the default)
and compares these outputs against existing snapshots with exact equality.
Readers and input fingerprints are identical for both backend paths.

## Figures

```python
from geecs_analysis.render import single, waterfall
from geecs_analysis.render.specs import FigureSpec

figure = single(result, FigureSpec(
    imshow={"cmap": "plasma"},
    colorbar={"label": "counts"},
    overlays={"com": {"marker": "+", "color": "cyan"}},
))
# The caller owns figure display/export; rendering never saves files.
```

FigureSpec groups matplotlib kwargs under `fig`, `axes`, `imshow`,
`pcolormesh`, `plot`, `colorbar` and per-id `overlays`. Preview rendering
validates these arguments and wraps drawing failures in RenderError.
Use overlay `hidden: true` to omit it, or projection `scale` for its fraction
of the displayed image. `colorbar.show: false` omits the colorbar.

Images retain calibrated sample-center geometry. Uniform axes use `imshow`;
nonuniform axes use midpoint cell edges and `pcolormesh`, with the respective
keyword group (set common palette/limits in both groups if either is possible).
Axes must be strictly monotonic; traces may retain arbitrary sample order.
The image extent is derived from coordinates and cannot be overridden.

`waterfall(frames, positions, style)` stacks already-grouped 1D frames at
explicit shot/bin coordinates. All traces must share the exact x grid and
units. It never silently resamples, bins or averages. Both layouts use fresh
matplotlib Figure objects and are suitable for the portal's worker threads.
