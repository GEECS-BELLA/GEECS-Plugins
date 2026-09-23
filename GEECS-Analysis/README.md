# GEECS Analysis

The new analysis core under development on the analysis integration branch.
This first slice provides pure, ordered processing for coordinate-aware 1D
traces and 2D images. GeecsBluesky's optimizer uses this core directly. The portal
uses it for supported processing/preview recipes, retaining legacy fallback for
unported recipes. Explicit scan execution still uses ScanAnalysis.

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

`circular_mask` uses a `(y, x)` center and local sample indices by default;
`units: axis` uses physical coordinates. It preserves the image axes.
`interpolate` resamples a trace onto `count` uniformly spaced coordinates,
with optional `lower`/`upper` bounds and zero padding outside the input range.
It preserves the legacy `numpy.interp` ordering semantics: samples are not
sorted or reversed automatically. Both steps preserve units and shot identity.

Reading inputs and resolving backgrounds belong to data-utils. These processing
steps neither read nor write files and have no scan, GUI, or device dependencies.
See [the migration plan](../Planning/analysis_refactor.md) for the next layers.

Loaded backgrounds can be bound by name to `background_frame` steps:

```python
pipeline = Pipeline(steps=[{"step": "background_frame", "source": "dark"}])
processed = apply_pipeline(frame, pipeline, inputs={"dark": background_frame})
```

`analyze` accepts the same `inputs` mapping. Bindings contain immutable Frames;
the mapping is snapshotted before processing and every required name must exist.
Specs carry names only, never arrays or instructions to open files. By default,
backgrounds must match shape, coordinates and signal/axis units. Explicit
`alignment: samples` supports legacy backgrounds without coordinate metadata,
while still requiring identical shapes. Neither mode broadcasts or resamples.
Subtraction preserves negative samples, axes and shot identity; after a crop,
the caller must supply a background matching that cropped frame.

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
array, which Frame intentionally cannot represent. Identity camera transforms
are accepted; active geometric transforms remain unsupported. The optimizer
compiles supported camera recipes once before acquisition.

Source hosts can opt into camera file backgrounds with
`compile_v2(document, allow_file_backgrounds=True)`. The resulting
`file_backgrounds` tuple declares immutable binding names, path strings and
fallback levels; compilation itself never loads them. `analyze_v2` accepts the
same `inputs` mapping as pure analysis and refuses missing bindings. Default
compilation still rejects file-dependent recipes, including in the live
optimizer. Scan-context background aggregation remains unsupported.

`scan_analysis.core_inputs.prepare_v2` is the source adapter used by the portal.
It loads each background once through data-utils, preserves load-failure
constant fallback and additional-offset order, and returns a compiled recipe
plus bound Frames. A successfully loaded shape mismatch remains an error.
An explicit device `data_dir` resolves `{scan_dir}` for scan callers; previews
leave it literal, preserving their previous fallback behavior. No files are
written and the supplied document is unchanged.

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
