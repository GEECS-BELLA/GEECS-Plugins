# GEECS-Analysis

The replacement analysis core, developed on `codex/analysis-refactor`.
Read `../Planning/analysis_refactor.md` for scope, acceptance gates and migration.
The optimizer uses this core directly for supported v2 camera recipes. Portal
image processing and unsaved-document rendering use it for supported recipes,
with compilation-time fallback for unported recipes. Explicit scan execution
still uses ScanAnalysis until its runner/sinks and acceptance tests land.

## Boundaries

- Steps are pure `Frame -> Frame`; Frame/Axis/ShotMeta live in
  `geecs_data_utils.frames`. Axes follow numpy order: `(y, x)` or `(x,)`.
  Input-bound steps additionally receive one already-loaded immutable Frame.
  Declare the spec's binding-key field with `@step(input_field=...)`; execution
  preflights and snapshots required inputs before processing. Never read paths
  from inside a step. Backgrounds default to exact coordinates/units; explicit
  sample alignment supports metadata-free legacy backgrounds, with equal shapes.
- No readers, paths, config lookup, filesystem I/O or pyplot in steps, measures,
  algorithms or ephemeral execution. Data-utils resolves input and dependencies;
  explicit sinks will own output writes. Analysis never creates scan folders.
- `geecs_analysis.specs` must import without numpy/scipy/matplotlib or the
  data-utils package. Step specs sit beside their pure function; numerical
  imports belong inside the function and Frame annotations under TYPE_CHECKING.
- Register each builtin in `steps/__init__.py`; the registry constructs the
  discriminated spec union. Adding a builtin must not require a dispatcher edit.
  Runtime/plugin registration after spec construction is not supported yet.
- Preserve operation order and duplicates. New coordinate-changing operations
  must transform axes together with samples. Filtering acts on sample indices,
  even when coordinates are nonuniform; it does not resample.
- Preserve legacy scientific algorithms when migrating. Differential tests may
  import ImageAnalysis as an oracle; production code must not depend on it.
  Never equate NaNs to claim numerical parity; expose undefined measurements.

## Tests

Runs in the root Poetry environment and root CI leg:
`poetry run pytest GEECS-Analysis/tests -q`.
Pure invariants plus numerical comparisons with the legacy functions protect
this layer. Test schema imports in a fresh interpreter to detect eager imports.
Archived data acceptance is a separate integration gate, not simulated by unit
fixtures. Run `scripts/check.sh --all` before a PR, as required by root policy.

## Measurement compatibility

`run.analyze` processes a Frame and returns a Measurement; no file/scan state.
`specs.Analysis` adds a measure to Pipeline. `emitted_scalars()` is numpy-free
and must match the actual result keys and the existing optimizer contract.

The algorithms are intentionally ported with their numerical conventions:
line moments are calculated in index space, coordinates are interpolated at the
centroid, widths use local dx (even its sign on descending axes), and integrated
intensity is the sample sum after the RMS helper clips negatives. This refactor
must not silently redefine these quantities. Beam diagonal metrics and slopes
stay in local sample-index coordinates, as before.

LineBasicStats owns scratch because its RMS helper mutates samples. Measures
never mutate the processed Frame; the old float64 LineAnalyzer result could
alias that scratch, while its float32 result did not. The v2 adapter must
explicitly handle any output-array compatibility needed by float64 recipes;
this core preserves the input Frame and scalar math. Do not silently use a
mutable Frame to reproduce the old accidental aliasing.

Nonfinite scalars remain visible with notes. Never replace them with zero or
use matching NaNs as evidence of scientific parity. Overlays have stable ids;
centroid markers are omitted when their coordinates are nonfinite.

## The recipe (format 3)

`geecs_schemas.analysis.AnalysisRecipe` is this core's native document.
The schema types its frame (naming, `input`, `inputs`, `scan`, `figure`,
the summary kinds); its `steps` and `measure` are registry references
(`StepRef` / `MeasureRef`: the registered name plus parameters as written)
that `recipe.compile_recipe` binds here — unknown names or parameters,
undeclared or unused frame bindings, and a step or measure that does not
process the input's frames raise `RecipeError`. Never duplicate step or
measure specs into the schema package; the registry is the vocabulary and
the editor lists it from here. Both formats compile to the one in-memory
recipe (`V2Recipe`, renamed at A5) and run through `analyze_v2`; consumers
call `compile_document` / `figure_of` / `summaries_of` / `is_line` and
never ask which format they hold. `compat.convert.to_v3` converts a v2
diagnostic the core serves, built on the adapter's compile output and
checked by recompiling; it reports what v3 does not carry instead of
dropping it. A v3 recipe has no inactive sections: coordinates start where
the first crop starts.

## v2 compatibility

`compat.v2.compile_v2` compiles an already-validated schema document, without
I/O or numerical imports. `analyze_v2` takes loaded native-dtype arrays;
retaining native trace precision until axis scaling is deliberate. Active
unsupported operations raise UnsupportedRecipe before processing. Callers
choose whether to retain their old route; no hidden fallback lives here.
`allow_file_backgrounds=True` declares camera file requests without reading
them; only a source host that can bind those requests should opt in. Default
compilation still refuses them, including for the live optimizer. The source
adapter in `scan_analysis.core_inputs` loads via data-utils and applies the v2
constant fallback on reader/conversion failure. `analyze_v2(inputs=...)` never
loads a file or chooses fallback. Scan-background directives remain unported.
The compiler's supported subset is documented in its docstring and pinned by
differential tests. Never silently skip an active unported operation.
Explicit identity transforms compile to no steps; fixed-canvas rotation is
supported. Flips and distortion correction still raise UnsupportedRecipe.

Legacy camera ROI empty selections keep the full image, and beam coordinates
use the configured origin once even when ROI is inactive/repeated. Legacy
float64 line result values reflect RMS negative clipping; float32 results do
not. Preserve these values at the v2 boundary with private copies. Do not
propagate these quirks into pure steps or general Frame semantics.

## Rendering

`render.single` / `render.waterfall` return matplotlib Figure objects; no
pyplot and no file writes. Sources/runners own grouping, interpolation and
averaging. Waterfall requires matching trace grids and units; positions are
explicit coordinates, not inferred indices. Uniform images use imshow,
nonuniform rectilinear images use pcolormesh with midpoint cell edges;
nonmonotonic/duplicate image axes are refused. Both retain original samples.
`render.specs.FigureSpec` remains numpy-free. Matplotlib kwargs are intentionally
open-ended and validated by preview rendering, with RenderError at the layout
boundary. Deep-copy kwargs before passing to matplotlib; normalization objects
can be mutated by rendering. Preserve geometry rather than accepting arbitrary
extent overrides. New overlay families must carry data, never render callbacks.

`render.image_grid` draws image measurements with per-panel coordinates and a
single palette/colorbar. It accepts titles and a positive column count. Palette
keys in imshow/pcolormesh apply to both artist types; conflicting keys fail.
Missing limits autoscale over finite samples across all panels, and degenerate
limits expand before per-panel copies so colorbar mutation cannot desynchronize
them. Signal units must agree. The caller still owns aggregation and saving.

`render.specs.FigureSpec` is the schema's `FigureStyle` made immutable: the
recipe's `figure:` block, the per-frame draw every product and every summary
panel reuses. Keep the field list in the schema; add behaviour here.

`summaries/` holds the frozen summary kinds, one file each, registered with
`registry.summary(spec, consumes=..., filename=...)`: the schema option model,
the layout function `(results, positions, label, options, figure) -> Figure`,
whether it consumes the ordered panels or the noscan average, and the file
marker the sink appends to the device name (the portal's filename parser and
MCP's display-file contract read those markers — never change one). The
frame dimensionality a kind draws is the option model's `frame_ndim`, checked
by the document; adding a kind is one file here plus its option model in
GEECS-Schemas, no sink or dispatcher edit. The waterfall kind deliberately
keeps legacy geometry (index-wise stack on the first trace's x grid, midpoint
cell edges, a sort key implying even row spacing unless overridden) rather
than weakening the general renderer's same-grid rule.

`compat.v2_render` translates v2 `RendererOptions` into a `FigureSpec`
(`figure_v2`: the image palette rule as static keywords, a centred norm for
the diverging mode) and the fixed v2 summary pair (`summaries_v2`); its
`single_v2` / `image_grid_v2` / `waterfall_v2` draw through the kinds, so v2
and v3 documents share one layout per kind. Palette rules mirror the old
`_get_colormap_params_1d` and are pinned by a differential test that imports
ScanAnalysis as an oracle; production code must not.

## Mask and interpolation conventions

`circular_mask` defaults to local sample indices, with center in numpy `(y, x)`
order. `units="axis"` uses the existing Frame coordinates. The v2 adapter
reverses the legacy `(x, y)` center once and deliberately keeps index units,
including after a crop. Masking retains axes and provenance.

`interpolate` replaces the trace axis with a uniform grid and uses the legacy
`numpy.interp` convention with zero padding. It does not sort, deduplicate or
reverse source coordinates; do not silently change old numerical behavior
while migrating. Axis units/labels and signal units/provenance survive.

## Camera fiducials and rotation

`crosshair_mask` uses local integer `(y, x)` sample centers. Preserve the old
integer half-dimension bars, clipped-before-rotation geometry, OpenCV linear
warp and strict >0.5 mask threshold. Multiplicative masking deliberately retains
legacy nonfinite propagation. Numerical imports, including cv2, stay lazy.
`rotate` moves samples on a fixed output grid: SciPy cubic interpolation with
prefilter=False, reshape=False. It preserves axes, shape and metadata; its angle
is in sample-index space, not a transformation of world coordinates. The v2
adapter reverses crosshair centers once and retains every mask/rotation order.

## Execution units

`compat.v2_run` is the streaming v2 orchestration boundary. The host supplies
explicit groups and a loader; no path/config discovery or output writes belong
here. Raw native arrays must be averaged before v2 axis scaling and processing,
using mean rather than nanmean. Preserve both full bin membership (legacy
scalar propagation) and actual loaded contributors. Failures are explicit
outcomes, and raw buffers are released before yielding. Load in declared order
for reproducible sums; sources can reuse buffers, so snapshot each returned
array. This runner does not replace scan grouping, scalar/output sinks or the
factory by itself. Pure analyze/analyze_v2 remain loaded-input-only APIs.

`compat.v2_average.average_results` owns the post-analysis summary conventions:
noscan uses ordinary means and omits shot overlays; bin summaries use nanmean
and average projections. Never re-analyze the averaged processed frame to
produce mean scalars. Trace samples AND coordinates reduce at the recipe's
storage dtype, preserving float32 accumulation and the legacy index-wise axis
average. This is deliberately a v2 boundary, not a general alignment/resampling
operation. Mismatched shapes skip the averaged figure; unit/rank and camera-axis
mismatches raise. Aggregate frames have no individual shot identity.
