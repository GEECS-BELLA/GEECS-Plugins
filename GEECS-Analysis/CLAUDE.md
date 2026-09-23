# GEECS-Analysis

The replacement analysis core, developed on `codex/analysis-refactor`.
Read `../Planning/analysis_refactor.md` for scope, acceptance gates and migration.
Production consumers still use ImageAnalysis/ScanAnalysis until their adapters
and differential acceptance tests land. Do not switch consumers prematurely.

## Boundaries

- Steps are pure `Frame -> Frame`; Frame/Axis/ShotMeta live in
  `geecs_data_utils.frames`. Axes follow numpy order: `(y, x)` or `(x,)`.
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

## v2 compatibility

`compat.v2.compile_v2` compiles an already-validated schema document, without
I/O or numerical imports. `analyze_v2` takes loaded native-dtype arrays;
retaining native trace precision until axis scaling is deliberate. Active
unsupported operations raise UnsupportedRecipe before processing. Callers
choose whether to retain their old route; no hidden fallback lives here.
The compiler's supported subset is documented in its docstring and pinned by
differential tests. Never silently skip an active unported operation.

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
