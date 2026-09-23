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
