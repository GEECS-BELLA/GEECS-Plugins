# Changelog — geecs-scanner

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.31.0] — 2026-07-06

### Added

- **Xopt learns from measured readbacks, not proposals** —
  `SessionOptimizationBridge.observe()` now substitutes each VOCS
  variable's proposed value with the bin-mean measured readback from the
  session's bin rows (the variable's `Device:Variable` column, NaN rows
  excluded), falling back to the proposal when the column is missing,
  empty, or all-NaN. GEECS set convergence is tolerance-bounded (e.g.
  0.05 A on magnet PSUs), so the evaluated point can differ from the
  proposed one by up to the tolerance; the optimizer now sees the point
  actually visited. Substitutions that change a value are logged at DEBUG;
  `xopt_dump.yaml` / history shapes are unchanged.
- **Optimizer device requirements exposed on the session bridge (legacy
  parity)** — `SessionOptimizationBridge.device_requirements` surfaces
  `BaseOptimizer.device_requirements` (`{"Devices": {...}}`) so
  `BlueskyScanner` can auto-provision the objective's analyzer devices
  into the save-device set, mirroring the legacy ScanManager's
  `device_manager.load_from_dictionary` merge. Read duck-typed via
  `getattr` on the geecs_bluesky side — the dependency direction stays
  GUI → geecs_bluesky.
- **Device requirements are canonicalized against the experiment
  database** (live-observed 2026-07-06, first GUI optimization test) —
  the DB (and therefore the gateway's case-sensitive CA PV names) spelled
  the camera `UC_Amp4_IR_input` while the optimizer config said
  `UC_Amp4_IR_Input`, so the auto-provisioned device failed to connect on
  every PV and aborted the scan. `SessionOptimizationBridge.device_requirements`
  now case-insensitively matches each requirement device name against the
  experiment database dict (`DatabaseDictLookup`, default experiment from
  `config.ini`) and substitutes the canonical DB spelling, logging
  corrections at INFO. Strictly best-effort and cached per bridge: any DB
  failure (no config, off-network, empty dict) logs at DEBUG and returns
  the requirements verbatim, so headless/test use is unaffected.

## [0.30.1] — 2026-07-06

### Changed

- Native-file naming is now imported from `geecs_data_utils.native_files`:
  `optimization/session_bridge.py`'s `_expected_native_file` delegates to
  the shared `native_file_path`, retiring the "do not refactor from here"
  duplication note — the contract now lives in geecs-data-utils, shared with
  GeecsBluesky's asset registry and ScanAnalysis's timestamp file mapping.
  No behavior change; the `TestExpectedNativeFile` / `TestAwaitBinAssets`
  tests pass unchanged.


## [0.30.0] — 2026-07-05

### Added

- **Bluesky/CA optimization bridge** —
  `optimization/session_bridge.py`: runs the existing config-driven
  optimization stack (BaseOptimizer, Xopt 3.1 generators, evaluators with
  ScanAnalysis analyzers incl. per-bin image averaging) on a
  `GeecsSession.optimize` scan. `SessionBinSource` presents the session's
  schema-v1 event rows as the legacy DataLogger-shaped frame
  (`Device:Variable` columns, `Bin #`, `Shotnumber`); `SessionOptimizationBridge`
  adapts the optimizer's ask/tell surface to the session suggester protocol
  (with the legacy 2-random-init behavior). `RunControl` injects
  `load_session_optimization` into `BlueskyScanner`, so GUI optimization
  scans now work with `use_bluesky=True` using the same optimizer YAML
  configs.
  The bridge awaits each bin's expected native files (direct stat — immune
  to stale SMB directory listings) before running the evaluator, and its
  `finish()` writes `xopt_dump.yaml` into the scan folder at scan end
  (legacy parity; the `seed_dump_files` warm-start source).
- `BaseEvaluator` engine seam: `EvaluatorDataSource` protocol +
  `DataLoggerSource` (the legacy in-memory path, extracted verbatim from
  `get_current_data`). New optional `data_source=` / `scan_tag=` constructor
  kwargs; passing `data_logger` alone (including post-construction
  assignment) is unchanged behavior.

- `GEECS_USE_BLUESKY` env var (`1/true/yes/on`) switches a GUI session onto
  the Bluesky backend without touching source — the GUI constructs
  `RunControl` without the `use_bluesky` argument, so the env var is the
  supported switch during the legacy → Bluesky transition (mirroring
  `GEECS_BLUESKY_ACQUISITION_MODE`). Resolver lives in
  `engine/backend_selection.py` (PyQt5-free, testable headless).

### Fixed

- `_await_bin_assets` builds expected native-file paths with the
  analyzer's `data_device_name` (the asset registry's `directory_suffix`),
  so suffixed diagnostics (FROG/magspec-style) no longer burn the full
  timeout with spurious warnings every bin; the wait also refuses to block
  a thread with a running asyncio loop (the RunEngine-loop blocking itself
  is fixed plan-side in geecs-bluesky). (PR #449 review #12)
- Fresh installs can now run the Bluesky/CA backend: the `geecs-bluesky`
  path dependency gains the `ca` extra (aioca), and the lockfile — which
  predated both the GeecsBluesky→GeecsCAGateway dependency and the
  ophyd-async 0.19.3 bump — was regenerated (the root `geecs-docs` lock
  likewise). Previously a from-lock install produced an environment with
  no gateway library, no aioca, and ophyd-async 0.16.
- Defused a latent order-dependent import cycle:
  `base_evaluator → config_models` (module-level model rebuild) `→ engine →
  scan_executor → base_optimizer → base_evaluator`. Importing
  `base_evaluator` first now works (the `config_models` import is deferred
  into `__init__`).

## [0.29.0] — 2026-06-24

### Changed (BREAKING — Xopt 2.6 → 3.1 upgrade)
- Upgraded the `xopt` dependency from `^2.6` to `^3.1` (pulls `gest-api`,
  `botorch>=0.17`, `pydantic>=2.12`, `numpy>=2.3`). Xopt 3.x adopts the
  cross-package [GEST standard](https://github.com/campa-consortium/gest-api):
  `VOCS` now lives in `gest_api`, variables/objectives/constraints are typed
  objects (not bare lists/strings), and the generator — not the `Xopt` object —
  owns the VOCS.
- `BaseOptimizer._setup_xopt` no longer passes `vocs=` to `Xopt(...)` (the
  generator carries it); `BaseOptimizer.generate()` now uses the GEST
  `generator.suggest()` verb.
- `ScanStepExecutor.generate_next_step` uses the free function
  `xopt.vocs.random_inputs(vocs, n)` — the `VOCS.random_inputs()` method was
  removed in 3.x.
- New `optimization/vocs_utils.py` centralises typed-VOCS access
  (`is_maximize`, `variable_bounds`, `bounds_of`); inspection helpers
  (`slicing.py`, `surfaces.py`) and `dump_loader.py` now use it instead of
  unpacking `vocs.variables[name]` as `[lo, hi]` or comparing
  `str(objective) == "MAXIMIZE"` (which silently broke under typed objectives).
- `inspection/dump_loader.load_xopt_dump` reads VOCS from the 3.x dump layout
  (`generator.vocs`); 2.x top-level-`vocs` dumps are no longer supported.
- BAX: removed the `MultipointBAXGenerator` subclass. Xopt 3.x `BaxGenerator`
  is observables-only (requires zero objectives), which the GEECS BAX evaluator
  already matches (`output_key=None` returns observables, never an objective).
  The `make_multipoint_bax_alignment[_l2]` factories now return a stock
  `BaxGenerator`. **Config change:** `multipoint_bax_alignment[_l2]` optimizer
  YAMLs must use an observables-only VOCS — drop the (previously vestigial,
  2.x-only) `objectives:` block and keep `observables:`. `BaseOptimizer`'s
  `best_observed_setpoint()` / `get_best()` now return `None` for objective-less
  problems. Modernised the `MultipointProbeConfig` after-validator to an
  instance method.
- `BaseOptimizer.best_observed_setpoint()` and `get_best()` now share a
  `_best_row_index()` helper that delegates to Xopt's native
  `xopt.vocs.select_best` instead of hand-rolled logic. Besides removing
  duplicated code, this makes "move to best on finish" **respect constraints**
  (the previous version only filtered errored/NaN rows). It also fixes a latent
  bug in `get_best()`, which did an ascending `sort_values(...)[:1]` and so
  returned the *worst* point for a MAXIMIZE objective.

### Added
- `tests/optimization/test_xopt3_migration.py` — pins typed-VOCS access, the
  3.x dump round-trip + seeding, the generate/evaluate loop, and BAX
  construction/generation.

### Fixed
- Seed-dump compatibility now treats VOCS **observable** names as a hard check
  (like variables/objectives), and `seed_from_dumps` filters NaN rows across
  observables as well as objectives. Without this, an observables-only BAX VOCS
  (no objectives) would accept dumps with mismatched observable names and load
  rows with missing/NaN observables.
- `tests/conftest.py` now actually patches `GeecsDatabase.collect_exp_info`
  (the docstring had long promised this but no code did it), so importing the
  engine package during test collection is network-free on a developer machine
  that has a `config.ini` but is off the lab network — previously such imports
  blocked ~75 s and failed.
- Optimization test fakes no longer pass a `name=` kwarg to `CameraConfig` /
  `Line1DConfig`; those models dropped the `name` field in an ImageAnalysis
  refactor, so the fakes were raising `extra_forbidden` at collection. Device
  identity already lives on `DiagnosticAnalysisConfig.name`. Restores 14
  previously-erroring tests in `test_config_models.py` and
  `test_evaluator_create_scan_analyzer.py`.

## [0.28.2] — 2026-06-09

### Fixed
- Bluesky-mode `RunControl.get_database_dict()` now loads the experiment device
  dictionary directly, so save-element editors can populate variable completers
  without relying on the legacy `ScanManager` database cache.

## [0.28.1] — 2026-06-09

### Changed
- Bluesky-mode `RunControl` now passes the GUI scan-event callback and timing
  YAML into `BlueskyScanner`, allowing the GUI to receive lifecycle completion
  events and letting Bluesky scans use the same shot-control configuration.
- The Scanner GUI path dependency now requests the `geecs-bluesky[tiled]` extra,
  so Tiled client support is installed with the GUI environment.

## [0.28.0] — 2026-06-01

Companion to ImageAnalysis 1.8.0 + ScanAnalysis 1.12.0 (issue #412 —
scalar-key prefix/suffix moves to ScanAnalysis). The contract the
optimizer evaluator sees is unchanged: ``analyzer.results[N].scalars``
keys are still ``{prefix}_{key}{suffix}``-shaped (defaulting to
``{device_name}_{key}`` when no overrides are set). What changed is
*which layer* applies the namespacing. The pass-through behaviour in
``BaseEvaluator._get_value`` shipped in 0.27.0 stays correct.

### Changed
- Doc-only refresh of the ``BaseEvaluator._get_value`` slot-merge
  inline comment and the ``BaseEvaluator`` class docstring. The earlier
  "inversion pending when #412 lands" note is replaced by the
  post-#412 explanation: ImageAnalysis emits bare keys, ScanAnalysis's
  ``SingleDeviceScanAnalyzer._consume_result`` adds the prefix/suffix,
  the evaluator just forwards.

### Test count
- 67 tests pass unchanged.

## [0.27.0] — 2026-06-01

Unified evaluator architecture. `MultiDeviceScanEvaluator` and
`ScalarLogEvaluator` collapse into a single `BaseEvaluator` class that
handles both diagnostic-driven analyzers and direct s-file scalar
columns. The minimum viable subclass for a new objective is now ~5
lines of Python, down from ~30. Observables and objectives are peer
first-class hooks; the BAX (observables-only) path is no longer a
degenerate special case.

### Changed (breaking)
- `BaseEvaluator` is now a concrete class (not abstract) that absorbs
  the data-source orchestration that previously lived in
  `MultiDeviceScanEvaluator` and `ScalarLogEvaluator`. Construction
  accepts:
    - `analyzers: List[str | dict]` — diagnostic stems with optional
      `{diagnostic: X, ...patch}` override entries (unchanged from
      0.26.0)
    - `scalars: List[str]` — s-file column names pulled per-shot from
      `current_data_bin` (new — was `scalar_keys` on the old
      `ScalarLogEvaluator`)
    - `objective_tag: str | None` — human-readable label written to
      `log_entries` under `Objective:<tag>`. Defaults to the subclass
      name; override via YAML kwarg or subclass class attribute
- `compute_objective(self, scalars, bin_number)` — the parameter is
  now a flat dict with **device-prefixed keys for analyzer outputs**
  (`scalars["UC_TopView_x_fwhm"]`) and bare column names for s-file
  scalars (`scalars["U_Laser:Energy"]`). The previous loose `get_scalar`
  helper with three-naming-convention fallback is gone — just dict access.
  No collisions possible because analyzer keys are always device-prefixed.
- `compute_objective` defaults to returning `None` (was: `NotImplementedError`).
  Returning `None` signals "this evaluator has no objective" — the BAX
  case where `output_key = None`. Subclasses with an objective must
  override this OR `compute_objective_from_shots`.
- `compute_observables_from_shots(self, scalars_list, bin_number)` — new
  per-shot hook, peer to `compute_objective_from_shots`. Mean-aggregates
  and delegates to `compute_observables` by default; override for custom
  observable statistics or shot-level filtering (e.g. weighted median).
- s-file scalars are always per-shot (one slot per shot row in
  `current_data_bin`). Per-bin aggregation is purely an
  image-analyzer concern (analyzer's own `scan.mode`); for raw scalars
  the subclass decides how to aggregate the per-shot list (mean, median,
  filtered, …) via `compute_*_from_shots` — no framework-level mean
  policy on scalar data.

### Removed
- `MultiDeviceScanEvaluator` class. Absorbed into `BaseEvaluator`. All
  concrete subclasses (`BeamSizeEvaluator`, `MaxCountsEvaluator`,
  `BeamPositionEvaluator`, `BeamPositionSimulationEvaluator`,
  `EBeamSourceOpt`) now inherit from `BaseEvaluator` directly.
- `ScalarLogEvaluator` class and its `observables_only` classmethod
  factory. Same capability is now `BaseEvaluator` with `analyzers=[]`
  and a subclass implementing `compute_observables`.
- `get_scalar(device, metric, results)` helper with three-naming-convention
  fallback. Replaced by direct dict access on the prefixed scalars dict.
- `primary_device` is still exposed on `BaseEvaluator` as a convenience
  for single-analyzer subclasses, but reads from `self.diagnostics[0].name`
  rather than `self.analyzer_refs[0].device_name`.

### Fixed
- `BaseEvaluator._get_value` no longer double-prefixes analyzer scalars.
  The image analyzer already emits scalars with keys like
  `"UC_TopView_x_fwhm"` (the prefix comes from
  ``camera_config.name``); the earlier draft of `_get_value` added the
  prefix again, producing `"UC_TopView_UC_TopView_x_fwhm"` and a
  `KeyError` on every objective lookup. `result.scalars` keys are now
  forwarded through unchanged.
- RFC #412 will move prefixing from ImageAnalysis to ScanAnalysis. When
  that lands, this layer's pass-through behaviour reverses — the
  framework will need to prefix bare keys with the device name itself.
  Comment in `_get_value` flags the line that will need to change.

### Net
- Files: 9 → 7 in `evaluators/`
- Public classes: 3 evaluator-base classes (`BaseEvaluator`,
  `MultiDeviceScanEvaluator`, `ScalarLogEvaluator`) → 1 (`BaseEvaluator`)
- LoC (production code): 995 → ~800

### Migration
No production YAMLs need changes — all currently subclass
`MultiDeviceScanEvaluator`, which is replaced by `BaseEvaluator` but
the subclass names are preserved. The only optimizer YAML feature that
required `ScalarLogEvaluator` directly was the simulation evaluator's
column reads, which now use the new `scalars:` config block (auto-added
by the subclass `__init__`).

## [0.26.0] — 2026-05-30

Reintroduce per-analyzer overrides on the optimizer YAML — but as a
generic `scan:` (or any-field) patch routed through the loader, not as
a magic single-field knob. Companion to ImageAnalysis 1.7.0.

### Added
- Optimizer YAML's `evaluator.kwargs.analyzers` accepts dict-form
  entries alongside bare strings:
  ```yaml
  analyzers:
    - UC_TopView                          # diagnostic as-is
    - diagnostic: UC_FROG                 # patch scan block
      scan:
        mode: per_bin
  ```
  Any field on the diagnostic can be overridden; the patch is
  deep-merged via `load_diagnostic`'s new `overrides` kwarg
  (ImageAnalysis 1.7.0) and re-validated. No code change here is
  optimization-specific — the override mechanism lives on the loader.
- `OptimizerAnalyzerEntry` — three-line envelope model
  (`diagnostic: str` + `ConfigDict(extra="allow")`) that validates the
  dict form's shape and surfaces the override patch via
  `model_extra`. Deliberately does not enumerate override fields, so
  future per-context overrides (priority, save flag, anything) need no
  code changes here.
- `_split_analyzer_entry(entry) -> (name, overrides)` — single
  envelope decoder used by both `BaseOptimizerConfig._load_and_check`
  (for `device_requirements` auto-generation) and
  `MultiDeviceScanEvaluator.__init__` (for analyzer construction).
  Keeps the two call sites in lockstep.

### Migration
Production optimizer YAMLs that previously needed `per_bin` and were
flattened to bare strings in 0.25.0 — and lost their override — are
restored to dict form with `scan: {mode: per_bin}` patches in
[GEECS-Plugins-Configs@148e222](https://github.com/GEECS-BELLA/GEECS-Plugins-Configs/commit/148e222).
Bare-string entries are still accepted and behave as before (use
diagnostic as-is).

### Note
This release reverses the override removal in 0.25.0. The reintroduced
mechanism is structurally cleaner than the 0.24.0 form it replaced:
the override capability lives in `image_analysis.config.load_diagnostic`
(generic, available to any consumer), the optimizer-side surface is a
three-line envelope model instead of the previous 95-LoC computed-field
wrapper, and the YAML vocabulary matches the diagnostic's own shape
(`scan:` block as the patch).

## [0.25.0] — 2026-05-30

Drop the optimizer's per-analyzer override surface. The diagnostic YAML
is now the single source of truth for how each analyzer runs, in both
canonical scan analysis and optimization. To change an analyzer's mode
for optimization, edit the diagnostic.

### Changed (breaking)
- Optimizer YAML's `evaluator.kwargs.analyzers` is now a flat list of
  diagnostic stems. The `{diagnostic: <stem>, analysis_mode: <mode>}`
  dict form from 0.24.0 is gone:
  ```yaml
  analyzers:
    - UC_TopView                   # was: {diagnostic: UC_TopView}
    - U_BCaveICT
  ```
- `OptimizerAnalyzerRef` (and its `to_device_requirement` method) and
  `_build_device_requirements` helper are gone from `config_models.py`.
  Device-requirements aggregation now happens inline in
  `BaseOptimizerConfig._load_and_check` — for each diagnostic stem the
  validator loads the YAML, reads `diag.name`, and templates a per-analyzer
  device block under that key.
- `MultiDeviceScanEvaluator.__init__` takes `analyzers: List[str]` instead
  of `List[dict]`. The evaluator stashes the loaded diagnostics on
  `self.diagnostics`; the `analyzer_refs` attribute is gone.
  `primary_device` now reads `self.diagnostics[0].name`. Downstream
  subclasses that touched `analyzer_refs[0].device_name` should use
  `primary_device` instead (`bcavemagspec_opt.EBeamSourceOpt` updated).
- The `analysis_mode` keyword on `create_scan_analyzer` (added in
  ScanAnalysis 1.10.0) is also gone — see ScanAnalysis 1.11.0.

### Migration
Production optimizer YAMLs under `scanner_configs/experiments/<exp>/optimizer_configs/`
need to drop their `analysis_mode:` lines and may collapse `{diagnostic: X}`
entries to bare `X` strings. Done in
[GEECS-Plugins-Configs@b2c14b1](https://github.com/GEECS-BELLA/GEECS-Plugins-Configs/commit/b2c14b1)
for the 10 Undulator YAMLs in production. If an analyzer needs a different mode in optimization than
in scan analysis, edit the diagnostic's `scan.mode` (or fork the
diagnostic if the modes legitimately differ between contexts).

## [0.24.0] — 2026-05-29

Optimizer-side modernization that exploits the post-PR-E unified-diagnostic
surface. Drops the heavyweight wrapper config the previous release introduced
in favour of a thin reference model that defers to the canonical scan-analyzer
factory.

### Changed (breaking)
- `SingleDeviceScanAnalyzerConfig` is gone. Replaced by
  `OptimizerAnalyzerRef` (~20 LoC vs the prior 170+) — just `diagnostic` +
  optional `analysis_mode`. The replaced model used `computed_field`
  shims (`device_name`, `analyzer_type`, `file_tail`, `image_analyzer`,
  `data_device_name`, plus an `image_config` property) to re-expose
  everything on the diagnostic; none of that earns its keep after
  `scan_analysis.config.create_scan_analyzer(diag)` became the canonical
  factory used by the task queue + LiveWatch. The optimizer now hands
  the loaded diagnostic straight to that factory with
  `use_injected_data=True`.
- `MultiDeviceScanEvaluatorConfig` is gone. Its only consumer
  (`BaseOptimizerConfig._load_and_check`) inlines into a small
  `_build_device_requirements` helper that walks the analyzer entries,
  validates each as `OptimizerAnalyzerRef`, and merges the per-analyzer
  device blocks. No public-API change for the optimizer YAML shape; the
  field disappearing is internal.
- `MultiDeviceScanEvaluator._create_scan_analyzer` is gone. The
  evaluator now calls
  `create_scan_analyzer(ref.diag, analysis_mode=ref.analysis_mode,
  use_injected_data=True)` directly — no manual analyzer-class
  importing, no kwarg-splicing of the typed image config into the
  analyzer's constructor, no `live_analysis` / `use_colon_scan_param`
  setattr lines (the latter were broken anyway after ScanAnalysis 1.10.0
  removed those attributes).
- Evaluator attribute renamed `analyzer_configs` → `analyzer_refs` to
  reflect the thinner role. The `_get_value` dispatch reads each
  analyzer's effective mode straight off `analyzer.analysis_mode` (the
  factory resolved `ref.analysis_mode` ↔ `scan.mode` at construction);
  no parallel cache to keep in sync. Subclasses + tests that touched
  the old attribute were updated (`bcavemagspec_opt.EBeamSourceOpt`,
  `test_multi_device_scan_evaluator.py`, `test_evaluator_bax_mode.py`,
  `test_concrete_evaluators.py`).

### Migration
The optimizer YAML shape from 0.23.0 is unchanged externally — analyzer
entries are still `diagnostic: <stem>` (+ optional `analysis_mode`).
Existing 0.23.0-style YAMLs work without edits.

## [0.23.0] — 2026-05-29

Diagnostic-driven optimizer analyzer configs.

### Changed (breaking)
- `SingleDeviceScanAnalyzerConfig` now references a unified diagnostic
  YAML by name (`diagnostic: <stem>`) instead of duplicating every
  field the diagnostic already defines. The model validator calls
  `image_analysis.config.load_diagnostic(stem)` at construction and
  exposes `device_name`, `analyzer_type`, `file_tail`,
  `image_analyzer`, `image_config`, and `data_device_name` as
  computed properties derived from the diagnostic's `image:` /
  `scan:` sections. The single override knob is `analysis_mode`,
  which beats the diagnostic's `scan.mode` when set (the common
  `per_shot` ↔ `per_bin` toggle you actually want per optimization
  run).
- `MultiDeviceScanEvaluator._create_scan_analyzer` consumes the
  diagnostic's typed image config directly (no second on-disk lookup
  via `load_camera_config` / `load_line_config`). Drops the
  `image_analysis_config.set_base_dir(...)` side effect entirely
  since `load_diagnostic` resolves its own base dir.
- The legacy optimizer YAML shape (`device_name`, `analyzer_type`,
  `file_tail`, `image_analyzer.{module, class, kwargs}`,
  `analysis_mode`, `data_device_name` as top-level analyzer fields)
  is gone. Existing optimizer YAMLs must migrate to the
  one-diagnostic-per-analyzer form:

  ```yaml
  # Before:
  analyzers:
    - device_name: UC_TopView
      analyzer_type: Array2DScanAnalyzer
      file_tail: .png
      image_analyzer:
        class_path: image_analysis.analyzers.beam_analyzer.BeamAnalyzer
      analysis_mode: per_bin

  # After:
  analyzers:
    - diagnostic: UC_TopView      # everything else inherited
      analysis_mode: per_bin      # optional override
  ```

Net win: optimizer YAMLs drop from ~8 fields per analyzer to 1–2,
analyzer configuration lives in one place (the diagnostic YAML)
instead of being copy-pasted between scan-analysis and optimization
contexts, and per-optimization-run tuning (specifically per_shot vs
per_bin) stays a one-line override.

## [0.22.1] — 2026-05-29

### Fixed
- `MultiDeviceScanEvaluator._create_scan_analyzer` now points the
  image-analysis loader at the unified diagnostic-config root
  (`scan_analysis_configs_path`) instead of the legacy
  `image_analysis_configs_path`. The optimizer code missed the
  base-dir migration that the PR-E loader-API refactor + unified-
  configs cutover landed: optimizer YAMLs were resolving
  `camera_config_name` against the empty/legacy
  `image_analysis_configs/` directory and raising `FileNotFoundError`
  on any real run. `load_camera_config` / `load_line_config` unwrap
  the `image:` subsection from unified diagnostic YAMLs transparently,
  so the new base dir works for the unified files that production
  configs have migrated to.

## [0.22.0] — 2026-05-27

Loader API consolidation (PR-E). Companion to ImageAnalysis 1.5.0 and
ScanAnalysis 1.7.0. The optimizer-side scan-analyzer construction path
now goes through the same resolved-spec shape that the diagnostic
factory uses.

### Changed
- `MultiDeviceScanEvaluator._create_scan_analyzer` reworked to consume
  the resolved `ImageAnalyzerSpec` (`class_path` + `kwargs`) directly
  and the new `image_analysis.analyzers.*` module layout. The evaluator
  no longer carries an ad-hoc local construction path; it shares the
  same spec resolution the diagnostic factory uses.
- `optimization/config_models.py` simplified: the `image_analyzer`
  field defers to the shared `ImageAnalyzerSpec` validator rather than
  re-implementing analyzer-class resolution locally.

### Added
- `tests/optimization/test_evaluator_create_scan_analyzer.py` — 264
  lines covering `MultiDeviceScanEvaluator._create_scan_analyzer`
  against the new spec-resolution path. This evaluator method was
  previously untested.

### Breaking
- Optimizer YAMLs that still reference
  `image_analysis.offline_analyzers.*` class paths must migrate to
  `image_analysis.analyzers.*` (PR-E rename).

## [0.21.0] — 2026-05-24

Companion release to the ScanAnalysis 1.6.0 unified-configs cutover.
Optimizer YAMLs that embed a `SingleDeviceScanAnalyzer` now use the
unified `ImageAnalyzerSpec` shape (string alias / alias-dict / verbose
`class_path` form) instead of the old separate `{module, class}` pair.

### Changed
- `SingleDeviceScanAnalyzerConfig.image_analyzer` now accepts the
  unified `ImageAnalyzerSpec` shape, with a `resolve_image_analyzer_value`
  field validator. The optimizer-side `_create_scan_analyzer` path uses
  the resolved `spec.class_path` / `spec.kwargs` directly — it doesn't
  go through the disk-loaded diagnostic factory, because optimizer YAMLs
  don't carry an embedded `image:` section.

### Removed
- The duplicate `ImageAnalyzerConfig` model in
  `optimization.config_models`. The single source of truth is now
  `scan_analysis.config.aliases.ImageAnalyzerSpec`.

### Breaking
- Optimizer YAMLs that embed `image_analyzer: {module: ..., class: ...}`
  must migrate to one of the unified forms — e.g.
  `image_analyzer: beam` (alias) or
  `image_analyzer: {class_path: image_analysis.analyzers.beam.BeamAnalyzer, kwargs: {...}}`.

## [0.20.0] — 2026-05-13

### Added

- **`move_to_best_on_finish` option for optimization scans.**
  `BaseOptimizerConfig` gains a `move_to_best_on_finish: bool` field (default
  `False`).  When `True`, at scan end — whether the scan completed normally or
  was stopped early via the GUI — `ScanManager.stop_scan()` calls
  `BaseOptimizer.best_observed_setpoint()` and sets each control device to that
  row's values instead of restoring the pre-scan state.  Useful for leaving the
  beamline at the empirically-best configuration the run found.  Falls back to
  initial-state restoration (with a warning log) if `X.data` is empty, not yet
  initialized, or all rows are errored / have a NaN objective.  Device-set
  failures are logged and emitted as `ScanRestoreFailedEvent` (same pattern as
  `restore_initial_state`), so the GUI accumulates them without aborting cleanup.
- **`BaseOptimizer.best_observed_setpoint()`** — returns
  `{variable_name: float}` for the best row in `X.data` (filtered for errors
  and NaN objective), or `None` if no usable rows exist.  Objective direction
  (`MAXIMIZE` / `MINIMIZE`) is respected.

## [0.19.0] — 2026-05-13

### Added

- **`bayes_ucb_explore` generator** — UCB preset with default ``beta=10.0``.
  As β grows, UCB's acquisition is dominated by predictive σ, so this
  configuration approximates pure exploration on single-objective problems
  (the surrogate-building workflow). xopt's own `BayesianExplorationGenerator`
  does **not** support single-objective VOCS — it's for constraint /
  observable exploration only — so a high-β UCB preset is the practical
  alternative.
- **`bayes_ucb` and `bayes_turbo_ucb` generators in `PREDEFINED_GENERATORS`.**
  Upper Confidence Bound (`UpperConfidenceBoundGenerator`) wired into the
  generator factory in two flavours: bare UCB and UCB inside a TuRBO trust
  region. The `beta` parameter is overridable via the standard config dict
  for both — `{"name": "bayes_ucb", "beta": 4.0}` for more exploration in
  noisy regimes where EI flattens. Default `beta=2.0`. UCB stays peaked
  under high observation noise where EI's improvement formulation goes
  flat (because expected improvement collapses when σ_noise is comparable
  to the objective signal range), making both UCB variants useful for
  comparing acquisition behaviour on the same model and data.
- **Inspection helpers promoted into `optimization/inspection`.**  The
  visualization, slicing, and GP-introspection helpers that previously lived
  inline in the example notebooks (`xopt_run_inspection.ipynb`,
  `xopt_from_scans.ipynb`) are now first-class modules:
  - `surfaces.evaluate_model_on_grid` — posterior mean/σ over a 2D slice of
    an N-D model.
  - `surfaces.acquisition_surface` — generator-specific acquisition surface
    over the same 2D slice.
  - `candidates.next_candidate` / `next_candidate_xy` — ask a generator for
    its proposed next point, optionally projected to two named variables.
  - `slicing.pick_top_varied_pair` / `best_observed_point` /
    `resolve_slice_and_fixed` / `print_slice_summary` — choose which two
    variables to slice over and how to pin the rest.
  - `column_match.match_vocs_to_sfile_column` — alias-tolerant VOCS-to-s-file
    column resolution.
  - `hypers.gp_hypers` / `gp_summary` — read GP noise / lengthscale /
    output-scale from any fitted xopt generator (handles `ModelListGP`).
  All are re-exported from `geecs_scanner.optimization.inspection` so each
  notebook collapses to a single import block.

### Changed

- **Inspection notebooks now import from the module.**  The three notebooks
  under `docs/geecs_scanner/examples/optimization/` (xopt_run_inspection,
  xopt_from_scans, magspec_objective_tuning) drop their inline copies of the
  promoted helpers in favour of the module imports.

## [0.18.0] — 2026-05-13

### Added

- **Warm-start optimization from prior dump files.**  `BaseOptimizerConfig` gains an
  optional `seed_dump_files` field (list of paths, resolved relative to the config
  YAML).  When set, `BaseOptimizer` loads each file's evaluated data, checks VOCS
  compatibility (hard error on variable or objective mismatch; warning on differing
  bounds), filters error and NaN-objective rows, then injects the combined history
  into `Xopt` via `add_data` before the scan loop begins.  The generator (e.g., a
  Bayesian GP) is pre-trained on this data so exploration starts informed rather than
  cold.  Multiple dump files are accepted; pairwise bound consistency is logged.
  Duplicate input rows that appear more than five times trigger a warning.
- **`optimization/inspection` sub-package.**  New
  `geecs_scanner.optimization.inspection` module containing:
  - `load_xopt_dump(path)` — parse an Xopt YAML dump into `(VOCS, DataFrame)`;
    used by both `BaseOptimizer.seed_from_dumps` and the inspection notebook.
  - `check_vocs_compatible(target, source, source_path)` — hard/soft VOCS
    compatibility checks with structured error messages.
  - `check_cross_dump_consistency(dump_vocs)` — pairwise bound-drift logging
    across multiple seed files.
- **Seed-aware initialization in `ScanStepExecutor`.**  `num_initialization_steps`
  is now `max(0, 2 - optimizer.n_seeded)`: runs seeded with ≥ 2 prior points skip
  random warm-up entirely and use the GP from step one.
- **`xopt_run_inspection.ipynb` updated.**  Cell 5 (dump load + Xopt rebuild) now
  delegates to `load_xopt_dump` from the new inspection module instead of inlining
  the parse logic.

## [0.17.2] — 2026-05-11

### Changed
- `ActionManager.add_action` no longer emits a WARNING when overwriting an
  existing action — this fires on every scan after the first and is expected
  behaviour. Downgraded to INFO so it remains visible in scan logs.

## [0.17.1] — 2026-05-11

### Changed
- **Improved scan log context**: shot control name/variables, full device list
  (sync / non-scalar / async), estimated acquisition time, and scan options are
  now logged at INFO level at the start of every scan log, making post-hoc
  troubleshooting easier without changing the log file's INFO threshold.
- **Clearer ECS dump messaging**: `generate_live_ECS_dump` now logs an INFO
  line before attempting the MC command and a descriptive WARNING (naming the
  MC IP) when the command goes unacknowledged, replacing the generic debug/warning
  messages that gave no indication of what was failing or why.

## [0.17.0] — 2026-05-11

### Changed
- **Python 3.11 now required.** `GeecsBluesky` (a hard dependency) requires
  `>=3.11` via `ophyd-async`; the `pyproject.toml` constraint is updated to
  reflect this. Users on 3.10 must upgrade before installing.

## [0.16.0] — 2026-05-08

### Changed — Bold Refactor: Delete and Extract

- **Phase 1 (ScanManager cleanup)**: Deleted `estimate_current_completion()` (never
  called anywhere) and the `trigger_off()` / `trigger_on()` proxy methods on
  `ScanManager` (pure one-line delegates to `TriggerController`).  The single
  internal call site in `_phase1_pre_scan()` was inlined.  Net: ~18 lines deleted.
- **Phase 2 (split `initialize_and_start_scan`)**: The 176-line method that mixed
  widget reads, YAML loading, config building, and scan submission was split into
  three named parts: `_collect_ui_scan_config()` (widget reads + YAML loading),
  `_build_exec_config()` (pure config construction), and a ~40-line orchestrator.
  Also fixed a latent `NameError` bug where `scan_config` was accessed outside its
  defining `if` block.
- **Phase 3 (AppController extraction)**: New `geecs_scanner/app/app_controller.py`.
  `AppController` owns the `RunControl` lifecycle (creation, all exception handling),
  database access (with `DatabaseDictLookup` fallback), scan submission, stop, and
  UI-coordination flags (`is_in_multiscan`, `is_in_action_library`, `is_starting`).
  `GEECSScannerWindow` now holds `self.controller: AppController` and exposes
  `@property RunControl` for backward-compatible read access.  Config-file write
  logic extracted to module-level `_write_config_if_changed()` helper.

## [0.15.0] — 2026-05-08

### Changed — Decompose Phase (D1–D5)

- **D1**: Added 18 behavioral tests for `DataLogger` in
  `tests/engine/test_data_logger.py` — covers `_log_device_data`, shot indexing,
  `FileMover` task queuing, and standby detection.
- **D2**: Extracted `FileMoveTask` and `FileMover` (~430 lines) from
  `data_logger.py` into `engine/file_mover.py`.  `ScanManager` now creates and
  owns the `FileMover`; `DataLogger.start_logging(file_mover=...)` accepts it as
  an injected dependency.  All `data_logger.file_mover.*` chains in `ScanManager`
  replaced with direct `self.file_mover.*` access.
- **D3**: Extracted `ScanLifecycleStateMachine` into `engine/lifecycle.py`.
  `ScanManager._state` / `_state_lock` / `_set_state()` / `current_state` all
  delegate to `self._lifecycle`.  Tests updated to test the state machine directly.
- **D4**: Split `ScanManager._start_scan()` (145 lines) into three named phase
  methods — `_phase1_pre_scan()`, `_resolve_scan_id()`, `_phase2_acquire()`.
  `_start_scan()` is now 12 lines and reads as a plain recipe.
- **D5**: Removed the 200 ms `QTimer` from `GEECSScannerWindow`.  Scan lifecycle
  state is now driven purely by events.  Mode transitions (multiscan, action library)
  and `RunControl` changes call `update_gui_status()` directly at the right moments.

## [0.14.0] — 2026-05-08

### Added
- Block 5: explicit state machine in `ScanManager`.  Added `_state: ScanState`,
  `_state_lock: threading.Lock`, `_set_state()`, and `current_state` property.
  All lifecycle transitions now go through `_set_state()`, which atomically
  updates `_state` and emits a `ScanLifecycleEvent`.
- `ScanState.PAUSED_ON_ERROR` — new state entered when `request_user_dialog()`
  blocks waiting for the operator's abort/continue decision.  The engine
  transitions to `STOPPING` (abort) or `RUNNING` (continue) after the dialog
  is dismissed.
- Block 7: event-driven GUI.  `GEECSScannerWindow` now subscribes to the scan
  event stream via a `pyqtSignal(object)` bridge; per-event handlers update
  status indicator colour, progress bar, button states, and restore-failure
  warnings instead of the 200 ms polling timer.
- `RunControl.__init__` accepts an `on_event` callback and passes it through
  to `ScanManager`.

### Removed
- `ScanManager.dialog_queue` — device-error dialogs are now delivered via
  `ScanDialogEvent` through the `on_event` callback / `pyqtSignal` bridge.
- `ScanManager.restore_failures` list — restore failures are now emitted as
  `ScanRestoreFailedEvent` s and accumulated by the GUI event handler.
- `RunControl.is_in_setup`, `is_busy()`, `is_in_stopping`, `is_stopping()`,
  `clear_stop_state()`, `get_progress()`, `is_active()` — all replaced by
  event-driven state tracking in `GEECSScannerWindow`.
- `GEECSScannerWindow._was_scanning` flag — no longer needed; the DONE/ABORTED
  lifecycle event is the authoritative transition signal.
- `import queue` from `scan_manager.py` and `geecs_scanner.py` — no longer
  used after `dialog_queue` removal.

## [0.13.0] — 2026-05-08

### Added
- Block 6: typed event stream for the scan engine.  `ScanEvent` dataclass
  hierarchy (`ScanLifecycleEvent`, `ScanStepEvent`, `DeviceCommandEvent`,
  `ScanErrorEvent`, `ScanRestoreFailedEvent`, `ScanDialogEvent`) defined in
  `engine/scan_events.py`.
- `ScanManager` accepts an optional `on_event: Callable[[ScanEvent], None]`
  callback (injected on construction).  Emits `ScanLifecycleEvent` at each
  state transition (INITIALIZING → RUNNING → DONE / ABORTED).
- `ScanStepExecutor.execute_scan_loop()` emits `ScanStepEvent` (phase
  "started" / "completed") before and after each step; carries `step_index`,
  `total_steps`, and `shots_completed`.
- `DeviceCommandExecutor.set()` / `.get()` emit `DeviceCommandEvent` on every
  outcome (accepted / rejected / timeout / failed).
- `_emit()` defensive wrapper — callback exceptions are caught and logged at
  DEBUG level; never propagate into the engine.
- Unit tests: `tests/engine/test_event_emission.py` — 17 network-free tests
  covering all `DeviceCommandExecutor` outcomes and the full
  `execute_scan_loop` event sequence.
- Hardware integration test skeleton:
  `tests/integration/hardware/test_scan_manager_hardware.py` — lifecycle and
  step-event assertions against live hardware (gated by
  `GEECS_HARDWARE_AVAILABLE` environment variable).
- Updated `tests/conftest.py` `hardware_available` fixture from always-skip
  stub to `GEECS_HARDWARE_AVAILABLE` env-var gate.
- Exported `ScanState`, `ScanLifecycleEvent`, `ScanStepEvent`,
  `DeviceCommandEvent`, `ScanEvent` from `geecs_scanner.engine`.

## [0.12.2] — 2026-05-08

### Changed
- `engine/data_logger.py`: replaced `geecs_python_api.tools.files.timestamping.extract_timestamp_from_file`
  with `geecs_data_utils.timestamp_from_filename`. Removes dependency on the
  deleted `GEECS-PythonAPI` timestamping module.

## [0.12.1] — 2026-05-08

### Changed
- Restored 14 log lines from DEBUG back to INFO across `action_manager`,
  `scan_data_manager`, `device_manager`, `scan_manager`, and `data_logger`.
  These were demoted in 0.9.1 to reduce terminal noise, but with the
  log-triage tooling in place the richer log signal is more valuable than
  the brevity. Restored lines: action sequence start/complete, pre-scan and
  closeout action attempts, configure-save-paths per device, scan-info file
  written, scan-info loaded, device unsubscribe lifecycle, orphan-file sweep
  start, FileMover worker started/stopped, per-device acq_timestamp during
  global time sync, and timestamp-tolerance-failed fallback message.

## [0.12.0] — 2026-05-08

### Added
- `DeviceCommandExecutor` — single policy object for all `device.set()` /
  `device.get()` calls during a scan.  Enforces a per-error-type retry
  policy: `GeecsDeviceCommandRejected` retries up to `max_retries` times;
  `GeecsDeviceExeTimeout` and `GeecsDeviceCommandFailed` escalate immediately
  (no retry, since retrying a hung or hardware-failed device makes it worse).
  The `escalate()` method routes failures to the operator dialog callback and
  sets the scan `stop_event` when the user chooses Abort.
- Exported from `geecs_scanner.engine` package.

### Changed
- `ScanStepExecutor`: replaced scattered `retry()` + `escalate_device_error()`
  calls in `move_devices_parallel_by_device` with `cmd_executor.set()` /
  `cmd_executor.escalate()`.  `max_retries` / `retry_delay` parameters removed
  from `move_devices_parallel_by_device` (now configured on the executor).
- `ScanDataManager.configure_device_save_paths`: fixed bug where the return
  value of the escalation callback was ignored, so an Abort choice never
  stopped the device-configuration loop.  Now correctly returns early on
  Abort and skips the failed device on Continue.
- `ActionManager._set_device`: routes device failures through
  `cmd_executor.set()` / `cmd_executor.escalate()`; re-raises on Abort so
  the enclosing action sequence fails cleanly.
- `ScanManager`: creates a `DeviceCommandExecutor` instance in `__init__` and
  injects it into `ScanStepExecutor`, `ScanDataManager`, and `ActionManager`.
  Also re-injects into the new `ScanDataManager` created during `reinitialize()`.

### Fixed
- `ActionControl`: `cmd_executor` was never wired into its standalone
  `ActionManager`, causing `AttributeError` on any action execution outside a
  scan.  Now creates a `DeviceCommandExecutor` at construction time.
- `ScanStepExecutor.move_devices_parallel_by_device`: `device.set()` can return
  `None` when the device raises `GeecsDeviceCommandFailed` in the UDP listener
  thread rather than the calling thread.  `None` is now treated as a command
  failure and raises `DeviceCommandError` so the normal escalation dialog fires
  instead of crashing with `TypeError` at the tolerance check.
- Scan log now includes scan config summary (device, range, step, wait, mode)
  at INFO level immediately after scan start — triage agent no longer needs to
  read the `.ini` file for context.
- Per-variable hardware set commands restored to INFO level in scan log
  (`[DeviceName] setting Var → value`); quieted previously by the async/file-mover
  noise reduction pass.
- Per-shot heartbeat (`shot N`) logged at INFO from `DataLogger` so acquisition
  progress is visible in the scan log during long wait-time steps.

## [0.11.0] — 2026-05-07

### Added
- `geecs_scanner.data_acquisition.trigger_controller.TriggerController` —
  encapsulates all shot-trigger interactions (`trigger_on`, `trigger_off`,
  `set_standby`, `singleshot`) previously spread across `ScanManager`.
  `ScanStepExecutor` now receives a `TriggerController` via constructor
  injection instead of dynamically-injected `trigger_on_fn`/`trigger_off_fn`
  callable attributes.

### Changed
- `ScanStepExecutor.__init__`: removed `shot_control` positional parameter;
  added `trigger_controller: Optional[TriggerController] = None`. The
  `hasattr` guard pattern for `trigger_on_fn`/`trigger_off_fn` is replaced
  with a typed `if self.trigger_controller is not None` check.
- `ScanManager`: `_set_trigger`, `trigger_on`, and `trigger_off` methods
  are now thin delegations to `self.trigger_controller`; `SINGLESHOT` and
  `STANDBY` states use `trigger_controller.singleshot()` and
  `trigger_controller.set_standby()`.
- `ActionManager`: removed `instantiated_devices` persistent device cache.
  Devices are now opened fresh at the start of each `execute_action` call
  (and `ping_devices_in_action_list` / `return_value`) and closed in a
  `finally` block regardless of success or failure.  This prevents stale
  TCP connections surviving across actions and removes the hidden cache that
  could hold devices open indefinitely.

## [0.10.0] — 2026-05-07

### Added
- `geecs_scanner.data_acquisition.scan_options.ScanOptions` — new Pydantic
  model for all engine-level execution options (`rep_rate_hz`,
  `enable_global_time_sync`, `global_time_tolerance_ms`, `master_control_ip`,
  `on_shot_tdms`, `save_direct_on_network`, `randomized_beeps`). Replaces the
  raw `options_dict: dict` that was previously threaded through
  `ScanManager` and `ScanStepExecutor`.

### Changed
- `ScanManager.__init__` and `reinitialize()` now accept and store a
  `ScanOptions` instance instead of a plain dict; all internal `.get("key")`
  accesses replaced with typed attribute access.
- `ScanStepExecutor.__init__` parameter `options_dict` renamed to `options:
  ScanOptions`; `on_shot_tdms` read via `options.on_shot_tdms`.
- `DataLogger.reinitialize_sound_player()` parameter changed from
  `Optional[dict]` to `Optional[ScanOptions]`.
- `SoundPlayer.__init__` no longer accepts an `options` dict; takes an explicit
  `randomized_beeps: bool` parameter instead.
- `RunControl.submit_run()` gains an optional `options: ScanOptions` parameter;
  when provided it is injected into the config dictionary before passing to
  `ScanManager.reinitialize()`.
- `GEECSScannerWindow._collect_options()` now returns a `ScanOptions` instance.
- Fixed bug where `On-Shot TDMS` and `Save Direct on Network` GUI toggles were
  defined in `BOOLEAN_OPTIONS` but absent from `OPTION_MAP`, so they were never
  collected and the engine always defaulted to `False`.
- Fixed bug where `scan_config.background` was set to a string `"True"/"False"`
  instead of a bool.

## [0.9.1] — 2026-05-07

### Changed
- Log level rationalization across all six `data_acquisition/` modules
  (`data_logger`, `device_manager`, `scan_manager`, `scan_executor`,
  `scan_data_manager`, `action_manager`).  Per-shot ticker events (standby
  checks, timestamp comparisons, async variable updates, file iteration,
  orphan retries, per-step device moves, optimizer chatter) demoted from INFO
  to DEBUG.  Two cases promoted to WARNING: file not found after all retries
  (orphaned to post-scan sweep), and missing `acq_timestamp` column during
  orphan sweep.  Lifecycle milestones (scan start/end/abort, device subscribed,
  config loaded, files written, devices synchronized) remain at INFO.
- Docstring purge in `data_acquisition/`: removed boilerplate NumPy-style
  docstrings from methods whose signatures and names are self-explanatory,
  leaving only non-obvious `Parameters`/`Returns`/`Raises` blocks and
  single-line summaries where warranted.

## [0.9.0] — 2026-05-07

### Added
- `geecs_scanner.utils.exceptions`: typed exception hierarchy rooted at `ScanError`.
  New types: `ConfigError`, `DeviceCommandError`, `TriggerError`,
  `DataFileError`. Existing names (`ActionError`, `ConflictingScanElements`,
  `ScanSetupError`, `OrphanProcessingTimeout`) are now subclasses of the
  hierarchy; all existing catch sites continue to work without change.
- `geecs_scanner.utils.retry`: `retry(fn, *, attempts, delay, backoff, catch,
  on_retry)` — centralizes retry-with-backoff logic for hardware call sites.
- `per_shot` analysis mode in `MultiDeviceScanEvaluator`: each image in a bin
  is analyzed individually instead of averaged, enabling richer per-shot
  statistical treatment (median, std dev, noise estimates for Xopt GP surrogate)
- `compute_objective_from_shots(scalar_results_list, bin_number)` hook on
  `MultiDeviceScanEvaluator`; default implementation mean-aggregates per-shot
  scalars and delegates to `compute_objective`, so existing subclasses require
  no changes when switching from `per_bin` to `per_shot`
- Mixed-mode support: when analyzers have different `analysis_mode` settings,
  `per_bin` scalars are merged into every shot dict before
  `compute_objective_from_shots` is called
- `ScalarLogEvaluator` — a new `BaseEvaluator` subclass that reads scalars
  directly from `log_entries` columns with no image analysis required; supports
  the same hook API (`compute_objective`, `compute_objective_from_shots`,
  `compute_observables`) and observables-only mode via `observables_only()`
- CI-friendly test suite (82 tests, no network or scan files):
  `test_base_evaluator`, `test_evaluator_get_scalar`, `test_evaluator_bax_mode`,
  `test_config_models`, `test_multi_device_scan_evaluator`,
  `test_scalar_log_evaluator`, `test_concrete_evaluators` (uses real
  `ImageAnalyzerResult` with synthetic scalars — no image files), plus shared
  fixtures (`FakeDataLogger`, `make_log_entries`) in `tests/optimization/conftest.py`

### Removed
- Legacy evaluators `ALine3_FWHM.py` and `HiResMagCam.py` (dead code, no known
  callers outside this repo; superseded by the `MultiDeviceScanEvaluator` pattern)
- `evaluation_mode` field removed from `BaseOptimizer` and `BaseOptimizerConfig`
  (was stored but never read; analysis mode is configured per-analyzer via
  `SingleDeviceScanAnalyzerConfig.analysis_mode`)
- Dead `move_devices()` method removed from `ScanStepExecutor`; only
  `move_devices_parallel_by_device()` remains (#291)

### Changed
- `DeviceSynchronizationError`, `DeviceSynchronizationTimeout`, and
  `ScanAbortedError` promoted from local definitions inside `scan_manager.py`
  to `geecs_scanner.utils.exceptions`. Import paths updated; no behaviour change.
- `move_devices_parallel_by_device()` uses `retry()` for hardware exceptions and
  raises `DeviceCommandError` (with chaining) on exhaustion; tolerance failures
  are logged as WARNING and no longer trigger a retry (#291)
- `_set_trigger()` in `ScanManager` uses `retry()` and raises `TriggerError` on
  exhaustion; `_start_scan()` catches `TriggerError` with `logger.critical` (#291)
- `FileMover._move_file()` retries `shutil.move()` on `OSError` (3 attempts,
  exponential backoff) and raises `DataFileError` on exhaustion; callers no longer
  silently discard move failures (#292)
- `FileMover._process_task()` and `_post_process_orphaned_files()` guard
  `home_dir.iterdir()` against `OSError` so a disconnected network share raises a
  typed exception rather than crashing a worker thread silently (#292)
- `ScanDataManager` filesystem failures (`initialize_tdms_writers`, `save_to_txt_and_h5`,
  `_make_sFile`) now chain into `DataFileError`; `process_results()` catches
  `DataFileError` explicitly before the broad handler (#292)
- `BaseEvaluator` stripped of dead-code methods (`_gather_shot_entries`,
  `validate_variable_keys_against_requirements`, `log_objective_result`,
  `get_device_shot_path`, `convert_log_entries_to_df`, `get_shotnumbers_for_bin`);
  `pandas` import moved inside `get_current_data` to avoid module-level import cost
- `BaseEvaluator` now owns the shared hook API: `compute_objective`,
  `compute_objective_from_shots` (default mean-aggregation), `compute_observables`
  (default empty dict), and `_compute_outputs` helper that handles objective
  computation, observable merging, and output-key shadowing checks; eliminates
  duplication that previously existed in both `MultiDeviceScanEvaluator` and
  `ScalarLogEvaluator`
- `MultiDeviceScanEvaluator`: unified `merged` slots approach replaces the
  `has_per_shot` branching; added `primary_device` property; `_get_value` now
  delegates to `_compute_outputs` after building the shot list
- `config_models.py`: `SaveDeviceConfig` import moved to `TYPE_CHECKING` + lazy
  inside `_load_and_check` to break the module-level chain to live DB connections

### Fixed
- `pint` pinned to `>=0.24` in `pyproject.toml`; lock file updated from 0.22 to
  0.24.4, resolving a `NumPy 2.0` incompatibility (`np.cumproduct` removal) that
  prevented `image_analysis.types` from being imported in tests
- `BaseOptimizerConfig.model_rebuild()` called at module load so Pydantic v2 can
  resolve the `SaveDeviceConfig` forward reference at validation time; previously
  raised `PydanticUserError` and prevented any optimization run from starting
- Closes #339

## [0.8.2] — 2026-04-15

### Fixed
- Orphan file-move tasks now drain in parallel instead of serially. The
  0.5 s retry delay was previously applied inside `move_files_by_timestamp`
  (the queueing call), which caused `_post_process_orphan_task` to sleep
  0.5 s per task before each enqueue — serialising the entire drain through
  a single thread regardless of the 16-worker pool. The sleep is now applied
  inside `_process_task` (the worker), so all orphan tasks are queued
  immediately and processed concurrently, improving end-of-scan drain
  throughput from ~2 files/s to ~20 files/s
- `_post_process_orphan_task` now resets `retry_count` to 0 before
  re-queuing each task, eliminating the 0.5 s per-worker sleep during the
  post-scan drain. No new files are written after the scan ends, so the
  delay serves no purpose and was costing ~2 s of wall time for a typical
  60-task backlog across 16 workers
- `PermissionError` on `file.is_file()` inside `_process_task` no longer
  crashes the entire task. When a device holds a write lock on a file at
  the moment the worker tries to stat it, the file is now skipped and the
  task continues; the retry or end-of-scan orphan sweep picks it up once
  the lock is released

## [0.8.1] — 2026-04-15

### Fixed
- **Qt thread-safety**: device-error dialogs are now shown on the Qt main thread
  instead of the scan worker thread, eliminating the production hang where
  `exec_()` blocked indefinitely (closes #312). Worker threads submit a
  `DialogRequest` to `ScanManager.dialog_queue` and block on a `threading.Event`;
  the 200 ms `update_gui_status` timer drains the queue and shows the dialog safely
- **Full command-error coverage**: all three error types (`GeecsDeviceExeTimeout`,
  `GeecsDeviceCommandRejected`, `GeecsDeviceCommandFailed`) are now caught and
  escalated with a Continue / Abort dialog at every device-set call site:
  `ScanStepExecutor`, `ScanManager._set_trigger`, `ScanDataManager.configure_device_save_paths`,
  `ActionManager._set_device`, and `ScanManager.restore_initial_state`
- **All queued variables shown in dialog**: when a device set fails, the dialog
  lists every variable queued for that device so the operator knows the full
  hardware state to check
- **Root-cause exception preserved across retries**: the first exception
  (e.g. `GeecsDeviceExeTimeout`) is tracked and shown even if subsequent retry
  attempts produce a different error type (e.g. `GeecsDeviceCommandRejected`)
- **`restore_initial_state` deadlock eliminated**: calling a blocking dialog from
  the scan worker thread while `stop_scanning_thread().join()` waited on the main
  thread caused a hard freeze. Restore failures are now collected non-blocking into
  `ScanManager.restore_failures`; a one-shot `QMessageBox` is shown by the main
  thread once the scan thread has exited
- **`GeecsDeviceInstantiationError` device name surfaced**: the reinitialize-failure
  dialog now shows the failing device name (e.g. "TCP connection test failed for
  UC_ModeImager") instead of the generic "Check log for problem device(s)"
- **Action library error consistency**: three previously inconsistent outcomes
  (GUI crash, silent log, popup) unified — `GeecsDeviceInstantiationError` no
  longer crashes via a missing `.message` attribute; `DEVICE_COMMAND_ERRORS`
  during action execution now surface the same error dialog via the
  `on_user_prompt` callback instead of silently logging

## [0.8.0] — 2026-04-13

### Fixed
- `stop_scan()` now writes scalar data files before any device interaction,
  ensuring `ScanData*.txt` and `s*.txt` are always produced even if a closeout
  action fails (closes #309)
- Closeout action failure no longer aborts the remaining shutdown sequence
- `_clear_existing_devices()` now disconnects devices in parallel via
  `ThreadPoolExecutor`, eliminating O(N) serial teardown (closes #308)
- `_stop_saving_devices()` dispatches `save=off` to all camera devices in
  parallel; per-command exceptions are caught individually so one failure does
  not skip remaining devices

### Removed
- Deprecated `save_data` boolean flag removed from `ScanManager`
