# Optimization on the native plan layer — the arc brief

> Implementation amendments (owner conversation, 2026-09-15): use GEST's
> actual VOCS model in GEECS-Schemas; the lightweight dependency is approved.
> Relative pseudo variables restore on unstage, with an explicit Set to best
> action afterward using physical positions captured in the staged frame.
> The implementation branch is `codex/native-optimization`, based on
> `feature/native-bluesky-plans` at `22961377`.
>
> Execution corrections: random cold start until two valid samples; no single
> feasible best restores initial positions; seed bests are filtered to current
> bounds and relative-pseudo seed reuse is refused. Existing required device
> rows are upgraded to essential/saved. Generator validation precedes claim;
> dump location comes from the emitted start document. Acquisition replay ends
> before evaluation/tell, which must not be replayed. Complete rows alone count
> toward the bin after a refire. Corpus deployment and hardware acceptance
> remain after schema landing; local v1 examples are prepared in the tests.


**Status (2026-09-16): implemented in PR #920 on `codex/native-optimization`.
The beam-free smoke test passed; corpus rollout and beam-on acceptance remain
owed. This is the historical design brief, not a new execution instruction;
`12_optimization_implementation.md` and package contracts record the final state.** Original brief written from a two-part
survey of this branch at `aaae524e` (the pseudo arc merged: #912, #913,
#914, #918; GeecsBluesky 0.91.1, GEECS-Schemas 0.26.0, ImageAnalysis
2.2.0, GeecsScanner 0.8.0) and the owner's rulings of 2026-09-15 (§2).
The original four-PR sequence (§7) was combined into PR #920. The executing agent reads §2
and §8 before touching a file: the rulings are closed, and §8 lists the
traps the survey found.

**Why now.** `03_clean_room_rebuild.md` §10.6 trimmed optimization to
"option B" in #816: the Xopt core stayed importable, the glue that ran
it on the funnel (`plans/optimize.py`, `optimize.py`,
`optimization/session_bridge.py`, `optimization/worker_loader.py`) was
deleted, and "optimization is broken until it is re-glued to the native
scan path in its own phase". This is that phase. `10_web_scanner.md` §5
kept the door open on the scanner side (#880) and this brief walks
through it.

**What it is not.** Not the PV-producer node of #744 (that remains the
way image-derived scalars reach Badger and OSPREY, and is not needed
here); not gated optimization; not traces over PVA (a separate gateway
+ DB item, §9); not the MCP optimize verb (#727's rewire).

---

## 1. Where things stand (survey, 2026-09-15)

**Survived #816** — `GeecsBluesky/geecs_bluesky/optimization/`: the Xopt
assembly (`base_optimizer.py`: `BaseOptimizer` with `generate`,
`seed_from_dumps`, `best_observed_setpoint`, `xopt.dump`), the
file-based evaluator (`base_evaluator.py`: `EvaluatorDataSource` seam,
ScanAnalysis `run_analysis(scan_tag)` per bin), `config_models.py`
(`BaseOptimizerConfig`, the corpus shape), five evaluator classes, the
generator recipes (`generators/generator_factory.py`: `random`,
`bayes_default`, `bayes_ucb`, `bayes_turbo_*`, `multipoint_bax_alignment*`,
+ `bax/`), `vocs_utils.py`, `inspection/`. 91 tests pass. Nothing outside
the package imports it. The worker host installs
`poetry install --extras "ca tiled qserver"` (`qserver/deploy/DEPLOYMENT.md:63`)
— **no `optimize`**.

**Deleted by #816** (commit `e8b428ca`; read with
`git show e8b428ca^:GeecsBluesky/geecs_bluesky/plans/optimize.py` and
`…/optimization/session_bridge.py`): the adaptive plan (one run,
iteration = bin, `propose` on a one-worker thread while the plan
`bps.sleep(0.05)`s so pause stays responsive), the suggester protocol, the
bridge (event rows → s-file frame; bounded wait for native files;
measured readbacks substituted for proposals in `add_data`;
`xopt_dump.yaml` into the scan folder), the worker loader. The
*mechanics* worth copying are the thread + sleep loop, the measured-readback
substitution and the dump; the file wait and the s-file frame are not.

**The plan layer already provides the acquire block.**
`plans/strict.py:213` `geecs_take_reading(shot_control, *, max_refires=2,
name="primary", shot_period=None)` returns `take_reading(devices)`, a plan
that prepares, arms, fires between trigger and wait, refires on a missed
frame, emits `create/read/save`, and **returns the merged readings dict**
(survives `contingency_wrapper` and `rewindable_wrapper`). `BinCounter`
(`strict.py:395`, `name="bin_number"`) must be in every row or the s-file
loses `Bin #`. The stock per-step hook cannot return a suggestion, so an
adaptive plan owns its loop. Gated acquisition is documents-only, so this
arc is **strict-only** (§8).

**Registration is small.** `plan_names.py:31-70` (`NON_SCAN_PLAN_NAMES`,
`GEECS_PLAN_NAMES`), `plans/registry.py:521` `bind_plans(profiles, *,
resolver=None, settables=None)` with the non-stock `elif` chain at
`:536-547`, `qserver/user_group_permissions.yaml:41` (the operator regex),
`qserver_ready.py:250` (readiness asserts every name), `qs_client/client.py:709`
(`submit_plan` refuses unknown names), pinned by
`tests/test_plan_registry.py:64,70,118` and `tests/test_deploy_templates.py:33`.

**Presets cannot name it yet.** `qs_client/presets.py:61-63`
`PRESET_PLAN_NAMES = GEECS_PLAN_NAMES − NON_SCAN_PLAN_NAMES`; `expand_preset`
(`:102`) builds `args[0]` purely from the preset's `devices` list
(`:142-161`), and `_resolve` (`:187`) converts a string only when it holds
`:` or is a catalog name — a config name passes through untouched. No hook
exists for "devices the plan needs".

**Pseudo variables resolve like any other.** `presets.py:82`
`scan_variable_reference(target, catalog)` → `identifier_name(name)` for a
`kind: pseudo` entry, `device_reference(device, variable)` otherwise;
worker-side the namespace binds every pseudo as a `CaPseudoPositioner`
(`namespace.py:657 add_pseudos`; `devices/ca/pseudo.py:186`) and any plain
name through `namespace.resolve(target)` (`namespace.py:797`). A custom
plan moves either with `bps.mv`. **A `mode: relative` pseudo zeroes at
`stage()` and restores at `unstage()`** (`pseudo.py:315-365`) — the stock
plans stage motors through `stage_wrapper`; a custom plan must stage its
movables itself or a bump never zeroes and never restores.

**The live-frame pieces exist on both sides of the join.** The PVA
gateway posts every frame with the camera's own `acq_timestamp`,
converted to Unix epoch (`GeecsPvaGateway/geecs_pva_gateway/server.py:67`
`_frame_timestamp`: `float(value) − 2_082_844_800`, `post(image,
timestamp=ts)` at `:435`; the NTNDArray carries **no** other attribute).
p4p 4.2.2 exposes it as `value.timestamp` (float seconds) on the
unwrapped array (verified in the GeecsBluesky env). The detector's
reading carries the raw LabVIEW-epoch stamp as `<safe_name>-acq_timestamp`
(`devices/detector.py:104,652`). `p4p` is already in the `ca` extra
(`pyproject.toml:79`) and `epics_env.py:51 apply_epics_address_config`
already sets `EPICS_PVA_ADDR_LIST` from `[pva] addr_list` +
`file_plugin_addr_list`. ImageAnalysis has the write-free in-memory seam:
`ImageAnalysis/image_analysis/ephemeral.py:72 run_diagnostic_ephemeral(
name_or_path, frames, *, config_dir=None, overrides=None,
auxiliary_data=None) -> list[ImageAnalyzerResult]`, one result per frame,
**bare scalar keys** (`x_CoM`, not `UC_TopView_x_CoM` — the prefixing is
ScanAnalysis's, `single_device_scan_analyzer.py:993`, and does not apply
here). Denylist: `haso`, `frog_retrieval`. Per-bin in ScanAnalysis is
load → mean → analyze once.

**Scalars are not declared anywhere.** No analyzer kind declares the keys
it emits (`image_analysis/base.py`, confirmed). `beam` emits 18 keys by
default (`algorithms/basic_beam_stats.py:216 flatten_beam_stats` over
`image_{total,peak_value}` and `{x,y,x_45,y_45}_{CoM,rms,fwhm,peak_location}`),
narrowed by `BeamAnalyzerSpec.enabled_stats` and widened by
`compute_slopes` (`geecs_schemas/analysis/analyzers.py:76-88`) — the set is
**derivable from the validated spec**. `standard` emits none. 15 kinds in
`ANALYZER_SPECS` (`analyzers.py:445`) mirrored by
`image_analysis/config/registry.py:25`.

**Schema conventions** (`GEECS-Schemas/README.md`, no CLAUDE.md there):
`VersionedSchemaModel` (`_base.py:36`, `schema_version: int`,
`extra="forbid"`), registered in three places — `SCHEMA_REGISTRY`
(`__init__.py:161-174`), `EXPORTED_SCHEMAS` (`schema_export.py:53-64`,
artifacts under `docs/geecs_schemas/`, regenerated by
`tests/generate_schema_artifacts.py`), `EXAMPLES` in `docgen.py:53` — and
pinned by `tests/test_docgen.py` (every field described, examples validate)
and `tests/test_schema_export.py` (byte-for-byte). `pyproject.toml:26-28`:
pydantic only under the then-current rule (superseded by the owner-approved
GEST dependency). The legacy pieces: `scan_request.py:371-470`
(`EvaluatorSpec`, `GeneratorSpec`, `OptimizationSpec`, the
`ScanRequest.optimization` field and its `mode == "optimize"` validator),
`convert/optimizer_configs.py` (**no production caller**), the golden
`tests/golden/hexapod_optimization_spec.json`, fixtures
`tests/fixtures/optimizer_configs/`. GEECS-MCP imports `ScanRequest` but
never reads `.optimization`; its two "optimize" mentions are message
strings.

**Expressions.** `geecs_schemas/restricted_expr.py:150 compile_expression(
expression, symbols, whitelist, *, filename)` → `CompiledExpression.evaluate(
values)`. `_validate` (`:73-112`) refuses `ast.Attribute` outright, so a
reference like `topview.x_CoM` is a **new feature**, not a whitelist toggle
(§4.3). Two consumers build their own whitelists: `forward_expr.py:78` and
`GeecsCAGateway/geecs_ca_gateway/derived.py:43`.

**The corpus** (`GEECS-Plugins-Configs/scanner_configs/experiments/Undulator/optimizer_configs/`,
11 files, all `BaseOptimizerConfig`-shaped with `geecs_scanner.optimization.*`
module paths and save-set-shaped `device_requirements`): keepers per Sam =
`bax_alignment_{S1H,S1V,S2H,S2V}` (their TODO headers are stale —
`scan_analysis_configs/analyzers/HTU/UC_ALineEBeam3.yaml` exists, `kind:
beam`), `TopViewMax` (`UC_TopView.yaml` exists), `hi_res_mag_cam` +
`hi_res_mag_cam_max_counts` (one names a deleted evaluator module; **no
`UC_HiResMagCam` diagnostic YAML was found** — needs the owner before it
is regenerated); `bax_alignment_simulation` (synthetic objective from the
setpoints — the hardware-safe acceptance run, §7 PR 3). Dropped:
`ebeam_source_opt` (a LabVIEW TSV trace, not a frame — §9),
`hexapod_alignment`, `multi_device_example`.

**Scanner side.** `GeecsScanner/geecs_scanner/service/streams.py:110-118`
already computes `planned_total = (num_points or max_iterations) ×
shots_per_step`; `templates/console.html:129` has the greyed Optimize
button; `static/scanner.js:504-527, 601-633` map mode → plan call; the
submit route (`web/api.py:109`) is **preset-only** (`SubmitIn`,
`service/models.py:118-130`); `summaries.py:84 summarize_item` keeps an
unknown plan's name as its text. The Now panel has **no** live-values list
in the SSE stream (`web/events.py`: `status` + `progress` frames only;
readbacks are polled per variable through `GET /api/readback`).

---

## 2. Rulings (Sam, 2026-09-15) — closed, do not reopen

1. **The objective's data comes from the PVA live path**, never from
   files mid-run. "We used to read from disk because that was the only
   way." Cameras used by an optimization are ordinary run devices: in the
   device group, **essential, images saved to the record** (post-mortem of
   an optimization is essential). The optimizer consumes the streamed
   frames; it never touches the written ones.
2. **The optimizer config declares its resources; submit ensures those
   devices are in the run.** Not the other way round.
3. **In-plan, one run per optimization, iteration = bin.** The
   out-of-process agent shape (bluesky-adaptive's) is rejected: every
   queue item claims a scan number.
4. **One schema, not image-shaped.** Scalar-only optimizations, traces
   (later), and arbitrary combinations of cameras and scalars ("measures
   things on two cameras") must all be the *same* document. "Getting the
   schema right is the hardest part."
5. **Xopt-shaped where Xopt has a word for it** (`vocs`, `generator`); the
   Xopt `evaluator` callable is replaced by a declarative, GEECS-aware
   measurements block.
6. **Analyzer kinds declare the scalars they emit** ("an obvious thing
   that was missing") — enables load-time validation and picklists.
7. **A measurement names a diagnostic document, never a device.** The
   document's `analyzer.kind` fixes the scalars; its `name` is the device.
   The diagnostic-naming question (device-named files, several pipelines
   per device) is a separate ImageAnalysis arc; not this one.
8. **End of run: move-to-best by default**, plus a surface showing the
   best positions with a "set to best" action for the operator.
9. **After the pseudo arc**: optimizations over pseudo variables wanted.
10. The magspec-spectrum optimizer is **dropped from this arc**; serving
    every non-scalar device type over PVA is a separate gateway + DB item.

My calls, taken under the delegation of ruling 5 (each may be revisited
by the owner, none by the executing agent): the `vocs` choice (superseded by the owner amendment to embed GEST, §4.1); dotted symbols in `restricted_expr`
(§4.3); measurement-local names in references (`topview.x_CoM`); drop a
missed shot within a bin and reduce over the rest, with a `min_shots`
floor below which the bin's outputs are NaN (§4.2); the `optimization`
event stream as the in-run record of proposals, outputs and best-so-far
(§5.5); `shots_per_step` as the per-iteration budget's name (the scanner
already multiplies it).

---

## 3. The shape in one page

```
OptimizerConfig (YAML in the configs repo, GEECS-Schemas document)
   ├─ vocs          variables (bounds over catalog names — plain, alias or pseudo),
   │                objectives, observables, constraints, constants   [Xopt's words]
   ├─ measurements  name → signal | diagnostic  (+ frames, reduce, min_shots)
   ├─ derived       name → expression | python            [replaces Xopt's evaluator]
   ├─ generator     name + options                        [Xopt's words, our recipes]
   └─ run           on_finish, seed_dumps, default shots_per_step / max_iterations

optimize(detectors, *, optimizer_config, max_iterations, shots_per_step,
         trigger_profile, shot_period=None, non_essential=(), md=None)
   worker: resolve + compile the config (movables from the namespace, live-frame
           sources per diagnostic, expressions), refuse before any claim if a
           required device is missing from `detectors`; liveness gate; stage
           movables + detectors; open sources; one run under the ARMED bracket:
             iteration i:  checkpoint → ask (thread) → mv → shots_per_step ×
                           take_reading([*detectors, *movables, bins]) →
                           await frames → analyze (thread) → reduce → derived →
                           tell → emit one `optimization` event
           on_finish → close → xopt_dump.yaml into the scan folder → unstage
   client: expand_preset("optimize") merges the config's devices into args[0]
           as essential/save; the scanner's Optimize mode is that preset.
```

---

## 4. The schema — `OptimizerConfig` v1 (GEECS-Schemas)

New module `geecs_schemas/optimizer_config.py`. Every model subclasses
`SchemaModel` (`extra="forbid"`); the document subclasses
`VersionedSchemaModel` with `CURRENT_SCHEMA_VERSION = 1`. Every field
carries a `description=` (docgen test). Registered as kind
`"optimizer_config"` in `SCHEMA_REGISTRY`, `EXPORTED_SCHEMAS` and
`docgen.EXAMPLES` (one example per §4.6).

### 4.1 `vocs` — the search space, in Xopt's words

```yaml
vocs:
  variables:                       # catalog names: Device:Variable, an alias, or a pseudo
    "U_S1H:Current": [-4.5, -1.5]  # [lo, hi], lo < hi
    "ALine_e_beam_angle_offset_x": [-0.5, 0.5]
  objectives: {brightness: MAXIMIZE}       # name → MINIMIZE | MAXIMIZE; may be empty (BAX)
  observables: [aline.x_CoM]               # names the generator models but does not optimize
  constraints: {charge: [GREATER_THAN, 20.0]}   # name → [GREATER_THAN | LESS_THAN, value]
  constants: {}                            # name → float, passed through to Xopt
```

**Owner amendment: embed `gest_api.vocs.VOCS` directly.** GEST adds only
Pydantic. Compact `[lo, hi]` bounds are accepted; canonical serialization
uses GEST's typed objects. The validated worker version is GEST 0.1; its
custom mappings lack JSON Schema hooks, so the field carries schema metadata
for docs/export without introducing a duplicate runtime model.

### 4.2 `measurements` — what is measured, per shot

```yaml
measurements:
  charge:                                   # a scalar already in the reading
    signal: "U_BCaveICT:Python Results.ChA"
    reduce: median                          # mean (default) | median | min | max | sum | std
    min_shots: 3                            # default 1
  topview:                                  # a diagnostic over a live non-scalar variable
    diagnostic: UC_TopView                  # AnalysisDiagnostic document ID (file stem)
    frames: per_bin                         # per_bin (default): mean the frames, analyze once
                                            # per_shot: analyze each, then `reduce` per scalar
    reduce: mean
    min_shots: 1
    overrides: {}                           # deep-merged into the diagnostic YAML (load_diagnostic's `overrides`)
```

A measurement is a discriminated union on `signal` vs `diagnostic`
(exactly one). `signal` is a `Device:Variable` string (the schema's
`split_device_variable` from GEECS-Schemas 0.25.0 validates the shape);
its device is a required run device. `diagnostic` is a document stem; its
**device is the diagnostic document's `name`** (`analysis/diagnostic.py:56`)
and its **scalars are `spec.emitted_scalars()`** (§4.5) — both resolved
by the consumer at load time, because GEECS-Schemas cannot read the
configs tree. Rule for names: `^[A-Za-z_][A-Za-z0-9_]*$`, unique, never
containing `.`.

Missing data (my call): a shot whose frame never arrives or whose analyzer
raises is dropped from the bin; if fewer than `min_shots` remain, every
scalar of that measurement is NaN for the bin, which makes every derived
value over it NaN and the iteration a failed evaluation (Xopt's own
semantics; the plan logs it at WARNING and continues). A bin is **never
retaken** by the optimizer — the strict refire already covers a missed
frame once.

### 4.3 `derived` — expressions over measurements

```yaml
derived:
  size2: "(cam_a.x_fwhm * 24.4e-3)**2 + (cam_a.y_fwhm * 24.4e-3)**2"
  brightness: "cam_b.image_total / size2"
  sim_x: {python: "geecs_bluesky.optimization.simulations:aline_x_com"}   # the escape hatch
```

Values are either an expression string or `{python: "module:function"}`.
Expressions compile through `geecs_schemas.restricted_expr` with the
arithmetic whitelist `forward_expr.py:78` uses (binary ops, unary sign,
`sqrt`/`abs`/`exp`/`log`/trig, `pi`/`e`), extended with `min`/`max`.
Symbols are every measurement reference (§4.4) plus every derived name
declared *earlier* in the mapping (no cycles: order is the dependency
order, validated). Since `cam_a.x_fwhm` parses as `ast.Attribute`,
**`restricted_expr` gains dotted symbols**: `_validate` accepts an
`Attribute` chain whose flattened dotted name is in `symbols`, and
`compile_expression` rewrites each such chain to a mangled `Name`
(`cam_a__x_fwhm`-style, collision-checked) before compiling, with
`evaluate(values)` accepting the dotted keys. Existing consumers pass
undotted symbols and are unaffected (their tests prove it). The Python
form names a function `(scalars: Mapping[str, float]) -> float` importable
in the worker; it exists so the simulation config (§7 PR 3's acceptance)
and any future non-expressible objective have a home, and it is
**imported at compile time**, before the claim.

### 4.4 References

A reference is `<measurement>` (a `signal` measurement's one value),
`<measurement>.<scalar>` (a `diagnostic` measurement's emitted scalar), or
`<derived>`. The document validates syntax and intra-document existence;
the consumer validates `<scalar>` against the diagnostic's
`emitted_scalars()` when it loads the diagnostic (`GeecsConfigurationError`
naming the reference and the emitted set).

### 4.5 Declared scalars (ImageAnalysis + GEECS-Schemas, PR 1)

Each `*AnalyzerSpec` in `geecs_schemas/analysis/analyzers.py` gets
`def emitted_scalars(self) -> frozenset[str]` — the keys its analyzer's
`ImageAnalyzerResult.scalars` will contain for this spec. `beam`: the 18
from `enabled_stats` (all when unset) ∪ the four slope keys when
`compute_slopes`. `standard`: empty. Every other kind: read its
`analyze_image` and declare the keys it always emits (a key emitted only
on some data is *not* declared — the declaration is the guaranteed set).
A base default that raises `NotImplementedError` is not acceptable: every
kind declares, even if empty. ImageAnalysis pins the declaration against
reality: `tests/test_emitted_scalars.py` runs every non-denylisted kind
once over a synthetic frame (or line) and asserts
`spec.emitted_scalars() <= set(result.scalars)` and, for `beam` and
`standard`, equality. **Prove it bites**: temporarily add a nonexistent key to a
declaration and watch the test fail before claiming the pin. Where a
kind needs vendor DLLs or data the test cannot synthesize (`haso`,
`frog_*`), the test skips with the reason, and the declaration is
reviewed by hand in the PR.

### 4.6 `generator` and `run`

```yaml
generator:
  name: bayes_turbo_standard          # a recipe in generators/generator_factory.py
  options: {}                         # the recipe's overrides (today's `xopt_config_overrides[name]`)
run:
  on_finish: best                     # best (default) | hold
  seed_dumps: []                      # xopt_dump.yaml paths to warm-start from
  shots_per_step: 5                   # default for the plan kwarg
  max_iterations: 30                  # default for the plan kwarg
```

`generator.name` is validated against a **static tuple** the schema
carries (`GENERATOR_NAMES`), mirrored from the factory's mapping and
pinned equal by a GeecsBluesky test (the same discipline as
`GEECS_PLAN_NAMES` vs the permissions regex).

### 4.7 The six keepers in v1 (the corpus regeneration)

```yaml
# TopViewMax.yaml
schema_version: 1
vocs:
  variables: {"U_ESP_JetXYZ:Position.Axis 1": [3, 6]}
  objectives: {topview.image_total: MAXIMIZE}
  observables: [topview.x_CoM, topview.y_CoM, topview.image_peak_value]
measurements:
  topview: {diagnostic: UC_TopView, frames: per_bin}
generator: {name: bayes_turbo_standard}
```

```yaml
# bax_alignment_S1H.yaml   (S1V / S2H / S2V differ only in the names and bounds)
schema_version: 1
vocs:
  variables:
    "U_S1H:Current": [-4.5, -1.5]
    "U_EMQTripletBipolar:Current_Limit.Ch1": [1.2, 1.7]
  observables: [aline.x_CoM]
measurements:
  aline: {diagnostic: UC_ALineEBeam3, frames: per_bin}
generator:
  name: multipoint_bax_alignment_l2
  options:
    control_names: ["U_S1H:Current"]
    measurement_name: "U_EMQTripletBipolar:Current_Limit.Ch1"
    observable_names: [aline.x_CoM]
    probe_nominal: 1.5
    probe_grid_absolute: [-0.2, 0.0, 0.2]
    n_control_mesh: 21
    mesh_measurement: true
    n_measurement_mesh: 5
    n_monte_carlo_samples: 32
    use_low_noise_prior: false
```

```yaml
# bax_alignment_simulation.yaml — real magnets, synthetic centroid (the acceptance run)
schema_version: 1
vocs:
  variables:
    "U_S1V:Current": [-4, 4]
    "U_EMQTripletBipolar:Current_Limit.Ch1": [1.2, 1.7]
  observables: [x_com_sim]
measurements:
  s1v: {signal: "U_S1V:Current"}
  emq: {signal: "U_EMQTripletBipolar:Current_Limit.Ch1"}
derived:
  x_com_sim: {python: "geecs_bluesky.optimization.simulations:aline_x_com"}
generator: {name: multipoint_bax_alignment_l2, options: {...as above, observable_names: [x_com_sim]}}
```

The HiResMagCam pair is regenerated only once its diagnostic exists
(owner's call; leave the two legacy files in place with a one-line
`# LEGACY — not loadable on v1; see 11_optimization.md` header rather
than deleting them). `ebeam_source_opt`, `hexapod_alignment`,
`multi_device_example` are deleted from the corpus. The corpus commit
goes to the configs repo's `main` directly (the legacy-configs rule),
after PR 2 merges and before PR 3's acceptance.

---

## 5. The worker — `geecs_bluesky.optimization` rebuilt (PR 3)

### 5.1 What goes and what stays

**Deleted** (with their tests): `base_evaluator.py`, `evaluators/`,
`config_models.py`, `_legacy_models_actions.py`,
`_legacy_models_save_devices.py`, `tests/optimization/test_base_evaluator.py`,
`test_concrete_evaluators.py`, `test_config_models.py`,
`test_evaluator_bax_mode.py`, `test_evaluator_create_scan_analyzer.py`.
`base_optimizer.py` loses `from_config_file` / `from_config` / the
evaluate-function plumbing; what remains is the Xopt assembly
(`_setup_xopt`, `seed_from_dumps`, `generate`, `add_data`,
`best_observed_setpoint`, `dump`) — rename it `driver.py` /
`XoptDriver` if that reads better; keep `test_xopt3_migration.py` and
`test_base_optimizer_from_config.py`'s assembly cases, rewritten over the
new constructor. **Kept unchanged**: `generators/` (+ the
`GENERATOR_NAMES` parity test), `vocs_utils.py`, `inspection/`. The
`scananalysis` path dependency leaves the `optimize` extra
(`pyproject.toml:54,80`); `imageanalysis` stays (the ephemeral seam).
The stale `bayes_cheetah` branch (`generator_factory.py:319` imports a
package that does not exist) is deleted in the same PR.

### 5.2 `optimization/live_frames.py` — the frame source

```python
class LiveFrameSource:
    """One held PVA monitor on a camera's image PV; frames kept by stamp."""
    def __init__(self, pv: str, *, keep: int = 64) -> None: ...
    def open(self) -> None            # p4p.client.thread.Context("pva").monitor(pv, self._on_update)
    def close(self) -> None
    def wait_connected(self, timeout: float) -> None     # first update (the cached frame) or GeecsDeviceDownError
    def frame_at(self, stamp: float, tolerance: float = 1e-3) -> np.ndarray | None
    def await_frames(self, stamps: Sequence[float], timeout: float) -> dict[float, np.ndarray]
```

The callback stores `(value.timestamp, np.asarray(value))` in a bounded
deque under a lock; **every** update is kept (latest-wins is the
gateway's behaviour, not ours). Stamps are Unix seconds; the plan
converts a row's `<cam>-acq_timestamp` (raw LabVIEW epoch) with the same
offset the gateway uses. **That constant gets one home**:
`geecs_core.db.variable_types.LABVIEW_EPOCH_OFFSET = 2_082_844_800`
(GEECS-Core patch), imported by the gateway (`server.py:41` becomes an
import) and by this module. The PV is `pv_name(experiment, device,
primary_image_variable(rows)[0])` (`geecs_core.pv_naming:58`,
`namespace.py:128`); the rows come from the namespace's DB runtime, the
same call the detector build uses (find it in `namespace.py`, do not
open a second DB path). The monitor is opened **before the liveness
gate** and `wait_connected(5.0)` runs before the first fire — one gating
round trip (`docs/geecs_gateway/image_pvs.md` §Reading images). A source
that never connects refuses the run before any claim, naming the PV.
Headless tests use a `FakeFrameSource` with the same surface; the p4p
class is exercised only by the hardware acceptance and one
`@pytest.mark.pva` test against the Mac twin if available.

### 5.3 `optimization/measurements.py` — compile the config

`compile_measurements(cfg, *, namespace, resolver, experiment, config_dir)
-> CompiledMeasurements` resolves, **before any claim** and raising
`GeecsConfigurationError` with the offending reference: every `signal` to
its device + event key (`<safe_name>-<variable>`, via the namespace's
naming); every `diagnostic` to `(AnalysisDiagnostic, device, LiveFrameSource,
emitted_scalars)` through `image_analysis.config.load_diagnostic(stem,
config_dir=…, overrides=…)`; every reference against the emitted sets;
every expression through `restricted_expr`; every `python:` import. It
also returns `required_devices: frozenset[str]` (GEECS spellings). At run
time `evaluate_bin(rows: list[dict], frames: dict[str, dict[float,
ndarray]]) -> dict[str, float]` applies per-shot extraction, the
`per_bin` frame mean (the same `np.mean(axis=0)`
`single_device_scan_analyzer.py:1223` uses) or per-shot analysis through
`run_diagnostic_ephemeral`, the `reduce`, the `min_shots` floor, then the
derived values in order. Pure functions over dicts and arrays; the tests
are table-driven and need no RunEngine.

### 5.4 `plans/optimize.py` — the plan

```python
def optimize_plan(profiles: TriggerProfiles, resolver, namespace) -> Callable:
    def optimize(detectors, *, optimizer_config: str, max_iterations: int | None = None,
                 shots_per_step: int | None = None, trigger_profile: str,
                 shot_period: float | None = None, non_essential=(), md=None):
```

Registered in `bind_plans` as `"optimize"` (the `elif` chain,
`registry.py:536-547`, receiving `profiles`, `resolver`, `settables`) and in
`NON_SCAN_PLAN_NAMES`; a synthesized `__signature__` like `strict_plan`'s so
the manager validates queue items. Body, in order:

1. `cfg = resolver.resolve_optimizer_config(optimizer_config)` (new
   `ConfigsRepoResolver` method, fresh per call like presets; the folder
   constant already exists at `config_resolver.py:196`); kwargs default
   from `cfg.run`; `max_iterations` must end up set (refuse otherwise —
   the MCP rule already says so).
2. `movables = [namespace.resolve(name) for name in cfg.vocs.variables]`
   — pseudos included; `compiled = compile_measurements(...)`;
   `missing = compiled.required_devices − {GEECS name of d for d in detectors}`
   → `GeecsConfigurationError` naming them (the client should have merged
   them, §6; the worker does not trust it). The GEECS name is
   `_geecs_device_name` today (`detector.py:641`, and on pseudos); give it
   one public read-only accessor on the namespace members rather than
   reading the private attribute from the plan.
3. `shot_control = profiles.resolve(trigger_profile)`; open every frame
   source; `wait_connected`; `liveness_gate(shot_control, [*detectors,
   *movables])`.
4. `md`: the strict keys (`trigger_profile`, `shots_per_step`,
   `acquisition="strict"`, `non_essential`, `shot_period`) **plus**
   `max_iterations`, `optimizer_config`, `geecs.optimizer = cfg.model_dump()`
   (the whole document rides in the start doc — the record of what ran).
5. `stage_wrapper` over `[*detectors, *movables]` (this is what zeroes and
   restores a relative pseudo); `run_bracket(inner, shot_control, ARMED)`
   wrapped in `name_failed_status`, then `bpp.run_wrapper` — mirror
   `strict_plan`'s order at `registry.py:411-420`.
6. `inner`: `driver = XoptDriver(to_vocs(cfg.vocs), cfg.generator,
   seed_dumps)`; `take = geecs_take_reading(shot_control,
   shot_period=shot_period)`; `bins = BinCounter()`; a one-worker
   `ThreadPoolExecutor`. For `i in range(max_iterations)`:
   `bps.checkpoint()`; `future = pool.submit(driver.ask)` and `yield from
   bps.sleep(0.05)` until done (the deleted plan's loop); `bins.value =
   i + 1`; `yield from bps.mv(*flat(proposal))`; rows = `[ (yield from
   take([*detectors, *movables, bins])) for _ in range(shots_per_step) ]`;
   stamps per camera from the rows; `frames = source.await_frames(stamps,
   timeout=2.0)` on the thread; `outputs = compiled.evaluate_bin(rows,
   frames)` on the thread (analysis takes seconds; the RE loop must keep
   sleeping); `measured = mean readbacks of the movables over the rows`
   (the bridge's substitution — Xopt learns where the magnets *were*);
   `driver.tell(measured, outputs)`; emit one `optimization` event (§5.5).
   `on_finish == "best"` → `bps.mv(*flat(driver.best))` after the loop,
   inside the bracket. `finalize_wrapper` closes the sources and the
   pool, and writes `xopt_dump.yaml` into `md["scan_folder"]` (the worker
   is the producer; `mkdir` never — the folder exists).
7. Pause: `bps.checkpoint()` at the top of each iteration is the only
   rewind point; a deferred pause lands there, an immediate pause repeats
   the iteration (the row loop is rewindable via `take_reading`'s own
   wrapper — verify on the mock RE, this is the one place the deleted
   plan was never tested under pause).

### 5.5 The `optimization` stream

One `Readable`-shaped soft device, `OptimizationRecord` (name
`optimization`), with float signals for every variable's proposal and
measured value, every output (measurement scalars used, derived, the
objectives), `iteration`, `n_valid_shots` per measurement, and
`best_<variable>` / `best_<objective>` so far; read once per iteration
with `bps.trigger_and_read([record], name="optimization")`. Consequences
that fall out for free: Tiled has the iteration table; the scanner's
document bridge can show objective and best-so-far live (§6); the
s-file stays per-shot (`primary` rows carry `bin_number` = iteration).
Descriptors are declared once, before the loop, so the columns cannot
change mid-run — a NaN is the value of a failed evaluation.

### 5.6 Registration, readiness, deploy

`plan_names.py` (`"optimize"` in `NATIVE_SCAN_PLAN_NAMES`),
`user_group_permissions.yaml:41` (add `optimize` to the alternation),
`test_plan_registry.py` (the per-name signature assert for `optimize`),
`test_deploy_templates.py` (passes once the regex is edited),
`qserver_ready.py` (nothing: it asserts the tuple). Startup profile: a
profile-thread warm import of `xopt` + `torch` before readiness,
with missing optional imports tolerated and other failures logged (the deleted
`warm_up_optimization_stack`, minus the session bridge). Deployment: the
worker's install string gains `optimize`
(`qserver/deploy/DEPLOYMENT.md:63`, `render_units.sh` if it carries the
extras, the fleet map's worker row); the worker's `config.ini` must carry
`[pva] addr_list` (the plugin list alone reaches only the plugin PVs) —
a runbook line, verified on the host before acceptance. Torch on the
worker box is a memory cost to watch beside the portal
(`reference_host_oom_tiled_dataframe_probe` lesson): note RSS before and
after the first `optimize` item in the PR's hardware section.

---

## 6. The clients (PR 3's `expand_preset` half, PR 4 the scanner)

**`expand_preset` learns `optimize`.** `PRESET_PLAN_NAMES` gains it. When
`preset.plan.name == "optimize"`, `expand_preset(preset, *, catalog, md,
required_devices=…)` **appends every required device not already in the
preset** as `essential=True, save_images=True` (ruling 2). The required
set is computed by one GEECS-Schemas helper,
`optimizer_required_devices(cfg, diagnostic_devices: Mapping[str, str])
-> frozenset[str]` — the `signal` measurements' devices plus each
`diagnostic` measurement's device looked up in `diagnostic_devices`
(stem → the diagnostic document's `name`). The scanner does not and must
not import ImageAnalysis, so `ConfigsRepoResolver` gains
`diagnostic_device(stem) -> str`, which reads
`scan_analysis_configs/analyzers/**/<stem>.yaml` as plain YAML and returns
its `name`; the scanner's service resolves the config, builds the mapping
through it, and passes the set in. The worker computes the same set with
the same helper (from the diagnostics it has already loaded) and still
refuses a group that lacks a device — the two cannot drift, and a raw
`submit_plan` that bypasses `expand_preset` is still caught. `kwargs`
carry `optimizer_config`, `max_iterations`, `shots_per_step`,
`trigger_profile`; `references` gains `optimizer_config:<name>` so
preflight checks it resolves.

**The scanner** (closes #880): Optimize mode ungreys; its form section =
optimizer config `<select>` (`GET /api/configs/optimizer_configs` exists),
`max_iterations`, `shots_per_step` (defaults filled from the config's
`run` via a new `GET /api/configs/optimizer_configs/{name}`), trigger
profile, description, and the devices table showing the merged group with
the config-required rows locked (they cannot be unticked). `POST /api/submit`
stays preset-only — an optimization *is* a preset whose plan is `optimize`.
Progress: `planned_total = max_iterations × shots_per_step` already; the
meter label reads "of ≤ N". The Now panel gains an **Optimization**
block fed by a new `optimization` SSE frame the document bridge emits from
each `optimization` event: iteration, the objective(s), best-so-far with
its variable values, valid-shot counts. After the run: **"Set to best"**
in that block submits one `mv` item with every variable → best (idle-only,
the existing gate); the values come from the last `optimization` event the
bridge saw (kept in the service's per-run memory) or, for a run finished
before the page loaded, from Tiled through the portal link — not this PR;
say so in the block's empty state. Kit: no new tokens; `.meter`, `.live`,
`.chip` cover it.

---

## 7. Original PR sequence (superseded by PR #920)

The table below records the initial proposed split, not pending work. Schema
0.27.0 was an intermediate draft, never released; the combined schema changes
first shipped on this branch as 0.28.0. Current package versions are in their
pyproject files and changelogs.

Original process per `/land`: branch off `feature/native-bluesky-plans`, scope
check, version bump + CHANGELOG, tests as CI runs them, `scripts/commit.sh`,
fresh-context adversarial review with a severity floor set *before* the
review and at most two rounds, CI watch, maintainer merges. LOC per concern
in the PR body when a PR bundles.

| PR | Packages (bump) | Content | Acceptance |
|---|---|---|---|
| **1 — declared scalars** | ImageAnalysis 2.3.0, GEECS-Schemas 0.27.0 | `emitted_scalars()` on every analyzer spec (§4.5); the parity test; docgen/export regenerated | headless: parity test green for every non-denylisted kind; the bite check recorded in the PR body |
| **2 — the document** | GEECS-Schemas 0.28.0 | `OptimizerConfig` v1 + dotted symbols in `restricted_expr` (§4); deletion of `OptimizationSpec`/`EvaluatorSpec`/`GeneratorSpec`, the `ScanRequest.optimization` field + validator, `convert/optimizer_configs.py`, the golden, the old fixtures; new fixtures = the v1 keepers (§4.7); `optimizer_required_devices` helper; GEECS-MCP stays importable (run its suite) | headless: every fixture validates; the old shape is refused with a message naming this brief; forward_expr and derived-channel tests unchanged and green |
| **corpus** | configs repo `main` | the v1 keepers committed, the three drops deleted, the HiResMagCam pair headed LEGACY | `list_optimizer_configs` on the box lists the v1 set |
| **3 — the worker** | GeecsBluesky 0.92.0, GEECS-Core 0.8.x (the epoch constant), GeecsPvaGateway patch (imports it) | §5 entire + `expand_preset` (§6 first paragraph) + `resolve_optimizer_config` + `diagnostic_device`; deploy docs; the warm import | headless: the plan on the mock RE with `FakeFrameSource` and a fake namespace — documents (`primary` rows with `bin_number`, `optimization` events, start-doc md), pause/resume at an iteration boundary, immediate pause repeating the iteration, refusal paths (missing device, unknown reference, source never connects) all before any claim. **Hardware, in order:** (a) `bax_alignment_simulation` on HTU with the box ARMED and no beam required — real S1V/EMQ moves, synthetic observable, `xopt_dump.yaml` in the folder, s-file rows per shot, the setpoint restored by `on_finish`; (b) `TopViewMax` with beam, 5 shots × 10 iterations: frames joined (the `optimization` event's `n_valid_shots` = 5 every iteration), objective plausible against the saved PNGs post hoc, RSS before/after noted |
| **4 — the scanner** | GeecsScanner 0.9.0 | §6 second paragraph; closes #880 | hardware: TopViewMax submitted from the page, progress "of ≤ N", the Optimization block live, "Set to best" observed once |

After PR 4: the BAX alignments on the real ALine camera (an operator day;
not a PR gate), then the HiResMagCam pair once its diagnostic exists.

---

## 8. Traps the survey found — read before building

- **Never read the record mid-run.** Not the PNGs, not the plugin's HDF5
  stack (finalized at `unstage`, after the stop document). The frame
  source is the only image path. Ruling 1 and `#738`'s ruling both say so.
- **Do not re-implement the fire.** `geecs_take_reading` is the acquire;
  the plan calls it, per shot, with `BinCounter` in the device list.
- **Stage your movables.** A `mode: relative` pseudo that is not staged
  never zeroes and never restores; a plain one skips its disagreement
  check. Use `stage_wrapper` over detectors *and* movables, like the stock
  plans.
- **Bare keys from the ephemeral seam.** `run_diagnostic_ephemeral`
  returns `x_CoM`, not `UC_TopView_x_CoM`; the measurement-local name
  (`topview.x_CoM`) is the namespacing. Do not import ScanAnalysis to get
  the prefix back.
- **Two epochs.** The reading's `acq_timestamp` is raw LabVIEW; the frame's
  `value.timestamp` is Unix. One constant, one home (GEECS-Core), both
  sides import it. Join by equality within 1 ms, never "nearest".
- **The frame may land after the reading.** CA and PVA are separate paths
  from separate hosts. `await_frames` with a bounded timeout after the
  bin's last shot; never block the RE thread — the thread pool + `bps.sleep`
  loop is the pattern.
- **Descriptors before the loop.** The `optimization` stream's columns
  are fixed at the first `describe`; build the record device from the
  compiled config, not from the first result.
- **Refuse before the claim.** Every configuration error (unknown
  generator, unknown reference, missing device, a source that never
  connects, a `python:` import failure) surfaces before `open_run`; a
  scan number is never burned on a bad config.
- **`GENERATOR_NAMES` and the permissions regex are hand-maintained
  tuples**; each has a parity test. Edit both sides in the same commit.
- **No lab literals** in code or defaults (experiment names, hosts, paths);
  the fixtures use HTU names as examples, labelled so.
- **No `Any` free dicts** across a seam: the compiled measurement, the
  bin result and the SSE frame are typed models or dataclasses.
- **Do not revive `ScanRequest`'s optimize mode** in GEECS-MCP; the MCP
  rewire is #727. PR 2 only keeps MCP importable.
- **Do not prune `extras/` or the legacy corpus files beyond §4.7.**

---

## 9. Spawned, not in scope

- **Every non-scalar device type over PVA** (magspec traces, scope
  traces; the DB's `1darray` kind is in `SKIP_VARTYPES` today,
  `geecs_core/db/variable_types.py:50`): gateway + DB + a `1darray`
  branch in `LiveFrameSource` and a `line` diagnostic path in
  `measurements.py`. The schema needs no change. Unfiled at the time of
  writing.
- **Gated optimization** (per-iteration batches at rep rate): documents
  only today; would need the frame source to feed per-bin analysis from
  the `shots` stream's stamps. Not before the strict path has run for a
  while.
- **Warm-start UI** (pick a previous run's `xopt_dump.yaml`), Tiled-backed
  "Set to best" for runs finished before the page loaded, an objective
  trend chart in the Now panel.
- **The MCP `optimize` verb** — with #727.
- **Diagnostic naming** (several pipelines per device) — an ImageAnalysis
  arc if a case forces it.
