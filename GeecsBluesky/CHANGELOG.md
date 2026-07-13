# Changelog

All notable changes to `geecs-bluesky` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.31.1] - 2026-07-13

### Changed

- Docstring condensation over the M4 range (docs-only in effect): the
  claim-before-bind rationale is stated once in full
  (`_run_optimize_request`) with one-line cross-references at the other
  three sites; `gateway_put.py`'s module docstring opens with the
  contract instead of its consolidation history (and drops a reference to
  the deleted legacy `DeviceCommandExecutor`);
  `BlueskyScanner._run_delegated_request` keeps only the bridge-specific
  facts.
- The claimed-scan-failure error message ("folder is left in place, never
  deleted") now lives in one shared helper,
  `scan_log.log_claimed_scan_failure`, used by both the bridge and the
  optimize runner (was duplicated verbatim).
- `BlueskyScanner.__init__`'s `optimization_loader` type hint updated to
  `Callable[[str | OptimizationSpec], Any]`, matching the 0.31.0
  delegated path (the docstring already said so).

## [0.31.0] - 2026-07-12

### Added

- **Optimize-mode ScanRequests run through the delegated GUI-bridge path**
  (M4 GUI-submission step iii). `BlueskyScanner.reinitialize(ScanRequest)`
  no longer refuses `mode: optimize`: the scan thread hands the request's
  `OptimizationSpec` to the GUI-injected `optimization_loader` (the same
  seam as legacy exec_config optimization — the Xopt/evaluator stack stays
  in `geecs_scanner.optimization`; this package still never imports it,
  now pinned by an AST-level test) and threads the returned bridge's
  `bind` into `run_scan_request` as the new `optimization_binder` hook.
  The runner claims the scan itself just before binding — after every
  fail-fast resolution and device connect — because the binder's
  analyzers need the real `ScanTag`; it then passes the pre-claimed
  number/folder to `session.optimize` and owns the `scan.log` attach
  (the session only self-attaches when *it* claimed). The bridge's
  optional `finish()` bookkeeping (legacy `xopt_dump.yaml`) runs after a
  successful run. Preserved behaviors: fail-fast pre-claim name
  resolution (VOCS catalog names now validate at reinitialize, pseudo
  variables refused there), actions-skipped-with-warning on optimize
  (recorded under `skipped_action_plans`), and the `db_scan_runtime`
  metadata. The GUI progress-totals hook now fires on the optimize path
  too, with the `(max_iterations, max_iterations × shots_per_step)`
  upper bound — the suggester may stop early. Reinitializing an optimize
  request on a scanner constructed *without* an `optimization_loader` is
  still refused with an explicit error (headless callers keep using
  `GeecsSession.run(request, resolver, objective=..., suggester=...)`).

## [0.30.1] - 2026-07-12

### Fixed

- **Foreign RunEngine event documents no longer mutate GUI scan progress**
  ([#511](https://github.com/GEECS-BELLA/GEECS-Plugins/issues/511)).
  `BlueskyScanner._on_document` guarded foreign start documents (the
  RUNNING-state guard on scan-number pickup, 0.30.0) but still counted
  *every* event document into `_completed_shots` and emitted a
  `ScanStepEvent` for it — a headless/foreign run driven directly on the
  shared session RunEngine could advance an idle or completed scanner's
  progress and contaminate `estimate_current_completion()`. The scanner
  now tracks ownership: a start document is claimed only while this
  scanner is RUNNING (every scan path sets RUNNING before its plan
  reaches the serial RunEngine), the claimed run's descriptor uids are
  collected from descriptor documents, and event documents are matched
  back through their `descriptor` uid — unmatched events are ignored,
  and the claim is cleared on the run's stop document. Residual
  limitation (documented in `_on_document`): a foreign run squeezed in
  after RUNNING is set but before this scanner's plan opens its run
  would still be mis-claimed; the RunEngine executes plans serially, so
  that window is only the bridge's pre-plan setup.

## [0.30.0] - 2026-07-11

### Added

- **Lifecycle events carry the claimed scan number.** `ScanLifecycleEvent`
  (`geecs_bluesky/events.py`) gains an optional `scan_number: int | None`
  field (default `None` — additive, consumers dispatching on event class
  names are unaffected) so GUIs can display "Scan NNN" from the event
  stream alone. `BlueskyScanner` stamps it onto every lifecycle emission:
  the exec_config paths (`_execute_scan`, `_run_optimization`) set it at
  their claim sites, so INITIALIZING/RUNNING/DONE/ABORTED all carry the
  number; the delegated ScanRequest path (where `session.scan` claims
  inside the engine, after INITIALIZING/RUNNING were emitted) picks the
  number up from the run start document (`geecs_run_wrapper` metadata) and
  re-emits RUNNING carrying it, so consumers get the number mid-scan, not
  only on DONE. Emissions before the claim — and every emission of a scan
  whose claim failed (NetApp unreachable) — carry `None`. A guard ignores
  start documents while the scanner is not RUNNING, so a headless run on
  the shared RunEngine cannot flip an idle GUI or plant a stale number.

## [0.29.0] - 2026-07-10

### Changed

- **One blessed gateway `:SP` put primitive.** Three call sites implemented
  "write a value to a gateway setpoint with GEECS-completion semantics"
  independently — `CaMotor`'s Layer-1 ophyd signal put, `ShotController`'s
  `CaPutSetter` raw caput, and the action factory's `_WireSettable` raw caput
  — and each independently decided the PV addressing dialect (ophyd `ca://`
  URI vs bare EPICS name), which tripled the suspect list during the
  closeout-hang diagnosis and produced the actual bug (issue #490; see the
  put-pathway discussion in PR #489). All three now delegate to
  `devices/ca/gateway_put.GatewaySetpointPut`, the single owner of the
  addressing rule (`bare_pv` strips `ca://`, rejects other schemes), the
  wire-value conventions (`str` for shot control, `wire_value` for actions,
  pass-through for typed motor signals), the timeout policy, `AsyncStatus`
  wrapping, and mock support. Per-consumer semantics are preserved exactly:
  `CaPutSetter` is now a thin pin of the primitive with byte-identical wire
  behavior (always-string puts, 10 s budget), `CaSettable`/`CaMotor`/
  `CaConfirmSettable` keep their typed setpoint signal as the transport and
  mock seam (`CaMotor` keeps `move_timeout` as the put budget), and action
  settables keep the wire-value convention and the issue-#490 bare-name
  invariant. New `tests/test_gateway_put.py` pins the addressing rule
  centrally, including that the primitive strips/rejects schemed names.

## [0.28.1] - 2026-07-10

### Fixed

- **Action set/check steps now work on numeric PVs.** `CaActionSignalFactory`
  built `str`-typed ophyd signals for `:SP` settables and readbacks; a float
  PV (e.g. `Position.Axis 1:SP`) then failed at connect ("inferred datatype
  float cannot be coerced to str") — found by the first hardware set-step on
  a numeric variable (M4 step-(i) smoke). Settables now put the wire string
  via raw CA (the hardware-proven `CaPutSetter` convention — CA converts to
  the PV's native type), with a dtype-inferred probe signal preserving the
  pre-claim fail-fast; readables are dtype-inferred (like telemetry).
  `values_match` handles native numerics with the legacy quirks intact.
- **The raw put uses the bare PV name.** `ca_pv()` returns the ophyd
  signal-URI form (`ca://<name>`); ophyd strips the scheme, raw aioca does
  not — a schemed name CA-searches for a PV that does not exist and hangs
  for the full timeout (live-found during the M4 step-(i) smoke as "the
  closeout hang", issue #490; the smoke now passes end-to-end on hardware,
  closeout set-steps included). New `tests/test_action_signals.py` pins the
  factory, including the bare-name invariant (it previously had no hermetic
  coverage — which is how both bugs survived to hardware).

## [0.28.0] - 2026-07-10

M4 step (i) — the GUI bridge reaches ScanRequest parity
(`Planning/gui_retrofit/00_overview.md` §3–4).

### Changed

- **`BlueskyScanner` delegates ScanRequest execution to `run_scan_request`.**
  `_reinitialize_from_scan_request` no longer synthesizes a legacy
  scan-config namespace: it validates every referenced name fail-fast
  (defaults, actions, save sets + rituals, trigger profile, scan variables)
  and stores the **original pre-defaults** request; the scan thread then
  runs it through the one engine definition — actions, entry rituals,
  multi-axis grids, db_scalars, and telemetry now all run through the GUI
  bridge. The legacy `exec_config` path is untouched.
- **Two new optional runner hooks on `run_scan_request`** (headless callers
  unchanged): `preflight(detectors, strict) -> list | None` — the
  scanner-layer operator-dialog seam, called pre-claim with the fully
  assembled detector list (`None` aborts; a reduced list is honored) — and
  `on_scan_start(total_steps, total_shots)` — the GUI progress-totals seam.
  Neither is called on the optimize path (an optimize preflight is a later
  seam).
- The bridge's preflight pipeline gained `disconnect_on_drop=False` for the
  delegated path (the runner's `finally` owns disconnection there); the
  exec_config path is byte-identical (default `True`).
- The retired byte-parity pin (request noscan ≡ synthesized exec_config) is
  replaced by the delegation-parity pin: a request through the bridge
  produces `session.scan` kwargs identical to headless `run_scan_request`
  (and the bridge never pre-claims — `session.scan` owns the claim and
  self-attaches `scan.log`).

### Removed

- The now-orphaned GUI-refusal cluster: `raise_if_actions_present`,
  `resolve_save_sets_checked`, and `MULTI_AXIS_MESSAGE` (their only callers
  were the bridge refusals this release removes), plus the bridge's
  `_request_step` machinery and `_run_request_step_scan`.

### Retained

- The optimize-mode refusal at `reinitialize(ScanRequest)` — wiring the
  GUI's `optimization_loader` into the delegated path is GUI-submission
  step (iii).

## [0.27.1] - 2026-07-10

Cleanup pass 2 — the docstring condensation (audit:
`Planning/cleanup_vision_v1/00_overview.md`). Docstrings/comments only;
every modified file verified AST-identical to the previous version with
docstrings stripped (zero code change), full suite green.

### Changed

- **~880 net lines of documentation redundancy removed across 27 files.**
  Docstrings now state the contract (what/args/raises + non-obvious
  invariants); design rationale, history narration, and verification
  anecdotes whose canonical copy lives in `CLAUDE.md` are cut to one-line
  pointers. Highlights: the save-set union rule is stated once (on
  `merge_save_sets`) instead of four times; `db_runtime`'s module docstring
  no longer duplicates the CLAUDE.md M3c section; `single_shot`'s 40-line
  RunEngine source quote is an 8-line conclusion; `action_compiler`'s
  legacy-equivalence story is told once instead of three times;
  `shot_controller`'s ordered-writes semantics live once, on `from_writes`.
- Load-bearing warnings whose only copy is the code were kept verbatim and
  are now inventoried in the audit doc (e.g. the cold-cache baseline-get
  race in `CaTriggerable._wait_for_shot`, the refire no-cancellation
  conclusion in `single_shot`, the `Reference()` re-parenting note in
  `contributor`).
- Stale claims fixed in passing: `settable.py`/`motor.py` no longer describe
  the shipped `CaConfirmSettable` as a future milestone; `shot_id.py` no
  longer references the deleted `geecs_device.GeecsDevice`.

## [0.27.0] - 2026-07-10

Cleanup pass 1 (audit: `Planning/cleanup_vision_v1/00_overview.md`) — no
behavior change.

### Changed

- **`ConfigResolver` / `ConfigsRepoResolver` moved to
  `geecs_bluesky/config_resolver.py`** (a pure relocation — the resolver
  layer shared no private helpers with the execution code). Both names are
  still importable from `scan_request_runner` (re-exported), so the existing
  import surface is unchanged.
- `BlueskyScanner._build_positions` now delegates to the session's
  `_positions` (was a near-verbatim copy of the start/end/step → linspace
  rule).

### Removed

- Dead GUI-compat shims on `BlueskyScanner`: `dialog_queue` and
  `restore_failures` (the GUI stopped reading them when `ScanManager`
  dropped them; `last_reinit_error` stays — the GUI reads it), plus the
  unused `_ScanConfig` import shim and `_CONNECT_TIMEOUT` constant.
- The singular `resolve_save_set_checked` / `resolve_save_set_and_rituals`
  pair — superseded by the plural M4 step-0 forms
  (`resolve_save_sets_checked` / `resolve_save_sets_and_rituals`); no
  production caller remained. Their tests were ported to the plural forms.
- `utils.build_signal_attrs` — its attr-collision disambiguation was never
  wired into any detector (all call `safe_name` directly); test-only.

### Fixed

- **Stale-reference sweep (~20 sites):** docstring cross-references to the
  deleted direct-backend device classes (`GeecsTriggerable`, `GeecsMotor`,
  `GeecsSettable`, `GeecsGenericDetector`, `GeecsTimestampedReadable`,
  `GeecsSnapshotReadable`) now point at the `Ca*` classes; the
  `devices/ca/*` module headers describe what each device *is* instead of
  comparing it to a deleted class; `step_scan.py`'s module example no longer
  shows the deleted host/port constructor (rewritten against `GeecsSession`
  factories); `CLAUDE.md` acquisition-mode dispatch names corrected.

## [0.26.0] - 2026-07-10

### Added

- **`CaConfirmSettable` — the topology-C device: set X, confirm on Y**
  (`devices/ca/confirm.py`). Acts on `ScanVariable.confirm` (declared in the
  schema since M1, unenforced until now): writes one variable but polls a
  *different* variable's streamed readback for the physical result — the EMQ
  triplet's `Current_Limit.ChN` (a software limit GEECS's own blocking set
  trivially confirms) vs its measured `Current.ChN`. Analog (float) match by
  tolerance, discrete (string/enum) match by exact equality — unifies with a
  future `CaShutter`. Defaults (`tolerance=0.05`, `timeout=10.0`) come from a
  live characterization of `U_EMQTripletBipolar:Current.ChN` (no beam,
  2026-07-09: jitter 0.01 A, response lag <1 s, settles within ~3 frames —
  see `Planning/scan_variable_metadata/00_overview.md` Deferred #5).
  `GeecsSession.confirm_settable(...)` constructs it; `resolve_movable_target`
  now returns the entry's `confirm` alongside `(device, variable, kind)`, and
  a new `build_movable` helper dispatches on it (confirm takes precedence over
  `kind`) for both `run_scan_request`'s grid-axis and optimize-mode movable
  construction. New `GeecsConfirmTimeoutError`.

### Fixed

Review pass on `CaConfirmSettable` (PR #477):

- **Optimize `on_finish` now goes through the movable's `set()`.**
  `GeecsSession._move_movables` (used for `on_finish="initial"`/`"best"`)
  used to put directly to `m._setpoint`, bypassing every movable's own
  completion semantics — for `CaConfirmSettable` that meant the exact "the
  limit register converged but nothing physically moved" failure this device
  exists to catch would go silently unconfirmed (only a warning if the raw
  put itself failed) on the final optimize move.
- **Discrete confirm matching no longer coerces numeric-looking strings.**
  `_matches` tried `float()` on both sides before falling back to equality,
  so a `datatype=str` confirm target could tolerance-match `"1.04"` against
  `"1.0"` — silently reintroducing analog matching for a discrete variable.
  Dispatch is now on the device's declared `datatype`, never on whether the
  values happen to be parseable as numbers.
- **`GeecsConfirmTimeoutError` added to `exceptions.__all__`** (it was
  defined but omitted from the exported exception surface).

## [0.25.0] - 2026-07-10

### Changed

- **Multi-save-set union in the runner (M4 step 0).** `ScanRequest` now
  carries `save_sets: list[str]` (was `save_set`); `run_scan_request` and the
  optimize path resolve **each** named save set and union them into one
  effective `SaveSet` before deriving the recorded device set. Per-device
  union rule (documented in the `scan_request_runner` module docstring and
  `merge_save_sets`): `scalars` union order-preserving/deduped, `images` /
  `db_scalars` / `all_scalars` OR together (True wins), the single non-`None`
  `role` is used — **conflicting explicit roles across the sets raise** (role
  sets the pacemaker/contributor/snapshot semantics, so a device required by
  more than one set must not disagree on it) — and entry-level
  `setup`/`closeout` ritual name lists union
  (deduped). Entry-level rituals are collected across *all* named sets,
  deduped by plan name so a shared ritual runs once
  (`resolve_save_sets_and_rituals`). `save_set_to_devices_config`, the
  reserved-boundary warning, and the run metadata (`save_sets` provenance
  key) all operate on the merged set.
- **Telemetry exclusion spans all named sets (M3c).**
  `select_telemetry_variables` is now passed the merged save set, so Tier-2
  background telemetry correctly excludes every device in *any* of the named
  sets.
- **GUI bridge is list-aware.** `BlueskyScanner._reinitialize_from_scan_request`
  resolves `save_sets` via the new `resolve_save_sets_checked` (union of
  devices; still refuses entry rituals / actions / multi-axis grids — routing
  those through the bridge remains a later milestone).

## [0.24.1] - 2026-07-09

### Documentation

- **CaSettable / CaMotor docstring honesty.** Corrected the framing that
  `motor` is *the* way to declare a positioner and `CaSettable` is unsafe for
  stages. A plain `CaSettable` already waits for GEECS's native blocking
  set-completion; `CaMotor`'s readback poll only adds information when the
  readback is an *independent* measurement of the same variable (a stage
  encoder), and is near-cosmetic when the readback merely echoes the command.
  Both docstrings now also note the topology neither class covers — a device
  whose measured variable differs from the one set (the EMQ triplet) — which
  the schema names via `ScanVariable.confirm`. No code change.

## [0.24.0] - 2026-07-08

M3c — the DB-integration runtime tier lands, wiring the two-tier recording
model (`SaveSetEntry` runtime contract, `ExperimentDefaults`) into
`GeecsSession.run`.  **Get-side only:** `db_scalars` (Tier 1 recorded
scalars) and background telemetry (Tier 2) are honored; the DB **set-side**
(scan start/end writes from the `set='yes'` rows) is intentionally
**disabled** in this version.  Live DB inspection showed the boundary writes
would race the shot controller / TriggerProfile on the DG645
(`U_DG645_ShotControl`'s `set='yes'` rows are the `Trigger.Source` /
`Amplitude.Ch AB` variables the shot controller already drives), and the
remaining `set='yes'` rows are almost all `save` / `localsavingpath` (the
scanner owns saving via its save-windowing).  The reserved schema fields
(`SaveSetEntry.at_scan_start` / `at_scan_end`,
`ExperimentDefaults.apply_db_scan_defaults`) are kept on record for a
possible future re-enable but are not applied.

### Added

- **`acq_timestamp` now appears in the legacy s-file for image/file-saving
  devices.**  A device that saves non-scalar data (`save_nonscalar_data=True` —
  `CaGenericDetector`, `CaTimestampedReadable`) surfaces its `acq_timestamp` as
  a legacy scalar column (`<device> acq_timestamp`), so saved files tie back to
  scan rows (the saved image filenames are stamped with it).  Previously an
  images-only save produced an s-file with only `Bin # / scan / Shotnumber` and
  no way to correlate rows to files from the s-file alone.  `acq_timestamp`
  stays an excluded companion column for pure-scalar devices.
- **db_scalars resolution (Tier 1).**  A save-set entry's recorded scalars are
  now its DB `get='yes'` variables (from `expt_device_variable`) unioned with
  its explicit `scalars` list (`db_scalars=True`, the default);
  `all_scalars=True` unions every DB variable instead; `db_scalars=False` (what
  the legacy converter pins) records only the explicit list.  Resolution is a
  pure function (`db_runtime.resolve_entry_scalars`) threaded through
  `save_set_to_devices_config(save_set, scalar_policy)`.  With no DB policy
  (the GUI-bridge path, or off the lab network) only the explicit list is
  recorded — strictly additive over M3b.  `all_scalars` is no longer a blanket
  `NotImplementedError`: it resolves when a policy is present, and only raises
  on the no-policy path.
- **Scan start/end DB writes — set-side, intentionally disabled.**  The DB
  `set='yes'` boundary writes are **not** applied by the engine in this
  version (see the version summary above for the DG645-conflict rationale).
  A save-set entry that still carries `at_scan_start` / `at_scan_end` is not
  an error — the engine logs one WARNING naming the device and records
  nothing; the values are inert.  No `db_scan_writes` metadata is produced.
- **Background telemetry tier (Tier 2).**  Every live experiment device with a
  `get='yes'` variable not in the save set is recorded as best-effort snapshot
  columns via the new soft `CaTelemetryReadable`: read-only, sampled once per
  event row, never waited on — a signal read that fails degrades to a
  dtype-appropriate null cell (NaN for a numeric column, `""` for a string
  column) instead of aborting, and a device unreachable at scan start is dropped
  with a log line (`GeecsSession.telemetry` returns `None`).  Telemetry is
  **dtype-tolerant, per-variable**: each signal's type is inferred from its PV
  (`epics_signal_r(datatype=None, …)`), so numeric variables stay numeric
  (float) for downstream analysis while enum/string/path variables — e.g.
  `U_VisaPlungers` `DigitalOutput.Channel N` — are logged as their string/label
  value.  A single non-numeric `get='yes'` variable no longer fails to connect
  and take the whole device (including its numeric columns) down with it: **no
  telemetry variable or device is dropped for a type reason** — only a
  genuinely unreachable device degrades to a dropped device.  If we `get` it,
  we log it.  Telemetry columns carry the `telemetry_<device>-` name prefix so
  they are distinguishable from Tier-1 save-set data (see `EVENT_SCHEMA.md`);
  selection is recorded in run metadata under `background_telemetry`.  Gated on
  `ScanRequest.background_telemetry` (else
  `ExperimentDefaults.background_telemetry`, default true).
- New `geecs_bluesky/db_runtime.py` — the pure resolution logic plus the
  DB-backed `GeecsDbScalarPolicy` provider (the one place touching `GeecsDb`),
  failure-tolerant: a DB lookup that fails degrades to empty policy with a
  warning, so a scan never aborts because the DB was briefly unreachable.

### Notes

- Optimize mode resolves `db_scalars` (recorded-scalar consistency) but does
  **not** run background telemetry yet (no scan-boundary hook on
  `GeecsSession.optimize`) — skip-and-record; recorded in run metadata under
  `db_scan_runtime`.  The set-side is disabled everywhere in this version.
- Requires the gateway ≥ 0.9.0 (new `GeecsDb.get_all_experiment_variables`).
  `GeecsDb.get_scan_boundary_writes` also lands in 0.9.0 as a reserved,
  currently-unused query (the set-side is disabled) — not consumed by the
  engine.

## [0.23.0] - 2026-07-07

M3b — every engine-pending ScanRequest seam closes: actions execute inside
scans, multi-axis grids run, and multi-device trigger profiles drive the
shot controller. `GeecsSession.run(request)` now executes the full schema
surface except the documented v1 gaps (pseudo variables, `all_scalars`,
optimize-mode injection/actions). Also folds in the M3a action-compiler
entry that PR #461/#464 intentionally shipped without.

### Added

- **ActionPlan → Bluesky plan-stub compiler** (M3a, PR #461/#464 —
  `plans/action_compiler.py`): the successor of the legacy `ActionManager`
  executor. `compile_action_plan` turns a validated
  `geecs_schemas.ActionPlan` into a plain message generator — `set` becomes
  `abs_set(wait=wait_for_execution)` (the legacy `sync=` semantics), `wait`
  becomes an RE-interruptible `bps.sleep`, `check` reads and compares with
  the exact legacy `interpret_value` + `==` rules (`values_match`, quirks
  pinned verbatim), `run` recurses through a registry with a cycle guard
  legacy never had. Purity contract: no CA, no PV strings — signals come
  from an injected `SettableFactory`. Fidelity pinned end to end against
  the converted legacy corpus (`Amp4_DUMP_HP`: nested plans, wait
  durations, check-mismatch abort).
- **Production `SettableFactory`** (`devices/ca/action_signals.py`,
  `CaActionSignalFactory` via `session.action_signal_factory()`): `set`
  steps put wire strings to the variable's gateway `:SP` PV — CA
  put-completion rides GEECS's blocking set, so `wait_for_execution` keeps
  its exact legacy meaning; `check` steps read the streamed readback as a
  string (matching the legacy coercion pipeline). Signals are cached per
  (device, variable), connected on the RE loop **before** the plan runs
  (`prefetch_action_signals` — a lazy connect inside the RE loop would
  deadlock; prefetching also fail-fasts unreachable targets pre-claim), and
  ride the scan's device cleanup.
- **Action execution wired into scans.** `build_step_scan_plan` /
  `GeecsSession.scan` grew `setup` / `per_step` / `closeout` plan-stub
  hooks with documented placement: *setup* runs first thing in the composed
  plan (after device connect + pre-flight, before free-run quiesce/t0-sync
  and the first step); *per_step* is yielded by both step plans at every
  step boundary — after the move, before that step's shots (free-run
  brackets steps with arm/disarm so per-step actions run disarmed; strict
  is quiescent between plan-owned shots); *closeout* is the outermost
  `finalize_wrapper`, so it runs even on mid-scan abort (legacy
  ActionControl parity) and always after the trigger disarm.
  `run_scan_request` assembles the §4.4b layers in nesting order — setup:
  ExperimentDefaults → SaveSet entry rituals (collected across entries,
  de-duplicated by name, each once) → the request's own; closeout: the
  exact mirror (request → entries → defaults) — and records the assembled
  order in the run metadata (`action_plans`) for provenance.
  `apply_experiment_defaults` now appends default closeout plans after the
  scan's own (the mirrored merge rule ratified in geecs-schemas 0.2.0).
- **Multi-axis grid execution.** `len(axes) >= 2` runs an outer-product
  grid (first axis outermost/slowest — the schema's documented semantics):
  one movable per axis, explicit grid-point tuples, one bin per grid point,
  per-step actions at every grid point. The step plans accept a motor
  *sequence*; only the axes whose target changed are re-moved (innermost
  varies fastest), changed axes move concurrently via `bps.mv`. Every
  motor's readback lands in every event row (exactly like the single-motor
  path); run metadata carries `scan_axes` / `grid_shape` /
  `num_grid_points`, and the legacy 1-D ScanInfo fields describe the
  outermost axis (`scan_parameter` = comma-joined targets).
- **Multi-device trigger profiles.** `ShotControlWrites`
  (`models/shot_control.py`) — per-state **ordered** `(device, variable,
  value)` write lists — plus `ShotController.from_writes`: one `CaPutSetter`
  per distinct target (cached across states), transitions replayed in
  declared order with each write completing before the next
  (schema-documented ordering). `trigger_writes_from_profile` replaces the
  single-device `shot_control_config_from_trigger_profile` pivot for
  request execution; `GeecsSession.shot_control` accepts `ShotControlWrites`
  alongside the legacy shapes, and the GUI bridge's `reinitialize(ScanRequest)`
  stores the generalized writes too. The legacy `ShotControlConfig` path
  (concurrent per-state writes, single device) is untouched.
- **Hardware verification test** (`tests/test_scan_request_hardware.py`,
  integration-marked, skipped in CI): one noscan ScanRequest through
  `session.run()` against the real gateway, corpus configs converted on the
  fly; device/config names parameterizable via `GEECS_HW_*` env vars
  (defaults: `UC_Amp4_IR_input` camera, `HTU-LaserOFF` trigger profile).

### Changed

- `scan_request_runner` no longer refuses actions, entry rituals,
  multi-axis grids, or multi-device trigger profiles — the
  validate-then-`NotImplementedError` treatment is gone from the engine
  path (names still resolve fail-fast pre-claim). The GUI bridge's
  `reinitialize(ScanRequest)` still refuses actions/multi-axis with a
  pointer to `GeecsSession.run` (routing them through the bridge is the GUI
  submission milestone). Optimize-mode requests with action bindings or
  entry rituals are refused loudly (`GeecsSession.optimize` has no action
  hooks yet — new documented gap).

### Fixed

Review-pass fixes (max-effort review of this milestone):

- **Optimize-mode requests with actions no longer refuse — they run and skip
  the actions (logged + recorded in run metadata).** `apply_experiment_defaults`
  merges default setup/closeout into the request before the optimize check, so
  the old `NotImplementedError` would have blocked *every* optimization the
  moment an experiment defined default bracket actions (via a future
  `experiment_defaults.yaml`), even with no per-request actions. Optimize has no
  action hooks yet, so the actions (request, defaults, and save-set rituals) are
  skipped rather than executed; the skip is a WARNING and lands in run metadata
  under `skipped_action_plans`. Unknown action names still fail fast.
- **Free-run scan with saving detectors but no shot controller now warns.**
  Native-save windowing needs the controller's quiesce to stop the trigger;
  with no controller there is no such point, so `build_step_scan_plan` logs a
  clear warning that frames captured during t0-sync/moves may be orphaned
  (rather than silently leaking or refusing the supported controllerless config).
- **Lazy action-plan registry now only masks genuine "not found".** The bare
  `except Exception` in `_LazyResolverRegistry` turned any resolver fault into a
  `KeyError`/miss (misdirecting debugging to "plan not found" with no
  candidates); it now catches only `GeecsConfigurationError` and lets unexpected
  faults propagate.
- Removed dead code: an unreachable `return` in `GeecsSession.optimize` and the
  superseded `collect_save_set_action_names` (production uses
  `collect_save_set_rituals`).

Both items below were found during Gate-2 hardware verification of this release
(2026-07-07: Scans 013–016 — the first ScanRequest-driven hardware runs).

- **Native-save windowing** — saving is now enabled only while the trigger
  cannot free-run. The run wrapper's eager save-on (before arming) let
  free-running frames be saved as orphans that join no event row: Scan015
  (strict) saved 6 images for 3 shots (three 1 Hz STANDBY frames during the
  ~5 s setup-action window), Scan013 (free-run) saved 6 for 5 (one frame in
  the window between save-on and quiesce[OFF]). `geecs_run_wrapper` grew
  `defer_save_on` + a public `save_enable_plan` stub; the step plans grew an
  `enable_saving` hook yielded at the first orphan-free moment — strict:
  after ARMED + quiescence confirmation; free-run: immediately after
  quiesce[OFF], before t0-sync. Setup actions run before that point by
  construction (their duration was producing the orphans). Save-off is
  unchanged: the innermost finalize, before the disarm — so on completion
  *and* abort the order is save-off → disarm[STANDBY] → closeout. Direct
  `geecs_run_wrapper` users (no `defer_save_on`) keep the eager behavior;
  `GeecsSession.optimize`'s adaptive scan still enables saving eagerly
  (known remaining window — the step-scan paths were the hardware-verified
  offenders).
  **Tail closed too** (Gate-2 re-verify on 26_0708: strict Scan001 was
  perfect — exactly 3 images for 3 shots — but free-run Scan002 saved 7
  images for 5 shots: STANDBY passes external edges, so frames kept
  arriving between the last per-step disarm and the finalize save-off,
  stretched by close_run/Tiled document writes): the free-run plan now
  quiesces to OFF **after the last step's shots, before the tail flush**
  (the flush reads cached last values by design, so flushing while OFF is
  safe), and an internal abort-parity finalize quiesces before the caller's
  cleanup when the scan dies mid-plan — so completion and abort share one
  end order: quiesce[OFF] → tail flush → save-off → finalize
  disarm[STANDBY] (legacy free-running end state) → closeout.
  **Accepted, deliberately not fixed**: between-step STANDBY frames in
  multi-step free-run scans — per-step disarm during moves is legacy-parity
  behavior (jet off during moves); those frames join by timestamp and
  orphans are ignorable. Do not turn this into per-step save toggling.
- **`scan.log` for headless GeecsSession runs** — Scans 013–016 had no
  per-scan log because the handler lived bridge-side only. The helper moved
  to a shared `geecs_bluesky/scan_log.py` (`scan_log()` context manager +
  `ScanLogContextFilter`, verbatim behavior; `BlueskyScanner._scan_log`
  delegates to it), and `GeecsSession.scan`/`optimize` now attach it around
  the run + exports whenever the session itself claimed the scan number.
  Pre-claimed scans (the GUI bridge path, or any caller that opened its own
  per-scan log) deliberately do not self-attach — the claiming caller owns
  the handler, and a second one would duplicate every line.

### Removed

- `shot_control_config_from_trigger_profile` and its multi-device
  `NotImplementedError` (superseded by `trigger_writes_from_profile` +
  `ShotController.from_writes`); the engine-path multi-axis and
  action-refusal raises in `run_scan_request` / `resolve_save_set_checked`'s
  runner usage.

## [0.22.0] - 2026-07-07

Engine consolidation (target-architecture vision §2): the engine owns its
event vocabulary, operator interaction is one injected seam, pre-flight is a
declarative pipeline, and a `ScanRequest` can drive a scan end to end.
Everything is behind compatibility shims — no GUI changes, no behavior
changes on the existing paths.

### Added

- **`geecs_bluesky/events.py`** — the typed scan event vocabulary
  (`ScanEvent` hierarchy, `ScanState`, `DialogRequest`) moved here verbatim
  from `geecs_scanner.engine.scan_events` / `dialog_request` (semantics
  preserved exactly; `DialogRequest` no longer needs geecs_python_api —
  the `DEVICE_COMMAND_ERRORS` tuple stayed behind in the legacy shim).
  `geecs_scanner`'s modules are now re-export shims of the *same class
  objects*, so every existing import path and isinstance check keeps
  working. The engine's defensive try/except imports of its own vocabulary
  are gone — `bluesky_scanner` imports directly from `geecs_bluesky.events`.
- **`geecs_bluesky/operator_channel.py`** — the one seam through which the
  engine asks the operator anything: `OperatorChannel.ask(OperatorQuestion)
  → "continue" | "abort" | default`. Two implementations:
  `EventStreamOperator` (today's exact GUI behavior — DialogRequest inside
  a ScanDialogEvent via `on_event`, blocking on `response_event` with a
  timeout) and `NullOperator` (headless: logs, returns the question's
  default). `BlueskyScanner._request_operator_decision` and the pre-flight
  now route through the channel; the scanner builds `EventStreamOperator`
  when `on_event` is wired, else `NullOperator`. Byte-identical behavior,
  pinned by the existing dialog tests.
- **`geecs_bluesky/preflight.py`** — pre-flight checks as a pipeline:
  checks return pass / ask(question, handlers) / abort, and `run_preflight`
  executes the list, routing questions through the OperatorChannel. The two
  existing checks moved in with their exact semantics
  (`GatewayLivenessCheck`: CONNECTED PV both modes, fail-open on unreadable
  status, abort-only for a dead free-run reference;
  `FreeRunStalenessCheck`: trigger-running staleness with the recheck
  grace, all-stale trigger-off wording, abort-only stale reference).
  `BlueskyScanner._preflight_check_sync_liveness` is now a thin call into
  the pipeline; new checks are list entries, not new scanner branches.
- **`geecs_bluesky/scan_request_runner.py` + `GeecsSession.run(request)`** —
  execute a `geecs_schemas.ScanRequest` end to end: a `ConfigResolver`
  protocol resolves the request's names (save set, trigger profile, scan
  variable, action plans) to schema models, and `ConfigsRepoResolver` reads
  the real configs repo — new-schema YAML (`schema_version` present) loads
  directly, legacy YAML converts via `geecs_schemas.convert`, so the whole
  existing corpus works immediately. Mapping helpers derive the engine
  shapes (`save_set_to_devices_config` with the documented role-derivation
  rules; `shot_control_config_from_trigger_profile`, living bluesky-side to
  respect the dependency direction). Documented v1 gaps refuse loudly
  (`NotImplementedError`): multi-axis grids, action bindings (names are
  *validated* now, executed when the ActionPlan compiler lands), pseudo
  scan variables, `all_scalars`, and optimize mode without an injected
  objective/suggester.
- **`BlueskyScanner.reinitialize` accepts a ScanRequest** (duck-detected by
  type) as a parallel entry beside the untouched `exec_config` path; a
  request is resolved fail-fast and mapped onto the same internal
  machinery. Parity pin: `tests/test_bluesky_scanner_scan_request_seam.py::
  test_scan_request_noscan_parity_with_exec_config` asserts a ScanRequest
  noscan drives the identical fake-session `scan()` call as the equivalent
  exec_config.
- **geecs-schemas is now a dependency** (path dep; pydantic-only).
- **Scan-setup schema addendum support** (multi-device trigger states,
  SaveSet action refs, ExperimentDefaults — landed in geecs-schemas the
  same day): the trigger adapter handles both TriggerProfile generations —
  the retired single-device shape and the landed multi-device shape
  (per-state ordered write lists carrying their own `device`) — via a
  **single-device fast path**; writes spanning devices raise
  `NotImplementedError` ("multi-device trigger profiles land with a later
  milestone"), the same schema-accepts/engine-pending pattern as
  multi-axis. Resolvers gained `resolve_experiment_defaults()`
  (`ConfigsRepoResolver` reads and validates
  `<experiment>/experiment_defaults.yaml`); the merge rule is the model's
  — a default trigger profile applies only when the request names none,
  default setup/closeout plans are *prepended* to the request's own — and
  every applied default is recorded into the run metadata
  (`md["applied_defaults"]`) for provenance. SaveSet **entry-level**
  setup/closeout plan references get the same
  validate-now/refuse-until-the-actions-milestone treatment as
  request-level bindings (`resolve_save_set_checked`; plans the
  save-element converter extracts from legacy `setup_action`/`scan_setup`
  blocks resolve too).

### Changed

- The unanswered-dialog warning is logged by `EventStreamOperator` as
  "Operator dialog … got no response" (was "Pre-flight dialog …") — the
  channel is no longer pre-flight-specific. Log wording only.

## [0.21.0] - 2026-07-07

### Added

- **GUI progress events in Bluesky mode** — `BlueskyScanner._on_document`
  now emits a `ScanStepEvent` (shot-level, `phase="completed"`) for every
  Bluesky event document through the same `on_event` callback as the
  lifecycle events, so the Scanner GUI progress bar advances identically
  for both backends with no GUI changes (`shots_completed` = running
  event-document count, clamped at `total_shots` so the free-run
  tail-flush overcount stays cosmetic; step index derived from the
  schema-v1 `bin_number` column). Closes the "progress bar never advances
  in Bluesky mode" gap — see `Planning/gui_stewardship/00_overview.md` §5.
- **Gateway-liveness pre-flight dialog (both acquisition modes)** — every
  CA sync device now carries a non-readable `connected_status` child on the
  gateway's per-device `[Experiment:]Device:CONNECTED` status PV (enum
  `Disconnected`/`Connected`, MAJOR severity while the device's TCP stream
  is down — PV_CONTRACT.md §1/§5), created outside
  `add_children_as_readables()` so it never appears in event rows or
  `describe()`. This is the authoritative, mode-independent liveness
  signal: **the gateway serves every DB device's data PVs whether or not
  the device is up, so CA-connect success never implied device liveness**
  (an OFF camera's PVs connect fine — the root cause of the 2026-07-07
  strict Scan006 incident, where a dead camera burned all three refires
  post-claim). Before a scan claims its folder (so an abort burns no scan
  number), `BlueskyScanner._preflight_check_sync_liveness` reads each sync
  device's `connected_status` from the scan thread
  (`run_coroutine_threadsafe` on the RE loop, 2 s budget, **fail-open**: an
  unreadable CONNECTED PV — e.g. an old gateway without status PVs — logs
  at DEBUG and reads as live). Devices reporting DISCONNECTED raise an
  operator dialog through the legacy channel (`DialogRequest` inside a
  `ScanDialogEvent`): drop-and-continue (disconnected, removed from the
  detector list, logged loudly) vs abort; a disconnected free-run reference
  (pacemaker) is abort-only in v1 (second button is a clearly-labeled
  "Try Anyway"). In free-run mode a second stage keeps the `acq_timestamp`
  staleness check (threshold 10 s — now relevant *only* here — with one
  ~2 s re-check grace) for the trigger-must-be-free-running requirement,
  now unambiguous: all devices CONNECTED but all frames stale → "trigger
  appears to be off" dialog (Start Anyway / Abort); the residual
  CONNECTED-but-stale contributor with a fresh reference keeps the drop
  dialog (the fresh reference proves the trigger runs, so it is a
  per-device acquisition problem). Strict mode is liveness-only — the
  previous differential-staleness heuristic is removed (CONNECTED is
  authoritative; frames are not needed pre-scan since the trigger may sit
  OFF until ARMED). Headless (`on_event=None`), missing geecs_scanner, or
  an unanswered dialog (30 s timeout) preserve today's behavior — proceed
  and fail loudly downstream. New `GeecsDeviceDownError` carries the
  disconnected-device message; `GeecsStaleDevicesError` remains for the
  free-run staleness dialogs. Implements the stewardship plan's first
  concrete use case (`Planning/gui_stewardship/00_overview.md` §4).
- **Refire gated on gateway liveness** — `geecs_single_shot`'s bounded
  refire now reads the frameless device's `connected_status` (via `bps.rd`
  in plan context) before re-firing: a device reporting DISCONNECTED gets
  no refire — the plan raises `GeecsDeviceDownError` immediately, naming
  the device as down ("went down mid-scan — not a frame drop"), instead of
  burning ~3 s attempts against hardware that cannot answer. A live or
  unreadable status (fail-open) keeps the existing `max_refires` semantics
  for genuine frame drops.

## [0.20.0] - 2026-07-06

### Added

- **Analyzer-device auto-provisioning for optimization scans (legacy
  parity)** — `BlueskyScanner._run_optimization` now merges the optimizer
  config's `device_requirements` (exposed duck-typed on the optimization
  bridge, like `on_finish`/`finish`) into the save-device set before
  session devices are built, mirroring the legacy
  `device_manager.load_from_dictionary` path: a required device absent
  from the GUI save list is added with the requirement's config
  (synchronous, `save_nonscalar_data`, variable list — the auto-generated
  analyzer template), while a device already on the GUI list keeps its GUI
  settings and only gains missing required variables. Auto-provisioning is
  logged at INFO. Objectives' cameras no longer need to be added to the
  save list manually.
- **Case-insensitive device-requirement merge** (live-observed 2026-07-06,
  first GUI optimization test) — GEECS is internally case-inconsistent
  about device-name spelling (the DB said `UC_Amp4_IR_input`, the
  optimizer config `UC_Amp4_IR_Input`), and the case-sensitive merge added
  a duplicate wrong-case device whose gateway PVs could never connect (CA
  names are case-sensitive). Requirement names now match existing GUI save
  devices via `str.casefold`; on a hit the requirement merges into the
  existing entry under the GUI's spelling (the one that connects), with
  the case difference logged at INFO. Genuinely new devices keep the
  requirement's spelling, and the auto-provisioning log now hints to
  verify the spelling against the GEECS database.
- **Bounded refire for strict-mode single shots**
  (`geecs_single_shot(..., max_refires=2)`) — live evidence from the first
  GUI optimization campaign (2026-07-06, Undulator Scan011): 84 SINGLESHOT
  fires produced 83 camera frames; the one no-frame fire (the Basler's
  known ~1% frame-drop intermittency) failed the group wait and aborted
  the whole optimization. A missed pulse never yields a frame, so a
  device that produces nothing now gets the shot re-fired (fresh trigger
  group per attempt, WARNING naming the device) up to `max_refires` times
  before the `FailedStatus` propagates. Strict semantics are preserved:
  a failed attempt records nothing, and any orphan frame from a partial
  miss is drained by the next attempt's `trigger()` baseline, so every
  recorded row is one physical shot. The 3.0 s trigger timeout was
  measured adequate (fire→frame offset median 0.21 s, max 1.06 s across
  the 83 good shots), so no timeout knob was added — waiting longer
  cannot recover a dropped frame; re-firing can.

## [0.19.2] - 2026-07-06

### Changed

- Native-file naming is now imported from `geecs_data_utils.native_files`:
  `assets/registry.py`'s `native_file_filename` and the internal
  `_native_file_path_builder` delegate to the shared contract module
  (`native_file_name` / `native_file_path`, which folds `directory_suffix`
  into both the folder and the filename stem). The registry keeps its
  bare-extension normalization as a local wrapper. No behavior change; the
  registry/asset tests pass unchanged.


## [0.19.1] - 2026-07-06

### Fixed

- **`CaTriggerable` cold-cache race: the first shot after connect can no
  longer be lost or mis-baselined** — with a cold monitor cache
  (`_last_acq is None`, i.e. no acquisition delivered since subscribe), the
  cold path of `_wait_for_shot` took a CA-get baseline inside the returned
  coroutine and then drained the shot queue. A strict-mode shot fired
  immediately after `trigger()` could land before the coroutine ran and be
  (a) drained away, or (b) become the baseline itself — either way the
  single shot timed out (`GeecsTriggerTimeoutError`). The cold path now
  takes **no CA-get baseline at all** (review follow-up: the get raced the
  shot — a first acquisition landing inside the get's round-trip became the
  baseline). Instead, `trigger()` drains the queue synchronously on both
  paths — on a cold cache the drain can only discard a stale CA subscribe
  replay, never a requested shot, since nothing fires before `trigger()`
  returns — and the first positive monitor update after `trigger()` is the
  shot. Warm-path behavior (synchronous drain + `!= t0` baseline test in
  `trigger()`) is unchanged.
- **Off-network session/scanner construction no longer blocks on the Tiled
  server** — `subscribe_tiled` (used by `GeecsSession(tiled=True)`, and
  therefore `BlueskyScanner`) called `tiled.client.from_uri` unconditionally,
  hanging for the full HTTP connect timeout when the catalog server was
  unreachable (the routine off-lab-network case). A bounded TCP reachability
  pre-check (`tiled_server_reachable`, budget
  `TILED_REACHABILITY_TIMEOUT_S` = 2 s) now runs before the client is
  created; an unreachable server logs a clear warning ("Tiled server <uri>
  unreachable; Tiled persistence disabled for this session") and skips the
  subscription, leaving the session fully usable. On-network behavior,
  `tiled=False`, and the missing-config/missing-package paths are unchanged.
- **CA transport pinned on every device PV** — all PV strings handed to
  ophyd-async `epics_signal_r/rw` in `devices/ca/*` now carry an explicit
  `ca://` prefix (new shared helper `devices/ca/_pv.py::ca_pv`). ophyd-async
  selects the default transport for un-prefixed names by what is importable:
  with `p4p` installed but `aioca` missing, every PV silently flipped to PVA
  and each connect against the CA-only gateway timed out with generic
  errors. Pinned, such an environment now fails loudly at signal
  construction ("Protocol ca not available, did you `pip install
  ophyd_async[ca]`?"). The prefix is stripped before the backend stores the
  PV, so event keys and source strings are unchanged.
  `ShotController.over_ca`'s setters intentionally keep bare PV names — they
  talk to aioca directly, which treats a prefix as part of the PV name.


## [0.19.0] - 2026-07-05

### Added

- **GUI optimization bridge support** — `BlueskyScanner` now runs
  OPTIMIZATION scans: a GUI-injected `optimization_loader`
  (`geecs_scanner.optimization.session_bridge.load_session_optimization`,
  wired in `RunControl`) supplies the config-driven Xopt 3.1 stack
  (evaluators, ScanAnalysis analyzers, generator factory) while the scanner
  maps the request onto `GeecsSession.optimize` — VOCS variables become
  session settables, save devices are the detectors, iterations come from the
  configured step count. Dependency direction stays GUI → geecs_bluesky.
- `plans/run_wrapper.py::claim_scan` — like `claim_scan_number` but returns
  the full `ScanTag` (analyzers load native files by tag);
  `claim_scan_number` is now a thin wrapper over it.
- `GeecsSession.optimize` accepts pre-claimed `scan_number`/`scan_folder`
  (mirroring `scan()`), so the scanner's per-scan log and the bridge's
  `ScanTag` cover the whole run.
- The per-scan log (`scan.log`) now also captures the
  `geecs_scanner.optimization`, `scan_analysis`, and `image_analysis`
  loggers, so an optimization scan's per-bin story (file mapping, analyzer
  runs, objective values) is visible from the scan folder.
- After `session.optimize` returns, `BlueskyScanner` invokes the bridge's
  optional `finish()` hook (post-run bookkeeping, e.g. the legacy
  `xopt_dump.yaml`).
- `_write_scan_info` stamps `Scanner = "bluesky"` into ScanInfo — metadata
  only (nothing depends on it for correctness), so tooling can tell
  Bluesky-produced scans from legacy MC ones.

- **Gateway address from the shared GEECS config** — clients resolve the CA
  gateway like they resolve the database: `[epics] ca_addr_list` (and
  optional `ca_auto_addr_list`, default `NO`) in
  `~/.config/geecs_python_api/config.ini`, applied at package import before
  aioca creates its CA context. An exported `EPICS_CA_ADDR_LIST` always
  wins. Removes the per-shell env-var requirement for Windows GUI clients.
- t0-sync failures now name the stale device(s) and their lag ("U_CamA
  (5.000s behind U_CamB)") instead of only reporting the anonymous spread —
  a dead/off contributor serves its cached timestamp forever, and with N
  cameras the bare spread doesn't say which one to go look at.

### Fixed

- A synchronous save device with an empty `variable_list` (e.g. an
  image-only camera element) is no longer silently dropped by
  `BlueskyScanner`: `acq_timestamp` is always created as a dedicated child,
  so the device is built normally — matching the legacy scanner, which
  force-appends `acq_timestamp` to every synchronous device. Only an
  asynchronous snapshot device with no variables is skipped, now with a
  warning instead of a debug line. Found by the first unreachable-reference
  live check: a healthy image-only camera was skipped at DEBUG and the scan
  aborted blaming a connect failure that never happened; the pacemaker
  abort message now also names each device's actual failure.
- **Free-run pacing survives a reference connect failure** (PR #449 review
  #2) — when the designated reference (pacemaker) fails to connect, the
  next synchronous device is promoted to the reference role (built
  Triggerable via `session.detector`); if none connects, the scan raises
  `GeecsConfigurationError` instead of recording unpaced duplicate rows of
  cached frames. `geecs_free_run_step_scan` additionally rejects any
  non-`Triggerable` reference outright.
- Stop works before the plan reaches the RunEngine (review #8): the scan
  thread checks the abort flag after device connect and before claiming a
  folder; `RE.abort()` is only called on a non-idle engine; and a timed-out
  thread join keeps the handle so `is_scanning_active()` stays `True`
  rather than letting a second scan start on a busy engine.
- Early exits are ordered before `claim_scan_number` (review #14), so a
  validation failure no longer leaves an empty claimed `ScanNNN/` folder;
  VOCS settables join the cleanup list at connect time (no leaked CA
  monitors); unavoidable post-claim failures log the claimed-but-incomplete
  folder loudly (it is never deleted).
- Strict-mode fail-fast gaps closed (review #11): `optimize()` validates
  shot control like `scan()` and both validate *before* claiming; the
  validator also requires a non-empty `SINGLESHOT` state (`fire_shot` would
  be a silent no-op); shot-control setter PVs are reachability-checked when
  `shot_control()` attaches, so a typo'd device fails in seconds instead of
  blocking every mid-plan caput.
- `optimization.json` is always valid JSON (review #15): non-finite
  objective values serialize as `null`, with `allow_nan=False` so a
  sanitizer regression fails loudly.
- `scan()`/`optimize()` with `scan_number` but no `scan_folder` raise a
  clear `GeecsConfigurationError` instead of crashing on `Path(None)`;
  `shot_control({})` detaches cleanly like `shot_control(None)`.
- One TiledWriter exception no longer kills Tiled persistence for the rest
  of the session (review #9): `SafeDocumentCallback` re-enables at the next
  run's start document and logs which run lost persistence.
- CA devices bound their acq_timestamp monitor queue (drop-oldest ring,
  32 entries) so idle contributors no longer grow memory every machine
  shot, and every CA device type implements `disconnect()` (via ophyd-async
  `SignalR.clear_sub`) so per-scan teardown really unsubscribes monitors —
  it previously raised a silently-swallowed `AttributeError` (review #10).
- `geecs_adaptive_scan` runs `propose()` (asset wait + analysis + Xopt) on
  a worker thread, idling with `bps.sleep` — the RunEngine loop stays
  responsive to pause/abort, CA monitors, and TiledWriter between bins
  (review #12).

### Removed

- `BinData.images()` / `BinData.averaged_image()` and the `assets` plumbing —
  redundant with the evaluator path: image/diagnostic analysis (including the
  bin-average-then-analyze pattern) is config-driven through ScanAnalysis
  analyzers, which load natively saved files by scan tag. `BinData` is now
  pure scalar-row access (`rows` / `valid_rows` / `column`).

## [0.18.0] - 2026-07-04

### Added

- **Optimization as a scan** — `GeecsSession.optimize()` +
  `plans/optimize.py::geecs_adaptive_scan`: one scan number, one Tiled run,
  iteration = `bin_number`, the same schema-v1 shot-matched rows and
  acquisition modes as any scan (free-run reference-paced or strict
  single-shot; requirement from Sam — no side-channel optimizer data à la
  Badger). Between bins the objective is evaluated on that iteration's
  `BinData` (rows + native images: `bin.images("cam")`,
  `bin.averaged_image("cam")` for the average-then-analyze ImageAnalysis
  pattern, matched to rows by filename `acq_timestamp` with a wait for
  late-written files) and fed to the suggester (ask/tell protocol:
  dependency-free `RandomSuggester`, `XoptSuggester` adapter behind the new
  `optimize` extra, or any duck-typed generator). A failed objective records
  NaN instead of aborting. The per-iteration history is returned and written
  to `optimization.json` in the scan folder.
- `on_finish` policy on `optimize()`: `"hold"` (scan convention, default),
  `"initial"` (restore pre-optimization values; also applied on
  abort/failure), `"best"` (move to the highest-objective inputs).
- Verified live (laser off, physics-free objective): 6 random-search
  iterations steering U_S1H toward 0.3 A found best I=0.276 A, all data as
  one Tiled run, `on_finish='initial'` restored the magnet.

## [0.17.0] - 2026-07-04

### Removed

- **The direct UDP/TCP device backend is deleted** — the CA backend reached
  verified live parity (Scans 007–015), and per project direction the bespoke
  path dies once the standard path wins. Gone: `GeecsDevice`, `GeecsSettable`,
  `GeecsMotor`, `GeecsGenericDetector`, `GeecsTimestampedReadable`,
  `GeecsSnapshotReadable`, `GeecsTriggerable`, `signals.py`, `backends/`,
  `NonScalarSaveSupport._init_save_signals`, `ShotController.over_udp` /
  `UdpSetter`, and the `GEECS_BLUESKY_DEVICE_BACKEND` selector (setting it to
  anything but `ca` now raises). `BlueskyScanner` and `GeecsSession` are both
  CA-only; the gateway is the one component speaking GEECS wire protocol.
- **The GEECS access-layer core moved to GeecsCAGateway** (`transport/`,
  `db/`, `testing/fake_device_server.py`, `pv_naming.py`, and the wire-level
  exceptions), flipping the package dependency: geecs-bluesky now depends on
  geecs-ca-gateway (library: `GeecsDb`, `pv_naming`, exceptions — re-exported
  from `geecs_bluesky.exceptions` for compatibility; service: the PVs). This
  package is now a pure EPICS/Bluesky consumer.

### Changed

- **`BlueskyScanner` is now the thin GUI adapter over `GeecsSession`** (the
  endgame the deletion unblocked): the session owns the RunEngine, Tiled
  subscription, device factories, saving/asset wiring, ScanInfo, and s-file
  export; the scanner keeps only `exec_config` parsing, role classification,
  thread/progress/lifecycle plumbing, and the per-scan log. `_execute_scan`
  maps the GUI request onto `session.scan()` (with pre-claimed scan numbers so
  the log wraps the run, and legacy-format ScanInfo field fidelity). The
  scanner shrank ~990 → ~666 lines with zero duplicated discipline. Verified
  live post-rewrite: NOSCAN and STANDARD scans through the GUI bridge.
- The hermetic suite runs on ophyd-async mock backends
  (`tests/ca_mock_helpers.py`: `set_mock_value` shots, an RE-loop pacer as the
  free-running trigger, a setpoint→readback follower for motor convergence) —
  no real sockets in device/plan tests, roughly halving suite runtime. The
  plan/schema/domain tests (t0 sync, contributor labeling, strict single-shot
  ownership, arm/disarm ordering, drift immunity) were ported, not deleted.
- `CaAcqTimestampReadable` ignores non-positive `acq_timestamp` monitor values:
  `0.0` is the gateway channel's pre-acquisition placeholder, so "never
  acquired" now reads as `None` on CA exactly as it did on the direct cache
  (and the placeholder→first-frame jump can't fake a shot).
- Live re-verified post-deletion: scanner free-run NOSCAN over the gateway.

## [0.16.0] - 2026-07-03

### Added

- `geecs_bluesky/devices/ca/` — CA-backed ophyd-async devices that consume the
  GeecsCAGateway PVs as a stock EPICS IOC (no GEECS UDP/TCP): `CaReadable`
  (scalar readbacks), `CaSettable` (put to the `…:SP` PV, read the streamed
  readback), and `CaTriggerable` (whose `trigger()` gates on `acq_timestamp`
  advancing via a persistent CA monitor). Verified live against the gateway: one
  Bluesky row per real shot at 1 Hz. Requires the `ca` extra. These are the CA
  counterpart of the direct UDP/TCP devices; shot-id/save-path/schema logic
  stays shared, selected by backend rather than duplicated.
- `geecs_bluesky/pv_naming.py` — the shared GEECS-name → PV naming contract
  (`normalize_component` / `pv_name`), imported by both the CA devices and the
  gateway (which now delegates to it) so the producer and consumer can't drift.
- `CaGenericDetector` — the scanner's triggered detector over CA, composing the
  same `ShotIdSupport` mixin as the direct `GeecsGenericDetector` (same tracker,
  data keys, and NaN/valid semantics; only the `acq_timestamp` source differs).
- **Backend selector**: `GEECS_BLUESKY_DEVICE_BACKEND=direct|ca` (default
  `direct`) chooses the device family at `BlueskyScanner` construction — the one
  seam where backends differ; plans, schema, scan numbering, and Tiled stay
  shared. The CA backend currently supports reference/triggered scalar roles;
  contributor/snapshot roles, `save_nonscalar_data`, and STANDARD-scan motors
  (`CaMotor`) fail loud as not-yet-implemented rather than silently degrading.
- **Backend equivalence verified live**: the same NOSCAN (free-run, laser off,
  no shot control) run on both backends produced identical event counts
  (5 primary + 1 flush) and a verbatim-identical event key set, with matching
  shot_id/offset/valid behavior (Scan007 = CA, Scan008 = direct).
- `CaMotor` — position-feedback motor over the gateway: the `…:SP` put rides
  the blocking GEECS UDP set (native tolerance convergence) with the full
  `move_timeout` as its CA budget, then a readback poll confirms the streamed
  position arrived (belt-and-suspenders for devices whose set-timeout semantics
  are ambiguous). Wired into `_run_standard_scan` for the `ca` backend.
- **STANDARD-scan equivalence verified live**: jet 4→5 mm × 3 shots/step on
  both backends → identical event counts (9 primary + 1 flush), verbatim-
  identical key sets, motor readback in every event, and the same
  shot-id-gap-across-moves semantics (Scan010 = CA, Scan011 = direct).
- **Native file saving on the CA backend**: `CaGenericDetector` now composes
  the shared `NonScalarSaveSupport` mixin (same save-path column and
  Resource/Datum asset documents as the direct detector); only the
  `localsavingpath` / `save` controls differ — CA signals that read the gateway
  readback and write its `:SP` setpoint. The scanner's post-construction saving
  block (save paths, asset definitions, `_saving_detectors`) is now shared
  verbatim between backends. Requires gateway ≥ 0.3.0 (`include_settable` for
  the control-surface PVs, long-string path PVs for >40-char save paths).
  **Verified live (Scan013)**: a CA-backend NOSCAN with `save_nonscalar_data`
  drove the camera's save controls over CA, native PNGs landed in the
  `Y/MM/scans/ScanNNN/<device>/` layout with `device_<acq_timestamp>` names,
  events carried `nonscalar_save_path` + image datum-id columns (Resource/Datum
  asset docs), documents persisted to Tiled, and the legacy
  `ScanDataScanNNN.txt` / `sNN.txt` exports were written back from Tiled — the
  full-output contract in one run.

- **Free-run contributor/snapshot roles on the CA backend.** The
  reference-relative labeling semantics (row shot-id peeking, bounded grace
  wait, offset/valid emission) moved verbatim from `GeecsTimestampedReadable`
  into the shared `FreeRunContributorSupport` mixin
  (`geecs_bluesky/devices/contributor.py`); the direct class and the new
  `CaTimestampedReadable` both compose it, so the two backends cannot diverge.
  `CaSnapshotReadable` covers async devices; `CaTriggerable`'s monitor plumbing
  was factored into `CaAcqTimestampReadable` for the contributor to reuse.
  The scanner's CA branch now dispatches all four roles
  (`_build_ca_detector`). **Verified live (Scan014)**: a three-role free-run
  NOSCAN (reference + contributor + snapshot) with coordinated t0 sync —
  contributor shot_id equaled the reference's on every row (offset 0,
  valid True), snapshot column present, Tiled + s-files written.
- **Strict single-shot verified live on the CA backend (Scan015)**, using the
  HTU-LaserOFF shot-control config: ARMED confirmed quiescent, three
  plan-owned SINGLESHOT fires each captured by `CaTriggerable`'s
  synchronous-baseline trigger (shot spacing ~0.4 s — commanded shots, not
  free-run), finalize returned STANDBY, and the DG645 was restored to
  Internal afterwards via the gateway's own `Trigger_Source:SP` PV.

- **`GeecsSession` — headless scan execution** (`geecs_bluesky/session.py`;
  design note in `Planning/geecs_session/00_overview.md`): the full GUI-scan
  run discipline (scan numbering, ScanInfo, save-path layout, schema v1,
  Tiled, s-file export, shot-control bracketing) from a notebook/script, CA-only
  by design. Verified live: a free-run NOSCAN (reference + contributor +
  snapshot, images saving) and a strict NOSCAN (HTU-LaserOFF) from six lines of
  session code.
- **`ShotController` extracted** (`geecs_bluesky/shot_controller.py`) — the
  arm/disarm/quiesce/single-shot plan stubs left `BlueskyScanner` (closing the
  long-standing "shot-control bracketing not extracted" gap). Two transports:
  `over_udp` (the original path) and `over_ca` — puts to the gateway `:SP` PVs,
  used automatically by the scanner on the `ca` backend and by sessions.
  Verified live driving the DG645 through ARMED/SINGLESHOT/STANDBY over CA.
- Supporting extractions, all delegated to by the scanner so the GUI path is
  unchanged: `tiled_integration.py` (TiledWriter subscription + descriptor
  patch), `data_paths.py` (local ↔ device-server path mapping, asset roots),
  `scanner_configs.py` (configs-repo resolution + validated shot-control YAML
  loading; the hardware test now uses it instead of its own copy).
- **One orchestration recipe** — the scan composition (mode dispatch → run
  wrapper → finalize disarm) extracted to
  `plans/orchestration.py::build_step_scan_plan` and called by both
  `GeecsSession.scan()` and `BlueskyScanner._run_step_scan`; the scanner's
  duplicate recipe and its per-state plan stubs were deleted. Both front
  doors re-verified live on the shared recipe (scanner ca-backend free-run;
  session free-run with images and strict single-shot).

### Notes

- `CaTriggerable` closes the strict single-shot race the same way
  `GeecsTriggerable` does: a persistent monitor on `acq_timestamp` feeds a local
  cache/queue, and `trigger()` drains stale updates and captures the baseline
  **synchronously before returning** — so a shot fired immediately after
  `bps.trigger` (trigger → fire → wait) cannot land in a blind window and be
  missed. Pinned by a mock race test (shot fired with zero awaits after
  `trigger()`).

## [0.15.0] - 2026-07-03

### Added

- Optional `ca` extra (`aioca`) for the forthcoming CA-backed device family
  (`geecs_bluesky/devices/ca/`), which consumes the GeecsCAGateway PVs like any
  EPICS IOC. `aioca` bundles libca via `epicscorelibs`, so no system EPICS base
  is required. The direct UDP/TCP backend does not need it.

### Changed

- Bumped the `ophyd-async` floor from `>=0.16` to `>=0.19.3` to track the current
  API (`init_devices`, `ophyd_async.epics.core`, `observe_value`) and stay
  consistent with the GeecsCAGateway environment. The existing device/backend
  code required no changes; the full hermetic suite passes on 0.19.3.
- `pytest` now defaults to the hermetic FakeGeecsServer unit tests under `tests/`
  only (`testpaths`), with hardware/integration markers deselected, so a fresh
  checkout is green with no lab network or live-device access. The top-level
  hardware scripts (`test_bluesky_scanner.py`, `test_hardware.py`) are run
  explicitly.
- The hardware integration test now loads its shot-control config from the
  configs repo (the production path) via a `GEECS_BLUESKY_LASER=on|off` toggle
  (default `off` → internal single-shot `HTU-LaserOFF`; `on` → external-timing
  `HTU-Normal`), validated against `ShotControlConfig`. This replaces a hardcoded
  inline config that had drifted (it was missing the `Amplitude.Ch AB` gating)
  and prevents laser-off runs from stranding the DG645 in an external mode.

## [0.14.0] - 2026-06-30

### Added

- Added a post-run analysis contract for Bluesky camera runs, including
  sidecar metadata/features writers, ImageAnalysis analyzer adapters, optional
  derived analysis-run documents, and tests for event-scope and scan-scope
  analysis execution.
- Added a local handler for native text-array external asset specs, plus generic
  Tiled readback helpers for registered single-asset/event-field assets. TDMS
  event assets remain file-backed until analysis supplies the required 1D
  loader configuration.
- Added `load_asset_from_tiled(...)` as the canonical date/scan raw-readback
  helper for registered external assets; camera-specific readback helpers remain
  compatibility wrappers.
- Added generic Tiled asset-analysis helpers that run analyzers over registered
  non-camera asset fields and load provenance-aware 1D assets, such as
  `tdms_scope`, from registry defaults plus optional analyzer overrides.
- Asset registry entries now describe payload shape, provenance-aware loader
  names, loader config defaults, and whether analysis-time loader configuration
  or SDK capabilities are required.
- Synthetic local-fill Resource/Datum/Event streams now use an
  `ExternalAssetDocumentSpec` request model and explicit
  `geecs_external_asset_document_schema` marker.
- Added `tiled_camera_analysis_sidecar.ipynb` to exercise local Tiled camera
  asset fill, BeamAnalyzer execution, sidecar writing, and optional analysis
  run publication.

### Changed

- Analysis config resolution now uses the unified scan-analysis config root
  instead of falling back to legacy image-analysis config paths.
- Tiled raw-run lookup now ignores derived analysis runs so analysis records do
  not collide with acquisition runs that share the same date and scan number.

### Documentation

- Added planning notes for sidecar-first analysis results and linked them from
  the external-assets roadmap.

## [0.13.6] - 2026-07-02

### Added

- `GeecsDb.get_device_variables` now also returns `tolerance` (numeric, or
  `None`) — useful as a monitor deadband.

## [0.13.5] - 2026-07-02

### Added

- `GeecsDb.get_subscribed_variables(experiment)` — returns `{device: [var, ...]}`
  for `get='yes'` variables in `expt_device_variable` (the per-shot monitoring
  subset), in one query. Useful for down-selecting a sensible variable set.

## [0.13.4] - 2026-07-02

### Added

- `GeecsDb.get_device_variables` now also returns `variabletype` (`numeric`,
  `choice`, `string`, `path`, `image`, `1darray`, …) and `choices` (the
  comma-separated option string from the `choice` table for `choice` variables),
  so callers can map GEECS types onto typed PVs. Numeric `min`/`max` parsing is
  now tolerant of non-numeric strings.

## [0.13.3] - 2026-07-01

### Added

- `GeecsDb.list_devices(experiment, enabled_only=True)` — optionally filter to
  devices whose `expt_device.enabled` is `"yes"` (a device may belong to an
  experiment but be disabled). Default `False` preserves existing behavior.

## [0.13.2] - 2026-06-26

### Changed

- External asset Resource documents now use the configured device-server data
  root as their canonical `root` when available, with POSIX `resource_path`
  values below that root, instead of always using the scan folder as root.
- Resource path construction now normalizes Windows and POSIX separators before
  computing relative paths.

### Documentation

- Updated the external-assets roadmap to describe canonical Resource writing,
  reader-side root mapping, and the pre-production/test status of current Tiled
  data.

## [0.13.1] - 2026-06-25

### Fixed

- Tiled-backed local camera readback now maps Windows/device-server data roots
  such as `Z:/data` to local data mounts such as `/Volumes/hdna2/data` before
  constructing Resource/Datum documents, avoiding OS-dependent
  `Path.relative_to` failures.

### Documentation

- Added the external-assets roadmap/status document with the current
  acquisition, local-fill, root-mapping, and post-run-analysis next steps.

## [0.13.0] - 2026-06-24

### Changed

- `strict_shot_control` now requires a reachable shot-control device with a
  non-empty `ARMED` state and aborts configuration when that requirement is not
  met, instead of falling back to free-running `trigger_and_read`.
- Unknown acquisition-mode values now raise a configuration error instead of
  silently falling back to strict mode.
- The standalone hardware smoke harness now runs no-shot-control scenarios in
  explicit `free_run_time_sync` mode and uses true ARMED strict mode for
  shot-control/full-output checks.

### Added

- Added Tiled-backed local camera asset readback helpers. Archived Bluesky runs
  can now be found by GEECS scan identity, a shot can be selected by
  `scan_event_index`, and the event's device `acq_timestamp` is used with the
  asset registry to fill the native camera PNG through local handlers.
- Added `tiled_external_asset_readback.ipynb`, a thin notebook for querying a
  Tiled run by date, scan number, device, and shot, then loading the camera
  image locally.

### Fixed

- Missing-shot Tiled readback errors now report the available
  `scan_event_index` values, and the notebook prints lookup failures without a
  traceback.

## [0.12.2] - 2026-06-24

### Changed

- Split GeecsBluesky pytest selection into pure unit tests and socket-based
  `FakeGeecsServer` TCP/UDP integration tests via a dedicated `fake_server`
  marker, so unit-test CI can avoid opening localhost sockets.

### Fixed

- Hardened fake-server tests and socket teardown with bounded per-test timeouts,
  explicit background server shutdown, TCP subscriber cleanup, and retry logic
  for local UDP/TCP port collisions.

## [0.12.1] - 2026-06-23

### Added

- Local external asset readback helpers for registering GEECS handlers with
  `event_model.Filler` and filling ordered Bluesky document streams.
- Camera shot document helpers for building fillable Resource/Datum docs from
  existing legacy scan folders by date, scan number, device, and shot number.
- `external_asset_readback.ipynb` to demonstrate local camera asset filling,
  including a parameterized existing-scan lookup and a no-hardware synthetic
  PNG smoke test.

### Fixed

- `GeecsCameraImageHandler` now accepts Resource document metadata such as
  `data_key`, matching how `event_model.Filler` instantiates handlers from
  GEECS Resource documents.

## [0.12.0] - 2026-06-23

### Added

- Native-file-saving sync devices now emit Bluesky external asset references
  when their database device type is registered in `geecs_bluesky.assets`.
  Acquisition still records the existing `nonscalar_save_path` string column;
  registered assets add datum-id event fields plus matching Resource/Datum docs.
- `NonScalarSaveSupport.collect_asset_docs()` queues one Resource/Datum pair per
  native file and records `.tdms_index` companion paths for TDMS assets.
- The standalone `test_bluesky_scanner.py` hardware script now preflights the
  required lab devices and reports unreachable hardware before running
  scenarios. Its camera device can be overridden with
  `GEECS_BLUESKY_TEST_CAMERA`.

### Fixed

- Tiled persistence failures no longer abort scans. GEECS native-file asset
  datum IDs are stored as ordinary Tiled event metadata until the Tiled server
  has readers for the custom GEECS asset specs.
- Native-save device commands now translate scanner-local save folders to
  `geecs_device_server_data_base_path` from the user config before writing
  `localsavingpath`, so tests run from macOS/Linux can still command
  Windows-visible device paths such as `Z:\data`.
- External asset paths now use the direct native device filename
  (`Device_<acq_timestamp>.<ext>`) rather than the legacy post-move renamed
  filename.

## [0.11.0] - 2026-06-23

### Added

- Expanded `geecs_bluesky.assets` registry coverage for native multi-file save
  devices: `FROG`, `PicoscopeV2`, `Thorlabs CCS175 Spectrometer`,
  `RohdeSchwarz_RTA4000`, `ThorlabsWFS`, `MagSpecCamera`, and
  `MagSpecStitcher`.
- Added asset specs for TDMS primary files and text-array variant files. TDMS
  assets record `.tdms_index` as a companion extension while treating the
  `.tdms` file as the primary resource.
- Added registry path builders for FROG `-Spatial` / `-Temporal` image
  directories and MagSpec `-interp`, `-interpSpec`, and `-interpDiv` variant
  directories.

## [0.10.0] - 2026-06-23

### Added

- **External asset foundation.** Added `geecs_bluesky.assets` with a
  device-type registry, `GEECS_CAMERA_IMAGE` spec, `Point Grey Camera` native PNG
  path construction, and `GeecsCameraImageHandler` backed by
  `geecs_data_utils.io.images.read_imaq_image`. This is the first step toward
  emitting formal Bluesky external asset docs for native GEECS camera files.
- `GeecsDb.get_device_type(device_name)` to query the database
  `device.devicetype` value without depending on `GEECS-PythonAPI`.
- Real-database integration coverage for the `UC_TopView` device type so
  database string mismatches are caught when tests run with lab DB access.

## [0.9.0] - 2026-06-15

### Added

- **Legacy GEECS scalar files for Bluesky scans.** A scan now writes the
  on-disk files downstream GEECS analysis still consumes:
  - `ScanInfoScanNNN.ini` is written into the claimed `scans/ScanNNN/` folder at
    scan start, replicating the legacy `[Scan Info]` format
    (`BlueskyScanner._write_scan_info_ini`).
  - `ScanDataScanNNN.txt` and the mutable `analysis/sNNN.txt` are written at
    scan end by reading the run back from Tiled via the new
    `geecs_data_utils.write_scalar_files_from_tiled` exporter
    (`BlueskyScanner._export_scalar_files`, best-effort: failures are logged,
    never fatal).
- **`geecs_scalar_headers` start-doc metadata** — `geecs_run_wrapper` now
  collects each device's `_column_headers` (event data key → legacy
  `Device Variable`) and injects them so the exporter can recover legacy headers
  despite `safe_name()` mangling being irreversible.  Documented in
  `EVENT_SCHEMA.md`.
- **`build_signal_attrs`** (`utils.py`) — centralises the device signal
  attr-naming/disambiguation loop so signal creation and the header map cannot
  drift; adopted by the generic-detector and snapshot device classes.

### Changed

- **`geecs_data_utils` is now a declared path dependency** (`../GEECS-Data-Utils`,
  `develop = true`) rather than a manual install.  It supplies scan numbering
  (`claim_scan_number`), the Tiled→s-file exporter, and `pandas` / `nptdms`
  transitively — so the previously declared (and unused) `pandas` and `nptdms`
  pins are removed.  This also resolves the pandas version skew that surfaced
  when both packages were installed side by side.

## [0.8.2] - 2026-06-16

### Fixed

- TCP subscriptions now warn and continue when a subscribed variable is absent
  from a push frame instead of letting the listener fail.

## [0.8.1] - 2026-06-16

### Removed

- Removed the unused `GeecsCameraBase` device wrapper and its camera-specific
  tests. Scanner-created detectors now use `GeecsGenericDetector`,
  `GeecsTimestampedReadable`, or `GeecsSnapshotReadable`.

### Changed

- Updated step-scan examples and detector tests to use the active generic
  detector path.

## [0.8.0] - 2026-06-14

### Added

- **`geecs_run_wrapper`** (`plans/run_wrapper.py`) — reusable run bookkeeping
  shared by the scanner and notebook workflows: injects the scan-number
  metadata (`scan_number`, `scan_folder`, `experiment`, and **`scan_id` set to
  the GEECS scan number**) into the run's start document and brackets the plan
  with per-detector native file saving (save on before, off in a finalize that
  runs even on abort).  `claim_scan_number(experiment)` is the shared
  scanner-side claim.  `BlueskyScanner` now dogfoods both — its inline
  `_scan_with_saving` / metadata assembly are removed in favour of the wrapper.
- **`EVENT_SCHEMA.md`** — the canonical in-package event-schema v1 contract
  (start-doc metadata + per-device companion columns), graduated from
  `Planning/acquisition_modes/01_event_schema_contract.md`.

### Changed

- Bluesky `scan_id` is now set to the claimed GEECS day-scoped scan number
  (via the run wrapper) instead of the RunEngine's internal counter.

## [0.7.0] - 2026-06-13

### Added

- **True plan-owned single-shot for strict mode (fire-and-wait).** When the
  shot-control config defines an `ARMED` state, strict STANDARD/statistics
  scans now: arm the controller into single-shot mode at data-taking output
  (`ARMED` — e.g. gas jet on + `Trigger.Source` → single-shot, halting the
  free-run), confirm the trigger has stopped, then fire one shot per row and
  await every device (`geecs_single_shot`).  A device that misses the plan's
  shot is a hard, attributable failure.
  - `geecs_confirm_quiescent` (`plans/single_shot.py`) — the inverse of
    `trigger()`: waits until no sync device's `acq_timestamp` advances for a
    quiet window, raising `GeecsQuiescenceTimeoutError` if the trigger never
    stops.  This is the "watch acq_timestamp go quiet" confirmation.
  - `geecs_step_scan` gains a `setup_trigger` hook (run once at scan start)
    and records `fires_own_shots` in run metadata.
  - `ShotControlState.ARMED` added; `BlueskyScanner` dispatches strict to
    single-shot when `ARMED` is defined, else falls back to the free-running
    `trigger_and_read` contract (logged).
  - **Requires a config addition** to use: add an `ARMED` state to the
    shot-control YAML (see `Planning/acquisition_modes/03_strict_shot_control.md`).
    Configs without `ARMED` keep the free-running fallback unchanged.

## [0.6.0] - 2026-06-13

### Changed

- **Free-run t0 sync now quiesces with a dedicated `quiesce_trigger`** that
  *stops* the free-running trigger (DG645 `OFF` / single-shot source) before
  reading per-device t0 timestamps — the legacy "disable the trigger, then
  read `acq_timestamp`" procedure.  The previous disarm-to-`STANDBY` left the
  trigger free-running on real hardware (STANDBY only drops the gas-jet
  amplitude), so the t0 read could race advancing timestamps.  `BlueskyScanner`
  passes `_quiesce_trigger` (OFF) for free-run scans; falls back to
  `disarm_trigger` when no dedicated quiesce is supplied.
- **NOSCAN unified into the step-scan path.** `motor` is now optional in
  `geecs_step_scan` / `geecs_free_run_step_scan` (a `None` position is a bin
  with no move).  Statistics collection (formerly NOSCAN) is just a motorless
  scan with one no-move bin, routed through the same plan — so it works
  identically in **both** acquisition modes, including free-run with t0 sync
  and tail flush. The separate `_run_noscan` inline plan is gone;
  `BlueskyScanner` shares one `_run_step_scan` body for both modes.

### Added

- **`ShotControlConfig` model** (`models/shot_control.py`) — a Pydantic v2
  model for the shot-controller (DG645) YAML, replacing the bare untyped
  `{device, variables: {var: {state: value}}}` dict that was passed around.
  `from_information()` coerces the legacy dict (or `None`); `values_for_state()`
  returns just the non-empty writes for a state (empty = no-op);
  `defines_state()` reports whether a state does anything.  Pure data — no
  hardware or GEECS-engine imports — so it is reusable without dragging in the
  legacy `TriggerController`.  `ShotControlState` enumerates `OFF` / `SCAN` /
  `STANDBY` / `SINGLESHOT`.
- `BlueskyScanner` now validates `shot_control_information` into a
  `ShotControlConfig` on construction and drives trigger states through
  `values_for_state()` instead of digging the raw dict.

## [0.5.0] - 2026-06-12

### Added

- **Acquisition-mode dispatch in `BlueskyScanner`** — `reinitialize` resolves
  `acquisition_mode` from `options.acquisition_mode`, overridable by the
  `GEECS_BLUESKY_ACQUISITION_MODE` env var, defaulting to
  `strict_shot_control`.  STANDARD scans dispatch to `geecs_free_run_step_scan`
  vs `geecs_step_scan` accordingly.
- **Automatic reference selection** — `_classify_device_roles` assigns the
  first synchronous device as the free-run reference (built as a
  `GeecsGenericDetector` pacemaker) and later synchronous devices as
  `GeecsTimestampedReadable` contributors anchored to it; asynchronous devices
  stay snapshots.  No YAML field; the choice is recorded in run metadata.

### Changed

- **Free-run plan disarms the shot control before t0 sync** so every device's
  cache holds a settled frame from the same last physical shot (matching the
  legacy "disable trigger, then read `acq_timestamp`" procedure).  No-op when
  there is no shot control.

### Known gaps

- Strict plan-owned single-shot needs an `ARMED` state in the shot-control
  YAML; the experiment configs gained one on the `geecs-plugins-configs`
  branch `add-bluesky-armed-shot-control` (pending merge).  Until that merges,
  configs on `main` lack `ARMED` and strict uses the free-running
  `trigger_and_read` fallback.  See
  `Planning/acquisition_modes/03_strict_shot_control.md`.
- General per-scan setup/teardown of arbitrary device variables (the clean
  replacement for the amplitude-as-gas-jet-switch hack) is deferred future
  work, not part of this branch.

## [0.4.0] - 2026-06-12

### Added

- **`ShotIdTracker`** (`devices/shot_id.py`) — incremental per-device shot-ID
  derivation from `acq_timestamp` history.  IDs advance by
  `round(Δt × rep_rate)` per event, so rep-rate mismatch never accumulates
  (the absolute `(ts − t0) × rep_rate` method misquantizes after ~30 min at
  1 Hz with a 0.05% rate error).  Repeated timestamps (device timeouts) are
  idempotent; cross-device matching is shot-ID equality.
- **Coordinated t0 sync plan stage** (`plans/t0_sync.py`) —
  `geecs_t0_sync(devices)` seeds every sync device's tracker from one
  physical trigger: with the shot control disarmed, cached `acq_timestamp`
  values within the acceptance window (default 0.2 s) are the same shot.
  Retries while frames propagate; raises `GeecsT0SyncError` rather than ever
  proceeding unseeded.
- **Sync-device companion columns** — `GeecsGenericDetector` now emits
  `<dev>-shot_id`, `<dev>-shot_offset`, and `<dev>-valid` alongside
  `<dev>-acq_timestamp` on every read (event schema contract v1 — see
  `Planning/acquisition_modes/01_event_schema_contract.md`).  Keys are
  stable: unavailable values are NaN / `False`, never omitted.
- **`GeecsTimestampedReadable`** (`devices/timestamped_readable.py`) — the
  free-run sync contributor: snapshot-style read (no blocking `trigger()`)
  that labels its latest data with reference-relative `shot_offset` /
  `valid`, computed by peeking the pacemaker device's cached shot.  A
  bounded grace wait (default 0.3 s, ~one TCP push period) lets a late
  frame for the row's shot arrive; lagging devices emit real data at
  negative offsets for downstream realignment by `shot_id`.
- **`ShotIdSupport` mixin** — shared shot-ID configuration, t0 seeding, and
  companion-column emission used by both `GeecsGenericDetector` and
  `GeecsTimestampedReadable`.  Devices opt into the `acq_timestamp` TCP
  subscription via a `GeecsDevice._subscribe_acq_timestamp` class flag
  (replaces the `isinstance(GeecsTriggerable)` gate).
- **`geecs_free_run_step_scan`** (`plans/free_run_step_scan.py`) — the
  free-run time-sync plan: t0-sync stage before the run opens (captured
  `device_t0s` land in the start document), the same move/arm/shots/disarm
  bracketing as the strict plan with only the reference Triggerable,
  contributor auto-anchoring to the reference, and a tail-flush event on a
  separate `flush` stream after the final disarm so lagging contributors'
  final shot is recorded.  `geecs_step_scan` start metadata now carries
  `acquisition_mode="strict_shot_control"` and `geecs_event_schema: 1`.

- **`geecs_single_shot`** (`plans/single_shot.py`) — the strict-shot-control
  primitive: arm detector waiters → fire (DG645 `SINGLESHOT` state) → await
  every detector → one complete event row.  `geecs_step_scan` gains a
  `fire_shot` plan-stub parameter; when provided the plan owns every shot,
  and a device missing the plan's own shot is a hard, attributable failure.
  Without it, behaviour is unchanged (free-running trigger, internal-trigger
  test mode).  `GeecsTriggerable.trigger()` now drains stale frames and
  baselines `acq_timestamp` synchronously at call time, so a shot fired
  immediately after `bps.trigger` can never be missed.

### Fixed

- **Reference adoption** — storing the pacemaker on a contributor tripped
  ophyd-async's `Device.__setattr__` child-adoption (re-parent + rename),
  after which bluesky's `separate_devices` silently dropped the reference
  from `trigger_and_read`.  `set_reference` now holds the pacemaker via
  `ophyd_async.core.Reference` (the sanctioned opt-out for peer devices);
  a regression test pins that the reference stays an unparented peer.

### Changed

- **`configure_shot_numbering()` → `configure_shot_id()`**, and the derived
  `<dev>-shotnumber` column (dtype integer, absolute derivation) is replaced
  by `<dev>-shot_id` (dtype number, incremental derivation).  Shot IDs are
  matching machinery and diagnostics, not a file-join key — files still join
  to events by device `acq_timestamp`.

## [0.3.6] - 2026-06-09

### Fixed

- **Synchronous save devices with empty variable lists** — `BlueskyScanner`
  now mirrors the legacy scanner by adding `acq_timestamp` for synchronous
  save devices before deciding whether the device has variables to read.  This
  lets non-scalar cameras save files even when no scalar variables are selected
  in the save-element editor.

## [0.3.5] - 2026-06-09

### Added

- **Plan-owned scan context** — step-scan and NOSCAN events now include
  `bin_number`, `shot_index_in_bin`, and `scan_event_index` fields emitted by
  the Bluesky plan at acquisition time.
- **Asynchronous snapshot readables** — save devices with `synchronous: false`
  are now read as snapshots in each shot event instead of being treated as
  triggerable detectors.  They do not require `acq_timestamp` and do not emit
  derived device shotnumbers.

### Tests

- Added step-scan fake-server coverage for scan-context columns and snapshot
  readbacks recorded in the same events as triggered detector data.

## [0.3.4] - 2026-06-09

### Added

- **Physical shotnumber metadata** — `GeecsGenericDetector` can now derive a
  device-prefixed integer `shotnumber` from the detector's own
  `acq_timestamp`, the first scan-read `t0_acq_timestamp`, and the configured
  scan repetition rate.  This lets missed device triggers appear as shotnumber
  jumps instead of being hidden by the Bluesky event counter.

### Tests

- Added fake-server coverage showing that a two-period `acq_timestamp` jump
  produces a `shotnumber` jump from 1 to 3 across two detector events.

## [0.3.3] - 2026-06-09

### Changed

- **GUI lifecycle events** — `BlueskyScanner` now accepts the GUI `on_event`
  callback, exposes `current_state`, and emits lifecycle transitions for
  initializing, running, completed, and aborted scans.  This lets the Scanner GUI
  re-enable its controls when a Bluesky-backed scan finishes.

## [0.3.2] - 2026-06-09

### Fixed

- **Windows MySQL connector crash** — GEECS database lookups now force
  `mysql-connector-python` to use its pure-Python implementation
  (`use_pure=True`), matching the legacy API layer.  The connector's C extension
  has crashed silently on lab Windows machines with 9.x.

### Tests

- Added a DB lookup regression test that verifies `use_pure=True` is passed to
  `mysql.connector.connect()`.

## [0.3.1] - 2026-06-08

### Added

- **Non-scalar save-path event metadata** — `GeecsGenericDetector` now emits
  derived per-event fields for non-scalar devices: device `acq_timestamp` and
  the configured save directory.  `BlueskyScanner` configures these fields when
  it assigns `localsavingpath` for `save_nonscalar_data=True` detectors, and
  includes the per-device save paths in run-start metadata.  File names remain
  hardware-native; downstream readers should join files to events by device
  `acq_timestamp`, not by a synthetic shot counter.

### Tests

- Added offline `FakeGeecsServer` coverage for non-scalar save-path metadata.

## [0.3.0] - 2026-05-08

### Added

- **DG645 shot control — per-step arm/disarm** — `BlueskyScanner` accepts an
  optional `shot_control_information` dict (matching the GEECS Scanner GUI timing
  YAML format).  The DG645 is armed to `SCAN` state after each motor move and
  disarmed to `STANDBY` after shots are collected, keeping the trigger off during
  motion.  A `bpp.finalize_wrapper` guarantees disarm even on mid-step abort.
- **`_UdpSetter`** — minimal Bluesky `Movable` wrapping a single GEECS UDP
  variable as a string-typed settable.  Used internally for shot control; avoids
  ophyd device overhead and works for both numeric delays and string state words.
- **`geecs_step_scan` arm/disarm parameters** — `arm_trigger` and `disarm_trigger`
  optional callables added to `geecs_step_scan`.  Each is a plan generator called
  after the motor move (arm) and after shots are collected (disarm) per step.
- **`BlueskyScanner._build_shot_controller()`** — resolves the shot control device
  from the GEECS MySQL DB and creates one `_UdpSetter` per configured variable.
- **`_set_trigger_state(state)`** — Bluesky plan stub that drives all shot control
  variables to a named state (`SCAN`, `STANDBY`, `OFF`, `SINGLESHOT`).  Empty-string
  values in the YAML are skipped (matching legacy `TriggerController` behaviour).
  Uses `bps.abs_set` + `bps.wait` rather than `bps.mv` to avoid the `.parent`
  attribute requirement of `bps.mv`.
- **`tests/test_shot_control.py`** — 10 unit tests covering `_UdpSetter`,
  `_set_trigger_state`, and arm/disarm ordering in `geecs_step_scan`.  All run
  against `FakeGeecsServer` — no hardware required.
- **`test_bluesky_scanner.py`** — hardware integration test with three scenarios:
  NOSCAN (UC_TopView), STANDARD step scan (U_ESP_JetXYZ 4→5 mm), and NOSCAN with
  DG645 shot control.  Verifies event counts, motor readback, `acq_timestamp`
  presence, and post-scan DG645 `Trigger.Source` state.  All 6 checks pass on
  real lab hardware.
- **`mysql-connector-python`** added as a direct Poetry dependency (previously
  required manual installation).

### Changed

- **`BlueskyScanner.reinitialize(exec_config)`** — now accepts a duck-typed
  `ScanExecutionConfig` object (or any `SimpleNamespace` with `.scan_config`,
  `.options`, `.save_config` attributes).  `shots_per_step` is derived from
  `rep_rate_hz × wait_time` (rounded, minimum 1) since `ScanOptions` has no
  explicit shots field.  Replaces the previous `config_dictionary` dict handoff.
- **`BlueskyScanner.start_scan_thread()`** — takes no arguments; uses the config
  stored by `reinitialize()`.

## [0.2.0] - 2026-05-07

### Added

- **TiledWriter integration** — `BlueskyScanner.__init__` now accepts optional
  `tiled_uri` and `tiled_api_key` parameters.  When `tiled_uri` is provided, a
  `bluesky.callbacks.tiled_writer.TiledWriter` is subscribed to the RunEngine
  so every scan is persisted to the Tiled catalog automatically.  Gracefully
  skips (logs a warning) if `tiled[client]` is not installed or the server is
  unreachable, so the scanner remains functional without Tiled.
- `tiled[client]` added as an optional Poetry dependency
  (`poetry install -E tiled` to enable).

## [0.1.0] - 2026-04-21

### Added

- Initial release: BlueskyScanner bridge, GeecsMotor, GeecsSettable,
  GeecsGenericDetector, GeecsTriggerable, TCP-backed signal cache, scan
  numbering, per-device image saving, STANDARD and NOSCAN scan modes.
