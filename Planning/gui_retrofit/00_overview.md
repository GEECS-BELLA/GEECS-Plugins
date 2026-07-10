# M4 — Retrofit GEECS-Scanner-GUI to submit a `ScanRequest`

This is the actionable plan for **milestone M4** of the vision rebuild:
teach the *existing* GEECS-Scanner-GUI to build and submit a
`geecs_schemas.ScanRequest`, running its scan through the schema front door
(`GeecsSession.run` / the bridge's `reinitialize(ScanRequest)`) instead of
the legacy `ScanExecutionConfig` (`exec_config`) path.

It updates the July-2026 stewardship audit
(`Planning/gui_stewardship/00_overview.md`) to the current post-M3c state
and turns its "durable jobs" framing into a step list. Read that audit
first; this note does not repeat it. Every claim about current behavior
below is cited to code.

---

## 1. Framing — the decision, and where we already are

**The approved M4 shape (not up for re-litigation here):**

- **Retrofit, not green-field.** The existing PyQt GUI keeps its editors,
  event stream, dialogs, and optimization setup — the "durable jobs" the
  stewardship audit identified (§3 of that doc). What changes is *what the
  GUI hands to the backend*: a `ScanRequest`, not a `ScanExecutionConfig`.
- **Front-ends-as-clients.** All scan logic lives in the engine
  (GeecsBluesky). The GUI's durable job narrows to (a) assembling a
  `ScanRequest` from form state and (b) editing the schema configs (config
  editing is **M5**, not M4).
- **Add, don't delete.** M4 adds the new submission path *alongside* the
  legacy `exec_config` path. Both coexist behind a flag. Nothing is deleted.
  Deletion/cutover is **M6**.

**What already landed (the M4 head start).** Since the stewardship audit,
M3b/M3c moved the whole schema execution surface into the engine:

- The bridge already accepts a `ScanRequest`.
  `BlueskyScanner.reinitialize()` duck-detects `isinstance(exec_config,
  ScanRequest)` and routes to `_reinitialize_from_scan_request`
  (`GeecsBluesky/geecs_bluesky/scanner_bridge/bluesky_scanner.py:289,352`).
  It resolves names fail-fast (save set → `devices_config`, trigger profile
  → `ShotControlWrites`, axis → device/variable/kind), synthesizes a
  legacy-shaped scan-config namespace, and runs the *same* scan-thread body
  as the exec_config path (pinned by a noscan parity test).
- **But the bridge still refuses the interesting cases.**
  `_reinitialize_from_scan_request` raises `NotImplementedError` on: any
  action bindings (`raise_if_actions_present`), multi-axis grids
  (`len(request.axes) > 1` → `MULTI_AXIS_MESSAGE`), save-set entry rituals
  (`resolve_save_set_checked`), and optimize mode
  (`bluesky_scanner.py:409-421,430-436`). Each refusal points the caller at
  `GeecsSession.run`.
- **The engine already honors all of it.** `GeecsSession.run` /
  `run_scan_request`
  (`GeecsBluesky/geecs_bluesky/scan_request_runner.py:1333`) executes the
  full schema surface: request-level + entry-ritual + defaults actions
  (assembled in `assemble_action_slots`, compiled, signals prefetched
  pre-claim), multi-axis outer-product grids, multi-device trigger profiles,
  db_scalars, and background telemetry. Remaining engine-side v1 gaps:
  pseudo scan variables, `all_scalars` without a DB policy, and optimize
  without an injected `objective`/`suggester`.
- **The event-stream seam is done and backend-agnostic.** `RunControl`
  passes the same `on_event` to both backends
  (`GEECS-Scanner-GUI/geecs_scanner/app/run_control.py:72,88`);
  `BlueskyScanner` emits lifecycle + step/progress + pre-flight dialog
  events. **M4 does not touch this seam** — it already works for both
  backends and for both submission shapes.

So the gap M4 closes is narrow and well-defined: **the GUI does not build a
`ScanRequest`, and the bridge refuses the request features the engine
already runs.** M4 is mostly plumbing two ends toward a seam that already
exists in the middle.

---

## 2. Editor → schema mapping

Where each GUI form area's state must land in the schema. Grounded in the
current build path
(`geecs_scanner.py::_collect_ui_scan_config` /`_build_exec_config`, lines
1866-1994) and the schema definitions in `GEECS-Schemas/geecs_schemas/`.

| GUI form area (today) | Produces today | M4 target schema | Notes / gaps |
|---|---|---|---|
| Selected save-elements list (`selectedDevices`, one YAML each, merged by `combine_elements`) | `run_config["Devices"]` → `SaveDeviceConfig.Devices` | `save_set: str` (name) → `SaveSet`/`SaveSetEntry` | **Naming gap.** A `ScanRequest` references *one named* `save_set`; the GUI merges *several* elements ad hoc. See §3 open question — either persist a synthetic save set or add a multi-name resolver. Per-device `synchronous`/`save_nonscalar_data` map to `SaveSetEntry` role/`images` (derived, not hand-set). |
| Scan-variable UI: nickname + start/stop/step (`scanRadioButton`, `read_device_tag_from_nickname`, `scan_start/stop/step_size`) | `ScanConfig(device_var, start, end, step, scan_mode=STANDARD)` | `axes: [ScanAxis(variable, PositionRange)]`, `mode=step` | Single axis today (1-D). Nickname → `ScanAxis.variable` (the catalog friendly name), start/stop/step → `PositionRange`. Grid (multi-axis) and explicit `PositionList` have **no GUI concept yet** — schema supports both. |
| Composite / pseudo variables (composite editor) | composite `device_var` numexpr | `PseudoScanVariable` in the `ScanVariables` catalog | **Engine gap:** pseudo execution raises `NotImplementedError` both in the bridge and `run_scan_request` (`resolve_movable_target`, runner:1152). Out of scope for M4 submission; keep on the legacy path until the pseudo-positioner is built. |
| Shot-control selection (shot-control YAML → `shot_control_information`) | `RunControl` loads YAML → `ShotControlConfig` | `trigger_profile: str` (+ `trigger_variant`) → `TriggerProfile` | Today the GUI picks a shot-control *file* at RunControl init, not per-scan. `ScanRequest` names a profile *per request*. Variant (`laser_off`) has no GUI control today. |
| Setup / closeout actions (per-element `setup_action`/`closeout_action` blocks; action library) | `run_config["setup_action"]/["closeout_action"]` (inline steps) | `actions: ActionBindings{setup, per_step, closeout}` (plan **names**) + action library | Legacy inlines step lists; schema references *named* `ActionPlan`s. The **converter** already extracts legacy element actions into named plans (`convert_save_element`), so a converted save set carries its rituals. `per_step` is new — no GUI concept. |
| Scan mode radios (No-Scan / 1D / Optimization / Background) | `ScanMode.{NOSCAN,STANDARD,OPTIMIZATION}` + `background` flag | `ScanRequest.mode ∈ {noscan, step, optimize}` + `background: bool` | Background = `noscan` + `background=True` (schema docstring confirms this was never a distinct mode). Clean 1:1. |
| Shots/step & num-shots spinners | `wait_time = (shots + 0.5) / rep_rate` (derivation) | `shots_per_step: int` (declared intent) | Schema declares shots directly; drops the rep-rate×wait derivation (schema docstring). Cleaner — GUI passes the integer straight through. |
| Optimization config combo (`comboOptimizationConfig` → `optimizer_config_path`) | `scan_config.optimizer_config_path` + injected `optimization_loader` | `ScanRequest.optimization: OptimizationSpec` | See §3(c). The GUI's Xopt/evaluator stack can't be described *fully* by `OptimizationSpec` fields alone today — recommend keeping loader injection. |
| Rep rate / time-sync / TDMS / beeps (`ScanOptions`) | `ScanOptions` | *(no direct home)* | `ScanRequest` has no rep-rate/TDMS/beeps fields. Rep rate is a session concern (`session.rep_rate_hz`); the rest are legacy-engine knobs. Flag: these stay out of the request (see §5). |
| Acquisition mode (strict vs free-run) | **env var only** (`GEECS_BLUESKY_ACQUISITION_MODE`); no GUI widget (grep confirms none in `geecs_scanner/`) | `ScanRequest.acquisition: AcquisitionMode` | **DECIDED: add a GUI strict/free-run toggle** (the schema field exists; the GUI just never had a widget). Small; lands with the ScanRequest submission path (step ii/iii). Closes a schema-ahead-of-GUI gap. |
| Description text (`textEditScanInfo`) | `scan_information["description"]` | `ScanRequest.description` | 1:1. |

**Direction of the gaps:** schema is *ahead* of the GUI on multi-axis
grids, `PositionList`, `per_step` actions, trigger variants, and a
first-class acquisition toggle. The GUI is *ahead* of the current
single-named-save-set model with its multi-element merge. The pseudo/composite
case is a gap in *both* the engine and the GUI, and stays on the legacy path.

---

## 3. The submission path

**Concrete flow M4 introduces (behind a flag), alongside the unchanged
legacy path:**

```
GUI form state
  → _collect_ui_scan_config()               (unchanged UI read)
  → NEW: _build_scan_request(config_data)    → geecs_schemas.ScanRequest
  → RunControl.submit_run(request)           (accepts ScanRequest OR exec_config)
  → BlueskyScanner.reinitialize(request)     (already duck-detects ScanRequest)
  → start_scan_thread()                      (unchanged)
  events → on_event → pyqtSignal → GUI       (unchanged seam)
```

### 3(a) Which entry point — `reinitialize(ScanRequest)` vs `GeecsSession.run`?

**Recommendation: finish the bridge's `reinitialize(ScanRequest)` path;
do not route the GUI through `GeecsSession.run` directly.**

Why: `RunControl` already owns a `BlueskyScanner` and speaks its
duck-typed surface (`reinitialize`/`start_scan_thread`/`stop_scanning_thread`/
`is_scanning_active`) — the same surface the legacy `ScanManager` exposes, and
the same one the event seam, pre-flight dialogs, progress, and abort are all
wired against (`run_control.py`, stewardship §3). `GeecsSession.run` is a
*blocking headless* call with none of that GUI machinery (thread management,
`on_event` lifecycle emission, `stop_scanning_thread`, pre-flight operator
dialogs). Routing the GUI at `session.run` would mean re-implementing the
scanner's thread/abort/event adapter — throwing away the exact code M3b/M3c
built. The bridge is the adapter; M4's job is to stop it refusing.

The bridge should keep its current internal strategy: resolve the request
(names → engine shapes) and drive `GeecsSession`'s scan/optimize plans —
but via the **same `run_scan_request` machinery** the headless path uses,
so there is one execution definition, not two. (See 3(b).)

### 3(b) What must change in the bridge to stop refusing

The bridge's `_reinitialize_from_scan_request` currently reproduces a
*subset* of `run_scan_request`'s resolution and then refuses the rest. To
reach parity, the bridge must **delegate to the engine's request machinery**
rather than re-deriving a legacy namespace. Concretely:

1. **Actions.** Remove `raise_if_actions_present` /
   `resolve_save_set_checked`'s ritual refusal. Instead, assemble +
   compile the action slots exactly as `run_scan_request` does
   (`resolve_defaults_for` → `resolve_save_set_and_rituals` →
   `assemble_action_slots` → `compile_action_slot` → `prefetch_action_signals`),
   and pass the compiled `setup`/`per_step`/`closeout` stubs into the
   session scan call. All of this is already pre-claim and fail-fast.
2. **Multi-axis grids.** Remove the `len(request.axes) > 1` refusal; build
   the movable list and outer-product positions the way `run_scan_request`
   does (runner:1525-1559) and pass `motor_arg`/`positions` to
   `session.scan`.
3. **Optimize mode.** Remove the optimize refusal; wire the GUI's
   `optimization_loader` bridge (already injected into `BlueskyScanner`,
   `run_control.py:72`) to supply the `objective`/`suggester` that
   `run_scan_request`'s optimize path requires (runner:1620). This is where
   the GUI's contribution is load-bearing — the engine *cannot* build them
   (dependency direction; runner docstring §gaps).

**The cleanest way to do all three at once** (recommendation): have the
bridge's scan-thread body call `run_scan_request(self._session, request,
resolver, objective=..., suggester=...)` inside its threaded/pre-flight/
scan-log wrapper, instead of synthesizing a legacy scan-config namespace and
calling `_execute_scan`. That collapses the two code paths into one engine
definition and deletes the bridge's request→legacy-namespace adapter
(`_reinitialize_from_scan_request`'s namespace synthesis,
bluesky_scanner.py:449-500) rather than growing it. **Open question for the
maintainer** (§5): does the bridge's pre-flight liveness/staleness pipeline
(`_preflight_check_sync_liveness`) need to wrap the `run_scan_request` call?
Today pre-flight runs inside `_execute_scan`; `run_scan_request` →
`session.scan` has its own claim discipline but the operator-dialog pre-flight
is a scanner-layer concern. This is the one real integration seam to design,
not just delete-a-refusal.

Keep the event/progress/dialog seam **as-is** — it is downstream of both
`_execute_scan` and `session.scan` (both emit through the same `RE.subscribe`
+ `on_event`), so it works unchanged.

---

## 4. Incremental sequence

Each step leaves the GUI working and the legacy path fully intact. Each is
independently verifiable: a hermetic test (schema build / bridge parity) plus
a hardware smoke on the daily-driver mode.

- **(0) Multi-save-set foundation (schema + resolver).** `ScanRequest`
  references a *list* of named save sets; the resolver unions them (dedupe by
  device, merge entries). Its own focused PR before the rest — schema +
  `save_set_to_devices_config` + converter/goldens + docgen no-drift
  reference. Verified hermetically (a request naming two save sets records the
  union). This is the one schema evolution M4 needs; everything after it is
  wiring. (§5 "Save-set naming".)
- **(i) Bridge reaches ScanRequest parity (engine-side).** Remove the four
  refusals in `_reinitialize_from_scan_request` by delegating to
  `run_scan_request` (§3b). *No GUI change yet* — verified with the existing
  headless ScanRequest tests plus a new bridge-parity test asserting a
  request with actions + a 2-axis grid runs through the bridge identically to
  `session.run`. Hardware smoke: a headless multi-axis + action request via
  the bridge. **This is the biggest engine step and unblocks everything
  after it.**
- **(ii) Behind-a-flag "submit as ScanRequest" for a simple scan.** Add
  `_build_scan_request()` to the GUI and a submission flag
  (`GEECS_USE_SCAN_REQUEST`, mirroring `GEECS_USE_BLUESKY`, resolved next to
  it in `backend_selection.py`). When set, `initialize_and_start_scan` builds
  a `ScanRequest` and calls `submit_run(request)`; when unset, the legacy
  `exec_config` path is byte-identical. Scope: 1-D step + noscan + background
  only. Verify: hermetic test that a fixed GUI form state produces the
  expected `ScanRequest`; hardware smoke reproducing a current simple scan
  through the flag.
- **(iii) Expand GUI coverage one feature at a time.** Each independently
  shippable, each gated by the same flag: setup/closeout actions (map
  merged-element rituals → save set + `ActionBindings`); the save-set naming
  resolution (§5); optimization (`OptimizationSpec` + loader wiring);
  trigger profile + variant selection; then the schema-ahead features
  (multi-axis grid UI, `PositionList`, acquisition toggle) as demand appears.
- **(iv) Make ScanRequest the default; legacy still available.** Flip the
  flag default to ScanRequest for Bluesky sessions; the legacy `exec_config`
  path stays reachable (flag off) as the fallback. No deletion — that is M6.

Steps (ii)-(iv) touch only the GUI submission cluster; step (i) is
engine-only. The two halves can proceed in parallel once (i) lands.

---

## 5. Risks / open decisions

- **The big-file problem — is the concern-extraction refactor forced?**
  **Recommendation: no, M4 does not force it, and should not trigger it.**
  M4's GUI change is contained to *one* of `geecs_scanner.py`'s seven
  concern clusters — scan submission (`_collect_ui_scan_config` /
  `_build_exec_config` / `initialize_and_start_scan`, lines 1822-1994). It
  *adds* a sibling `_build_scan_request` next to `_build_exec_config` and a
  flag branch in `initialize_and_start_scan`; it does not cross into the
  save-element, preset, or toolbar clusters. The root-CLAUDE.md forcing
  function ("a feature touching three-plus concern clusters at once") is not
  met. Keep the deferred refactor deferred. *Watch item:* if the save-set
  naming decision (below) pulls in the save-element-list cluster too, re-check
  this call.
- **Save-set naming — DECIDED: (b), a list of named save sets.** The GUI
  merges *several* selected save elements into one `Devices` dict
  (`combine_elements`, `_collect_ui_scan_config:1889-1901`); a `ScanRequest`
  names *one* `save_set`. The maintainer's workflow is *named diagnostic
  groups* (laser cams, aux diagnostics, e-beam profiles) mixed and matched
  per scan — which is exactly a scan referencing a **list** of named save
  sets, each group a reusable `SaveSet`, the engine unioning their devices
  (deduped). Chosen over (a) a synthetic per-scan merged set (throws away the
  named groups) and (c) single-element (too limiting).
  **This is a schema evolution done as its own focused PR *before* M4 step
  (i)**, so the GUI is built against the clean model from the start:
  `ScanRequest.save_set: str` → `save_sets: list[str]` (or a plural field);
  `save_set_to_devices_config` unions the resolved sets (dedupe by device,
  merge entries; entry-level rituals/`db_scalars` per entry); converter,
  goldens, and the docgen no-drift reference regenerate. Bounded, but it
  touches the schema + resolver + runner, so it is **not** free — it is the
  first M4-foundation task.
- **Backward-compat with existing presets/configs.** Presets are saved
  `exec_config`-shaped scan configs today; the config corpus is legacy-YAML.
  The `ConfigsRepoResolver` already converts legacy save elements / shot
  control / scan variables / action libraries on the fly
  (runner:162-448), so the request path consumes the existing corpus with no
  flag day. **But:** existing *presets* (full saved scan configs) are not
  `ScanRequest`s. Decision: M4 keeps presets on the legacy path (they still
  work, flag off) and does not migrate them — preset migration is naturally
  M5 work (config editors). Flag this so nobody tries to round-trip a preset
  through the request path in M4.
- **Optimize-mode gaps.** `OptimizationSpec` carries `evaluator`/`generator`
  specs, but the GUI's Xopt/evaluator/ScanAnalysis stack is loaded from the
  `optimizer_config_path` YAML via the injected `optimization_loader`
  (`session_bridge.load_session_optimization`) — the engine can't instantiate
  them. So the request's `optimization` block is partly *descriptive* while
  the loader remains *operative*. Open question: for M4, does the GUI (a)
  keep pointing at `optimizer_config_path` and inject the loader (minimal,
  recommended), or (b) fully populate `OptimizationSpec` from the YAML so the
  request is self-describing? (b) is more faithful to front-ends-as-clients
  but needs the loader to accept a spec, not a path — defer to when the
  optimizer stack is next touched.
- **`ScanOptions` has no home in `ScanRequest`.** rep_rate, time-sync,
  TDMS, beeps live in `ScanOptions` but the schema has no fields for them.
  Rep rate → `session.rep_rate_hz` (already threaded); the rest are legacy
  knobs the Bluesky path ignores anyway. No schema change needed — just
  confirm nothing silently drops a setting an operator relies on.
- **Possible schema additions (flag early — schema changes ripple to the
  no-drift docs guard).** None are *required* for M4 as scoped. Candidates if
  scope grows: a first-class GUI acquisition toggle needs no schema change
  (`acquisition` exists); multi-name save sets (option b above) would be an
  engine/resolver change, not a schema one; a `per_step` action UI needs no
  schema change (`ActionBindings.per_step` exists). **If** the save-set
  naming decision lands on a new schema field, that ripples to the
  `GEECS-Schemas` docgen reference — call it out in the PR.

---

## 6. What M4 is explicitly NOT

M4 **does not** delete any legacy code (the `exec_config` path,
`ScanExecutionConfig`, `ScanManager`, or the legacy config models all stay —
deletion is **M6**). M4 **does not** add schema-native config editors — the
GUI in M4 still edits legacy YAML configs and relies on the on-the-fly
converter in `ConfigsRepoResolver`, or authors new-schema configs by hand;
building real schema-backed editors is **M5**. M4 **does not** migrate saved
presets (they remain legacy-shaped and run on the legacy path).

**The sequence in one line:** **M4** = the GUI *submits* a `ScanRequest`
(new path added, legacy kept, converter bridges the config corpus) →
**M5** = the GUI *edits* schema configs natively (new-schema editors replace
the legacy dialogs; presets migrate) → **M6** = *cutover* (ScanRequest
becomes the only path; the legacy `exec_config` path, `ScanManager`, and the
legacy config models are deleted).
