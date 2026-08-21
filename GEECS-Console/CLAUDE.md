# GEECS-Console — Developer Context for Claude

The greenfield PySide6 operator console (decided 2026-07-10).  **The screen
map is the spec**: `Planning/cutover_strategy/00_overview.md` (settled
parameters), `Planning/cutover_strategy/01_gui_feature_inventory.md` (the
capability inventory with dispositions), and the approved screen-map
artifact (regions R1–R7).  This package is the "one working screen" half of
the commit/abort checkpoint's criterion (c).

## The screen map (regions → widgets)

One main window, menu bar (Ops / Actions / Editors / Preferences / Help),
status bar (gateway addr, configs path, version).  Object names in the `.ui`
are prefixed by region (`r3_radio_1d`, `r5_start_button`, …).

- **R1 session bar** — experiment combo, rep-rate field, trigger-profile +
  variant combos, gateway/tiled/db health chips.
- **R2 save sets** — available/selected lists, Add/Remove, union preview
  line ("union: N devices"), role-conflict/reference hint line.
- **R3 scan form** — mode radios (No-scan / 1D / Grid / Optimization /
  Background), variable picker + start/stop/step (two axis rows; row 2 is
  Grid-only), an optimizer-config combo (visible in Optimization mode only —
  see Implemented seams), shots per step, acquisition combo (free_run
  default, strict — the request declares intent), live shot count with the
  `MAXIMUM_SCAN_SIZE = 1e6` guard, description.
- **R4 presets** — combo + Apply + Save-as + Delete.  A preset IS a saved
  `ScanRequest`; **persistence live** (see Implemented seams): YAML files
  in the configs repo's per-experiment `presets/` dir.
- **R5 submit row** — Stop (danger) + Start (primary).  Start requires: no
  active plan on the manager (running *or* paused — any client's, the
  polled `re_state` is the truth), no submission already in flight, ≥1
  selected save set (**except in Optimization mode** — the worker
  auto-provisions the optimizer's `device_requirements`, so zero selected
  sets is valid there), valid shot count within the guard, and in
  Optimization mode a selected optimizer config.  **Start runs the
  pre-submit pipeline** (queueserver decision 3): the check phase
  (`run_submit_preflight` — engine validation, unserved variables,
  CONNECTED liveness, free-run staleness) on the submit worker → each
  question as one modal (`_ask_binary`; a render failure reads as abort)
  → `stamp_submission` writes the `SubmissionRecord` → the queue submit
  on the worker, with the **failed-item-at-front question** ("Remove &&
  submit?") on a non-empty queue.  Both worker callables capture their
  own exceptions into refusal/failure results — `BackgroundResult`
  swallows raises without emitting, which would strand the pipeline
  in-flight (pinned by test).  **Stop is asynchronous** (the #571 rule
  with a longer worst case): `Submitter.stop_scan` sequences
  deferred-pause → stop from a running scan, so it runs on the stop
  worker with a result slot (a *failed* sequencing releases the hold for
  a retry); the button shows "Stopping…" until a terminal state
  (`_TERMINAL_SCAN_STATES` = aborted/done from the stop document, plus
  idle — the status poll's fallback) releases the hold.
- **R6 now panel** — state pill, progress bar, "Scan NNN" with 10 s expiry
  to "(previous)" (**live**: driven by the start document's
  `scan_number`), compact log tail.
  When idle (startup / experiment change) the label shows
  "Scan NNN (previous)" from a **strictly read-only** peek at today's
  daily `scans/` folder (see Implemented seams), or "No scans today".
  State comes from two sources with a deliberate split
  (`_on_queue_status` / `_on_scan_document`): the **document stream**
  narrates transitions (start → running + totals
  (`num_points × shots_per_step`) + scan number; primary-stream event
  `seq_num` → progress + beeps; stop → done/aborted by `exit_status`),
  and the **manager status poll** is the fallback narrator — it asserts
  every live RE state (running/paused and the transitional
  pausing/stopping/…, rendered as-is; authoritative for the pill and
  Pause button, including scans other clients started), falls an active
  pill back to idle when the RE is idle (stream down, or another
  client's stop), and reads "unknown" when the manager is unreachable
  *or* `re_state` is `None` (worker environment gone mid-scan — the
  crash case must never leave a RUNNING pill lying; a status-bar line
  says the worker is down).  Live-state asserts are suppressed for
  `_TERMINAL_GRACE_S` after a terminal document so a stale pre-stop
  snapshot cannot narrate the transition backwards.  A `paused` pill's *why* comes from the
  console-output stream's failed-move reason line (`_on_pause_reason`).
  **The log tail is NOT the worker's scan.log** (the canonical statement
  of this — the engine CLAUDE.md points here): it is the manager's
  captured stdout/stderr stream (port 60625) plus the window's own
  narration lines — delivered over the network with its own timing and
  granularity; the two stories will drift, and shot counts must never be
  reconciled across them.  The tail suppresses exact consecutive repeats
  at its convergence point, `NowPanelController.append_log` (the cache
  re-arms on set_totals so a new scan's first line always renders); data
  signals never dedupe.
- **R7 movable panel** (middle column, below R4/R5 since 0.19.1 — the
  right column is R6's alone) — an editable combo (catalog scan-variable names
  first, then `device:variable` completions from `GeecsDbCompletions`),
  readback label, set field + button — owned by
  `app/movable_panel.py::MovablePanelController` since 0.19.0 (see
  Implemented seams).  **Catalog-aware**: a catalog name (plain, confirm,
  or pseudo/composite) monitors its target readback(s) and sets through
  the engine's `Submitter.move_variable`; a raw `device:variable` string
  keeps the historical direct path — CA monitor on the readback, put to
  `:SP` riding GEECS's native blocking set — never `geecs_python_api`'s
  ScanDevice.  The R3 axis combos auto-select the panel (the legacy
  scanner behavior), composites included.

## Architecture rules

- **Never import `geecs_python_api`** — pinned by
  `tests/test_no_geecs_python_api.py` (source grep + sys.modules check;
  the grep blesses exactly one string, the
  `~/.config/geecs_python_api/config.ini` path literal the Ops menu opens).
  DB autocompletes go through `GeecsDb` (`geecs_core`, a declared direct
  dependency); errors from the bluesky/gateway tree.
- **The one submission shape is `geecs_schemas.ScanRequest`.**
  `request_builder.build_scan_request` is the only place form state becomes
  a request; keep it a pure function, keep widgets out of it.
- **The window depends on seams, not implementations**: `Submitter`
  (the queueserver-client protocol since #648 — `QueueSubmitter` over
  the RE Manager: queue submission with the failed-item guard, sequenced
  stop, pause/resume, worker-side actions and `move_variable`, and a
  never-raises `status()` the monitor polls), `ConsoleConfigs`,
  `HealthProbe`, `PresetStore`, `ConsoleSettings`.  All
  constructor-injectable; every test drives the window with fakes (and
  disposes the window's `ScanMonitorController` so state slots are
  driven directly, never raced by the background poll).
- **Offline-first**: the window must open and run with zero network and
  zero configs.  `geecs_bluesky` is imported lazily (function-level) in
  `services/submit_preflight.py` and `services/configs.py` — it pulls
  `aioca` at package import, so a module-level import would couple
  opening the window to the `ca` extra; `bluesky-queueserver-api`
  imports live inside `services/queue_client.py` methods the same way.
  Without a `[qserver]` config section the stub client refuses
  submission with a message naming the missing section — everything
  else works.
- PySide6 only (LGPL, agent-editable `.ui` XML).  Never PyQt.
- The `.ui` is hand-authored XML loaded at runtime via `QUiLoader` — no
  generated `*_ui.py` files to keep in sync.

## Implemented seams

- **Health chips (R1)** are live via `GatewayTiledDbHealth` (real probe) or
  `StubHealth` (all-unknown offline/test default).  The real probe runs three
  guarded checks — CA read of `{experiment}:cagateway:heartbeat` (OK; WARN
  when `DEVICES_CONNECTED == 0`; DOWN on failure; UNKNOWN with no experiment),
  HTTP GET of the `[tiled] uri`, and a cheap `GeecsDb` query — each with a
  short timeout; `poll()` **never raises** and lazily imports
  `aioca`/`httpx`/`GeecsDb` inside itself so the module is import-safe offline.
  Polling is **background**: a GUI-thread `QTimer` dispatches each blocking
  `poll()` to a short-lived daemon thread (`HealthPoller`), and the result is
  marshaled back to the chips via a **queued** `report_ready(object)` signal
  (`_apply_health_report` is `@Slot(object)` and connected `QueuedConnection`
  — an undecorated bound method could otherwise wire *direct* and paint
  QLabels off the GUI thread, a hard crash).  Deliberately **no** worker
  QThread/event-loop or cross-thread QTimer — that pattern aborted under
  offscreen pytest ("QThread destroyed while running").  The experiment combo
  pushes the selection into the probe (guarded `hasattr`/`setattr`, since
  StubHealth has no `experiment`); `closeEvent` stops the timer.  Inject the
  real probe in `main.py`; keep `StubHealth` as the window's default.
- **Pre-submit preflight dialogs** (queueserver decision 3 — the old
  mid-run `ScanDialogEvent`/`DialogRequest` transport is gone with the
  bridge): the checks run *before* queueing on the submit worker
  (`services/submit_preflight.py` — the engine's own
  `validate_scan_request` and `UnservedVariablesCheck` reused, plus
  client-side CONNECTED liveness (fail-open; the read passes
  `datatype=str` because CONNECTED is a DBR_ENUM — a native read returns
  the index and can never match "Disconnected") and a free-run
  staleness sample), and each question is an ordinary synchronous modal
  (`_ask_binary`).  Answers are stamped into the request's
  `SubmissionRecord` (geecs-schemas ≥ 0.10.0, aware timestamp pinned)
  so run metadata records who was asked what.  The worker re-runs
  validation authoritatively at execution — duplication by design.
- **Scan monitor** (`app/scan_monitor.py::ScanMonitorController`, #534
  controller shape): the manager status poll (`HealthPoller` pattern
  over `Submitter.status()`), the document stream (`RemoteDispatcher`
  on the worker's proxy out-port — per-shot progress, totals, scan
  number, terminal states), and the manager console-output stream (log
  tail + failed-move pause reasons).  Stream workers emit on
  worker-owned signals connected `QueuedConnection`;
  `stop()`/`dispose()` only gate emission and **abandon** the daemon
  threads — a zmq socket must never be closed from another thread (a
  libzmq assertion aborts the process; #653 review).  Stream setup
  failure emits `stream_failed(str)`, which the window surfaces via
  `_report` — degraded mode (no progress/log stream) is never silent.
- **Movable panel (R7)** — owned by
  `app/movable_panel.py::MovablePanelController` since 0.19.0 (the #534
  controller shape: no Qt parent, injected widgets + callables, its own
  `BackgroundResult` set worker and queued `value_ready` signal, and a
  `dispose()` the window's `closeEvent` calls).  Selection resolves
  catalog-first (`ConsoleConfigs.scan_variable_specs()` — the same
  catalog the R3 axis picker lists): a plain `ScanVariable` monitors its
  target, a `PseudoScanVariable` monitors **every** component
  (`subscribe_many`, one live readback per target, rendered compactly by
  `format_target_readbacks`), and an unresolvable text falls back to the
  raw `device:variable` parse.  **Sets fork by selection kind**: catalog
  names dispatch `Submitter.move_variable(name, value)` on the set
  worker — since #648 that is the queueserver worker's
  `geecs_move_variable` via `function_execute` (idle manager only;
  scan-identical completion semantics — motor poll, confirm poll, pseudo
  fan-out) with the worker's refusals ("scan in progress — move not
  started" / "manual move in progress — …") surfaced verbatim; raw
  `device:variable` strings keep the direct gateway put (no manager
  required).  The R3 axis combos' `currentTextChanged` auto-selects the
  panel (populate churn is signal-blocked; unresolvable names never
  hijack it).  The transport backend stays `GatewayDevicePanel` (real) or
  `StubDevicePanel` (offline/test default) behind `DevicePanelBackend`
  in `services/device_panel.py`.  Readback is
  a persistent `aioca.camonitor` per target PV; because aioca
  is asyncio-based and the monitor is long-lived, the backend owns **one
  persistent asyncio event loop in one daemon `threading.Thread`**
  (`run_forever`; camonitor open/close submitted via
  `run_coroutine_threadsafe`) — the same **no-QThread** rule as the health
  poller (a worker QThread aborted under offscreen pytest: "QThread
  destroyed while running").  Values reach the GUI through the
  controller's `value_ready(int, object)` signal, connected **queued** to
  a `@Slot(int, object)` — never paint widgets off the GUI thread.  Selection
  commits (dropdown pick / Enter / focus leave) resubscribe; per-keystroke
  edits only regate the Set button (no CA-monitor churn while typing); a
  generation counter drops straggler callbacks from retired monitors.  Set
  goes through `GatewaySetpointPut` (the one blessed `:SP` put primitive,
  `wire_value` coercion) dispatched through a dedicated `BackgroundResult`
  set worker whose queued `result_ready` delivers `(ok, message)` back on
  the GUI thread — the worker owns the cross-thread emission, never the
  window (issue #510, resolved in 0.7.0); the button is disabled while a
  put is in flight.  PV names come only from `ca_pv`/`bare_pv`
  (never hand-built — the `ca://`-vs-bare addressing rule, issue #490).
  All real imports are lazy (module import-safe offline); `closeEvent`
  unsubscribes and disconnects, never joins.  Inject the real backend in
  `main.py`; keep `StubDevicePanel` as the window's default.
- **Presets (R4)** are live via `PresetStore` (`services/presets.py`), the
  constructor-injectable persistence seam.  A preset IS a saved
  `ScanRequest`: one YAML file per preset at
  `scanner_configs/experiments/<Experiment>/presets/<name>.yaml` (beside
  the config kinds `ConfigsRepoResolver` reads), written as
  `model_dump(mode="json")` and loaded through
  `ScanRequest.model_validate`.  Save-as goes current form →
  `build_scan_request` → store (name from a `QInputDialog`, overwrite
  allowed); Apply goes store → `form_state_from_request` — the **pure
  inverse** of `build_scan_request`, next to it in `request_builder.py`,
  widgets kept out — → `_apply_form_state`, which validates everything the
  widgets cannot express *before* touching any of them (optimize presets,
  action bindings, explicit position lists, >2 axes ⇒ status-bar error,
  form untouched; unknown save-set names are skipped with a warning).
  Listing never raises (missing configs repo ⇒ empty); save/load/delete
  raise `PresetStoreError` surfaced in the status bar.  Creating the
  `presets/` dir with `mkdir(parents=True, exist_ok=True)` is deliberate —
  it is a config dir, not a `scans/ScanNNN/` folder, so the repo's
  scan-folder invariant does not apply.  The combo repopulates on
  experiment change and after save/delete.
- **Last-experiment memory**: `ConsoleSettings` (`services/settings.py`) is
  a tiny QSettings-backed helper (`GEECS`/`GEECS-Console`, **INI format**
  so `QSettings.setPath` redirection works in tests) — deliberately not a
  framework; future GUI state becomes more properties on it.  The window
  writes `last_experiment` on every experiment change and restores it at
  startup only when no experiment was passed explicitly and the name is
  still in the combo (restoring fires the normal experiment-changed path,
  so configs, presets, health probe, and device panel all follow).
  Constructor-injectable; `tests/conftest.py` isolates the user scope to a
  per-test tmp path so no test touches real settings.  Also carries the
  Preferences beep options (`per_shot_beep`, `randomized_beeps`).
- **Scan number (R6)**: the run's start document carries `scan_number`
  (recorded by the worker at the claim); `_on_scan_document` feeds it to
  `set_scan_number`, which delegates to `NowPanelController` (10 s
  expiry to "(previous)").
- **Ops menu**: four items, handlers in `main_window.py`, path resolution
  factored into `services/ops_paths.py` as small pure `-> Path | None`
  functions (unit-tested against tmp trees, no Finder).  *Open experiment
  config folder* (configs-repo dir for the current experiment); *Open user
  config* — the shared `config.ini` **by path literal only** (the
  no-geecs-python-api pin blesses exactly that one string; opens the folder
  with a note when the file is absent); *Open today's scan folder* —
  **strictly read-only**: builds the daily `scans/` path via
  `geecs_data_utils.ScanPaths.get_daily_scan_folder` (lazy import, pure
  path construction) and NEVER creates directories — a missing folder
  reports "no scans today" (repo scan-folder invariant, pinned by
  tree-unchanged tests in `tests/test_ops_paths.py`); *GEECS-Plugins on
  GitHub*.  All open via `QDesktopServices.openUrl`.  Menus created in
  `_build_menus` must be referenced on the window (`self._menus`) —
  PySide6 garbage-collects the `addMenu` wrapper and tears down the C++
  menu with it.
- **Per-shot beeps (Preferences)**: two checkable actions persisted via
  `ConsoleSettings`, both default off.  "Per-shot beep" sounds
  `QApplication.beep()` (no sound assets, no multimedia dep) on every
  `shots_completed` increment in the progress stream; "Randomized beeps"
  thins that to a random ~1-in-4 subset.  The RNG is a constructor
  parameter (`rng: random.Random`) so tests inject a seeded instance.
- **File logging**: `main.py` has a `--log-level` flag (default INFO) and a
  `RotatingFileHandler` at `~/.config/geecs_console/logs/console.log`
  (2 MB × 3 backups) beside the stderr handler.  Creating that log dir with
  `parents=True` is deliberate — a user config dir, not a scan folder.
  `configure_logging` caps the `httpx` logger at WARNING — the Tiled
  health probe otherwise logs one INFO line per 5 s poll, forever.
- **The four config editors (Editors menu)** — all implemented (built on
  their own branches, #504–#507; wired in 0.6.0):
  `editors/save_set_editor.py::open_save_set_editor`,
  `editors/scan_variable_editor.py::open_scan_variable_editor`,
  `editors/shot_control_editor.py::open_shot_control_editor` (trigger
  profiles), `editors/action_library_editor.py::open_action_library_editor`.
  Each entry point takes `(parent, experiment, configs_base=None,
  completions=None)`, shows a **non-modal** dialog (`show()`, not
  `exec()`), and returns it.  The menu handlers call
  `open_*_editor(self, experiment=<current combo text>)` and append the
  returned dialog to `self._open_editors` (pruned when closed) — see the
  PySide6 ownership hazards below.  Actions are disabled while no
  experiment is selected.  Tests monkeypatch the `open_*` names on
  `app.main_window`.
- **R7 device:variable completions**: the device combo's dropdown lists
  sorted `device:variable` strings from a `CompletionsProvider`
  (`GeecsDbCompletions` in production, constructor-injectable
  `completions_factory` in tests), fetched at startup and on experiment
  change via a `BackgroundResult` worker (below).  The combo stays
  editable; typed text survives repopulation; results tagged with a
  no-longer-selected experiment are dropped.  An unparsable committed
  selection shows "Device format: DeviceName:Variable Name" in the status
  bar (both on commit and on a Set attempt) instead of a silent no-op.
- **R6 idle scan number** — owned by
  `app/now_panel.py::NowPanelController` since 0.18.2 (issue #534 step 5,
  with the pill/progress/log-tail rendering and the expiry timer; the
  window keeps the widget attributes, thin `append_log` /
  `set_scan_number` delegates, and the scan-lifecycle hub, and its
  `closeEvent` calls the controller's `dispose()` — same lifetime rule
  as the actions-menu controller): at startup and on experiment change
  the controller's `BackgroundResult` worker runs the injectable
  `scan_number_lookup` (default: `ops_paths.todays_scan_folder` +
  `ops_paths.highest_scan_number`, resolved at probe time so test
  patching works), **one probe in flight per experiment** (the same
  native-first-import race dedupe as the actions fetch), and the label
  shows "Scan NNN (previous)" or "No scans today".  **Strictly read-only** —
  resolution + `is_dir()`/`iterdir()` only, never creating anything on the
  scans path (repo scan-folder invariant; pinned by tree-untouched tests
  in `tests/test_ops_paths.py` and
  `tests/test_main_window_editors_integration.py`).  A live scan number
  (10 s expiry timer running) is never clobbered.  `tests/conftest.py`
  patches the module-level default lookup (and the completions factory) so
  hermetic tests never touch the real data root or DB.
- **`BackgroundResult`** (`services/background.py`, extracted from
  `app/main_window.py` in 0.10.0 — the shared home recorded on issue
  #510): the one blessed daemon-thread → queued-signal worker for one-shot
  background calls (the `HealthPoller` shape, generalized —
  `HealthPoller` itself, the interval variant with in-flight skip, lives
  beside it in the same module since 0.15.1 / issue #534 step 1).  **The daemon
  thread must emit on the worker QObject, never on the window**: emitting
  a window-owned signal from a daemon thread races window teardown and
  segfaults under offscreen pytest (observed directly when the idle scan
  probe emitted a `MainWindow` signal; the R7 device-set completion was
  the last such emission and moved to a `BackgroundResult` worker in
  0.7.0 — issue #510).  `closeEvent` disconnects each window-owned
  worker's `result_ready`; the actions-menu and now-panel controllers'
  workers are detached inside their `dispose()` instead.
- **Actions menu (G-actions v1)** — owned by
  `app/actions_menu.py::ActionsMenuController` since 0.18.1 (issue #534
  step 3): the window creates the QMenu (kept in `self._menus`) and the
  controller owns its contents, its `BackgroundResult` fetch worker, and
  the open dialogs; the window keeps thin `enable_actions_action` /
  `_open_action_dialogs` properties as the test surface, and its
  `closeEvent` calls the controller's `dispose()` (severing every
  controller → window reference — the cycle would otherwise defer
  dead-window teardown to the cyclic GC mid-event-processing, a
  segfault under offscreen pytest).  Lists the current experiment's
  action-plan names — fetched from `ActionLibraryStore.list_names()` (the
  same `action_library/actions.yaml` the Action Library editor edits;
  constructor-injectable `action_store` seam), refreshed on experiment
  change, stale results dropped by experiment tag, **one fetch in
  flight per experiment** — startup requests twice back-to-back, and
  two concurrent fetch threads race the lazy `geecs_bluesky` import
  inside the store's configs-root resolution, a native init that
  aborts the process when raced (found 2026-07-20; the now-panel idle
  probe is deduped the same way since 0.18.2, the completions
  double-fetch is not yet); empty/offline/failed renders one disabled
  "(no actions)" entry.  On top sits **"Enable action execution"** — the
  accidental-click guard: checkable, **default OFF at every launch and
  deliberately NOT persisted** (a fresh session must never start armed;
  do not "fix" this by adding it to `ConsoleSettings`).  Clicking a plan
  opens the non-modal `ActionRunDialog` (`app/action_dialog.py`, kept on
  the controller — the GC hazard): a dry-run steps table
  from the engine's `describe_action(name) -> list[dict]` (keys `kind` /
  `device` / `variable` / `value` / `wait_s` / `from_plan`, execution
  order) plus Run/Close.  Run is enabled only while armed and dispatches
  the blocking `run_action(name)` on a dialog-owned worker — in flight
  the button disables and the status bar shows "running action
  '<name>'…"; success reports "action '<name>' done"; failures/refusals
  land in the status bar AND inline.  The
  preview and run outcomes render on **separate labels** — a slow
  describe arriving late must never clobber a refusal (pinned by test).
  Both methods are `Submitter` protocol members (`submission.py`) —
  since #648, `run_action` queues `geecs_run_action_plan` and
  `describe_action` runs the worker's `geecs_describe_action`.
  **Actions are idle-only queue items since the queueserver migration**
  (decision 2 dropped the G-actions v2 pause-window flow — `Run` queues
  `geecs_run_action_plan` via `Submitter.run_action`, the preview comes
  from the worker's `geecs_describe_action`, and with a scan active the
  Run button simply disables; the window's state hub pushes
  `set_scanning` into the controller, which forwards it to open dialogs
  and seeds newly opened ones).  The old
  `request_action_during_scan`/three-way-decision machinery is deleted.
  **Misfire hardening (#575):** name in larger type, Run gated on the
  preview having loaded (no firing beside an empty table — kills the old
  late-preview-clobbers-run race), step count on the Run label.  Pinned
  by `tests/test_actions_menu.py`.
- **Optimization (R3) — end to end**: the Optimization radio shows a
  config combo listing the YAML stems of the experiment's
  `optimizer_configs/` folder (legacy scanner-GUI folder name; part of
  `ConfigListing`).  `ConsoleConfigs.optimization_spec` loads a named
  config as a validated `OptimizationSpec` — new-schema documents
  directly, the legacy `vocs` dialect through
  `geecs_schemas.convert.convert_optimizer_config`.  `form_state()`
  resolves the selected name into the spec (`ConsoleFormState.optimization`)
  so `build_scan_request` stays pure; optimize requests round-trip through
  `form_state_from_request`, and applying an optimize preset matches its
  inline spec against the listed configs by content (no match ⇒ status-bar
  error, form untouched; `max_iterations` is neutral in the match — it
  belongs to the spinner below).  The **Iterations spinner** (`r3_iterations_spin`,
  visible with the mode) owns the submitted spec's `max_iterations`
  (`ConsoleFormState.max_iterations`; 0 renders as "auto" ⇒ `None`, the
  engine's default budget — deliberately NOT the old GUI's derive-from-1D-
  limits hack): the builder writes it onto the spec, picking a config seeds
  the spinner from the config's own limit, presets restore it, and the R3
  shot count shows `iterations × shots/step` (the runaway guard applies) or
  "auto".  **Execution is worker-side since the queueserver migration**
  (decision 5): the request's inline `OptimizationSpec` travels in the
  queue item, and the *worker's* loader
  (`geecs_bluesky.optimization.worker_loader`, registered at qserver
  startup) builds the Xopt/evaluator stack there — the console injects
  nothing and needs no heavy dependencies to submit an optimization.  A
  worker without the `optimize` extra refuses the queued request loudly;
  the refusal message surfaces like any other submission failure.
  `services/optimization.py` (the old GUI-process loader and its
  `optimization` extra) is now **consumer-less** — kept only until the
  W5 cleanup deletes it; do not wire it anywhere new.  **The worker
  auto-provisions the optimizer's `device_requirements`** (GeecsBluesky
  ≥ 0.38.0, reversing the #520 deferral after the 2026-07-15
  NaN-objectives field incident), recorded in run metadata as
  `provisioned_device_requirements` — so in Optimization mode Start
  requires no selected save set, and the R2 union line notes the
  optimizer's contribution ("diagnostics from optimizer config" /
  "+ optimizer diagnostics").
- **Tooltips (issue #497 phase 1)**: editor form fields get their tooltips
  from the geecs-schemas `Field(description=...)` texts via
  `services/schema_tooltips.py::apply_schema_tooltips` — single source of
  truth; a mapping to a missing or description-less field raises at editor
  construction.  When a tooltip reads poorly, fix the schema description,
  never hardcode GUI text.  Main-window operator controls carry
  hand-written operator-language tooltips
  (`app/tooltips.py::apply_operator_tooltips`, an attribute-name → text
  catalog that raises loudly on a missing widget; `ToolTipSuppressor`
  lives in the same module) — those are GUI concepts with no schema
  counterpart.  **Preferences →
  Show tooltips** (persisted, default on) gates them all via
  `ToolTipSuppressor`, an application-level event filter that swallows
  `QEvent.ToolTip`; it is installed on the `QApplication` **only while
  tooltips are off** — an always-installed per-window app filter
  measurably slowed the offscreen suite (every event crossing into a
  Python `eventFilter`), so presence = suppression.  It is parented to
  the window (Qt auto-removes a destroyed filter) and `closeEvent`
  removes it explicitly.  Pinned by `tests/test_tooltips.py`.

## Standing PySide6 ownership hazards (GC eats live C++ objects)

Python wrappers PySide6 does not parent-track are garbage-collected, and
shiboken tears down the underlying C++ object with them.  Two recurrences
of the same bug class are load-bearing here; hold a Python reference on
the window for anything in these families:

- **Menus**: the `QMenu` returned by `menuBar().addMenu(...)` must be kept
  (`self._menus`) or the menu and all its actions vanish.
- **Non-modal dialogs**: a dialog shown with `show()` (all four editors)
  must be kept (`self._open_editors`) or it closes/dies at the next GC.
- **QCompleter** (inside the editors): a completer set on a line edit via
  `setCompleter` is not owned by the widget — the editors keep their
  completers (and their model) on `self`.  The same applies to any
  `QValidator`, proxy model, or event filter created without a parent.

## The Scan Browser (`geecs_console/browser/`, regions B1–B7)

A second window in this package: the quick-look Tiled client (day → scan →
plot/table/metadata/drift), per its own approved screen map (regions
`B1`–`B7`; object names `b1_`…`b7_`; B7 is the scan-metadata panel below
B5 in the middle column, added on issue #559).  Own entry points — `geecs-scan-browser`
(console script) and `python -m geecs_console.browser` — deliberately NOT
wired into the operator console's Ops menu yet (deferred; the browser must
stay usable by analysts who never run the console).

Structure:

- `browser_window.py` — `ScanBrowserWindow`, layout built in code (no
  `.ui`: the pyqtgraph central widget doesn't suit `QUiLoader` promotion).
  Dark screen-map palette QSS applied at window level, over the console
  family stylesheet the entry point sets application-wide.
- `__main__.py` — `main()`; injects the real catalog
  (`TiledScanCatalog.from_config()`).  Loads `app/style.qss` with its own
  tiny loader (same behavior as `load_stylesheet`) because of the
  main-window import ban below.

Rules (inherit all Architecture rules above, plus):

- **The ScanCatalog seam**: the window depends on
  `geecs_data_utils.tiled_catalog.ScanCatalog`
  (`probe`/`list_runs`/`load_run`) — never on `tiled` directly.  Offline
  default is `StubCatalog`; every catalog call runs on a daemon thread
  through the shared `BackgroundResult` worker (`services/background.py`)
  with generation counters dropping superseded results — the GUI thread
  never blocks on Tiled (VPN latency is real).  `BackgroundResult`
  swallows exceptions, so the browser wraps every callable in
  `_capture_outcome` to deliver `(result, error)` tuples — catalog
  failures reach the status bar instead of hanging a spinner.
  `closeEvent` disconnects the workers so a straggling slow read lands
  nowhere.
- **Schema knowledge lives in `geecs_data_utils.tiled_schema`** (one
  version-tagged module; `GeecsBluesky/EVENT_SCHEMA.md` is the contract)
  and **drift analysis in `geecs_data_utils.tiled_drift`** — the browser
  interprets no column names on its own.  The pure layer's tests live in
  GEECS-Data-Utils; this package tests window behavior against
  `tests/fake_catalog.py`.
- **No imports from `app/main_window.py`** — the browser duplicates
  nothing from it except the one deliberate twin noted below.
- **`pyqtgraph` is imported lazily** with
  `os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")` set first (the
  `_pg()` helper) so it can never bind a stray PyQt install.
- **Open scan folder is strictly read-only** (`resolve_scan_folder`):
  run-metadata `scan_folder` first, else `ops_paths.todays_scan_folder`
  for the *selected* date + `ScanNNN`; only an existing dir is returned,
  nothing on the scans path is ever created (repo scan-folder invariant,
  pinned by tree-untouched tests in `tests/test_browser_scan_folder.py`).
- **B7 (scan metadata) renders from the already-loaded `RunDetail` only**
  (`metadata_rows`, a pure function over summary + start/stop docs) —
  never a second Tiled fetch; absent/empty keys are omitted, not rendered
  blank, so legacy/aborted runs get a shorter list.

Kit boundary — shared-intent console modules the browser imports (the
shared-package candidates for a future extraction; extend these rather
than copying):

- `services/settings.py` (`ConsoleSettings` — the shared last-experiment
  memory)
- `services/ops_paths.py` (read-only daily-folder resolution)
- `app/style.qss` (read-only; loaded by the browser's own loader)
- `geecs_data_utils.tiled_catalog` / `tiled_schema` / `tiled_drift` (the
  data layer, already extracted downward)

Deliberate temporary twin of `app/main_window.py` internals (kept because
the browser must not import that file — another stream owns it):

- `__main__._load_console_stylesheet` ↔ `load_stylesheet`.

(The other twin, `browser/_background.py::BrowserWorker` ↔
`BackgroundResult`, was retired once the shared `services/background.py`
extraction landed in 0.10.0 — the browser now uses `BackgroundResult`
directly.)

## Stubbed seams (intentional, wire later)

- An `OptimizationSpec` *editor* (authoring configs in the GUI) remains
  out of scope — configs are YAML files in `optimizer_configs/`.  The
  optimization stack behind the loader (`geecs_bluesky.optimization` —
  relocated out of geecs_scanner 2026-08-20) is legacy machinery kept for
  parity; a redesigned hook (bluesky-adaptive direction) is planned, at
  which point `services/optimization.py` and the `optimization` extra are
  deleted together.
- `ConsoleConfigs.scan_variable_specs()` uses the public
  `ConfigsRepoResolver.scan_variable_catalog()` accessor (promoted in
  geecs-bluesky 0.49.0, closing the old reach-into-private debt note);
  a getattr fallback to the private method keeps older engines working.
- **Remaining M5 item — config bootstrap/repair dialog**: deliberately
  deferred.  When the configs repo (or an experiment's folder inside it)
  is missing/broken, the console currently reports and degrades to empty
  listings; a guided create/repair dialog is the outstanding piece of M5.

## Testing

`QT_QPA_PLATFORM=offscreen poetry run pytest -q` — hermetic, pytest-qt,
`qt_api = "pyside6"` pinned in pyproject.  The request-builder tests are the
important ones: they validate the exact `ScanRequest` shapes against the
real schema.  CI also runs this suite on `windows-latest` (the
`console-windows` job in `.github/workflows/unit-tests.yml`) — the console
deploys to Windows control-room machines, so keep the suite green there too.
