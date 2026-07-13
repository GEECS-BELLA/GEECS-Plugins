# Changelog

All notable changes to GEECS-Console are documented here.  Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
semantic.

## [0.9.2] - 2026-07-13

### Changed

- Docs-only: `browser/_background.py`'s module docstring drops the stale
  agent-coordination note ("another session owns that file") and the
  claim that a #510 PR would land `services/background.py` (#510 closed
  without it — the extraction is simply pending); `CLAUDE.md` corrects the
  optimization-loader wiring version to 0.9.0.

## [0.9.1] - 2026-07-12

### Fixed

- **Scan Browser: stale run detail cleared on context change** (post-merge
  review finding on #523): reloading the run list (date/experiment change),
  an emptied selection, and list/load errors now all route through a
  centralized `_clear_detail()` — B3 actions disabled, plot/table/drift
  cleared, in-flight loads invalidated. Previously a previously selected
  scan's detail survived a refresh, and *Open scan folder* could resolve
  the old run against the newly selected day (wrong-ScanNNN risk). Load
  failures now carry their request generation so a late error from a
  superseded load cannot clobber a newer selection.
## [0.9.0] - 2026-07-13

### Added

- **Optimization runs end-to-end (the engine-side loader is now wired)** —
  closes the last gap of M4 step (iii).  `make_bluesky_submitter` injects
  an `optimization_loader` into `BlueskyScanner`, implemented in the new
  `services/optimization.py`: `optimizer_config_from_spec` maps the
  request's `OptimizationSpec` onto the `BaseOptimizerConfig` dict shape
  (pinned as the exact inverse of
  `geecs_schemas.convert.convert_optimizer_config` — `generator.options`
  ↔ `xopt_config_overrides[generator.name]`; `max_iterations` deliberately
  unmapped, the engine consumes it from the spec), and
  `load_console_optimization` builds `BaseOptimizer.from_config(...)`
  (new in geecs-scanner 0.34.0) wrapped in the `SessionOptimizationBridge`.
  The stack lives behind the new **`optimization` extra**
  (`poetry install --extras optimization` pulls `geecs-scanner-gui` — the
  only place the legacy package appears; the import is confined to
  `services/optimization.py`, lazy, and availability-gated by a light
  `find_spec` probe so submitter construction never pays the Xopt import).
  Without the extra the loader is `None` and the engine refuses
  optimize-mode submissions at `reinitialize` with its explicit
  needs-a-loader message, surfaced in the status bar as before; every
  other scan mode is unaffected.  Relative `seed_dump_files` entries are
  not resolved on this path (no config-file directory once the spec is
  inline) — use absolute paths in warm-starting optimizer configs.

## [0.8.0] - 2026-07-12

### Added

- **Scan Browser v1** (`geecs_console/browser/`) — a quick-look Tiled client
  per the approved screen map (regions B1–B6): pick a day, pick a scan, see
  what happened, without mounting the data share or writing a notebook.
  New entry points: `geecs-scan-browser` console script and
  `python -m geecs_console.browser`.
  - **B1 session bar** — editable experiment combo (last value remembered
    via the shared `ConsoleSettings`), calendar date picker (default
    today), Tiled connection chip (off-thread probe), metadata filter box.
  - **B2 run list** — the selected day's runs newest first (status dot from
    the stop-doc `exit_status`, scan number, time, mode chip, shots);
    metadata-only until a row is selected.
  - **B3 identity strip** — mode/shots/acquisition/duration/save
    sets/short uid from the start/stop docs, Copy uid (clipboard), Open
    scan folder (**strictly read-only** resolution — run-metadata
    `scan_folder` first, else the daily folder for the *selected* date via
    `ops_paths`; never creates, pinned by tree-untouched tests).
  - **B4 plot** — pyqtgraph (`PYQTGRAPH_QT_LIB=PySide6` pinned before
    import): X = shot sequence or any numeric column, multi-series Y via a
    contains-matching completer over all data columns, per-series
    mean ± σ in the legend/series list; stepped scans default X to the
    scan variable with per-step mean ± σ error bars.  Non-numeric columns
    (dtype-tolerant telemetry) refused with a status message, never a
    crash.
  - **B5 table** — pinned columns (shot sequence, reference
    `acq_timestamp`) + plotted columns; Export CSV of the visible
    selection via a file dialog.
  - **B6 "Moved during scan" rail** — `geecs_data_utils.tiled_drift` over
    the numeric telemetry columns (3σ first-vs-last, significance-sorted,
    signed Δ with % when meaningful, "N of M steady" line); click adds the
    column to the plot.
- **`browser/_background.py`** — a thin private daemon-thread →
  queued-signal worker shim (the `BackgroundResult` shape).  Temporary
  twin, to be replaced by the shared `services/background.py` extraction
  planned on issue #510; API kept tiny so the swap is mechanical.
- New dependencies: `pyqtgraph ^0.13` and a direct
  `geecs-data-utils[tiled]` path dep (previously transitive) — the browser
  consumes its `tiled_catalog` / `tiled_schema` / `tiled_drift` layer
  (geecs-data-utils 0.13.0) and depends on the `ScanCatalog` protocol,
  never on `tiled` directly.
- Hermetic browser tests (fake catalog, offscreen): day listing +
  filtering, metadata-only listing, B3 population, clipboard uid, plot
  add/remove/replot, non-numeric guard, per-step error bars, CSV export,
  drift rail + click-to-plot, never-creates folder pin, offline
  StubCatalog default, stale-result drop, catalog-error surfacing,
  close-during-slow-read teardown, and entry-point subprocess smoke.

### Notes

- Ops-menu wiring of the browser into the operator console window is
  deliberately deferred (the browser is its own window/entry point in v1;
  `app/main_window.py` is untouched by this change).

## [0.7.0] - 2026-07-12

### Added

- **Tooltips everywhere** (issue #497 phase 1, text only — no doc links).
  Editor form fields derive their tooltips from the geecs-schemas pydantic
  `Field(description=...)` texts via the new
  `services/schema_tooltips.py::apply_schema_tooltips` helper, applied in
  all four editors (save sets, scan variables, shot control, action
  library) — single source of truth; a widget mapped to a description-less
  or renamed field fails loudly at editor construction.  (No schema
  descriptions needed backfilling — every geecs-schemas field already
  carries one, so GEECS-Schemas is untouched.)  The main window's operator
  controls (mode radios, acquisition combo, shots/step, save-set lists,
  device panel, Start/Stop, health chips, presets row, ...) get
  hand-written operator-language tooltips in `_apply_operator_tooltips`.
  Pinned by `tests/test_tooltips.py`: editor tooltips compared against the
  schema field descriptions, representative window controls asserted
  non-empty and not label restatements.  A **Preferences → Show tooltips**
  toggle (checkable, persisted via `ConsoleSettings.show_tooltips`,
  default on) turns them all off for operators who know the console: an
  application-level `ToolTipSuppressor` event filter swallows
  `QEvent.ToolTip` for every console widget — editors included — and is
  installed **only while tooltips are off**, so the default path adds no
  per-event filter overhead (an always-installed filter measurably slowed
  the offscreen suite).  Parented to the window and removed on close, so
  no dangling suppressor can outlive it.
- **Optimization mode dropdown (GUI half)**.  Selecting the R3
  Optimization radio now shows an optimizer-config combo listing the
  YAML files in the experiment's `optimizer_configs/` folder (the legacy
  GEECS-Scanner-GUI folder name; offline → empty combo and Start stays
  disabled).  `ConsoleConfigs.optimization_spec` loads a named config as a
  validated `OptimizationSpec`, accepting both new-schema documents and
  the legacy `vocs` dialect (via
  `geecs_schemas.convert.convert_optimizer_config`).
  `build_scan_request` maps optimize forms onto `mode: optimize` requests
  carrying the loaded spec (still a pure function — the window resolves
  name → spec when snapshotting the form), and `form_state_from_request`
  round-trips optimize instead of raising, so optimize presets save and
  apply (Apply matches the preset's inline spec against the experiment's
  configs by content; no match ⇒ status-bar error, form untouched).  The
  GUI does **not** pre-block submission: the current engine still refuses
  optimize requests, and that refusal is surfaced in the status bar — when
  the engine half lands, zero GUI changes are needed.

### Fixed

- **Device-set completion routed through a worker QObject** (issue #510,
  finished properly).  The 0.6.2 fix swallowed the torn-down-window
  `RuntimeError` around a `MainWindow`-owned `device_set_finished` signal
  emitted from the set-dispatch daemon thread; that window-owned
  cross-thread emission is now gone entirely.  The blocking set runs
  through a dedicated `BackgroundResult` worker whose `result_ready`
  signal delivers `(ok, message)` queued to the GUI thread — the blessed
  pattern (the worker owns the emission and survives window teardown).
  The deterministic regression test (blocked fake backend released after
  window close) stays green.

## [0.6.3] - 2026-07-12

### Fixed

- **Experiment-name path traversal in the config stores** (#513): the
  experiment selector is editable, and four stores (`PresetStore`,
  `SaveSetStore`, `TriggerProfileStore`, `ActionLibraryStore`) joined the
  raw text onto the experiments root — and created parent directories on
  save — without validating it, so a value like `../OtherExperiment`
  escaped the intended experiment folder and could corrupt another
  experiment's real lab configs.  Experiment-name validation is now
  centralized in `services/_experiment_name.py::check_experiment_name`
  (mirroring the guard `ScanVariableStore` already had, which now delegates
  to it) and applied by all five stores **before any path join**: names
  containing `/` or `\`, or equal to `.` / `..`, make load/save/list/delete
  raise the store's own error with no directory created.  Pinned by
  `tests/test_experiment_name_validation.py` (every store × traversal
  variants, asserting the tmp configs tree stays untouched).  Making the
  experiment combo non-editable (the other half of the issue) is deliberately
  deferred as a main-window follow-up.

## [0.6.2] - 2026-07-12

### Fixed

- Device-set completion no longer raises `RuntimeError: Signal source has
  been deleted` on its daemon thread when the window is closed and
  C++-deleted while the blocking set is still in flight (`closeEvent`
  deliberately never joins that thread).  The single
  `device_set_finished.emit` site in `_run_device_set` now swallows the
  torn-down-window `RuntimeError` — harmless in production, but it was an
  intermittent `PytestUnhandledThreadExceptionWarning` (~1 in 2 full
  offscreen suite runs), surfacing under whichever unrelated test the
  straggler thread finished during.  Pinned by a deterministic regression
  test that blocks a fake backend's `set()` until after the window is
  deleted.

## [0.6.1] - 2026-07-12

### Added

- **Windows CI job** (`console-windows` in `.github/workflows/unit-tests.yml`):
  the full GEECS-Console suite now runs on `windows-latest` (Python 3.11,
  Poetry, `QT_QPA_PLATFORM=offscreen`) on every push/PR — the console is the
  operator-facing GUI and control-room machines run Windows.  Console only;
  the ubuntu job continues to cover the rest of the monorepo.

### Fixed

- **Windows test portability** (surfaced by the new CI job's first run):
  the four `TestOpsMenu` open-URL assertions now compare as `Path` —
  `QUrl.toLocalFile()` always returns forward slashes while `str(tmp_path)`
  is backslashed on Windows — and the no-geecs-python-api source grep reads
  files as UTF-8 explicitly (`read_text()` defaults to cp1252 on Windows,
  which cannot decode the sources).

### Notes

- The monospace font stack in `app/style.qss` was audited for Windows: every
  `font-family` already carries the cross-platform fallback chain
  `"SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", monospace`, so no
  style change was needed.

## [0.6.0] - 2026-07-12

### Added

- **The four config editors** (built on their own branches, #504–#507, and
  now wired into the main window): the save-set editor
  (`editors/save_set_editor.py`), the scan-variable editor
  (`editors/scan_variable_editor.py`), the trigger-profile / shot-control
  editor (`editors/shot_control_editor.py`), and the action-library editor
  (`editors/action_library_editor.py`).  Each exposes an
  `open_*_editor(parent, experiment, configs_base=None, completions=None)`
  entry point that shows a non-modal dialog and returns it.
- **Editors menu wired** (replaces the placeholder): *Save Elements…*,
  *Scan Variables…*, *Shot Control…*, *Action Library…*, each opening the
  corresponding editor for the currently selected experiment.  Actions are
  disabled while no experiment is selected; every opened dialog is
  referenced on the window (`_open_editors`, pruned when closed) so PySide6
  garbage collection cannot tear down a live editor.
- **R7 device-combo autocomplete** (user report: "dropdown shows nothing"):
  the device panel combo now lists `device:variable` completions from
  `GeecsDbCompletions` for the current experiment, fetched at startup and
  on experiment change — one blocking provider call on a short-lived daemon
  thread, marshaled back through a queued signal (never on the GUI thread;
  offline degrades to an empty dropdown with free text still working).
  The combo stays editable; typed text survives repopulation; a stale fetch
  racing an experiment change is dropped by experiment tag.  The provider
  factory is constructor-injectable for tests.
- **R7 parse feedback** (user report: silent nothing): committing device
  text that doesn't parse as `Device:Variable` now shows
  "Device format: DeviceName:Variable Name" in the status bar instead of
  doing nothing (both on selection commit and on a Set attempt).
- **R6 idle scan-number display** (user report): on startup and experiment
  change the console peeks — strictly read-only — at today's daily `scans/`
  folder for the highest existing `ScanNNN` and shows "Scan NNN (previous)"
  (or keeps "No scans today").  The peek is resolution + listdir only and
  NEVER creates anything on that path (repo scan-folder invariant, pinned
  by tree-untouched tests); it runs on a daemon thread because the data
  root is typically a slow network mount.  New read-only helper
  `ops_paths.highest_scan_number`.  A live scan number on display is never
  clobbered by the idle peek.
- **Version from source** (user report: status bar said 0.1.0 while running
  0.5.0 code — `importlib.metadata` reflects the last `poetry install`, not
  the source tree): the status-bar/Help version now prefers the source tree
  — `geecs_console/version.py::console_version` reads the adjacent
  `pyproject.toml` (`[tool.poetry] version`) when present (dev checkout),
  falling back to `importlib.metadata.version`, then to "unknown".
- `BackgroundResult` (`app/main_window.py`): a small reusable
  daemon-thread → queued-signal worker (the `HealthPoller` shape,
  generalized) used by the completions fetch and the idle scan-number
  probe.  The daemon thread emits on the worker, never on the window —
  emitting a window-owned signal from a daemon thread races window teardown
  and segfaults under offscreen pytest.

### Fixed

- `httpx` request logging is capped at WARNING in `configure_logging` — the
  Tiled health probe's 5 s poll was writing one INFO line per request to
  the console log, forever.

## [0.5.0] - 2026-07-11

### Added

- Ops menu (replaces the placeholder): **Open experiment config folder**
  (the current experiment's configs-repo dir), **Open user config**
  (the shared `~/.config/…/config.ini`, path only — the no-import pin now
  blesses exactly that one path literal; opens its folder with a note when
  the file is absent), **Open today's scan folder**, and
  **GEECS-Plugins on GitHub**.  All open via `QDesktopServices.openUrl`;
  unresolvable targets report in the status bar.  Path resolution lives in
  `services/ops_paths.py` as small pure `-> Path | None` functions,
  unit-tested against tmp trees without launching Finder/Explorer.
- Today's-scan-folder resolution is **strictly read-only**: it builds the
  daily `scans/` path via `geecs_data_utils.ScanPaths.get_daily_scan_folder`
  (lazy import, pure path construction) and NEVER creates directories — a
  missing daily folder reports "no scans today" (repo scan-folder
  invariant: GUI code is a consumer of scan folders, never a producer).
  Pinned by tests that assert the tree is unchanged after resolving a
  missing folder.
- Per-shot beeps: a checkable **Per-shot beep** Preferences action sounds
  `QApplication.beep()` (no sound assets, no multimedia dep) whenever the
  progress stream's `shots_completed` increments; a second checkable
  **Randomized beeps** action thins that to a random ~1-in-4 subset of
  shots (the RNG is constructor-injectable for seeded tests).  Both persist
  via `ConsoleSettings` (`preferences/per_shot_beep`,
  `preferences/randomized_beeps`); both default off.
- "Scan NNN" wiring: `ScanEventsAdapter` reads the engine's new
  `ScanLifecycleEvent.scan_number` (duck-typed getattr; `None` and absent
  are tolerated) and emits `scan_number_known(int)`, which the window
  connects to the existing `set_scan_number` slot (10 s expiry to
  "(previous)").  The R6 label is now live end-to-end.
- File logging: `main.py` grows a `--log-level` CLI flag (default INFO) and
  a `RotatingFileHandler` writing to `~/.config/geecs_console/logs/console.log`
  (2 MB × 3 backups) alongside the stderr handler.  Creating that log dir
  is deliberate — a user config dir, not a scan folder.

### Fixed

- Menus created by `_build_menus` are now referenced on the window
  (`self._menus`) — PySide6 can garbage-collect the wrapper returned by
  `addMenu` and tear down the C++ menu (and its actions) with it.

## [0.4.0] - 2026-07-11

### Added

- Presets persistence (R4): the stub handlers are replaced by a
  `PresetStore` seam (`services/presets.py`).  A preset IS a saved
  `ScanRequest` — one YAML file per preset
  (`scanner_configs/experiments/<Experiment>/presets/<name>.yaml`, a plain
  `model_dump(mode="json")` document round-tripped through
  `ScanRequest.model_validate`), living beside the other per-experiment
  config kinds `ConfigsRepoResolver` reads.  Listing degrades to empty with
  no configs repo; load/save/delete raise `PresetStoreError` with a message
  the window surfaces in the status bar.  Constructor-injectable into
  `MainWindow` (tests drive a fake or a tmp-dir-backed store).
- Save-as: current form → `build_scan_request` → store, named via a
  `QInputDialog` (overwrite allowed); the combo repopulates and selects the
  new preset.  Apply: selected preset → `form_state_from_request` (the new
  pure inverse of `build_scan_request`, beside it in `request_builder.py`)
  → form widgets.  Content the form cannot express — an optimize preset,
  action bindings, explicit position lists, more than two axes — reports a
  clear status-bar error and leaves the form untouched; preset save-set
  names missing from the experiment's configs are skipped with a warning.
  Delete: a Delete button beside Save-as removes the selected preset.  The
  combo repopulates on experiment change and after save/delete.
- Last-experiment memory: the last selected experiment persists across
  sessions via `ConsoleSettings` (`services/settings.py`, a tiny
  QSettings-backed helper — `GEECS` / `GEECS-Console`, INI format) and is
  restored at startup when nothing was selected explicitly and the name is
  still in the combo's list — so the health probe, configs, presets, and
  device panel all reopen pointed at it.  Constructor-injectable; tests
  redirect QSettings to a per-test tmp path.

## [0.3.0] - 2026-07-11

### Added

- Live device panel (R7): the stubbed backend is replaced by a
  `DevicePanelBackend` seam (`services/device_panel.py`) with two
  implementations — `StubDevicePanel` (the offline/test default: readback
  never updates, sets report unwired) and `GatewayDevicePanel` (the real
  one, injected by `main.py`).
- Readback: a persistent `aioca.camonitor` on the gateway readback PV
  (`{experiment}:{device}:{variable}`, names via the blessed
  `ca_pv`/`bare_pv` helpers).  The backend owns ONE persistent asyncio
  event loop in a single daemon thread (no QThread — the health-probe
  teardown rule); monitor open/close are submitted via
  `run_coroutine_threadsafe` and values are marshaled to the GUI through a
  queued `device_value_ready(object)` signal.  Committing a new
  `device:variable` selection (dropdown pick / Enter / focus leave) closes
  the previous monitor first, and a generation guard drops straggler
  callbacks from retired monitors.  Floats render with 6 significant
  digits, strings as-is (scalar-only, per the package charter).
- Set: `GatewaySetpointPut` (the one blessed gateway `:SP` put primitive,
  `wire_value` coercion) on the `{...}:SP` PV — put-completion rides
  GEECS's native blocking set.  The blocking put is dispatched to a
  short-lived daemon thread (never the GUI thread); success/failure lands
  in the status bar + log via a queued `device_set_finished(bool, str)`
  signal, and the Set button is disabled while a put is in flight, when
  the selection is not a valid `device:variable`, or when the field is
  empty.
- The experiment combo re-points the readback monitor (the PV is
  experiment-prefixed); `closeEvent` unsubscribes the monitor and
  disconnects both queued signals without joining any thread.

## [0.2.0] - 2026-07-11

### Added

- Live health chips (R1): `GatewayTiledDbHealth` replaces the all-unknown
  stub — a CA read of `{experiment}:CAGateway:HEARTBEAT` (OK; WARN when
  `DEVICES_CONNECTED == 0`; DOWN on failure; UNKNOWN with no experiment
  selected), an HTTP GET of the configured `[tiled] uri` (2xx → OK), and a
  cheap `GeecsDb` query (OK / DOWN).  Each check is guarded with a short
  timeout; `poll()` never raises and lazily imports `aioca` / `httpx` /
  `GeecsDb` so the module stays import-safe offline.  `main.py` injects the
  real probe; `StubHealth` remains the offline/test default.
- Background health polling: a GUI-thread `QTimer` dispatches each blocking
  `poll()` to a short-lived daemon thread (`HealthPoller`); the result is
  marshaled back to the R1 chips via a queued signal, so a slow probe (e.g.
  over VPN) never blocks the event loop.  The experiment combo pushes the
  selected experiment into the probe so the next poll targets the right
  gateway PV.  `closeEvent` stops the timer for clean teardown.
- Operator / pre-flight dialogs: `ScanDialogEvent` now renders a modal
  `QMessageBox` (`ScanEventsAdapter.dialog_requested` → `_on_operator_dialog`)
  with the request's continue/abort labels.  Abort sets `request.abort[0]`;
  either choice sets `request.response_event`, unblocking the engine's scan
  thread (which is waiting on it).  The signal is delivered queued so the
  modal always runs on the GUI thread while the engine thread stays blocked.

## [0.1.1] - 2026-07-10

### Added

- Visual treatment to match screen map v0.1: a packaged Qt stylesheet
  (`geecs_console/app/style.qss`, applied application-wide at window
  construction) implementing the approved screen-map palette and widget
  treatments — panel group boxes with small uppercase dim legends, primary
  blue Start / danger-outline Stop, white 1px-bordered inputs and lists,
  accent radios, thin green progress bar, dim monospace status bar and
  hints, dark monospace log tail.
- Health chips render as rounded pills with a per-status colored dot
  (grey unknown / green ok / amber warn / red down); `HealthStatus` gains
  `WARN`.  The Now panel's state text renders as a pill (colored dot +
  uppercase state word).
- Screen-map layout parity in the `.ui`: session bar in its own panel,
  save-set lists stacked vertically with Add/Remove between them, submit
  row in a panel with the `request → validate → submit` hint, 26/46/28
  column proportions, 10px outer margins with 8px gaps.

### Changed

- Spin boxes use `NoButtons` (plain fields per the screen map; the native
  macOS button geometry drew outside the styled border).  Values adjust
  by typing, arrow keys, or wheel.
- The experiment combo shows a `select experiment…` placeholder when the
  configs repo lists experiments but none is selected yet (it previously
  rendered blank until the dropdown was opened).

## [0.1.0] - 2026-07-10

### Added

- Package scaffold: the greenfield PySide6 operator console
  (`geecs_console`), per the cutover-strategy decisions
  (`Planning/cutover_strategy/00_overview.md`) and the approved screen map.
- Main window implementing regions R1–R7 from a hand-authored Qt Designer
  `.ui` file: session bar (experiment / rep rate / trigger profile+variant /
  health chips), save-set lists with union preview, scan form (mode radios,
  axis rows, shots per step, acquisition combo, live shot count with the
  `MAXIMUM_SCAN_SIZE = 1e6` guard, description), presets row (stub), submit
  row, "Now" panel (state pill, progress bar, scan-number label with 10 s
  expiry, log tail), device panel (backend stubbed).
- `request_builder.build_scan_request`: the pure
  `ConsoleFormState → geecs_schemas.ScanRequest` mapping (noscan / 1D /
  grid / background; optimization refused until an editor exists).
- `submission.Submitter` protocol + lazy `make_bluesky_submitter` factory
  (the window opens without the `ca` extra or any network).
- `services.configs.ConsoleConfigs`: offline-safe configs-repo listings and
  resolver-backed union preview / trigger variants.
- `services.health`: `HealthProbe` protocol + all-unknown `StubHealth`.
- `events_adapter.ScanEventsAdapter`: engine `on_event` stream → Qt signals.
- Hermetic test suite (pytest-qt, offscreen) including a guard test that
  `geecs_python_api` is never imported.
