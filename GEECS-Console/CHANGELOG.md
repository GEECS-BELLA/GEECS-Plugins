# Changelog

All notable changes to GEECS-Console are documented here.  Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
semantic.

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
