# Changelog

All notable changes to GEECS-Console are documented here.  Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
semantic.

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
