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
  Grid-only), shots per step, acquisition combo (free_run default, strict —
  the request declares intent), live shot count with the
  `MAXIMUM_SCAN_SIZE = 1e6` guard, description.
- **R4 presets** — combo + Apply + Save-as.  A preset IS a saved
  `ScanRequest`; persistence not wired yet.
- **R5 submit row** — Stop (danger) + Start (primary).  Start requires: not
  scanning, ≥1 selected save set, valid shot count within the guard, mode
  not Optimization.
- **R6 now panel** — state pill, progress bar, "Scan NNN" with 10 s expiry
  to "(previous)", compact log tail.
- **R7 device panel** — device:variable combo, readback label, set field +
  button.  **Backend stubbed** — the real one is gateway PVs (CA monitor on
  the readback, put to `:SP` riding GEECS's native blocking set), never
  `geecs_python_api`'s ScanDevice.  Scalar-only at birth (composites arrive
  with the pseudo-variable runtime, if ever).

## Architecture rules

- **Never import `geecs_python_api`** — pinned by
  `tests/test_no_geecs_python_api.py` (source grep + sys.modules check).
  DB autocompletes go through `GeecsDb` (`geecs_ca_gateway`, an allowed
  transitive of `geecs-bluesky`); errors from the bluesky/gateway tree.
- **The one submission shape is `geecs_schemas.ScanRequest`.**
  `request_builder.build_scan_request` is the only place form state becomes
  a request; keep it a pure function, keep widgets out of it.
- **The window depends on seams, not implementations**: `Submitter`
  (protocol over `BlueskyScanner`'s four ScanManager-compatible methods),
  `ConsoleConfigs`, `HealthProbe`, `ScanEventsAdapter`.  All constructor-
  injectable; every test drives the window with fakes.
- **Offline-first**: the window must open and run with zero network and
  zero configs.  `geecs_bluesky` is imported lazily (function-level) in
  `submission.py` and `services/configs.py` — it pulls `aioca` at package
  import, so a module-level import would couple opening the window to the
  `ca` extra.  `events_adapter` dispatches on event class *names* for the
  same reason (and so hermetic fakes work).
- PySide6 only (LGPL, agent-editable `.ui` XML).  Never PyQt.
- The `.ui` is hand-authored XML loaded at runtime via `QUiLoader` — no
  generated `*_ui.py` files to keep in sync.

## Implemented seams

- **Health chips (R1)** are live via `GatewayTiledDbHealth` (real probe) or
  `StubHealth` (all-unknown offline/test default).  The real probe runs three
  guarded checks — CA read of `{experiment}:CAGateway:HEARTBEAT` (OK; WARN
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
- **Operator / pre-flight dialogs**: a `ScanDialogEvent` is rendered as a
  modal `QMessageBox`.  `ScanEventsAdapter.handle` (on the engine/scan thread)
  emits `dialog_requested(request)`; the queued signal delivers
  `_on_operator_dialog` on the GUI thread (where the modal must live) while the
  engine thread blocks on `request.response_event.wait()`.  Abort sets
  `request.abort[0] = True`; either choice calls `request.response_event.set()`
  to unblock the engine.  Duck-typed on the `DialogRequest` attributes — no
  `geecs_bluesky` import.

## Stubbed seams (intentional, wire later)

- Device panel backend (see R7 above).
- Presets persistence (save/load ScanRequest YAML; a preset dir in the
  configs repo is the natural home).
- Optimization mode: radio exists, submission refused with a clear error
  until an `OptimizationSpec` editor exists.
- Scan-number source: `set_scan_number` + the 10 s expiry exist, but no
  event carries the claimed number yet (engine-side follow-up).
- `ConsoleConfigs._scan_variable_names` reaches into the resolver's private
  `_scan_variables_catalog()` — promote a public "list variable names"
  method on `ConfigsRepoResolver` when next touching geecs-bluesky.

## Testing

`QT_QPA_PLATFORM=offscreen poetry run pytest -q` — hermetic, pytest-qt,
`qt_api = "pyside6"` pinned in pyproject.  The request-builder tests are the
important ones: they validate the exact `ScanRequest` shapes against the
real schema.
