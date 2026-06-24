# Changelog

All notable changes to `geecs-bluesky` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
