# Changelog

All notable changes to `geecs-bluesky` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.6.0] - 2026-06-13

### Changed

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

- Strict STANDARD uses `SCAN`/`STANDBY` arm/disarm with a free-running
  trigger + `trigger_and_read` — which is correct for the deployed
  amplitude-gated hardware (DG645 Ch AB amplitude gates the Undulator gas
  jet; `SINGLESHOT` fires at jet-off level and is only for t0/timing, not
  data). `geecs_single_shot` (plan-owned fire-per-shot) is built and tested
  but intentionally left unwired until a full-power single-shot config exists
  or it is used on a non-gated experiment (e.g. Thomson). See
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
