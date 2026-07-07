# Changelog

All notable changes to `geecs-bluesky` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.19.1] - 2026-07-06

### Fixed

- **`CaTriggerable` cold-cache race: the first shot after connect can no
  longer be lost or mis-baselined** — with a cold monitor cache
  (`_last_acq is None`, i.e. no acquisition delivered since subscribe), the
  cold path of `_wait_for_shot` took a CA-get baseline inside the returned
  coroutine and then drained the shot queue. A strict-mode shot fired
  immediately after `trigger()` could land before the coroutine ran and be
  (a) drained away, or (b) become the baseline itself — either way the
  single shot timed out (`GeecsTriggerTimeoutError`). Because the monitor
  callback filters non-positive placeholders, any positive queued value on a
  cold cache is a genuinely new acquisition: the cold path now treats an
  already-queued value as the shot, normalizes the gateway's `0.0`
  pre-acquisition placeholder baseline to `None`, and never drains the
  queue. Warm-path behavior (synchronous drain + baseline in `trigger()`)
  is unchanged.

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
