# GeecsBluesky — Roadmap

> **Historical note (2026-07-04):** the direct UDP/TCP device backend this
> roadmap tracked (`GeecsSignalBackend`, `GeecsDevice`, `GeecsTriggerable`,
> `GeecsMotor`, `GeecsGenericDetector`, `GeecsTimestampedReadable`, …) was
> **completed, verified, and then deleted** after the CA-gateway backend
> reached live parity. Devices are now CA-backed via GeecsCAGateway; the
> checked items below are kept as the historical record of that path.

> **Historical note (2026-07-16):** the `ScanExecutionConfig` /
> `exec_config` submission API and the RunControl integration described
> below were likewise completed, verified, and then **deleted** (G3):
> `reinitialize` accepts only a `geecs_schemas.ScanRequest`, and the
> consumer is GEECS-Console's `Submitter` seam. See `CLAUDE.md` for the
> current architecture; the items below are the historical record.


Current status (2026-06-14): the **two-acquisition-mode architecture is complete
and hardware-verified** (GeecsBluesky 0.8.0, branch
`geecs-bluesky-acquisition-modes`).  `BlueskyScanner` runs STANDARD and
statistics (NOSCAN) scans from the `GEECS-Scanner-GUI` (`use_bluesky=True`) in
both `free_run_time_sync` and `strict_shot_control` modes — including true
plan-owned single-shot (DG645 `ARMED` state) — with DG645 shot control, Tiled
persistence, scan numbering, and native file saving.  Both modes write one
versioned event schema (`EVENT_SCHEMA.md`).  Remaining work (below) is features /
tuning / data-pipeline, not architecture; see
`Planning/acquisition_modes/00_overview.md` for the design of record.

---

## Completed

- [x] **TCP-backed signal cache** — coherent per-shot reads; one shared TCP
      subscriber per device populates `_shot_cache`; `get_value()` reads from
      cache with UDP fallback.
- [x] **GeecsTriggerable** — event-driven trigger via `asyncio.Queue`; waits
      for `acq_timestamp` to advance; no polling.
- [x] **GeecsMotor / GeecsSettable** — generic `Movable`; UDP polling for
      position convergence; `GeecsMotor` inherits `GeecsSettable`.
- [x] **GeecsGenericDetector** — dynamically-signalled detector from a
      `variable_list`; `localsavingpath` / `save` signals when
      `save_nonscalar_data=True`.
- [x] **BlueskyScanner bridge** — ScanManager-compatible API; STANDARD and
      NOSCAN modes; `dialog_queue` / `restore_failures` shims for GUI compat.
- [x] **ScanExecutionConfig API** — `reinitialize(exec_config)` accepts the
      validated `ScanExecutionConfig` Pydantic model produced by the GUI; duck-
      typed at runtime so the hardware test works without importing
      `geecs_scanner`.  `shots_per_step` derived from `rep_rate_hz × wait_time`.
- [x] **Shot control — per-step arm/disarm** — `_UdpSetter` + `_set_trigger_state`;
      DG645 armed to SCAN after each motor move, disarmed to STANDBY after shots.
      `bpp.finalize_wrapper` guarantees disarm on abort.  `geecs_step_scan` has
      `arm_trigger` / `disarm_trigger` parameters.
- [x] **Scan numbering** — `ScanPaths.get_next_scan_tag()` claimed before
      detectors are built so save paths are known at device-build time.
- [x] **Per-device image saving** — sets `localsavingpath` and `save="on"`
      concurrently before scan; finalise wrapper sets `save="off"` even on
      abort.  (Now part of `geecs_run_wrapper`; the device-side signals live in
      the `NonScalarSaveSupport` mixin.)
- [x] **Non-scalar save-path event metadata** — `save_nonscalar_data=True`
      detectors emit per-event device `acq_timestamp` and configured save
      directory. `BlueskyScanner` also includes per-device save paths in
      run-start metadata.  File names remain hardware-native; downstream readers
      should join by `acq_timestamp` rather than a synthetic shot counter.
- [x] **Tiled integration** — `TiledWriter` subscribed to RunEngine on init;
      reads URI + API key from `~/.config/geecs_python_api/config.ini`; skips
      silently if server is unreachable.
- [x] **FakeGeecsServer** — in-process fake device server using real wire
      protocol; `device.fire_shot()` advances `acq_timestamp` and broadcasts
      TCP push; used for all unit tests.
- [x] **Unit tests** — `tests/test_shot_control.py` (10 tests covering
      `_UdpSetter`, `_set_trigger_state`, arm/disarm ordering; no hardware).
- [x] **Hardware integration test** — `test_bluesky_scanner.py`; 3 scenarios,
      6 checks; all pass on real lab hardware (UC_TopView, U_ESP_JetXYZ,
      U_DG645_ShotControl).

### Acquisition modes (0.4.0 → 0.8.0)

- [x] **Two acquisition modes** — `free_run_time_sync` and
      `strict_shot_control`, env-selected (`GEECS_BLUESKY_ACQUISITION_MODE`),
      dispatched by `BlueskyScanner`; one shared event schema (`EVENT_SCHEMA.md`).
- [x] **Incremental shot IDs + coordinated t0 sync** — `ShotIdTracker`,
      `ShotIdSupport`, `geecs_t0_sync`; cross-device matching by shot-id equality.
- [x] **Free-run plan** — `geecs_free_run_step_scan`: reference pacemaker,
      `GeecsTimestampedReadable` contributors with offset/valid + grace wait,
      quiesce-to-`OFF` before t0 sync, end-of-scan tail flush.
- [x] **Strict plan-owned single-shot** — `geecs_single_shot` +
      `geecs_confirm_quiescent`; requires a reachable shot-control device with
      an `ARMED` state.
- [x] **NOSCAN unified** as a motorless step scan (works in both modes).
- [x] **`ShotControlConfig` / `ShotControlState`** — validated shot-control YAML.
- [x] **`geecs_run_wrapper` + `claim_scan_number`** — reusable scan-number
      metadata (incl. `scan_id` = GEECS scan number) + native file saving.
- [x] **`EVENT_SCHEMA.md`** — canonical v1 data contract.
- [x] **GUI integration** — `RunControl(use_bluesky=True)` threads the
      shot-control YAML and `on_event` (lifecycle events emitted).
- [x] **`ARMED` config state** — added to the experiment shot-control YAMLs in
      `geecs-plugins-configs` (external single-shot laser-on, internal laser-off).

---

## Open Questions / Strategic Decisions

These require discussion before implementation — see brainstorm notes.

### Data pipeline transition

`ScanAnalysis` and everything downstream reads s-files and TDMS written by the
legacy `ScanManager`.  `BlueskyScanner` writes to Tiled, and scalar files are
exported from Tiled best-effort after each scan
(`geecs_data_utils.tiled_export.write_scalar_files_from_tiled` →
`ScanDataScan{NNN}.txt` + the `s{NNN}.txt` s-file); TDMS is not produced.

Options:
- **Tiled reader for ScanAnalysis** — add a Tiled-backed data source alongside
  the existing file reader.  This now needs to respect the post-#412 analyzer
  contract: ImageAnalysis emits bare scalar keys, ScanAnalysis owns
  prefix/suffix naming, and analysis code must never create missing scan
  folders.
- **BlueskyScanner also writes s-files** — **taken (transitional shim)**: the
  post-scan Tiled export above is live on all engine paths; buys time for the
  ScanAnalysis migration.
- **Accept cold-turkey** — new scans from Bluesky path only queryable via Tiled;
  old analysis tools stop working for new data until ported.

BlueskyScanner is now the production path (the legacy backend died with
G1/G3), and the s-file shim bridges the gap.  The strategic question that
remains open is the end state: does ScanAnalysis grow a Tiled reader, or
keep reading exported s-files long-term?

### GUI / RunControl integration — mostly done

`RunControl(use_bluesky=True)` now loads the selected shot-control YAML and
passes it as `shot_control_information`, and passes the `on_event` callback;
`BlueskyScanner` emits `ScanLifecycleEvent`s through it.  Acquisition mode is
chosen by the `GEECS_BLUESKY_ACQUISITION_MODE` env var (no GUI toggle —
intentional while bluesky is experimental).

Remaining: only **lifecycle** events are emitted — per-shot/step Bluesky
documents are not translated into the richer `ScanStepEvent` /
`DeviceCommandEvent` stream (decide if that's even wanted).  And the
arm/disarm/quiesce/fire shot-control callables still live inside
`BlueskyScanner`; extracting a reusable `ShotController` would give notebook
workflows full parity.

### Optimization scans

The scanner optimizer surface changed significantly after this package landed.
`MultiDeviceScanEvaluator` and `ScalarLogEvaluator` were folded into the unified
`BaseEvaluator`, which now consumes diagnostic analyzers and direct s-file
scalar columns through one hook surface.  Bluesky + Xopt may still be the right
long-term shape, but any port should start from the current `BaseEvaluator` /
`BaseOptimizer` contract rather than the older evaluator split.

---

## Near-Term Implementation Work

Small, well-scoped items that don't require strategic decisions:

- **Background scan mode** — BACKGROUND adds a flag to scan metadata; trivial
  addition as a third branch in `_run_scan`.  The GUI and `ScanMode` enum already
  expose `background`, but `BlueskyScanner._run_scan()` still supports only
  `standard` and `noscan`.
- **Scan number error visibility** — `claim_scan_number`
  (`plans/run_wrapper.py`) should log `ERROR` when the scan folder cannot be
  created.  It currently logs `WARNING` and returns `(None, None)`.
- **Pre/post-scan actions** — action sequences (set variable, wait, check
  condition) are not yet supported.  `SaveDeviceConfig` carries `setup_action`
  and `closeout_action`, and the legacy scanner executes them through
  `ActionManager` / `DeviceCommandExecutor`; Bluesky currently drops them.
- **Event stream bridge** — lifecycle `ScanEvent`s are emitted via `on_event`;
  decide whether per-shot/step Bluesky documents should also be translated into
  `ScanStepEvent` / `DeviceCommandEvent`, or left as raw Bluesky docs.
- **Reusable `ShotController`** — extract the arm/disarm/quiesce/fire plan-stub
  callables out of `BlueskyScanner` so notebook workflows get shot-control
  parity (jet gating / single-shot) with the GUI path.
- **Requested rep-rate throttling** — in strict single-shot, fire software
  triggers slower than the external rate (gas-jet economy); a per-shot
  inter-fire delay.  Free-run could subsample every Nth reference shot.
- **Windows install verification** — `poetry install` in `GeecsBluesky/` should
  be rechecked on Windows after the monorepo-wide Python 3.11 and
  `mysql-connector-python` updates.

---

## Longer-Term

- **Optimization scans** — wrap Xopt in a Bluesky plan; decisions pending on
  whether to port or deprecate the existing optimizer.
- **Composite variables** — two motors moving together; implement as a custom
  plan calling `bps.mv(m1, p1, m2, p2)`.
- **Live plotting** — `BestEffortCallback` or custom Bokeh subscriber.
- **Separate repo / PyPI** — once stable, publish `geecs-bluesky` to PyPI for
  use at other GEECS-based labs.
