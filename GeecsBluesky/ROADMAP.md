# GeecsBluesky — Roadmap

Current status (2026-06-08): MVP complete and hardware-verified, but not yet
integrated with the current production scanner surface on `master`.
`BlueskyScanner` runs STANDARD and NOSCAN scans end-to-end against real lab
hardware with per-step DG645 shot control when constructed directly with
`shot_control_information`, and persists Bluesky documents to Tiled.  The
`GEECS-Scanner-GUI` codebase has moved on since this package landed: it now has
typed scan events, a `DeviceCommandExecutor`, a refactored file finalization
path, and unified optimizer evaluators.  Treat the notes below as the current
bridge work needed after fast-forwarding this worktree to `origin/master`.

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
- [x] **Per-device image saving** — `_scan_with_saving` plan wrapper sets
      `localsavingpath` and `save="on"` concurrently before scan; finalise
      wrapper sets `save="off"` even on abort.
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

---

## Open Questions / Strategic Decisions

These require discussion before implementation — see brainstorm notes.

### Data pipeline transition

`ScanAnalysis` and everything downstream reads s-files and TDMS written by the
legacy `ScanManager`.  `BlueskyScanner` writes to Tiled only.

Options:
- **Tiled reader for ScanAnalysis** — add a Tiled-backed data source alongside
  the existing file reader.  This now needs to respect the post-#412 analyzer
  contract: ImageAnalysis emits bare scalar keys, ScanAnalysis owns
  prefix/suffix naming, and analysis code must never create missing scan
  folders.
- **BlueskyScanner also writes s-files** — transitional shim; buys time for
  ScanAnalysis migration.  If chosen, it must match the current scanner output
  convention (`ScanDataScan{NNN}.txt`, `ScanInfoScan{NNN}.ini`, per-device file
  folders under `scans/Scan{NNN}/`).
- **Accept cold-turkey** — new scans from Bluesky path only queryable via Tiled;
  old analysis tools stop working for new data until ported.

No decision yet.  Defer until BlueskyScanner is actually the production path.

### GUI / RunControl integration

`RunControl` has `use_bluesky=True` flag but `shot_control_information` is not
yet threaded through — BlueskyScanner gets no shot control in GUI mode.  The
current Bluesky branch also ignores the `on_event` callback in Bluesky mode, so
the scanner GUI's newer typed event stream does not yet apply to
`BlueskyScanner`.

The legacy scanner refactor has now landed in `master`; this is no longer
blocked on a parallel worktree.  The next implementation pass should update
`RunControl(use_bluesky=True)` to pass the timing YAML and decide whether
Bluesky documents should be translated into `ScanEvent` callbacks for GUI and
headless consumers.

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
- **Scan number error visibility** — `_claim_scan_number` should log `ERROR`
  when the scan folder cannot be created.  It currently logs `WARNING` and
  returns `(None, None)`.
- **Pre/post-scan actions** — action sequences (set variable, wait, check
  condition) are not yet supported.  `SaveDeviceConfig` carries `setup_action`
  and `closeout_action`, and the legacy scanner executes them through
  `ActionManager` / `DeviceCommandExecutor`; Bluesky currently drops them.
- **Event stream bridge** — decide whether Bluesky start/event/stop documents
  should be surfaced directly, translated into `ScanEvent`, or both.
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
