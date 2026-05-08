# GeecsBluesky — Roadmap

Current status (2026-05-08): MVP complete and hardware-verified.
`BlueskyScanner` runs STANDARD and NOSCAN scans end-to-end against real lab
hardware with per-step DG645 shot control and Tiled data persistence.
All unit tests pass without hardware; hardware integration test passes all checks.

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
  the existing file reader; cold-turkey cutover deferred.
- **BlueskyScanner also writes s-files** — transitional shim; buys time for
  ScanAnalysis migration; acknowledged as unpleasant.
- **Accept cold-turkey** — new scans from Bluesky path only queryable via Tiled;
  old analysis tools stop working for new data until ported.

No decision yet.  Defer until BlueskyScanner is actually the production path.

### GUI / RunControl integration

`RunControl` has `use_bluesky=True` flag but `shot_control_information` is not
yet threaded through — BlueskyScanner gets no shot control in GUI mode.
Deferred until the parallel GUI/RunControl refactor lands in master.

### Optimization scans

Bluesky + Xopt may naturally handle what `base_optimizer.py` does today.  The
existing optimization execution could become deprecated rather than ported.
Requires a deeper dive.

---

## Near-Term Implementation Work

Small, well-scoped items that don't require strategic decisions:

- **Background scan mode** — BACKGROUND adds a flag to scan metadata; trivial
  addition as a third branch in `_run_scan`.
- **Scan number error visibility** — `_claim_scan_number` should log `ERROR`
  (not silently return `None`) when the scan folder cannot be created.
- **Pre/post-scan actions** — action sequences (set variable, wait, check
  condition) are not yet supported; straightforward once the action model is
  understood.
- **Windows install verification** — `poetry install` in `GeecsBluesky/` has
  not been verified on Windows; `mysql-connector-python` platform behaviour
  worth checking before lab deployment.

---

## Longer-Term

- **Optimization scans** — wrap Xopt in a Bluesky plan; decisions pending on
  whether to port or deprecate the existing optimizer.
- **Composite variables** — two motors moving together; implement as a custom
  plan calling `bps.mv(m1, p1, m2, p2)`.
- **Live plotting** — `BestEffortCallback` or custom Bokeh subscriber.
- **Separate repo / PyPI** — once stable, publish `geecs-bluesky` to PyPI for
  use at other GEECS-based labs.
