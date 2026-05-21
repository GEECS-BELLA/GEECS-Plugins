# Refactor Status — GEECS-Scanner

**Read this at the start of every session before touching scanner code.**
**Update it as the last commit of every session.**

---

## Current direction: Decompose Phase

The support structures are now in place:
- Typed `ScanEvent` stream with defensive `_emit()` wrapper
- `DeviceCommandExecutor` as the single device-command policy point
- `ScanState` machine with `_set_state()` on `ScanManager`
- `pyqtSignal(object)` bridge from scan thread to Qt main thread

The next phase is to **use those structures to remove complexity**, not add
more. Breaking changes are expected and welcome. The right question before
adding anything is: does this make a class's purpose describable in one sentence
without using "and"?

**Tests will shift.** Tests that exercise behavior (what events fire, what
sequence happens) should survive refactors. Tests that reach into private methods
via `object.__new__` will need updating when those methods move to new classes —
that is acceptable and expected.

---

## Decompose steps (in order)

### D1 — Behavioral tests for DataLogger  `[x]`

`DataLogger` has zero test coverage and is the largest untested component.
Writing behavioral tests before decomposing it is the safety net that makes D2
non-risky.

**What to test:**
- `_log_device_data()` is called and a log entry is created for the right shot
- `FileMover` receives tasks when device files are present
- Standby mode detection suppresses logging correctly
- `get_current_shot()` returns the right value after N log calls

**First action:** Read `DataLogger.__init__` and `_log_device_data()` to find
the minimal fake device / fake filesystem setup needed to exercise these paths.
Use the `FakeScanDevice` pattern from `tests/engine/conftest.py`.

**Done when:** `pytest tests/engine/test_data_logger.py` covers the four paths
above without network access.

---

### D2 — Extract FileMover out of DataLogger  `[x]`

`FileMover` is ~400 lines of self-contained worker-thread + retry logic that
has no business living inside `DataLogger`. The class boundary is already
obvious: `DataLogger` creates tasks, `FileMover` executes them.

**What changes:**
- Move `FileMover` class to `geecs_scanner/engine/file_mover.py`
- `DataLogger.__init__` receives a `FileMover` instance (injected, not created)
- `ScanManager` (which currently calls `self.data_logger.file_mover.shutdown()`)
  calls `self.file_mover.shutdown()` directly
- No logic changes — only movement and injection

**First action:** Grep for every site that accesses `data_logger.file_mover.*`
in `scan_manager.py` — these become direct `file_mover.*` calls after extraction.

**Done when:** `DataLogger` constructor no longer creates a `FileMover`; all
existing tests still pass; `wc -l data_logger.py` is under 750.

**Depends on:** D1 (tests must catch regressions).

---

### D3 — Extract ScanLifecycleStateMachine  `[x]`

`ScanManager._set_state()` and `current_state` already exist — extracting
a `ScanLifecycleStateMachine` class is now mostly moving code to a new file.
This is the step that makes `ScanManager`'s remaining size drop sharply.

**What changes:**
- New class `geecs_scanner/engine/lifecycle.py`:
  `ScanLifecycleStateMachine(on_event)` owns `_state`, `_state_lock`, `_set_state()`,
  `current_state`, and optionally transition validation
- `ScanManager` holds `self._lifecycle = ScanLifecycleStateMachine(self._on_event)`
  and delegates: `self._set_state(s)` → `self._lifecycle.set_state(s)`,
  `self.current_state` → `self._lifecycle.current_state`
- `pause_scan_event` and `stop_scanning_thread_event` (threading.Events) remain
  as implementation details of `ScanManager` — they are coordination primitives,
  not public state

**First action:** Identify every external caller of `current_state` and
`_set_state()` to understand the public surface of the extracted class.

**Done when:** `ScanManager` no longer owns `_state` or `_state_lock`; the
`TestScanManagerStateMachine` tests in `test_event_emission.py` are updated to
point at `ScanLifecycleStateMachine` directly.

**Depends on:** D1 and D2 (want `ScanManager` stabilized before this move).

---

### D4 — Make ScanManager._start_scan() readable as a recipe  `[x]`

`_start_scan()` is ~150 lines mixing Phase 1 / Phase 2 orchestration,
try/except/finally scaffolding, and inline bookkeeping. The goal is not
necessarily a new class — just named phase methods that read as a script.

**What changes:**
- Extract `_phase1_pre_scan()` — trigger off, data folder setup
- Extract `_phase2_scan()` — pre-logging setup, DataLogger.start, scan loop, stop
- Extract `_phase3_teardown()` — restore state, close devices, set DONE/ABORTED
- `_start_scan()` becomes ~20 lines: call the three phases in sequence with
  clear exception handling at each boundary

**First action:** Read `_start_scan()` end-to-end and draw the phase boundaries
on paper before touching the code.

**Done when:** `_start_scan()` fits on one screen; each phase method has a
one-sentence purpose; exception handling is at phase boundaries, not inline.

---

### D5 — Drop the 200ms polling timer  `[x]`

Block 7 added event handlers but left the 200ms `QTimer` running. The timer
currently only covers the `RunControl is None` guard and multiscan/action-library
state — neither of which needs a periodic poll. Removing the timer completes
the event-driven migration.

**What changes:**
- The `RunControl is None` state is handled once on construction and on
  `reinitialize_run_control()` — not by a timer
- Multiscan and action-library mode changes trigger `update_gui_status()` via
  direct calls from `open_multiscanner()` / `open_action_library()` / exit paths
  (already the case — the timer is redundant)
- `self.timer.stop()` + remove `self.timer = QTimer(self)` from `__init__`

**First action:** Confirm that `update_gui_status()` is only called from the
timer and from `open_multiscanner()` / `open_action_library()` / their exit
handlers. If so, the timer can be removed without any other changes.

**Done when:** No periodic QTimer driving `update_gui_status()`; the GUI
responds purely to events and to direct method calls at mode transitions.

---

## Bold Refactor — Completed 2026-05-08

### Phase 1 — Delete dead code and proxy methods `[x]`

Deleted `estimate_current_completion()` (never called) and
`trigger_off()` / `trigger_on()` (pure proxies to `TriggerController`) from
`ScanManager`.  Single internal call site inlined.

### Phase 2 — Split `initialize_and_start_scan()` `[x]`

176-line method → `_collect_ui_scan_config()`, `_build_exec_config()`,
~40-line orchestrator.  Fixed latent `NameError` bug.

### Phase 3 — Create `AppController` `[x]`

New `geecs_scanner/app/app_controller.py`.  Owns RunControl lifecycle, database
access, scan control, and UI flags.  Window exposes `@property RunControl` for
backward compatibility.  Config-file write logic extracted.

**Next:** Lab verification (NOSCAN + 1D scan) before merging to master.

---

## Deferred (lower priority, independent)

### Block 4 — Engine runs without Qt  `[ ]`

After D1–D5, the engine is nearly headless already. This block formalizes it:
remove any remaining Qt imports from `geecs_scanner/engine/`, add a CI check
that runs `pytest tests/engine/` in an environment without PyQt5.

**Depends on:** D5 (polling timer must be gone first; everything else is clean).

---

### Block 2 — Config cleanup  `[ ]`

Move `ScanConfig` from `geecs_data_utils` into `geecs_scanner.engine.models`.
Independent of the decompose work — can happen in any session.

**First action:** `grep -r "from geecs_data_utils import.*ScanConfig"` across
the monorepo to understand migration scope.

---

## Completed

| Block | What | PR / branch |
|---|---|---|
| 0 | Tactical hardening | #313, #314 |
| 1 | Codebase audit | — |
| PR #366 | Engine reorg, `ScanExecutionConfig`, `TriggerController` | #366 |
| 6 | `DeviceCommandExecutor` — single command policy | #367 |
| 3 | Typed `ScanEvent` stream, `ScanState` enum | #370 |
| 5 | State machine: `_set_state`, `PAUSED_ON_ERROR` | scanner-finish-line |
| 7 | Event-driven GUI: `pyqtSignal` bridge, remove polling helpers | scanner-finish-line |
