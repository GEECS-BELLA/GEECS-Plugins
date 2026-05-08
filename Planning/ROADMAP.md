# GEECS-Plugins Architectural Roadmap

## Vision

The goal is a scan system that is:

- **Reliable** — hardware errors are surfaced to the operator with the opportunity to
  intervene, not silently swallowed or allowed to crash data collection
- **Comprehensible** — the scan lifecycle has an explicit, inspectable state; errors
  have clear origins; config is validated before a scan starts
- **Testable** — the scan engine runs without a GUI; components can be tested in
  isolation with mock devices
- **Coherent** — one object is responsible for sending commands to devices during a
  scan; one model describes what a scan is; one event stream describes what is
  happening

Breaking changes are acceptable. Better organisation and comprehensibility are
worth the cost of migration.

---

## Current State (as of May 2026)

### What works well
- `GeecsDevice` UDP/TCP communication layer is solid and well-typed
- Exception hierarchy (`GeecsDeviceExeTimeout`, `GeecsDeviceCommandRejected`,
  `GeecsDeviceCommandFailed`) is well-defined
- `ScanAnalysis` task queue and analyzer framework are clean and well-organised
- `geecs_data_utils` provides a good home for scan data structures (`ScanTag`,
  `ScanPaths`, `ScanData`)
- Pre-commit hooks, ruff formatting, and pydocstyle are enforced

### Core structural problems
- `scan_manager.py` is a ~1000-line monolith mixing scan control, device
  orchestration, file I/O, and state management
- `device.set()` is called directly from at least four different objects with no
  consistent error handling policy
- The scan's current state is implicit — inferred from scattered boolean flags
  (`_scanning_active`, `_pause_event`, `_stop_event`)
- GUI code (`app/`) is entangled with engine code; the engine cannot run without Qt
- Config is a mix of validated Pydantic models, raw dicts, and INI files depending
  on where you look
- `ScanConfig` is a plain dataclass in `geecs_data_utils` — a data utility package
  should not own scan execution config
- `save_hiatus` is a deprecated, unused code path that adds noise
- Trigger control (`_set_trigger`, `trigger_on`, `trigger_off`) is scattered across
  `scan_manager` and `scan_executor` with no central owner

---

## Completed Work

### Block 0 — Tactical hardening (PRs #313, #314)
Stabilised the existing system enough to refactor safely.

- `stop_scan()` reordered: scalar files written first, then device teardown
- `_stop_saving_devices()` parallelised via `ThreadPoolExecutor`
- `_clear_existing_devices()` parallelised via `ThreadPoolExecutor`
- `GeecsDeviceCommandRejected` caught in `action_manager._set_device()`
- `dequeue_command()` catches rejected commands so daemon threads never produce
  unhandled "Exception in thread" output
- `_process_command()` guards against `dev_udp is None` race with `device.close()`
- `exec_timeout` default reduced from 30s → 10s
- Deprecated `save_data` flag removed

---

## Roadmap

### PR #366 — Engine reorganisation + config model (branch: `worktree-refactor-blocks-2-4`)

Structural groundwork enabling Block 6 and the log-triage work.

- `data_acquisition/` renamed to `engine/`; `schemas/` consolidated into `engine/models/`
- `ScanExecutionConfig` — new top-level Pydantic model (scan config + options + save
  config) produced by the GUI and consumed by `ScanManager`.  Replaces the raw dict
  handoff that previously crossed the GUI ↔ engine boundary.
- `ScanOptions` and `SaveDeviceConfig` models completed and wired through `reinitialize()`
- `TriggerController` extracted from `ScanManager` — owns all trigger state transitions
  (`OFF / STANDBY / SCAN / SINGLESHOT`)
- `DialogRequest` dataclass for thread-safe GUI escalation (worker thread posts;
  Qt timer drains)
- `geecs_scanner/utils/config_utils.py` created from `data_acquisition/utils.py`
- Per-scan `scan.log` file: `pre_logging_setup()` moved inside `scan_log()` context
  so device subscription and action execution are captured in the scan file

### PR #??? — DeviceCommandExecutor + log-triage prep (branch: `feature/engine-refactor`)

Block 6 implementation and scan log improvements for the log-triage vision.

- `DeviceCommandExecutor` — single policy object for all `device.set()` / `device.get()`
  calls.  Per-error-type retry policy.  Injected into `ScanStepExecutor`,
  `ScanDataManager`, `ActionManager`, and `ActionControl`.
- Fixed `configure_device_save_paths` escalation bug (Abort was silently ignored)
- Fixed `None` return from `device.set()` crashing tolerance check when hardware
  raises in the UDP listener thread
- Scan log now shows: scan config summary at start, per-variable set commands at INFO,
  per-shot heartbeat — sufficient context for automated log triage without reading
  external files

### Block 1 — Full codebase audit ✓ (branch: `audit/block1-codebase-analysis`)

A thorough read of the entire codebase before any structural changes. Written up in
`AUDIT_BLOCK1.md`. Key questions returned to the user for decision at the end of
that document.

*Review gate: does the audit match your understanding? Are the structural problems
correctly identified?*

---

### Block 2 — Pydantic config models

**Goal:** Every config object is a validated Pydantic model. Config errors surface
at load time with a clear message, not mid-scan as a `KeyError`.

**What this includes:**
- Promote `ScanConfig` from a plain dataclass to a Pydantic model and move it from
  `geecs_data_utils` into the scanner engine (data-utils should contain data
  *navigation* structures, not scan *execution* config)
- Pydantic models for `ExperimentConfig` (experiment directory, INI paths, Drive IDs)
- Audit and complete existing device/action Pydantic schemas
- GUI becomes a form that produces a validated model; validation errors are shown to
  the user before a scan is allowed to start

**Why here:** Every downstream block depends on knowing exactly what a scan *is*
before it starts. Config model errors are also the most user-visible class of bugs
after hardware errors.

*Review gate: do the models capture all fields? Are the boundaries between
data-utils and scanner correct?*

---

### Block 3 — Scan event model

**Goal:** A single, typed event stream describes everything the scan engine does.
The GUI becomes a consumer of that stream.

**What this includes:**
- Define a `ScanEvent` hierarchy as plain Python dataclasses (no Qt):
  ```
  ScanEvent
  ├── ScanLifecycleEvent  (started, paused, resumed, completed, aborted)
  ├── ScanStepEvent       (step_started, step_completed)
  ├── DeviceCommandEvent  (sent, accepted, rejected, failed, timeout)
  └── ScanErrorEvent      (recoverable=True/False)
  ```
- `ScanManager` accepts an `on_event: Callable[[ScanEvent], None]` callback
- Wire the callback into the existing GUI as a thin adapter — emit events alongside
  existing behavior, don't replace it yet
- Identify every place the GUI currently reads engine state directly (these become
  the migration list for Block 4)

**Why here:** The event model is the contract between engine and GUI. Once it
exists, the operator intervention flow, live status display, and audit log all
become consumers of the same stream. The engine doesn't need to know about any of
them.

*Review gate: does the event set cover everything the GUI needs to know about?*

---

### Block 4 — Engine / GUI separation

**Goal:** The scan engine runs without Qt. The GUI is a thin subscriber.

**What this includes:**
- Move all core logic out of `geecs_scanner/app/` — `ScanManager`,
  `DeviceManager`, `ActionManager`, `DataLogger`, `ScanDataManager` are pure Python
- Remove Qt imports from engine code
- Consider restructuring the repo: an `apps/` directory at the root for GUIs
  (scanner GUI, config manager GUI), with the engine as a standalone importable
  package
- The GUI subscribes to the event stream from Block 3 and translates events to Qt
  signals; it sends user actions (start, stop, pause) to the engine via method calls

**Why here:** Without this boundary, the `DeviceCommandExecutor` (Block 6) can't
be cleanly injected, and the event model from Block 3 can't be properly implemented.
With this boundary, you can run a scan from a script or a test with no Qt
dependency.

*Review gate: run a real scan end-to-end, verify no regression.*

---

### Block 5 — Scan lifecycle state machine

**Goal:** The scan's state is explicit, inspectable, and transition-guarded.

**What this includes:**
- Define `ScanState` enum:
  ```
  IDLE → INITIALIZING → READY → RUNNING → PAUSED_ON_ERROR → STOPPING → DONE
                                         ↘ ABORTED
  ```
- All engine methods that mutate state assert the current state is valid for the
  requested transition; invalid transitions emit a fatal `ScanErrorEvent`
- Replace scattered boolean flags (`_scanning_active`, `_pause_event`,
  `_stop_event`) with state machine transitions
- `PAUSED_ON_ERROR` state is reachable (stub — the UI for it comes in Block 7)
- Remove `save_hiatus` entirely (deprecated, unused)
- `TriggerController` — already extracted in PR #366 (`engine/trigger_controller.py`),
  owning `OFF / STANDBY / SCAN / SINGLESHOT` transitions.  The state machine wires
  trigger transitions to `ScanState` changes.

**Why here:** The operator intervention flow (Block 7) *is* the
`PAUSED_ON_ERROR → RUNNING` transition. You can't implement it without the state
machine.

*Review gate: does the state diagram match how you think about scan lifecycle?*

---

### Block 6 — Device command policy layer ✓ (branch: `feature/engine-refactor`)

**Goal:** One object is responsible for all device commands during a scan.
`device.set()` is never called directly from scan logic.

**What this includes:**
- `DeviceCommandExecutor` in `geecs_scanner/engine/device_command_executor.py`
- Retry and escalation policy (implemented):
  - `GeecsDeviceExeTimeout` → escalate immediately (device hung; retrying makes it
    worse)
  - `GeecsDeviceCommandFailed` → escalate immediately (hardware error; retry won't help)
  - `GeecsDeviceCommandRejected` → retry N times, then escalate
- Escalation calls an injected `on_escalate(exc) -> bool` callback:
  - `True` = operator chose abort → sets `stop_event` on the scan thread
  - `False` = operator chose continue → returns `None` (best-effort)
- `ScanManager` instantiates one executor per scan and injects it into
  `ScanStepExecutor`, `ActionManager`, `ScanDataManager`.  Also wired into
  `ActionControl` for the standalone action-library path.
- All direct `device.set()` / `device.get()` calls in scan logic replaced with
  `cmd_executor.set()` / `cmd_executor.escalate()`

**Why this is not in `geecs-python-api`:** The API is a pure device control library
— it raises typed exceptions and knows nothing about retries, dialogs, or scan
context. The executor is a scanner-level policy object, not a library primitive.

**Note:** Two attributes (`on_device_error`) remain assigned on `ScanStepExecutor`
and `ScanDataManager` in `ScanManager.__init__` but are no longer read — dead code
from the pre-executor wiring.  Safe to remove in a follow-up cleanup (see Tech Debt
below).

*Status: complete. Validated on real hardware (NOSCAN, 1D scan, out-of-range
hardware error triggering escalation dialog, action library).*

---

### Block 7 — Operator intervention UI

**Goal:** When a device command fails, the operator sees exactly what failed and can
choose to fix it manually and continue, or abort.

**What this includes:**
- `PAUSED_ON_ERROR → RUNNING` transition implemented in the GUI
- Non-modal panel (or dialog) showing: device name, command, failure reason
- Options: **Retry**, **Continue** (skip this command), **Abort**
- Engine side: `ScanManager` blocks on a threading `Event`; the GUI sets it with the
  operator's choice
- This replaces the current `prompt_user_device_timeout` / `_prompt_user_quit_action`
  Qt-from-worker-thread approach (which works in practice but is technically unsafe
  — tracked as issue #312)
- All three failure modes (`ExeTimeout`, `CommandRejected`, `CommandFailed`) surface
  through the same UI

**Why here:** The executor (Block 6) defines the escalation contract; the state
machine (Block 5) provides the `PAUSED_ON_ERROR` state; the engine/GUI separation
(Block 4) enables safe signal-based dialog display. This block is the visible end
result but is the least code once the groundwork is in place.

*Review gate: does the UX match what operators actually need in the lab?*

---

### Block 8 — Automated log-triage agent (future)

**Goal:** A triage agent reads a per-scan `scan.log` and returns a structured
diagnosis — what happened, which device failed, what error was raised, and whether
the scan completed cleanly.

**Motivation:** After the logging improvements in PR #???, the scan log now contains:
- Scan config summary (device, range, step, wait, mode) at INFO immediately after
  scan start
- Per-variable hardware set commands at INFO (`[DeviceName] setting Var → value`)
- Per-shot heartbeat (`shot N`) so progress is visible
- All device command failures with error type and context
- Lifecycle milestones (start, stop, abort) at INFO

This is sufficient structured signal for an LLM-based triage agent without any
additional instrumentation.

**What this includes:**
- A triage agent that accepts a `scan.log` path and returns a structured report
- Test corpus: 2026-05-08 HTU scans 004–006 — a clean NOSCAN, a 1D scan with a
  hardware-rejected command (out-of-range value, operator chose Continue each step),
  and a 1D scan with a stale-socket cascade (`[WinError 10048]` pattern causing
  spurious timeouts that resolved on the second run)
- The stale-socket case is a good regression test: the agent should identify the
  `WinError 10048` pattern and note the socket was released before the second scan

**Why here:** Requires the log improvements from Block 6 / PR #??? to have enough
signal. Further improvements (Phase 1 messages inside `scan_log`, JSON format) can
be added as the agent's requirements become clearer.

---

## Known Tech Debt

Items surfaced during the Block 6 refactor that are safe to leave for now.

### Dead `on_device_error` attribute
`ScanManager.__init__` still assigns `self.executor.on_device_error` and
`self.scan_data_manager.on_device_error`.  Neither class reads this attribute
internally — both now route through `cmd_executor`.  The assignment is harmless but
confusing.  Remove in a follow-up PR alongside any other `ScanManager` cleanup.

### `ScanManager` size (~1100 lines)
The orchestrator still mixes scan lifecycle, state management, and threading
bookkeeping.  Block 5 (state machine) is the planned decomposition.  Don't attempt
to split it before the state machine is in place — the boundaries won't be clean.

### GeecsDevice API backlog
Three issues rooted in the `geecs-python-api` device layer that affect scanner
reliability.  These require API changes and are tracked separately:

1. **Stale UDP socket** (`[WinError 10048]`) — previous scan leaves port bound;
   next scan's `listen()` fails with `OSError`, causing immediate
   `GeecsDeviceExeTimeout` on every device command.  Resolves when OS releases the
   socket (usually <30 s).  Pattern in logs: `listen called with no socket bound`
   followed by a wave of timeout dialogs that disappear on the second attempt.

2. **`None` return on hardware rejection** — `GeecsDeviceCommandFailed` is raised
   in the UDP listener thread (not the calling thread), so `device.set()` returns
   `None` instead of raising.  The scanner now guards against this with an explicit
   `None` check that raises `DeviceCommandError`.  The root fix is in the API.

3. **Excessive `DEBUG`-level log noise** — the API layer emits verbose per-packet
   and per-variable messages.  These land in the terminal but not in the scan log
   (the scan log is gated at INFO).  Harmless today; worth pruning when the API is
   refactored.

---

## Notes on package structure

The current `GEECS-Scanner-GUI/` name is misleading — it contains both the engine
and the GUI. After Block 4, consider:

```
apps/
  geecs-scanner-gui/    ← Qt application only
  config-manager-gui/   ← already standalone
GEECS-Scanner/          ← pure engine package
GEECS-PythonAPI/
GEECS-Data-Utils/
ScanAnalysis/
ImageAnalysis/
LogMaker4GoogleDocs/
```

Whether `GEECS-Scanner` stays in this monorepo or becomes a separate installable
package is a decision for Block 4.

---

## Versioning policy

Each package is versioned independently using semantic versioning:
- `0.0.x` — bug fix, no behaviour or API change
- `0.x.0` — new feature or meaningful behaviour change (backwards-compatible)
- `1.0.0` — stable production API, deployed across multiple experiments

On every PR: run `poetry version patch|minor|major` in each affected package,
update its `CHANGELOG.md`, and commit both alongside the code changes.

Git tags on merge to master: `geecs-scanner-v0.8.0`, `geecs-python-api-v0.3.1`, etc.
