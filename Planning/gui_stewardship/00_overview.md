# GUI Stewardship — where GEECS-Scanner-GUI actually stands (2026-07-06)

This document answers the question "is this GUI good? is it correct? what do
we still owe it?" as of the day the first GUI-launched Bluesky scans ran on
Windows (2026-07-06). It replaces the old `Planning/STATUS.md` roadmap, which
was deleted on 2026-05-28 (PR #388) and never got a successor — which is
exactly why the old "Block 3 / Block 7" vocabulary started to feel stale: the
plan file disappeared while the package CLAUDE.md kept referring to it.

**Headline finding of the audit: the old roadmap is not stale because the
work stalled — it is stale because the work finished.** Every block and every
decompose step that STATUS.md tracked landed on the legacy scan path. The
docs (GEECS-Scanner-GUI/CLAUDE.md in particular) were never updated to say
so, and kept describing a pre-completion world: a 200 ms polling timer that
no longer exists, a GUI that "does not yet consume" an event stream it has
consumed for over a month. Those doc sections are fixed alongside this
document.

The open question is therefore not "finish the roadmap" — it is "what does
the GUI owe the *Bluesky* path, now that the engine underneath it is being
replaced?" That is what the rest of this document is about.

---

## 1. Glossary — the old jargon, decoded once

These terms come from the deleted `Planning/STATUS.md`. They are kept here
only so old commit messages and memories stay readable; new work should not
use them.

- **Block 3 — "typed event stream."** Instead of the GUI asking the scan
  engine "how are you doing?" every 200 ms, the engine *tells* interested
  parties what happened, as it happens, by calling an `on_event` callback
  with small typed objects (`ScanLifecycleEvent`, `ScanStepEvent`,
  `DeviceCommandEvent`, `ScanErrorEvent`, `ScanRestoreFailedEvent`,
  `ScanDialogEvent`). Defined in
  `GEECS-Scanner-GUI/geecs_scanner/engine/scan_events.py`.
- **Block 7 — "event-driven GUI."** Wire that event stream into the Qt main
  window: a `pyqtSignal` carries each event from the scan thread to the Qt
  main thread, handlers update the progress bar / status light / buttons,
  and the 200 ms polling `QTimer` is deleted.
- **D1–D5 — the "Decompose" steps.** Five ordered refactoring steps to
  shrink the legacy engine: D1 = behavioral tests for `DataLogger`; D2 =
  extract `FileMover` to its own module; D3 = extract the scan-state machine
  to `engine/lifecycle.py`; D4 = split `ScanManager._start_scan()` into
  named phases; D5 = remove the 200 ms polling timer (the last piece of
  Block 7).
- **Legacy path / legacy engine.** `ScanManager` + `DataLogger` +
  `DeviceManager` + friends in `geecs_scanner/engine/` — the original
  scan executor that talks to devices over the GEECS UDP/TCP API directly.
- **Bluesky path.** `BlueskyScanner`
  (`GeecsBluesky/geecs_bluesky/scanner_bridge/bluesky_scanner.py`), a
  drop-in replacement for `ScanManager` that delegates the actual run
  discipline to `GeecsSession` and the Bluesky `RunEngine`, with devices
  reached as EPICS PVs served by `GeecsCAGateway`.
- **RunControl.** The thin GUI-side object
  (`geecs_scanner/app/run_control.py`) that owns whichever scan backend is
  active and forwards start/stop/progress calls. It picks the backend at
  construction: legacy by default, Bluesky when the `GEECS_USE_BLUESKY`
  env var is truthy (`engine/backend_selection.py`).

---

## 2. Audit — the old roadmap vs. the code

Verified against the worktree at master `2c6c9abe` (2026-07-06). "STATUS.md"
below means its final version, recovered from git
(`git show 0bf0ed72~1:Planning/STATUS.md`).

| Roadmap item | Plain meaning | Verdict | Evidence |
|---|---|---|---|
| Block 3 | Typed event stream from the engine | **Landed** | `engine/scan_events.py`; emitted by `ScanManager`, `ScanStepExecutor`, `DeviceCommandExecutor`; pinned by `tests/engine/test_event_emission.py` |
| Block 5 | Scan state machine (`ScanState`, `PAUSED_ON_ERROR`) | **Landed** | `ScanState` in `scan_events.py`; state machine now lives in `engine/lifecycle.py` (D3) |
| Block 6 | `DeviceCommandExecutor` as the single device-command policy point | **Landed** | `engine/device_command_executor.py`; retry/escalate policy per error type |
| Block 7 | Event-driven GUI, kill the polling timer | **Landed (legacy path)** | `geecs_scanner.py`: `_scan_event_received = pyqtSignal(object)` (line ~116), `_handle_scan_event` dispatches lifecycle/step/restore/dialog events; **no 200 ms timer exists** — the only remaining `QTimer` is a 10 s `scan_number_timer` that expires a cached scan-number display, which is a display-freshness detail, not scan polling |
| D1 | Behavioral tests for `DataLogger` | **Landed** | `tests/engine/test_data_logger.py` |
| D2 | Extract `FileMover` | **Landed** | `engine/file_mover.py`; `DataLogger.file_mover` is injected (`Optional[FileMover] = None`) |
| D3 | Extract `ScanLifecycleStateMachine` | **Landed** | `engine/lifecycle.py` |
| D4 | `_start_scan()` as a readable recipe | **Landed** | `scan_manager.py`: `_phase1_pre_scan()` / `_phase2_acquire()` called from a short `_start_scan()`; the dead `on_device_error` attribute flagged for removal "during D4" is gone |
| D5 | Remove the 200 ms `QTimer` | **Landed** | See Block 7. `update_gui_status()` now covers only the RunControl-is-None and multiscan/action-library cases and is called directly at those transitions; its docstring says exactly this |
| Block 4 (deferred) | Engine importable without Qt | **Effectively landed in code; CI guard never added** | `grep PyQt5 engine/` finds only a docstring mention (`backend_selection.py` advertising itself as PyQt5-free); no CI job runs `tests/engine/` in a Qt-less env |
| Block 2 (deferred) | Move `ScanConfig` out of `geecs_data_utils` | **Not done — still deliberately deferred** | Root `CLAUDE.md` "Known debt we have deliberately deferred"; no forcing function yet |
| `Planning/STATUS.md` itself | The ordered plan file | **Deleted 2026-05-28 (PR #388)**, no successor until this document | `git log --diff-filter=D -- Planning/STATUS.md` |
| Dialog machinery (part of Blocks 6/7) | Thread-safe operator dialogs from the scan thread | **Landed (legacy path)** | `engine/dialog_request.py` (`DialogRequest` with `response_event` + `abort[0]`); `ScanDialogEvent` → GUI `_on_dialog_event` → `app/gui_dialogs.show_device_error_dialog`; the old `dialog_queue` is gone from `scan_manager.py` |

### What the access-layer pivot changed

Nothing in the roadmap was *invalidated* by the pivot to
GeecsCAGateway + GeecsSession + BlueskyScanner. What changed is the value of
each piece:

- **Made MORE valuable:** the Block 3/7 infrastructure. The
  `on_event` callback + `pyqtSignal` bridge is precisely the seam through
  which *any* backend talks to the GUI — `RunControl` passes the same
  `on_event` to `ScanManager` and `BlueskyScanner` alike, and
  `BlueskyScanner` already emits `ScanLifecycleEvent`s through it
  (`_set_state`). The event vocabulary is the GUI's future API.
- **Made LESS valuable:** any further decomposition or feature work on the
  legacy engine (`ScanManager`, `DataLogger`, `DeviceManager`). Those
  modules are now one of two backends, and the intended direction is that
  the Bluesky path replaces them. The root CLAUDE.md already warns against
  deep-testing `data_logger.py` / `device_manager.py` internals for this
  reason. The D-steps were finished before the pivot hardened, so nothing
  was wasted — but D-style work should not continue.
- **Superseded:** the *planning apparatus* (STATUS.md's block numbering).
  Current planning lives in per-topic folders under `Planning/`
  (`acquisition_modes/`, `geecs_session/`, `scan_finalization_refactor/`,
  `external_assets/`, and now `gui_stewardship/`).

### One consequence nobody wrote down

Deleting the polling timer (D5) quietly removed the only mechanism that
could have shown scan progress for a backend that doesn't emit step events.
The GUI progress bar is driven exclusively by `ScanStepEvent.shots_completed`
now — and `BlueskyScanner` emits **lifecycle events only**. So in Bluesky
mode today the status light works (INITIALIZING → orange, RUNNING → red,
DONE → green) but **the progress bar never advances** during the scan.
`BlueskyScanner.estimate_current_completion()` exists and is accurate (it
counts event documents in `_on_document`), but nothing calls it anymore.
This is the single cheapest, highest-visibility gap on the Bluesky path —
see §5.

---

## 3. The strategic frame

Today, both scan backends live behind `RunControl`:

```
GEECSScannerWindow ── RunControl ──┬── ScanManager        (legacy; default)
        ▲                          └── BlueskyScanner     (GEECS_USE_BLUESKY=1)
        │ on_event → pyqtSignal            │
        └──────────────────────────────────┘  GeecsSession → RunEngine → CA devices
                                                              → GeecsCAGateway PVs
```

Both backends receive the same `on_event` callback and satisfy the same
duck-typed surface (`reinitialize`, `start_scan_thread`,
`is_scanning_active`, `stop_scanning_thread`, ...). The GUI does not know or
care which is active.

**The GUI's durable jobs** — things no backend change removes:

- **Scan submission and config editing.** Save elements, scan variables and
  composites, timing/shot-control configs, presets, the MultiScanner queue,
  optimization configs. This is the bulk of `geecs_scanner.py` and all the
  editor dialogs, and it is genuinely useful, battle-tested UI.
- **Progress and status display.** Consuming the event stream.
- **Operator decision dialogs.** A scan thread hits something it can't
  decide alone ("device X failed — abort or continue?"); a human answers.
  The request/response machinery for this exists and works on the legacy
  path.
- **Optimization setup.** Building the Xopt config and injecting the
  `optimization_loader` bridge into `BlueskyScanner`.

**The GUI's shrinking jobs** — things moving out from under it:

- **Engine orchestration.** Run discipline (scan numbering, folder claiming,
  shot control bracketing, t0 sync, event schema, Tiled writes, s-file
  export) is owned by `GeecsSession`; `BlueskyScanner` is a thin adapter
  from `ScanExecutionConfig` shapes onto `GeecsSession.scan()`.
- **Device management.** Device I/O is now CA signals against gateway PVs;
  the GUI-side `DeviceManager` subscription machinery only matters to the
  legacy path.

So the honest answer to "is this GUI good?": **the front-end half is good
and getting more valuable; the engine half is good but scheduled to
fossilize.** Correctness on the legacy path is pinned by the engine test
suite (event emission, data logger, scan executor tests). The Bluesky path
is correct where exercised (hardware-verified both acquisition modes,
GUI-launched 2026-07-06) but its GUI integration is thin: lifecycle events
only, no progress, no dialogs, no setup/closeout actions.

---

## 4. First concrete use case: the dead-contributor dialog

**The problem, as observed live (2026-07-06):** a free-run scan with one
dead camera aborts at t0 sync. This is deliberate fail-loud behavior —
`geecs_t0_sync` (`GeecsBluesky/geecs_bluesky/plans/t0_sync.py`) retries,
names the stale device(s) whose cached `acq_timestamp` isn't advancing, and
raises `GeecsT0SyncError`. Correct, but operator-hostile in the common case:
the operator would usually rather drop the dead camera and take the scan
than lose the run. Worse, t0 sync runs inside the plan, *after* the scan
folder is claimed — so the abort burns a scan number on a scan that never
took data.

**Desired behavior:** after devices connect but **before the scan folder is
claimed**, check each contributor's `acq_timestamp` freshness. If one looks
dead, ask the operator: *"U_Cam4 looks dead (no shots for 12 s) — drop it
and continue, or abort?"* Then either proceed without it or abort cleanly,
in both cases without claiming a folder.

### What already exists (more than you'd think)

| Piece | Where | Status |
|---|---|---|
| Staleness signal | `CaAcqTimestampReadable` keeps a persistent CA monitor on each device's `acq_timestamp`; "never acquired" reads as `None` | Exists |
| Staleness detection logic | `geecs_t0_sync` already computes exactly this and names laggards — just too late in the sequence | Exists (wrong place) |
| The pre-claim seam | `BlueskyScanner._execute_scan()` is explicitly structured so "everything that can fail runs before the scan folder is claimed", and `_abort_before_acquisition()` is already checked at that point | Exists |
| Request/response channel | `DialogRequest` (`geecs_scanner/engine/dialog_request.py`): worker fills it, blocks on `response_event.wait()`; consumer writes `abort[0]` and sets the event. Thread-safe, headless-safe (`escalate_device_error` auto-aborts when no callback is wired) | Exists |
| Event type to carry it | `ScanDialogEvent(request=...)` in `engine/scan_events.py` | Exists |
| Scan-thread → Qt-main-thread bridge | `_scan_event_received` pyqtSignal → `_handle_scan_event` → `_on_dialog_event` → `show_device_error_dialog` | Exists **and already renders dialog events** on the legacy path |
| `on_event` plumbed to BlueskyScanner | `RunControl` passes the same callback to both backends | Exists |

### Landed (2026-07-07 — GeecsBluesky 0.21.0, GUI 0.32.0)

Liveness is **CONNECTED-based**, superseding the original staleness
heuristic (maintainer decision 2026-07-07): the gateway serves a per-device
`[Experiment:]Device:CONNECTED` status PV (enum Disconnected/Connected,
MAJOR severity while the device's TCP stream is down —
`GeecsCAGateway/PV_CONTRACT.md` §1/§5), and every CA sync device exposes it
as a non-readable `connected_status` child (never in event rows or
`describe()`). This closes the actual gap: the gateway serves every DB
device's data PVs whether or not the device is up, so CA-connect success
never detected a dead device.

`BlueskyScanner._preflight_check_sync_liveness` runs in `_execute_scan`'s
pre-claim seam, **both modes**: each sync device's `connected_status` is
read from the scan thread (`run_coroutine_threadsafe` on the RE loop, 2 s
budget, fail-open — an unreadable CONNECTED PV logs at DEBUG and reads as
live, so an old gateway can never block a scan). DISCONNECTED devices raise
a `DialogRequest`-in-`ScanDialogEvent` through the legacy channel:
drop-and-continue (disconnect + remove from the detector list) vs abort; a
disconnected free-run *reference* → abort-only v1 (the second button is a
clearly-labeled "Try Anyway" — promotion deferred). Free-run then keeps the
`acq_timestamp` staleness check (10 s threshold — only relevant here now;
~2 s re-check grace) purely for the trigger-must-be-free-running
requirement: all CONNECTED + all stale → "trigger appears to be off"
(Start Anyway / Abort); the residual CONNECTED-but-stale contributor with a
fresh reference keeps the drop dialog (the fresh reference proves the
trigger runs — a per-device acquisition problem, not a trigger problem).
Strict is liveness-only (frames are not needed pre-scan; the trigger may
sit OFF until ARMED — the earlier differential-staleness inference is
removed). Headless / no consumer / unanswered (30 s timeout) → today's
fail-loud proceed is preserved. Mid-scan, `geecs_single_shot`'s bounded
refire is gated on the same signal: a no-frame device reporting
DISCONNECTED raises `GeecsDeviceDownError` immediately ("went down
mid-scan — not a frame drop") instead of burning refires.
`DialogRequest` grew optional `title`/`continue_label`/`abort_label` fields
so the dialog wording is owned by the request; `show_device_error_dialog`
honors them (`_resolve_dialog_content`) and is otherwise unchanged. Pinned
by `GeecsBluesky/tests/test_bluesky_scanner_progress_and_preflight.py`,
`GeecsBluesky/tests/test_single_shot_plan.py` (refire gating), and
`GEECS-Scanner-GUI/tests/app/test_gui_dialogs.py`.

---

## 5. The investment question (solo-maintainer framing)

Three options, in increasing ambition:

**(a) Minimal.** Keep the PyQt GUI exactly as it is; add dialog events (and
similar operator affordances) only where lab operations actually demand
them — the dead-contributor dialog being the first. Cost: near zero beyond
each demanded feature. Risk: the Bluesky path's GUI experience stays
noticeably worse than legacy (no progress bar) even as it becomes the daily
driver.

**(b) "Finish Block 7 on the Bluesky path."** The audit reframes this
option: Block 7 is *done* — what remains is teaching `BlueskyScanner` to
speak the full event vocabulary the GUI already consumes. And most of that
is disproportionately cheap:

- **Step/progress events: LANDED 2026-07-07** (GeecsBluesky 0.21.0) —
  `_on_document` emits a shot-level `ScanStepEvent(phase="completed")` per
  event document (clamped at `total_shots` against the free-run tail-flush
  overcount; step index from `bin_number`), so the progress bar works in
  Bluesky mode with zero GUI changes.
- **Dialog events: LANDED 2026-07-07** — the §4 use case (see §4's
  "Landed" section).
- **`DeviceCommandEvent` translation:** probably *never* worth it — the
  GUI doesn't render them, and the GeecsBluesky CLAUDE.md already suspects
  they "may not need to be" translated. Skip unless something consumes them.

Meanwhile the legacy path fossilizes: bug fixes only, no new events, no
decomposition, deletion when the Bluesky path covers all daily modes
(needs: background mode, setup/closeout actions, and a season of routine
use).

**(c) Long-term thinning.** The GUI becomes a pure front-end over
`GeecsSession` (or a queueserver), opening the door to non-Qt alternatives
(web UI, bluesky-widgets). **This is named here for honesty, not for
action: it is not current-phase work.** It only makes sense after the
legacy path is deleted and if a second front-end consumer actually appears.

### Recommendation: (a) + (b)-lite

Concretely, in order:

1. Emit `ScanStepEvent` from `BlueskyScanner._on_document` (progress bar in
   Bluesky mode). Tiny. **Landed 2026-07-07.**
2. Implement the dead-contributor pre-flight dialog (§4). Small.
   **Landed 2026-07-07.**
3. Freeze the legacy engine: no further decomposition, no new event types,
   bug fixes only.
4. Skip `DeviceCommandEvent` translation and any speculative GUI
   restructuring (the seven-concerns debt in `geecs_scanner.py` stays
   deferred per root CLAUDE.md).

**Forcing functions to revisit:**

- *Revisit (c)* when the legacy path is actually deleted, or when a real
  second consumer of the event stream appears (web status page,
  queueserver, a second control room).
- *Revisit GUI restructuring* when a feature demands touching three or more
  of `geecs_scanner.py`'s concern clusters at once (e.g. "add a new scan
  mode").
- *Revisit option (b)-full* (rich per-device GUI feedback) if operators ask
  "which device is slow right now?" during scans — that is the question
  `DeviceCommandEvent` answers.

---

## 6. Documentation cleanup (what this change does, and what remains)

Done alongside this document (docs-only, this branch):

- **GEECS-Scanner-GUI/CLAUDE.md** — removed the 200 ms-timer/pre-Block-7
  descriptions (main-window bullet, scan-flow footer, event-system intro,
  stale Known Tech Debt entries); pointed "Current Direction" here instead
  of at the deleted `Planning/STATUS.md`; documented the
  `GEECS_USE_BLUESKY` backend switch.
- **GeecsBluesky/CLAUDE.md** — added the missing `analysis/` and `assets/`
  subpackages to the layout; re-dated "Known Gaps" (was "as of 0.8.0";
  package is at 0.19.0) and pruned gaps that closed; replaced the
  Transport-Layer section that described code now living in GeecsCAGateway;
  noted the `GEECS_USE_BLUESKY` switch.
- **Root CLAUDE.md** — dependency graph now shows the optional
  `GeecsBluesky → ImageAnalysis` edge (the `analysis` extra); the
  CHANGELOG enumeration includes `GeecsBluesky/` and `GeecsCAGateway/`.

Not done / for later:

- `Planning/STATUS.md` needs no banner — it was already deleted (2026-05-28,
  PR #388); its fate is recorded in §2. Do not resurrect it.
- Stale *docstrings* referencing the 200 ms timer remain in code
  (`ScanDialogEvent`'s "In Block 7 ..." note; the
  `engine/dialog_request.py` module docstring was fixed 2026-07-07 when
  the pre-flight dialog work touched that file). Docstrings are code; fix
  them in the next code PR that touches those files, not in this docs
  branch.
- Block 4's missing CI guard (engine tests in a Qt-less env) is a
  nice-to-have; add it opportunistically if CI is being touched anyway.
