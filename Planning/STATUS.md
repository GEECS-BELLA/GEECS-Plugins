# Refactor Status — GEECS-Scanner

**Read this at the start of every session before touching scanner code.**
**Update it as the last commit of every session.**

---

## Remaining blocks (in order)

### Block 5 — State machine  `[✓]` — merged in `worktree-scanner-finish-line`
Added `_state: ScanState`, `_state_lock`, `_set_state()`, `current_state` property.
All lifecycle transitions go through `_set_state()`. `PAUSED_ON_ERROR` added and wired
through `request_user_dialog()`. 4 new unit tests in `test_event_emission.py`. Bumped to 0.14.0.

---

### Block 7 — GUI event wiring + operator intervention  `[✓]` — merged in `worktree-scanner-finish-line`
`GEECSScannerWindow` now consumes `ScanEvent` stream via `pyqtSignal(object)` bridge.
`_handle_scan_event` dispatches to per-type handlers. `dialog_queue`, `restore_failures`,
`is_in_setup`, `is_busy()`, `is_stopping()`, `_was_scanning` all removed.
`RunControl` accepts `on_event` callback. Bumped to 0.14.0.

---

### Block 4 — Engine runs without Qt  `[ ]`
Remove all Qt imports from `geecs_scanner/engine/`. The engine must be
importable and runnable in a plain Python script or test with no PyQt5 present.

**First action:** `grep -r "PyQt5\|QtCore\|QtWidgets" geecs_scanner/engine/`
— find every Qt import in engine code.

**Done when:** `pytest tests/engine/` passes in an environment with no PyQt5
installed; a scan can be driven from a standalone Python script.

**Depends on:** Block 7 (GUI must have migrated off engine state reads first,
otherwise removing Qt breaks the GUI).

**Branch:** `worktree-block-4-headless-engine` (not yet created)

---

### Block 2 — Remaining config cleanup  `[ ]`
Move `ScanConfig` from `geecs_data_utils` into `geecs_scanner.engine.models`.
`geecs_data_utils` should contain data *navigation* structures, not scan
*execution* config. Add `ExperimentConfig` Pydantic model.

**First action:** Check every importer of `ScanConfig` across the monorepo
(`grep -r "from geecs_data_utils import.*ScanConfig"`) — understand migration
scope before moving anything.

**Done when:** `ScanConfig` is a Pydantic model in `geecs_scanner`; `geecs_data_utils`
has no scan-execution types; import paths updated across all packages.

**Depends on:** None — can be done in parallel with Block 5.

**Branch:** `worktree-block-2-config` (not yet created)

---

## Completed

| Block | What | PR |
|---|---|---|
| 0 | Tactical hardening | #313, #314 |
| 1 | Codebase audit | — |
| PR #366 | Engine reorganisation, `ScanExecutionConfig`, `TriggerController` | #366 |
| 6 | `DeviceCommandExecutor` — single command policy | #367 |
| 3 | Typed `ScanEvent` stream, `ScanState` enum | #370 |
| 5 | State machine: `_set_state`, `PAUSED_ON_ERROR` | scanner-finish-line |
| 7 | Event-driven GUI: `pyqtSignal` bridge, remove polling | scanner-finish-line |

---

## Recommended order

**5 → 2 (parallel) → 7 → 4**

Block 5 and Block 2 are independent — run them in parallel if possible.
Block 7 requires Block 5. Block 4 requires Block 7.

Block 4 is the most invasive and least user-visible. If time is limited,
stopping after Block 7 still delivers the main payoff (event-driven GUI,
operator intervention, deleted polling timer).
