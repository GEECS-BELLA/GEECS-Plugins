# Refactor Status — GEECS-Scanner

**Read this at the start of every session before touching scanner code.**
**Update it as the last commit of every session.**

---

## Remaining blocks (in order)

### Block 5 — State machine  `[ ]`
Replace `_scanning_active`, `_pause_event`, `_stop_event` boolean flags in
`scan_manager.py` with guarded `ScanState` transitions. Add `PAUSED_ON_ERROR`
as a reachable state. Remove dead `save_hiatus` code path.

**First action:** grep `_scanning_active` in `scan_manager.py` — understand
every read and write site before changing anything.

**Done when:** `ScanManager` has no boolean-flag state; all state reads go
through a single `self._state: ScanState` property; invalid transitions raise
or emit `ScanErrorEvent`.

**Branch:** `worktree-block-5-state-machine` (not yet created)

---

### Block 7 — GUI event wiring + operator intervention  `[ ]`
Wire the GUI to consume the `ScanEvent` stream. Delete the 200ms `QTimer` and
replace with event subscription. Delete `dialog_queue`, `restore_failures`.
Add non-modal operator intervention panel for device errors.

**First action:** Find the `QTimer` in `geecs_scanner/app/geecs_scanner.py`
and map every method it calls — these are the state reads that become event
handlers.

**Done when:** `QTimer` is removed; `dialog_queue` is removed; `restore_failures`
is removed; device-error failures surface as a non-modal panel in the GUI.

**Depends on:** Block 5 (needs `PAUSED_ON_ERROR` state for intervention flow).

**Branch:** `worktree-block-7-gui-events` (not yet created)

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

---

## Recommended order

**5 → 2 (parallel) → 7 → 4**

Block 5 and Block 2 are independent — run them in parallel if possible.
Block 7 requires Block 5. Block 4 requires Block 7.

Block 4 is the most invasive and least user-visible. If time is limited,
stopping after Block 7 still delivers the main payoff (event-driven GUI,
operator intervention, deleted polling timer).
