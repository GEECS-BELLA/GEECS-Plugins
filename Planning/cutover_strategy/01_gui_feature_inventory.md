# GEECS-Scanner-GUI — operator feature inventory (2026-07-10)

Full functional sweep of the GUI on feat/vision-v1 (main window
`geecs_scanner.py`, all dialogs/menus/.ui files). This is the requirements
document for the v2 GUI decision (greenfield vs retrofit): every capability,
its backing layer, and a disposition. Backing legend: LEGACY = legacy scan
engine (dies in the gut) · BLUESKY = already dual-backend · PYAPI =
geecs_python_api (backend-independent) · ACTIONMGR = ActionManager via
action_control (dark under bluesky today) · OS/WEB · GUISTATE = pure
GUI+YAML.

## Disposition summary

| Disposition | Count | Capabilities |
|---|---|---|
| **CARRY** (day-1 in any GUI) | 21 | experiment select · rep-rate · timing-config selector · config bootstrap dialog · save-element lists+delete · scan-mode radios (noscan/1D/optimization/background) · 1D variable picker (completer) · scan params + live shot-count · step-list popup · optimization-config dropdown · scan description · manual set/readback (ScanDevice) · presets save/apply/delete · start · stop · progress bar/status light/scan-number · device-error Continue/Abort dialog · restore-failures summary · reinit-failure dialog · dark mode · title/icon |
| **RE-HOME** (needs a bluesky/gateway backend) | 4 | ActionLibrary/save-element **live action execution** (ACTIONMGR — already dark under bluesky) · **per-shot beeps** (driven only by legacy DataLogger — also already dark under bluesky; re-drive from `ScanStepEvent` GUI-side) · ECS dump / Master Control IP (legacy `enable_live_ECS_dump` — or drop) |
| **M5-EDITOR** (config surfaces) | 5 | SaveElementEditor · ScanVariableEditor · ShotControlEditor · ActionLibrary (editing half) · config-repair dialog (overlaps CARRY) |
| **OPS-TOOL** (scan-independent; portable menu items) | ~7 | open experiment-config folder · open user config · open today's scan folder · GitHub page · Google-Docs scanlog (**HTU-hardcoded** via LogMaker `logid`) · randomized beeps · CLI logging flags/rotating file log |
| **REPLACE-WITH-STANDARD** | 1 | MultiScanner → **bluesky-queueserver** (confirmed: pure GUI polling of the main window — `apply_preset_from_name` + `is_ready_for_scan` — zero engine coupling of its own) |
| **DROP-CANDIDATE** (maintainer confirm) | ~8 | MultiScanner (alt) · scanlog for non-HTU · ECS dump (alt) · in-GUI log box (wired code commented out) · `run` action type (unimplemented) · `add_all_variables` flag (disabled) · `get-only` composite mode (editor authors it; runtime support unclear) · `btnRefreshOptConfigs` (never connected) |

Menu-bar Options (auto-persisted to config.ini): on-shot TDMS, save-direct-
on-network, global time sync + tolerance → CARRY as options; Master Control
IP → the ECS-dump decision.

## The long tail — conveniences that bite operators if lost in a rebuild

1. **Click-to-complete popups on every read-only field** (experiment, scan
   variable, timing, device/variable fields in all four editors) — operators
   rely on these instead of typing.
2. Step-list popup with **live TCP resolution of relative composites**.
3. abs/rel label swap on start/stop fields by composite mode.
4. 10 s scan-number expiry timer + "No Scans Today" / Current-vs-Previous.
5. `MAXIMUM_SCAN_SIZE=1e6` runaway-scan guard pre-submit.
6. Conflicting-save-element detection with a specific error dialog.
7. Assigned-action quick-access buttons (persisted, incl. import from a
   save-element's setup/closeout).
8. MultiScanner copy-row/copy-list/split-list + start-position ergonomics.
9. **Execute-gating checkboxes** on ActionLibrary/SaveElementEditor
   (prevent accidental live device commands).
10. Enter-key-swallowing dummy default buttons in editors.
11. Config auto-repair prompt on startup.
12. Dark mode applied to child windows too.
13. Randomized-beeps morale feature.
14. `known_scan_number` caching (avoids filesystem hammering).
15. Menu options auto-persist with no explicit save.

## Notable findings

- **Two features are ALREADY dark under bluesky today**: manual action
  execution (`action_control = None`) and per-shot beeps (SoundPlayer is
  driven only by legacy DataLogger). The gut changes nothing for them; the
  v2 work should light both up properly (actions → engine's
  `compile_action_plan` path; beeps → GUI-side off `ScanStepEvent`).
- `geecs_python_api` backs the manual set/readback (ScanDevice) and DB
  autocompletes — backend-independent, survives the gut, drops at M5+.
- The Google-Docs scanlog menu item is hardcoded to HTU/Undulator
  (`HTUparameters.ini` logid).

## Maintainer decisions (DECIDED 2026-07-10)

1. **ECS dump / Master Control IP — DROP.** The dump was a bespoke
   full-PV-space snapshot at scan start, requested over IP from the
   MasterControl client (buggy anyway). Its standard successor already
   exists in the stack: the **Bluesky baseline stream**
   (`SupplementalData.baseline` — declared devices read once at run start
   and once at run end, into the run's own document stream, queryable from
   Tiled). If/when a full-space snapshot is wanted, wire gateway PVs as
   baseline readables — no client commands, no separate dump file. The
   future archiver makes even that largely moot (snapshot = timestamp
   query). Not scheduled; recorded as the pattern to use.
2. **MultiScanner — plain DROP for now.** Queueserver remains the noted
   replacement pattern if batch sequencing is wanted later.
3. **`get-only` composite mode — DROP** (editor authors it, no runtime
   ever supported it, and it didn't make sense).
4. **Dead features — DROP** (in-GUI log box, `run` action type,
   `add_all_variables`, unwired refresh button).

Still open: ActionControl re-homing (rec: bluesky-native execution of named
ActionPlans via the engine compiler) — G-actions in `00_overview.md`.
