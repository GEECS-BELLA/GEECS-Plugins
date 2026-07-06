# GeecsSession — headless scan execution on the standard access layer

Status: **v1 landing on `feat/geecs-ca-gateway`** (2026-07-03). Optimization
intentionally deferred (separate discussion).

## Motivation

With the CA gateway serving GEECS as EPICS PVs and the CA device backend at
verified parity with the direct backend (Scans 007–015), a scan needs nothing
from `geecs_scanner` — but the run *discipline* (scan numbering, save paths,
Tiled, shot control, schema roles) is interleaved with GUI-bridge plumbing
inside `BlueskyScanner`. `GeecsSession` extracts that discipline into a small
headless layer:

```python
from geecs_bluesky.session import GeecsSession

s = GeecsSession("Undulator")                       # RE + Tiled subscription
cam = s.detector("UC_Amp2_IR_input", ["centroidx"], save_images=True)
top = s.contributor("UC_TopView", ["centroidx"])
jet = s.motor("U_ESP_JetXYZ", "Position.Axis 1")
s.shot_control("HTU-LaserOFF")                      # from the configs repo

s.scan(detectors=[cam, top], motor=jet, start=4.0, end=5.0, step=0.5,
       shots_per_step=3)                            # free-run, full discipline
s.noscan(detectors=[cam], shots=10, mode="strict")  # plan-owned single-shot
```

Same scan numbers, save-path layout, event schema v1, Tiled writes, and s-file
exports as a GUI scan — zero GUI, zero `geecs_scanner` import.

## Architecture

The session is **CA-only by design**: it is the future foundation, and the
gateway is the standard access layer. (The direct backend remains available
through `BlueskyScanner` until it is deleted.)

Extractions (shared by session and scanner — the scanner *delegates*, so the
GUI path keeps its hardware-verified behavior):

| New module | Extracted from `BlueskyScanner` | Contents |
|---|---|---|
| `geecs_bluesky/shot_controller.py` | `_build_shot_controller`, `_UdpSetter`, `_arm/_disarm/_quiesce/_arm_single_shot/_fire_single_shot/_set_trigger_state`, `_require_strict_single_shot` | `ShotController` with plan-stub methods; `over_udp()` (direct) and `over_ca()` (gateway `:SP` puts) factories |
| `geecs_bluesky/tiled_integration.py` | `_read_tiled_config`, `_subscribe_tiled`, `_prepare_descriptor_for_tiled`, `_SafeDocumentCallback` | `subscribe_tiled(RE)` one-call setup |
| `geecs_bluesky/scanner_configs.py` | (was only in the hardware test) | configs-repo resolution (`GEECS_SCANNER_CONFIG_DIR` → config.ini `scanner_config_root_path`) + `load_shot_control_config(name, experiment)` |
| `geecs_bluesky/data_paths.py` | `_translate_save_path_for_device_server`, `_device_server_save_path`, `_asset_resource_root_paths` and the config readers | local ↔ device-server path mapping + asset roots |

`GeecsSession` composes those with the existing plans (`geecs_step_scan`,
`geecs_free_run_step_scan`), `geecs_run_wrapper` (numbering + saving + md), and
the CA device family. `session.scan()` mirrors `_run_step_scan`'s recipe:
claim scan number → write ScanInfo ini → configure save paths/assets → role
wiring (first detector = free-run reference; contributors anchored to it) →
plan by mode → run wrapper → finalize disarm.

Shot control over CA: the gateway exposes every settable's `:SP` PV, and a put
rides GEECS's blocking UDP set — so `ShotController.over_ca()` drives the DG645
through the gateway with the same string-valued semantics as the YAML (labels
to enum PVs, numeric strings coerced server-side). In `ca` backend mode the
scanner uses it too, making a scan fully gateway-mediated.

## Phasing

1. **v1 (landed)**: extractions + `GeecsSession` with `scan()`/`noscan()`
   (free-run + strict), image saving, asset docs, Tiled, s-file export.
2. **v1.5 (landed)**: the orchestration recipe itself (mode dispatch → run
   wrapper → finalize disarm) extracted to
   `geecs_bluesky/plans/orchestration.py::build_step_scan_plan`, called by
   *both* `GeecsSession.scan()` and `BlueskyScanner._run_step_scan` — zero
   recipe duplication. The scanner keeps only what is legitimately its own:
   `exec_config` parsing, dual-backend device construction, and GUI plumbing
   (thread, progress, lifecycle events, per-scan logs).
3. **Endgame** (post direct-backend deletion): the scanner's device
   construction collapses onto the session factories, making it the full
   `exec_config → session` thin adapter; the gateway is the only thing that
   speaks GEECS TCP/UDP (`transport/` + `db/` migrate toward it or a lean base
   package); `geecs_bluesky` becomes a pure EPICS/Bluesky package whose
   GEECS-ness is the naming contract, event schema, and save conventions.
   (The full adapter is *only* possible then, because the session is CA-only
   while the scanner must serve both backends until direct dies.)

## Non-goals (for now)

- ~~Optimization~~ — landed as `session.optimize()` (optimization **as a
  scan**: iteration = bin, same schema/data tree; suggester ask/tell protocol
  with Xopt behind the `optimize` extra; `BinData` gives objectives the
  shot-matched scalar rows). Badger was evaluated and rejected for this:
  operator-centric, keeps its own evaluation archive outside the scan data
  stream. The GUI's config-driven optimization stack (BaseOptimizer /
  evaluators / ScanAnalysis analyzers, Xopt 3.1) now runs on top of it via
  `geecs_scanner.optimization.session_bridge` — the evaluator reads session
  bin rows through the `EvaluatorDataSource` seam and analyzers load native
  files by the claimed `ScanTag`, so image-based objectives (including
  bin-average-then-analyze) are the evaluator path's job, not `BinData`'s
  (its image helpers were removed as redundant).
- Per-scan log files and lifecycle `ScanEvent` emission — GUI concerns; they
  stay in `BlueskyScanner`.
- Background scan mode, setup/closeout actions — same status as before
  (pre-existing gaps in both backends).

## Open threads (as of the optimization-bridge merge, 2026-07-05)

Deliberately not blocking the PR; each has a natural forcing function.

- **Feed Xopt the measured readback, not the proposal.** GEECS set
  convergence is tolerance-bounded (e.g. 0.05 A on magnet PSUs), so the
  evaluated point can differ from the proposed one by up to the tolerance.
  Rows record the true readback; the bridge's `observe()` could substitute
  the bin-mean readback as the input Xopt learns from (~5 lines,
  `session_bridge.py`). Legacy has the same flaw, so this is an
  improvement, not parity. Decide when a real tuning problem exists.
- **Analyzer-device auto-provisioning for optimization scans.** Legacy
  merges `optimizer.device_requirements` into the save-device set
  (`device_manager.load_from_dictionary`); the Bluesky path currently uses
  only the GUI save list, so an objective's camera must be added manually.
  Fix in `BlueskyScanner._run_optimization` (merge into `_devices_config`)
  alongside the first GUI-launched optimization test.
- **Per-shot `Objective:`/`Observable:` s-file columns** are not exported
  on the Bluesky path (iteration-level values live in `optimization.json`
  + `xopt_dump.yaml`). Parked with the "analysis artifacts in Tiled"
  design discussion, as is **BAX `algorithm_results_file` rooting**
  (relative paths aren't scan-folder-rooted without a ScanDataManager).
- **Live re-verifications owed**: strict shot-control mode since the
  access-layer migration (logic pinned by tests; free-run heavily
  verified), a GUI-launched (PyQt) scan with `use_bluesky=True`, and a
  warm start from a scan's `xopt_dump.yaml`.
- **Diagnostic-config schema migration** (configs repo):
  `UC_Amp2_IR_input.yaml` and any other pre-DiagnosticConfig flat-schema
  YAMLs no longer validate against ImageAnalysis; migrate to the unified
  `image:`/`scan:` shape (an untracked `UC_Amp2_IR_input_dark.yaml` shows
  the target form).
- **Composite variables on the CA backend**: the standard shape is a
  pseudo-positioner (`CaCompositeSettable`) fanning one virtual axis onto
  N `CaSettable`s using the existing composite-variable config — a device-
  layer concern; the gateway stays a 1:1 GEECS mirror.
- **Archiver volume vs readback truth**: resolved for DAQ (deadband 0 as
  of gateway 0.5.1); the MDEL/ADEL-style plan for Archiver Appliance
  volume is recorded in `GeecsCAGateway/DESIGN.md`.
