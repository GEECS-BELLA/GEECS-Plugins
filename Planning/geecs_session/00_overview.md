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

1. **v1 (this branch)**: extractions + `GeecsSession` with `scan()`/`noscan()`
   (free-run + strict), image saving, asset docs, Tiled, s-file export.
   `BlueskyScanner` delegates to the extracted modules but keeps its own
   orchestration body.
2. **v2**: `BlueskyScanner._run_step_scan` collapses into `exec_config` →
   `session.scan()` adaptation (the "thin adapter"). Do this after v1 soaks —
   the GUI path is hardware-verified and should not churn twice in one PR.
3. **Endgame** (post direct-backend deletion): the gateway is the only thing
   that speaks GEECS TCP/UDP (`transport/` + `db/` migrate toward it or a lean
   base package); `geecs_bluesky` becomes a pure EPICS/Bluesky package whose
   GEECS-ness is the naming contract, event schema, and save conventions.

## Non-goals (for now)

- Optimization (Xopt/Badger) — deferred pending discussion; the session's
  `scan()`/device factories are the substrate it will sit on.
- Per-scan log files and lifecycle `ScanEvent` emission — GUI concerns; they
  stay in `BlueskyScanner`.
- Background scan mode, setup/closeout actions — same status as before
  (pre-existing gaps in both backends).
