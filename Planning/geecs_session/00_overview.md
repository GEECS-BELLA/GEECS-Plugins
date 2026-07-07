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

- **Feed Xopt the measured readback, not the proposal.** ~~Open~~ —
  **landed 2026-07-06** (geecs-scanner 0.31.0): `observe()` substitutes
  each variable's proposal with the bin-mean measured readback from the
  bin rows (NaN rows excluded; proposal fallback when the column is
  missing/empty/all-NaN).
- **Analyzer-device auto-provisioning for optimization scans.** ~~Open~~ —
  **landed 2026-07-06** (geecs-bluesky 0.20.0):
  `BlueskyScanner._run_optimization` merges the bridge-exposed
  `device_requirements` into `_devices_config` before device build (absent
  devices added per the requirement; GUI devices keep their config and
  gain missing variables).
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

## Open threads from the PR #449 max-effort review (2026-07-06)

The review's 15 reported findings (plus four below-cap confirmations) were
fixed in the 2026-07-06 fix wave (commits `df67c584..c51465cd`), each with a
pinning test. What follows is what was deliberately **not** fixed, with its
forcing function.

**Live checks — both verified 2026-07-06.** #1: legacy MC re-analysis
(Undulator 26_0219 Scan010) — loud fallback, all shots mapped, probe-skip
brought the per-device overhead from ~7 s to ~2.4 s. #2: from the GUI on
Windows (the first GUI-launched and first Windows Bluesky scans): healthy
camera scans normally; off camera aborts loudly at trigger with the
claimed-folder failure logged; a dead contributor fails t0-sync loudly with
the stale device now named. The session also produced four follow-up fixes
on the branch: `GEECS_USE_BLUESKY` env switch, variable-less synchronous
devices (image-only camera elements), config-driven `[epics] ca_addr_list`
client addressing, and t0-sync stale-device naming.

**Windows client recipe** (transition-period ergonomics): with
`[epics] ca_addr_list = 192.168.6.14` in config.ini, a GUI client needs only
`GEECS_USE_BLUESKY=1` (plus `GEECS_BLUESKY_ACQUISITION_MODE` when not
strict). Remaining cosmetic item: the `caRepeater` PATH warning — the
executable ships inside `epicscorelibs`; add its `bin/<arch>` dir to PATH on
machines that become regular clients. Windows machines are expected to be
the heaviest clients.

**Policy question (deliberately open): dead free-run contributor.** Today a
contributor whose device is off fails the whole scan at t0-sync (loud,
correct, no silent garbage). The alternative — drop the dead contributor
with a warning and scan without it — may be preferable operationally during
the transition. Decide when it first annoys an operator.

**Plausibles — mechanism verified, trigger unconfirmed.** Verify the two
shot-control-adjacent ones before the next strict live test: gateway push
fan-out does not deliberately write `acq_timestamp` last (gateway.py:257 —
a new shot id can pair with the previous shot's values if device echo order
ever changes), and the enum numeric-label-as-index fallback
(channels.py:118). The rest need one piece of live/DB evidence each:
non-ASCII variable name → endless supervise retries (tcp_subscriber);
a device variable literally named `CONNECTED` kills gateway construction;
UNC-root corruption in Resource documents (tiled_readback) and drive-root
`Z:/`→`Z:` mapping (readback) — both moot for the known `Z:/data`
deployments; `int(scan_event_index)` on pre-schema archives (camera.py);
duplicate/same-millisecond acq_timestamps double-mapping files
(single_device_scan_analyzer — the `valid`-column skip is the designed
mitigation).

**Gap-sweep candidates (unverified):** `GeecsSession()` blocks on the Tiled
server at construction when off-network (directly hits the off-network
workflow — check first); `subscribed_only=True` silently omits devices with
zero get-variables (loses their `:SP` control surface); CA devices don't pin
the `ca://` protocol (p4p-without-aioca environments flip to PVA and time
out generically); cold-cache race in `trigger()`'s first strict shot after
connect/INVALID.

**Cleanups, in rough priority order:** (1) the `{device}_{ts:.3f}{ext}`
native-filename contract lives in three packages and has already drifted —
review #1 proved it's easy to get wrong; consolidate into geecs-data-utils
soon (the fix wave left a pointer comment at each site). (2)
`analysis/camera.py` duplicates ~150–180 of 295 lines of the generic assets
path. (3) Per-event MySQL `device_type` lookups (tiled_readback) and ~166
sequential MySQL connections at gateway startup — batch/lru_cache them.
(4) `optimize()` copy-pastes `scan()`'s run-discipline preamble (drifted
once already). (5) ~10 stale docstring cross-references to the deleted
direct backend; two identically-named `read_tiled_config` readers among
five hand-rolled config.ini readers; 0.1 s polling loops; per-shot Tiled
table reads where batch iterators exist.

**Below-cap items left as-is:** `{:.12f}` truncates sub-picoscale setpoints
to zero (udp_client — decide the wire format the devices actually parse
first); `_completed_shots` counts the free-run tail flush (display is
clamped); the stale `geecs_bluesky.db.geecs_db` import in
`bluesky_hardware_smoke.ipynb`; the pre-existing `.`/`+` header-sniff quirk
that moved into `geecs_data_utils.io.array1d`.

**Docs/conventions:** GeecsCAGateway has no CLAUDE.md (root CLAUDE.md
promises one per subpackage); GeecsBluesky/CLAUDE.md omits the new
`analysis/` subpackage and still says "as of 0.8.0"; the root dependency
graph misses the optional GeecsBluesky → ImageAnalysis edge and the
changelog list omits the two new packages; a few new public signatures have
bare `Any` without the required comment.
