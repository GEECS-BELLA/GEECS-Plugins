# GeecsBluesky — Developer Context for Claude

Bridges the GEECS hardware control system to the
[Bluesky](https://blueskyproject.io/) experiment orchestration ecosystem.
The package is being rebuilt as a **native Bluesky application**
(GEECS-Plugins#807).  The plan of record is
`Planning/native_bluesky/03_clean_room_rebuild.md` — read its §3
(the evidence), §4 (the design), §10 (every decision) and §11 (the hardware
facts) before proposing a change here; it carries a staleness rule (a PR
that changes direction edits it in the same PR).  Phase-0 measurements are
in `04_phase0_measurements.md`.

**Where things stand (phase 1 complete and deployed, 2026-09-11):** a
queue item naming a stock plan and namespace devices runs a complete
strict GEECS scan — claimed scan number, the detectors' files in
`ScanNNN/<device>/` (an HDF5 stack from the PVA gateway's file plugin on
the rolled camera servers, LabVIEW-native files elsewhere), ScanInfo, the
s-file, `scan.log`, the baseline telemetry stream.  The worker registers
the stock `bluesky.plans` verbs under their own names with the strict
`take_reading` pre-bound (`plans/registry.py`); a client submits a stock
plan item or a saved preset (`qs_client.submit_plan` / `submit_preset`).
Hardware-accepted (`tests/test_phase1_hardware.py`,
`Planning/native_bluesky/05_phase1_acceptance.md` M4–M7; the file plugin
in `07_806_acceptance.md`); the worker runs the feature branch at the
#823 merge and nine camera-server gateways serve the plugin
(`03_clean_room_rebuild.md` §2).  The Console and GEECS-MCP are rewired
once, when the foundation is stable — not per step (their submit paths
call the removed funnel verbs meanwhile).  Per-shot budget: ~7 ms of
plan-layer work; the camera exposure sets the margin at 1 Hz — strict
single-shot is not the 1 Hz mode, phase 2's gated batch is
(`08_gated_batch.md`, designed 2026-09-11).

## The two rules

1. **One description of a scan** — the plan's arguments.  Nothing
   worker-side re-derives detectors or points from a request; the client
   expands a preset into a stock plan call and the request rides in
   `md["geecs"]` as provenance only.
2. **Devices own their per-run state** through the standard lifecycle
   (`stage → prepare → trigger/kickoff → unstage`).  Nothing outside a
   device configures it for a run — no preamble writes `save` or
   `localsavingpath`.

Twelve of #809's twenty-one review findings had those two causes (§3).

## Package Layout

```
geecs_bluesky/
  namespace.py              # GeecsNamespace: every DB device as a long-lived noun
  devices/detector.py       # GeecsDetector — the acquirer as a stock StandardDetector
  devices/shot_control.py   # ShotControl — the trigger box: Movable over the
                            #   profile's states, Pausable; CaPutSetter + the writes
  devices/ca/               # scalar devices + settable children: CaSnapshotReadable,
                            #   CaSettable, CaMotor, CaConfirmSettable, CaPseudoMovable,
                            #   CaActionSignalFactory, gateway_put, oneshot, liveness
  plans/strict.py           # geecs_take_reading (the fire between trigger and wait),
                            #   geecs_per_step (shots_per_step + bin_number), geecs_per_shot
  plans/registry.py         # the registration table: stock plan names bound strict,
                            #   TriggerProfiles (one ShotControl per configs-repo profile)
  plans/claim_scan.py       # the day-scoped claim (the ONE folder creator), the
                            #   claim_scan preprocessor, GeecsScanPathProvider
  plans/action_compiler.py  # ActionPlan → plan stubs
  run_engine.py             # make_run_engine: RE + claim + headers + baseline + callbacks
  preprocessors.py          # connect_on_demand (installed outermost), scalar_headers
  callbacks.py              # ScanInfo ini, the s-file, scan.log, the stack check — per run, best-effort
  scan_log.py               # ScanLogFile: the root-logger handler one run holds
  plan_names.py             # GEECS_PLAN_NAMES — what the profile exports; import-light
  qserver_ready.py          # geecs-qserver-ensure-ready (#793)
  qs_client/                # the RE Manager client every GEECS client uses
                            #   (+ presets.expand_preset: a Preset → the queue item)
  config_resolver.py        # ConfigsRepoResolver: presets, trigger profiles, catalogs, actions
  db_runtime.py             # the DB providers (served set, scalar policy, device types)
  tiled_integration.py      # subscribe_tiled (+ the geecs:// descriptor patch, goes with #806)
  data_paths.py, forward_expr.py, scanner_configs.py, epics_env.py, exceptions.py
  models/shot_control.py    # ShotControlWrites + QUIESCE_FROM (TriggerState names)
  devices/hdf_plugin.py     # the file plugin's worker side (#806): GeecsHdfIO (+Rewind),
                            #   PluginPathProvider (two paths per folder), file_plugin_hosts
  assets/                   # the geecs:// PNG asset registry — goes with PNG retirement (#738)
  optimization/             # the Xopt core; not runnable until re-glued (option B)
qserver/                    # the worker: launcher, startup profile, permissions, deploy/
```

## Devices (§4.A)

- **`GeecsDetector`** (`devices/detector.py`) — one GEECS acquirer as a
  `StandardDetector` composed of the three ophyd-async 0.19 logics:
  `GeecsTriggerLogic` (external edges only; the calibrated drain offset is
  its config signal and `get_deadtime`), `GeecsAcquireLogic` (a shot is
  `acq_timestamp` advancing; `wait_for_idle` is the shot wait; the baseline
  is taken **synchronously in `trigger()`** so the plan's very next message,
  the fire, can never land in a blind window — pinned by a mock race test),
  `ScalarsDataLogic` (the DB-subscribed scalars + the stamp as columns) and
  `LvNativeFileDataLogic` (LabVIEW-native saving: `save=on` at prepare from
  a `PathProvider`, `save=off` at stage and unstage; a per-event reading of
  the directory — there is no write-complete readback, §10.1).  It refuses
  a bare `bp.count([cam])` at prepare: a GEECS camera cannot self-trigger.
  `connected_status` reads the gateway's `CONNECTED` PV — the liveness
  signal, never a column.  `stage()` stages the scalar signals so per-shot
  reads come from the monitor cache (the 0.7 s/row regression of uncached
  reads, measured 2026-07-13, and the coherence argument for cached reads:
  the gateway posts data before the stamp, caproto and aioca deliver FIFO,
  so when the stamp advance arrives every cache holds that frame or newer).
- **`ShotControl`** — `Movable` over the trigger profile's states (`OFF`,
  `STANDBY`, `SCAN`, `ARMED`, `SINGLESHOT` — §11.1: OFF is the only quiet
  state, STANDBY is the machine's idle and passes edges), replaying each
  state's ordered writes through one cached `CaPutSetter` per target (every
  value as its wire string, 10 s budget — hardware-proven, pinned by
  `tests/test_gateway_put.py`); `Pausable` keyed on the standing state
  (§10.3: ARMED → nothing; SCAN/STANDBY → OFF and back).  Neither
  notification ever raises.  Not a flyer: the box has no counter, so in
  gated mode (phase 2, `08_gated_batch.md`) the plan drives it SCAN after
  the detectors' `kickoff` and OFF after their `complete`.
- **`GeecsNamespace`** — every enabled device of the experiment, built from
  the DB roster (loud on failure) and connected on first use by
  `connect_on_demand`.  Triggerable (`looks_triggerable`) → `GeecsDetector`
  with `native_save` iff the DB lists `save` and `localsavingpath` (the
  detector then owns those two; they are never scan-settable children);
  otherwise `CaSnapshotReadable`.  Every served settable is a Movable child
  (`U_S1H.current`: `CaMotor` with a DB tolerance, else `CaSettable`), and
  a subscribed settable's readback is a column of its parent.  Namespace
  bindings keep GEECS spelling (`U_S1H`); ophyd names and event keys are
  `safe_name` (lowercase, one lossy encoding shared with the gateway's PV
  naming).  Collisions raise.
- The scalar devices and children keep their contracts: `CaMotor` (readback
  convergence within the DB tolerance, no `stop()` — GEECS has no universal
  abort), `CaConfirmSettable` (writes one variable, confirms on another),
  `CaPseudoMovable` (composite axis via `forward_expr`; relative mode
  captures baselines at `stage`), `CaSnapshotReadable` (async readbacks
  sampled per row).  Every CA signal carries an explicit `ca://` source
  (`devices/ca/_pv.py` — transport by import luck is the trap).

## The scan path (§4.B)

Strict is the default and the only mode built: the stock plans with the
GEECS `per_step` / `per_shot` (`plans/strict.py`), registered under the
stock names by `plans/registry.py` — the stock parameters minus the hook,
plus `trigger_profile` and `shots_per_step` (keyword-only; both ride in
the start document), each run bracketed ARMED → STANDBY through the
profile's `ShotControl`.  Every shot is one row; `shots_per_step` rows per
position carry the same `bin_number` —

```
prepare(detectors, STRICT_TRIGGER_INFO)     # edge-triggered, one event; per shot,
trigger(detectors, wait=False)              #   because stage() resets the context
mv(shot_control, SINGLESHOT)                # the one GEECS line
wait(group)                                 # every detector saw its stamp
create / read / save
```

A no-frame wait (`FailedStatus` caused by `GeecsTriggerTimeoutError`)
re-fires the whole event up to `max_refires` times; a device whose
`CONNECTED` PV reads Disconnected raises `GeecsDeviceDownError` instead.
Any other failed status (a refused fire) re-raises untouched — a refire
must never issue an extra physical shot.  Phase-0 numbers: 1 Hz strict on a
count; every other edge on a scan with a motor move, because the fire put
(~200 ms) plus the move overruns the ~550 ms budget between stamp arrival
and the next edge (M1/M2) — the plan layer recovers it, not `take_reading`.

Free-run is gone.  Its two jobs return natively in phase 2: the rep-rate
job as gated batch (`bp.fly`-shaped, plugin-backed detectors that count)
and the contributor job as the non-essential stream
(a per-plan `non_essential=[…]` argument — `fly_during_wrapper` per plan
with the stage and unbounded prepare it lacks, never RunEngine-level
`SupplementalData.flyers` — joined by offset-corrected stamp, §11.5;
`08_gated_batch.md`).

## The GEECS scan (§4.C): one claim, three files, one telemetry stream

`make_run_engine(experiment, claim=True, path_provider=…, telemetry=…)`
installs, in this order: the `claim_scan` preprocessor (**every run
claims** a scan number on `open_run` — `scan_number`, `scan_folder`,
`experiment`, `scan_tag` into the start document, the shared
`GeecsScanPathProvider` pointed at `ScanNNN/`; a failed claim refuses the
run), `scalar_headers` (the staged devices' `Device Variable` headers
into `geecs_scalar_headers`), `SupplementalData(baseline=…)` (every
scalar-only device and every detector's scalar signals, read at open and
close — a detector itself is Triggerable and would wait for a shot ARMED
never delivers), and last `connect_on_demand`.  Three callbacks write
**into** the claimed folder, never creating it: `ScanInfoCallback` (the
legacy `[Scan Info]` keys downstream parses, `ScanEndInfo` filled at the
stop), `SFileCallback` (`ScanDataScanNNN.txt` + `analysis/sNNN.txt` from
the run's own primary events, for any exit status with rows — no Tiled
round trip), `ScanLogCallback`.  A detector's native files go to
`ScanNNN/<GEECS device>/`; `X.scalars` (a view every namespace device
carries) in the detector list records the same columns without files
(`save_images: false`).  The telemetry set is connected once at build and
an unconnectable member is dropped with a warning — never a per-run
failure after the claim.

## The worker (`qserver/`)

`launch_re_manager.sh` (Redis + the bluesky-0MQ-proxy document stream +
`start-re-manager --keep-re`), `startup/startup.py` (imports
`geecs_bluesky` first — load-bearing, it sets `EPICS_CA_ADDR_LIST` before
libca's context exists; builds `RE` through `make_run_engine(tiled=True,
sfile=True)`; publishes documents to the proxy; exports the namespace and
the stock plans — `plan_names.GEECS_PLAN_NAMES`, which the manager
discovers as every generator function in the namespace, so never import a
stray generator into the profile), `user_group_permissions.yaml`, and
`deploy/` (the manager and `geecs-qserver-ready` units + runbook).  **A
running service means ready (#793)**: the readiness unit runs
`geecs-qserver-ensure-ready` after every manager start — wait, open if
closed, wait for idle, assert `plans_allowed ⊇ GEECS_PLAN_NAMES`.  Read
`qserver/README.md` first; its Troubleshooting section is the empirical
contract (permissions file, `--keep-re`, manager-restart-after-install,
failed-items-requeue-at-front, CLI parses Python literals not JSON).

`qs_client/` is the client seam every GEECS client uses (`QueueClient`,
`ZmqQueueClient`/`StubQueueClient`, the `[qserver]` config reader,
`readiness_verdict` — the ONE definition of ready, `run_submit_preflight`
+ `build_submission_record`).  A client submits a **stock plan item**
(`submit_plan("scan", args=[["UC_Amp4_IR_input"], "U_S1H.current", -1, 1,
5], kwargs={"shots_per_step": 10})`) or a saved preset
(`submit_preset(preset)` → `presets.expand_preset`: device bindings,
`Device:Variable` / catalog names into `U_S1H.current`, the provenance
`md`).  The package import stays light (PEP 562-lazy device re-exports;
`bluesky-queueserver-api` behind the `qs-client` extra).
One-shot blocking CA reads go through `devices/ca/oneshot.py` (one
persistent reader loop, never a per-call `asyncio.run`).

## What stays GEECS (§6)

The DB as the roster's source of truth (`db_runtime`); day-scoped scan
numbering (`plans/claim_scan.py` — **the one place a `scans/ScanNNN/`
folder comes into existence**; analysis code never creates one, the
cross-package invariant in the root `CLAUDE.md`); the s-file format and its
`Device Variable` headers (every device's `_column_headers`); ScanInfo; PV
naming (owned by the gateways and `geecs_core`); the fire between trigger
and wait.  Transport, DB and PV naming live in GEECS-Core — this package
touches devices **only** through the gateway's CA PVs and never imports the
gateway (circular).

**Images:** a camera whose host serves the PVA gateway's file plugin
(#806, `Planning/native_bluesky/06_pva_file_plugin.md`) writes one HDF5
stack per scan through the **stock** `ADHDFDataLogic` over
`devices/hdf_plugin.GeecsHdfIO`; the run's stream documents reference it
and Tiled reads it with its stock adapter.  The rule is the namespace's:
DB image variable + endpoint in `config.ini [pva] file_plugin_addr_list`
(absent = no host; never the PVA fleet's `addr_list`).  A plugin-backed camera keeps writing its native
PNGs beside the stack (dual-write, the rollout's parity evidence) until
PNG retirement (#738); elsewhere per-shot data stays on the
LabVIEW-native file path (`LvNativeFileDataLogic`, named with the stamp)
— the non-image proprietary devices keep it for good.  Live frames are the
NTNDArray PVs.  A missed shot keeps its row (scalars, the missing
device's columns `NaN`, no frames) and the plan takes one more shot,
rewinding every plugin to its last referenced frame first
(`GeecsDetector.discard_uncollected`).  `assets/` (the `geecs://` PNG
asset docs) and the descriptor patch in `tiled_integration.py` go with
PNG retirement (#738).

## Configuration

`~/.config/geecs_python_api/config.ini`: `[epics] ca_addr_list`,
`[tiled] uri / api_key`, `[Paths]` (data root and the configs repo;
`geecs_pva_plugin_data_base_path` = the data root as the camera servers'
file-plugin *service* sees it, UNC), `[pva] file_plugin_addr_list` (the
camera servers whose gateway serves the file plugin; the rollout knob),
`[Database]`, `[Experiment] expt`.  Facility values have one home
(root `CLAUDE.md`); the worker's are in the host's `site.env`, rendered
into the units.

## Testing

Hermetic on ophyd-async mock backends (`tests/ca_mock_helpers.py`:
`set_mock_value` on `acq_timestamp` is a shot; a setter factory stands in
for the trigger box in `tests/test_strict_plans.py`).  Run one suite at a
time, unbuffered — `poetry run python -u -m pytest tests -v`; the whole
suite takes ~12 s.  `tests/test_phase0_hardware.py` **fires real shots**:
it is `hardware`-marked and gated on `GEECS_HW=1`, because an explicit `-m`
on the command line overrides the `addopts` deselect (it fired shots from
a laptop on the VPN once, 2026-09-10).  The startup-profile tests run with
`QS_DOC_PUBLISH_ADDR=OFF` — a peerless zmq PUB socket blocks the context's
teardown for the next test's whole timeout (#812).  `conftest.py` stops
the RunEngine loop threads a test leaves behind (#812).

## Do not

- Re-derive the scan from a request worker-side (a second description).
- Configure a device for a run from outside its lifecycle (a leak: #809's
  saving-mode / save-path / asset-definition P1).
- Read quiescence in a scan step — it costs the longest device timeout
  (§11.2); it belongs in the once-run calibration or a preflight.
- Treat monitor silence as liveness — the gateway posts no timeout events
  (M1); `CONNECTED` is the signal.
- Import anything from `geecs_scanner` (deleted 2026-08-20; pinned by
  `tests/test_dependency_direction.py`) or hold on to a funnel idiom
  because "we already built it" (§9).
