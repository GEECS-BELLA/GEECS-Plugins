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

**Where things stand (phase 1 PR 1, GeecsBluesky 0.79.0):** the `ScanRequest`
funnel, free-run, `GeecsSession` and every funnel-only device are deleted;
the scan path is the stock `bluesky.plans` verbs over the device namespace
with the strict `take_reading`; the worker registers the stock plans.  Next
(PR 2): the plan layer — the `claim_scan` preprocessor + `PathProvider`,
the ScanInfo / s-file / `scan.log` callbacks, the registration table with
the strict `take_reading` pre-bound, telemetry = everything.  Then (PR 3)
headless hardware acceptance and the worker flip.  **The deployed worker
stays on `master` until then**; the Console and GEECS-MCP are rewired once,
when the foundation is stable — not per step.

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
  plans/strict.py           # geecs_take_reading: the fire between trigger and wait
  plans/claim_scan.py       # the day-scoped scan-number claim (the ONE folder creator)
  plans/action_compiler.py  # ActionPlan → plan stubs
  run_engine.py             # make_run_engine: RE + connect_on_demand + callbacks
  preprocessors.py          # connect_on_demand (installed outermost)
  plan_names.py             # GEECS_PLAN_NAMES — what the profile exports; import-light
  qserver_ready.py          # geecs-qserver-ensure-ready (#793)
  qs_client/                # the RE Manager client every GEECS client uses
  config_resolver.py        # ConfigsRepoResolver: the configs-repo documents
  db_runtime.py             # the DB providers (served set, scalar policy, device types)
  tiled_integration.py      # subscribe_tiled (+ the geecs:// descriptor patch, goes with #806)
  sfile_callback.py         # the legacy s-file export from Tiled at stop
  scan_log.py               # per-scan scan.log (a callback in PR 2)
  data_paths.py, forward_expr.py, scanner_configs.py, epics_env.py, exceptions.py
  models/shot_control.py    # ShotControlWrites + QUIESCE_FROM (TriggerState names)
  assets/, capture/         # the capture daemon and the asset registry — go with #806
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
  notification ever raises.  A `FlyerController` for gated mode is phase 2.
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

Strict is the default and the only mode built: stock plans with the GEECS
`per_step` / `per_shot` (`plans/strict.py`) —

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
(`SupplementalData.flyers`, joined by offset-corrected stamp, §11.5).

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
+ `build_submission_record`).  Its submit verbs still name the retired
funnel plan, so `worker_ready` refuses against this worker — correct, and
rewired with the plan layer.  The package import stays light (PEP 562-lazy
device re-exports; `bluesky-queueserver-api` behind the `qs-client` extra).
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

**Images:** per-shot scan data is on the file path (the LabVIEW device
writes, named with the stamp); live frames are NTNDArray PVs from
`GeecsPvaGateway`.  `capture/` (the PVA frame-stack daemon) and `assets/`
are not required for production and go with #806 (the file plugin + stock
`ADHDFDataLogic`), which also retires the `geecs://` descriptor patch in
`tiled_integration.py`.

## Configuration

`~/.config/geecs_python_api/config.ini`: `[epics] ca_addr_list`,
`[tiled] uri / api_key`, `[Paths]` (data root and the configs repo),
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
