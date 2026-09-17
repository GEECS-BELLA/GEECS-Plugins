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
(`03_clean_room_rebuild.md` §2).  GEECS-MCP is rewired once, when the
foundation is stable — not per step (its submit path calls the removed
funnel verbs meanwhile); the Console was deleted instead (2026-09-14) and
GeecsScanner, the web scanner, submits presets.  Per-shot budget: ~7 ms of
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
                            #   CaSettable (+ the user offset), CaMotor, CaConfirmSettable,
                            #   CaPseudoPositioner (a catalog pseudo as a pseudo positioner),
                            #   gateway_put, oneshot, liveness
  plans/strict.py           # geecs_take_reading (the fire between trigger and wait),
                            #   geecs_per_step (shots_per_step + bin_number), geecs_per_shot
  plans/registry.py         # the registration table: stock plan names bound strict,
                            #   mv, run_action (the action library over the namespace),
                            #   TriggerProfiles (one ShotControl per configs-repo profile)
  plans/claim_scan.py       # the day-scoped claim (the ONE folder creator), the
                            #   claim_scan preprocessor, GeecsScanPathProvider
  plans/action_compiler.py  # ActionPlan → plan stubs; the namespace is its SettableFactory
  run_engine.py             # make_run_engine: RE + claim + headers + baseline + callbacks
  preprocessors.py          # connect_on_demand (installed outermost), scalar_headers
  callbacks.py              # ScanInfo ini, the s-file, scan.log, the stack check — per run, best-effort
  scan_log.py               # ScanLogFile: the root-logger handler one run holds
  plan_names.py             # GEECS_PLAN_NAMES — what the profile exports; import-light
  qserver_ready.py          # geecs-qserver-ensure-ready (#793)
  qs_client/                # the RE Manager client every GEECS client uses
                            #   (+ presets.expand_preset: a Preset → the queue item)
  config_resolver.py        # ConfigsRepoResolver: presets, trigger profiles, catalogs, actions
  db_runtime.py             # the DB providers (served set, device types; the scalar
                            #   policy lives in geecs_core.db.scalar_policy)
  tiled_integration.py      # subscribe_tiled (+ the geecs:// descriptor patch, goes with #806)
  data_paths.py, forward_expr.py, scanner_configs.py, epics_env.py, exceptions.py
  models/shot_control.py    # ShotControlWrites + QUIESCE_FROM (TriggerState names)
  devices/hdf_plugin.py     # the file plugin's worker side (#806): GeecsHdfIO (+Rewind),
                            #   PluginPathProvider (two paths per folder), file_plugin_hosts
  assets/                   # the geecs:// PNG asset registry — goes with PNG retirement (#738)
  optimization/             # native Xopt ask/tell, live PVA frames, measurement compiler
qserver/                    # the worker: launcher, startup profile, permissions, deploy/
```

## Devices (§4.A)

The bound `optimize` plan in `plans/optimize.py` runs strict acquisition in
one run, with one bin per iteration. `OptimizerConfig` lives in GEECS-Schemas
and embeds GEST's VOCS; Xopt and ImageAnalysis load only inside the worker's
optional optimize path. Validation and frame subscriptions precede the scan
claim. Measurements use actual readbacks and timestamp-matched live frames.
The `optimization` stream and JSON config provenance follow `EVENT_SCHEMA.md`.
Optimizer expansion takes the same configs resolver in preflight and
`submit_preset`; saving a preset validates its authored document without expansion.
Analysis diagnostics resolve through Data Utils' shared config-root manager.
Relative pseudos restore on unstage; the scanner's explicit Set to best uses
the recorded physical targets, not a relative coordinate after its zero moved.

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
  the detectors' `kickoff` and OFF after their `complete`; `pause_count`
  is how a gated step learns a pause interrupted its batch.
- **`ShotSampler`** (`devices/sampler.py`) — the gated run's record of
  every device without a plugin (phase 2b, `08` §4.7): Flyable +
  EventCollectable, clocked by an essential triggered device's
  `acq_timestamp`, one `shots` event per tick with the latest cached
  reading of every member (scalar-only devices, triggered scalars, `.scalars`
  views, the motors, `bin_number`) and the tick's stamp as the clock column;
  `complete` is done after the quota, fails when the clock stops.
- **`GeecsNamespace`** — every enabled device of the experiment, built from
  the DB roster (loud on failure) and connected on first use by
  `connect_on_demand`.  Triggerable (`looks_triggerable`) → `GeecsDetector`
  with `native_save` iff the DB lists `save` and `localsavingpath` (the
  detector then owns those two; they are never scan-settable children);
  otherwise `CaSnapshotReadable`.  Every served settable is a Movable child
  (`U_S1H.current`: `CaMotor` with a DB tolerance or a catalog `kind: motor`
  opt-in — the latter at the class default tolerance, WARNED for DB
  curation — else `CaSettable`), and
  a subscribed settable's readback is a column of its parent.  Namespace
  bindings keep GEECS spelling (`U_S1H`); ophyd names and event keys are
  `safe_name` (lowercase, one lossy encoding shared with the gateway's PV
  naming).  Collisions raise.  `add_pseudos(catalog.variables)` (the
  startup profile, after the roster) binds every catalog `kind: pseudo`
  as a `CaPseudoPositioner` over those same children under its catalog
  name (`ALine_e_beam_angle_offset_x`) — a bad entry is ERROR-logged and
  skipped, never fatal to environment open.
- The scalar devices and children keep their contracts: `CaMotor` (readback
  convergence within the DB tolerance, no `stop()` — GEECS has no universal
  abort), `CaConfirmSettable` (writes one variable, confirms on another),
  `CaSnapshotReadable` (async readbacks sampled per row).  Every CA signal
  carries an explicit `ca://` source (`devices/ca/_pv.py` — transport by
  import luck is the trap).
- **Every settable carries a user offset** (`CaSettable.offset`, a soft
  signal; `set_current_position(p)` redefines the user frame, ophyd's
  spelling — EPICS `.OFF`, `user = dial + offset`).  The raw GEECS value is
  the dial; `set`/`read`/`locate` stay in it.  Only the pseudo positioners
  consume the offset today; set-as-aligned + persistence + display for
  operators is the additive follow-on arc.
- **`CaPseudoPositioner`** (`devices/ca/pseudo.py`) — a catalog
  `kind: pseudo` entry as an ophyd-async `Transform` under a
  `DerivedSignalFactory`: `forward` formulas out, the inverse back (derived
  by `forward_expr.affine_coefficients` for `a*x + b`, else the entry's
  `inverse`), the components' offsets as parameters, moves through the
  components' own `set()`.  Two kinds, one class, frameworks' words only
  (never "absolute pseudo"): a *plain pseudo positioner* (`mode:
  absolute` — dial frame, R56) and a pseudo positioner over components
  *zeroed at stage* (`mode: relative` — the steering bumps: readback 0 by
  construction, `unstage` restores the baselines on success and abort).
  The **disagreement check** (`forward(inverse(readbacks))` vs the
  readbacks, per-component tolerance) fails the scan when a component
  moved under it; a relative `forward` is pinned `f(0) = 0` at build.
  `build_pseudo(name, spec, resolve)` is the one constructor from a
  catalog entry.
- **Pseudo positioner rulings** (the owner's, 2026-09-14/15; the arc #904
  and its brief are done — do not reopen):
  - A bump is a *deviation from today's alignment*, not a position:
    `mode: relative` zeroes the components' user offsets at every stage,
    so `scan` and `rel_scan` over it coincide (readback 0 at stage) and the
    end of the scan restores the baselines. Inverting a bump through one
    magnet and `rel_scan`-ing would snap the other onto the formula's
    absolute relation at the first step (the `U_S4H` restore incident
    class) — never.
  - **No reference component, no `reference:` field.** Each transform
    defines its inverse over all its components (affine: the identity
    target where one exists, else the first non-constant; otherwise the
    catalog's `inverse`); the disagreement check carries the weight, and
    its allowance includes what the inverse propagates. Least-squares was
    rejected.
  - Disagreement *after this pseudo moved its components* fails the scan;
    before the first move a plain pseudo warns and snaps, a relative one
    fails (its deviations were just zeroed). The restore runs at unstage
    on every exit path — end, abort, halt (the RunEngine sweeps leftover
    staged objects; on a halt without awaiting, so a failure there shows
    only in the journal). A restore that *failed* makes the next
    `stage()` refuse; `mv <pseudo> 0`, unstaged, is the recovery, the
    only move that skips the check, and the only thing that clears the
    owed restore (a staged scan point at 0 does not).
  - Two meanings of "relative", kept apart: the *scan choice*
    (`rel_scan`, about the current readback — any movable, R56 included)
    and the *definition* (the catalog's `mode: relative`: the value is a
    deviation with no absolute meaning — the bumps). R56 can be
    `rel_scan`ned but has no relative definition; a bump is the mirror.
  - The catalog carries the relations (targets, `forward` expressions,
    `inverse` for non-linear ones, a `description` with the geometry and
    assumptions behind a bump's coefficients); the maths is Python.
    Geometry-derived coefficients and magnet calibration are the
    physicists' job — never build them into the transforms.
  - Vocabulary is the frameworks': pseudo positioner, user offset (EPICS
    `.OFF`, ophyd `set_current_position`). The operator-facing
    set-as-aligned / persistence / display of the offset is an additive
    follow-on arc, not this one. Precedent for that arc, so it is not
    re-derived: spec gave every motor a user offset regardless of
    hardware; Sardana has `Offset`/`Sign` on every pool motor with pseudo
    motors on top; EPICS confined the idea to the motor record and
    bluesky/ophyd never added a generic layer.
  - Hardware-accepted 2026-09-15 (Scans 5–9 of 26_0915: bump, aborted bump
    with restore, `rel_scan` over a plain pseudo, R56 on the chicane).

## The scan path (§4.B)

### Sweep execution (2026-09-16 cutover; hardware acceptance owed)

The public scan taxonomy is `count`, `sweep`, `optimize`; moving stock verbs
are no longer registered. This supersedes the phase-1 stock-roster descriptions
above. `geecs_schemas.Sweep` carries the JSON trajectory; client expansion
resolves catalog/Device:Variable names once and records references for preflight.
The worker resolves only expanded bindings against its existing namespace.
`trajectory.sweep_to_cycler` remains the one pure numerical implementation.

`plans/sweep.py` validates and expands before the acquisition bracket moves
the box. It uses stock `scan_nd` through `stub_wrapper`, with relative/reset
preprocessors inside run_wrapper and stage_wrapper: baseline capture happens
after stage, reset before close and unstage. A reset failure fails the run.
Never put reset_positions_wrapper outside scan_nd's staging lifecycle.
The validated payload and ordered metadata follow `EVENT_SCHEMA.md`; old
run readers remain for history. Count retains its distinct noscan metadata.

### Deployed acquisition

Two modes, one keyword (`acquisition`, default `strict`), using a GEECS
`per_step` / `per_shot` bound by `plans/registry.py`. Count retains its stock
parameters; Sweep takes the validated trajectory payload; Optimize takes its
optimizer config. The acquisition binder adds
`trigger_profile`, `shots_per_step`, `acquisition`, `non_essential` and
`shot_period` (keyword-only; all ride in the start document).

**Strict** (`plans/strict.py`): each run bracketed ARMED → STANDBY through
the profile's `ShotControl`.  Every shot is one row; `shots_per_step` rows
per position carry the same `bin_number` —

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
must never issue an extra physical shot.
  Phase-0 numbers: 1 Hz strict on a
count; every other edge on a scan with a motor move, because the fire put
(~200 ms) plus the move overruns the ~550 ms budget between stamp arrival
and the next edge (M1/M2) — the plan layer recovers it, not `take_reading`.

**Before the bracket's first move** every bound plan runs the liveness
gate (`plans/registry.py::liveness_gate`, #852): one `CONNECTED` read for
the trigger profile's device(s) (`ShotControl.liveness_signals`), every
listed detector and every non-essential device, through the one verdict
rule in `devices/ca/liveness.py` (`read_disconnected`, fail-open — only
the exact `Disconnected` string counts). A dead device refuses the run
with `GeecsDeviceDownError` naming every dead one, before the box is
driven and before `open_run` claims a scan number — so nothing is
claimed and no folder exists. The §11.2 rule against reading quiescence
in a scan does not apply: this is the liveness PV, read once per run.
**A failure's name**: a `FailedStatus` is given its cause's `str` (plus
notes — `exceptions.failure_cause_text`, the one rendering) as it passes
the GEECS hooks (`plans/strict.py::name_failed_status`), inside the
stock `run_wrapper` that writes `str(exc)` into the stop document, so
`ScanEndInfo` and the portal read `CANothing: <pv>: …` rather than
`<AsyncStatus …>` (#868, #894); the settables log a failed set at ERROR
with the `:SP` PV before the status fails (`CaSettable._set_logged`).

**Gated** (`plans/gated.py`, phase 2b, `08_gated_batch.md` §4.2 / §4.7):
each run bracketed OFF → STANDBY; per step the box free-runs in SCAN while
the plugin-backed essential cameras count `shots_per_step` frames each —

```
mv(box, OFF); [repeat: drain wait, rewind_to_step_baseline]
prepare(cameras, gated_trigger_info(N)); prepare(sampler, N)
declare_stream(*cameras, "primary"); declare_stream(sampler, "shots")   # once
kickoff(*cameras, sampler); mv(box, SCAN)
complete(*cameras, sampler); mv(box, OFF); drain wait
truncate_to_quota; collect(*cameras, "primary"); collect(sampler, "shots")
```

`primary` is a datum stream (one datum per camera per step, the frames and
their per-frame scalars in the stack); `shots` carries one event per shot
from the `ShotSampler` (the clock stamp, the motors, `bin_number`, every
non-plugin scalar).  The step body is not rewindable: a deferred pause
lands between steps, an immediate pause drives OFF and the resume
**retakes the step** (the plan reads `ShotControl.pause_count`, settles
the batch's statuses through `abandon_step` / `cancel_step`, rewinds to the
step's baseline).  A stalled camera fails `complete` with the GEECS
timeout and the box goes OFF.  A gated step needs an essential triggered
device (the clock); a native camera cannot be essential there.

**Non-essential stream** (`non_essential=[…]`, strict or gated): the
listed plugin-backed detectors are staged, prepared unbounded, kicked off
right after `open_run` and each collected alone into `<name>_stream`
before `close_run` — `fly_during_wrapper`'s shape with the stage and
prepare it lacks, per plan, never RunEngine-level
`SupplementalData.flyers`; nothing waits on them.  `shot_period` is the
strict rep-rate throttle (#840).

**The s-file of a run with stream data** (phase 2c, `08` §4.5): the rows
are `primary`'s events when it has them and the sampler's `shots` events
otherwise, and every **datum-only** stream's per-frame columns are joined
onto them by offset-corrected stamp — the join itself is
`geecs_data_utils.shot_join`, shared with the offline re-export so the two
cannot drift, and fed **one** drain-offsets map (from the streams'
descriptor configuration) that covers both sides of every comparison.  One
row per essential shot: a camera's per-frame scalars and its own stamp come
from its stack under the names a *strict* row uses, each row takes the
nearest frame in its own window, an orphan frame stays in the stack and in
Tiled, a shot without a frame reads `NaN`, and a column the row already
carries wins.  Frames past what a stream's datums referenced are not s-file
data.  Such a run's
s-file is written on a thread (a stack may only be read once the plugin
finalizes it, which happens at `unstage`, after the stop document); a run
with no datum-only stream is still written synchronously.
`StackCheckCallback` checks a non-essential stream by count and a *gated*
stack by count **and** stamps — one frame per `shots` row, none orphaned.

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
the run's own per-shot rows joined to its stacks, for any exit status with
rows — no Tiled round trip), `ScanLogCallback`.  A detector's native files go to
`ScanNNN/<GEECS device>/`; `X.scalars` (a view every namespace device
carries) in the detector list records the same columns without files
(`save_images: false`).  The telemetry set is connected once at build and
an unconnectable member is dropped with a warning — never a per-run
failure after the claim.

## The worker (`qserver/`)

`launch_re_manager.sh` (Redis + the bluesky-0MQ-proxy document stream +
`start-re-manager --keep-re`), `startup/startup.py` (imports
`geecs_bluesky` first — load-bearing, it sets `EPICS_CA_ADDR_LIST` before
libca's context exists and `EPICS_PVA_ADDR_LIST` from the `[pva]` hosts
before the first plugin signal connects; builds `RE` through `make_run_engine(tiled=True,
sfile=True)`; publishes documents to the proxy; exports the namespace and
the plans — `plan_names.GEECS_PLAN_NAMES`: the stock verbs bound strict,
`mv`, and `run_action` (a named plan from the experiment's `actions.yaml`
compiled to stubs over the namespace devices; no run opened, nothing
claimed) — which the manager discovers as every generator function in the
namespace, so never import a stray generator into the profile),
`user_group_permissions.yaml`, and
`deploy/` (the manager and `geecs-qserver-ready` units + runbook).  **A
running service means ready (#793)**: the readiness unit runs
`geecs-qserver-ensure-ready` after every manager start — wait, open if
closed, wait for idle, assert `plans_allowed ⊇ GEECS_PLAN_NAMES` (and,
when the list is empty or incomplete with the environment up, restore it
once from the worker's on-disk copy via `permissions_reload`, #838).  Read
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

`~/.config/geecs_python_api/config.ini`: `[epics] ca_addr_list`, `[pva]
addr_list` + `file_plugin_addr_list` (both exported into
`EPICS_PVA_ADDR_LIST` at import, `epics_env`; the second is also the
namespace's plugin rule), `[tiled] uri / api_key`, `[Paths]` (data root and the configs repo;
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
- Put through `signal.set()` on a typed CA signal — go through
  `GatewaySetpointPut`. ophyd-async 0.19.3's `SignalW.set` runs the put
  inside a stamina/tenacity retry context whose outcome travels through
  a `concurrent.futures.Future`, and the stdlib re-raises only
  `if self._exception:` — a failed `aioca.CANothing` is *falsy*, so a
  refused put came back as success (found pinning #868). Known
  exceptions still on `signal.set()`, owed with the upstream report: the
  detector's `save` / `localsavingpath` puts (`LvNativeFileDataLogic`)
  and the HDF plugin's signals (`GeecsHdfIO`, the stock `ADHDFDataLogic`
  puts) — a refused one reads as success there today.
- Import anything from `geecs_scanner` (deleted 2026-08-20; pinned by
  `tests/test_dependency_direction.py`) or hold on to a funnel idiom
  because "we already built it" (§9).

Optimizer listings retain validation errors through `optimizer_config_listing()`;
the names-only listing delegates to it. The epoch equality test lives here because
this package already depends on Core and Data Utils. Those foundational packages
keep independent constants for their wire/file formats, with no dependency edge
between them.
