# GeecsBluesky — Developer Context for Claude

Bridges the GEECS hardware control system to the
[Bluesky](https://blueskyproject.io/) experiment orchestration ecosystem
as a **native Bluesky application** (#807): ophyd-async devices over the
CA gateway's PVs, a stock `RunEngine` with GEECS preprocessors and
callbacks, and a queueserver worker that registers a small set of plans.
This file is the design of record: the rules below are the ones the
rebuild settled, and a PR that changes direction edits them in the same
PR.  The derivation (evidence, alternatives, measurements) is in the
`CHANGELOG.md` entries and the merged history.

## The two rules

1. **One description of a scan** — the plan's arguments.  Nothing
   worker-side re-derives detectors or points from a request; the client
   expands a preset into a plan call and the request rides in
   `md["geecs"]` as provenance only.
2. **Devices own their per-run state** through the standard lifecycle
   (`stage → prepare → trigger/kickoff → unstage`).  Nothing outside a
   device configures it for a run — no preamble writes `save` or
   `localsavingpath`.  The one named carve-out is a **run-level switch the
   device exposes as a property** (`GeecsDetector.native_image_save`,
   #738): the plan flips it before `stage` and restores it after, and the
   device's own lifecycle still does every PV write.  A precedent for a
   flag the lifecycle honours, never for a write from outside it.

## The plans the worker registers

`plan_names.GEECS_PLAN_NAMES` is the one spelling of the roster; the
registry (`plans/registry.py::bind_plans`) binds every name and the
readiness check asserts the manager lists them all.

| Name | What it is | Opens a run |
|---|---|---|
| `count` | stock `bp.count`, bound strict (`strict_plan`) | yes |
| `sweep` | `plans/sweep.py`: a `geecs_schemas.Sweep` trajectory through stock `scan_nd`, bound strict | yes |
| `optimize` | `plans/optimize.py`: native Xopt ask/tell, one bin per iteration | yes |
| `mv` | the stock stub with its failure named | no |
| `run_action` | a named plan from the experiment's `actions.yaml`, compiled to stubs over the namespace | no |
| `measure_shot_offsets`, `check_shot_sync` | the once-run calibration and its preflight (`plans/calibration.py`) | no |

The moving stock verbs (`scan`, `list_scan`, `rel_scan`, …) are **not**
registered: `sweep` is the one moving plan, and it uses `scan_nd`
internally.  A preset can name only the three scan plans
(`qs_client.presets.PRESET_PLAN_NAMES`); the utilities take no detector
list or no positions.  `run_action` is compiled from
`geecs_bluesky.actions.steps` (the same walk the scanner previews with).

Every bound scan verb takes, keyword-only, `trigger_profile`,
`shots_per_step`, `acquisition` (`strict` default, or `gated`),
`non_essential`, `shot_period` (the strict rep-rate throttle, #840),
`native_image_save` and `background_telemetry`; all ride in the start
document (`EVENT_SCHEMA.md`).  Every GEECS-defined registered plan
passes through `utils.resolve_annotations`: the manager evaluates a
plan's annotations in its own namespace, so postponed (string)
annotations fail at `queue add` (#861).

## Package Layout

```
geecs_bluesky/
  __init__.py, epics_env.py # apply_epics_address_config at import: EPICS_CA/PVA_ADDR_LIST
                            #   from config.ini before libca's context exists
  namespace.py              # GeecsNamespace: every DB device as a long-lived noun;
                            #   capture_streams (the file-plugin rule), add_pseudos, telemetry()
  db_runtime.py             # the DB providers (served set, device types); the scalar
                            #   policy lives in geecs_core.db.scalar_policy
  devices/detector.py       # GeecsDetector — the acquirer as a stock StandardDetector
  devices/shot_control.py   # ShotControl — the trigger box: Movable over the profile's
                            #   states, Pausable; CaPutSetter + the writes
                            #   (ShotControlWrites, QUIESCE_FROM)
  devices/sampler.py        # ShotSampler (the gated run's per-shot record) + StampStream
                            #   (a non-essential device without a plugin)
  devices/background.py     # BackgroundSnapshot — the run's background telemetry (#1016),
                            #   warm_up / warm_up_on at environment open
  devices/hdf_plugin.py     # the file plugin's worker side (#806): GeecsHdfIO (+Rewind),
                            #   PluginPathProvider (two paths per folder), file_plugin_hosts,
                            #   GeecsHdfDataLogic (the stock logic; the stream's geometry read
                            #   at the first describe, after the first frame — #1023)
  devices/ca/               # scalar devices + settable children: CaSnapshotReadable,
                            #   CaSettable (+ the user offset), CaMotor, CaConfirmSettable,
                            #   CaPseudoPositioner, ScalarsView (_view), gateway_put,
                            #   oneshot, liveness, _pv (the explicit ca:// source),
                            #   forward_expr (a pseudo's forward/inverse formulas,
                            #   affine_coefficients)
  plans/strict.py           # geecs_take_reading (the fire between trigger and wait),
                            #   geecs_per_step / geecs_per_shot, name_failed_status
  plans/gated.py            # gated_take_reading, the run bracket, the non-essential
                            #   stream wrapper, shot_clock, native_essentials
  plans/sweep.py            # sweep_plan: the Sweep payload through scan_nd, relative/reset
  plans/optimize.py         # optimize_plan: native Xopt loop, one run, one bin per iteration
  plans/calibration.py      # measure_shot_offsets / check_shot_sync (once-run, never a step)
  plans/registry.py         # bind_plans: the registration table, strict_plan (the acquisition
                            #   binder), liveness_gate, TriggerProfiles, background_wrapper
  plans/claim_scan.py       # the day-scoped claim (the ONE folder creator), the
                            #   claim_scan preprocessor, GeecsScanPathProvider
  actions/compiler.py       # ActionPlan → plan stubs; the namespace is its SettableFactory
  run_engine.py             # make_run_engine: RE + claim + headers + callbacks (+ the spool)
  preprocessors.py          # connect_on_demand (installed outermost), scalar_headers
  callbacks/                # the run's GEECS outputs, per run: scan_info.py (ScanInfo ini),
                            #   sfile.py (the s-file), scan_log.py (scan.log: ScanLogFile, the
                            #   root-logger handler one run holds), stack_check.py,
                            #   outputs.py (subscribe_scan_outputs), _base.py (stream bookkeeping)
  qserver_ready.py          # geecs-qserver-ensure-ready (#793)
  qs_client/                # the RE Manager client every GEECS client uses: client.py
                            #   (QueueClient, readiness_verdict), presets.py (expand_preset),
                            #   submit_preflight.py (the pre-submit checks, SubmissionRecord)
  config_resolver.py        # ConfigsRepoResolver: presets, trigger profiles, catalogs,
                            #   actions, optimizer configs, analysis diagnostics
  tiled/integration.py      # subscribe_tiled_spool (the engine's whole Tiled path) +
                            #   the shared checks (tiled_server_reachable, SafeDocumentCallback)
  tiled/spool.py            # the per-run JSONL spool both sides share: layout, the RE
                            #   callback (+ the run's lock), read-back, the heartbeat model
  tiled/writer.py           # geecs-tiled-writer: the sweep that registers spooled runs
  tiled/parquet.py          # the stream table as ScanNNN/ScanDataScanNNN-<stream>.parquet,
                            #   registered like a camera stack (GeecsRunWriter / GeecsTiledWriter)
  data_paths.py             # local ↔ device-server data path mapping
  exceptions.py             # the scan-level exception tree, failure_cause_text
  optimization/             # native Xopt ask/tell (driver), live PVA frames, the
                            #   measurement compiler, simulations, generators/ (BAX),
                            #   inspection/ (dump loading, surrogate analysis)
  # import-light contract modules (the scanner imports them; stdlib only):
  plan_names.py             # GEECS_PLAN_NAMES and the roster's subsets, ACQUISITION_MODES
  log_markers.py            # log-line strings clients parse from the manager's text stream
  actions/steps.py          # flatten_action_steps: the one walk of an action plan
  optimization_events.py    # the optimization stream's column codec (OptimizationRole)
  trajectory.py             # sweep_to_cycler: the one numerical expansion of a Sweep
  utils.py                  # safe_name, identifier_name, resolve_annotations
qserver/                    # the worker: launcher, startup profile, permissions, deploy/
```

## Devices

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
  the directory — there is no write-complete readback; on a device with no
  plugin it is the one logic of a fly prepare, the gated run's run-long
  saving).  It refuses a bare `bp.count([cam])` at prepare: a GEECS camera
  cannot self-trigger.  `connected_status` reads the gateway's `CONNECTED`
  PV — the liveness signal, never a column.  `stage()` stages the scalar
  signals so per-shot reads come from the monitor cache: uncached reads
  cost ~0.7 s per row, and cached reads are coherent because the gateway
  posts data before the stamp and caproto and aioca deliver FIFO, so when
  the stamp advance arrives every cache holds that frame or newer.
- **`ShotControl`** — `Movable` over the trigger profile's states (`OFF`,
  `STANDBY`, `SCAN`, `ARMED`, `SINGLESHOT` — OFF is the only quiet
  state, STANDBY is the machine's idle and passes edges), replaying each
  state's ordered writes through one cached `CaPutSetter` per target (every
  value as its wire string, 10 s budget — pinned by
  `tests/test_gateway_put.py`); `Pausable` keyed on the standing state
  (ARMED → nothing; SCAN/STANDBY → OFF and back).  Neither
  notification ever raises.  Not a flyer: the box has no counter, so in
  gated mode the plan drives it SCAN after the detectors' `kickoff` and
  OFF after their `complete`; `pause_count` is how a gated step learns a
  pause interrupted its batch, and `hold_for_batch` makes that pause end
  the batch at once and leave the restart to the plan.
- **`ShotSampler`** (`devices/sampler.py`) — the gated run's record of
  every device without a plugin: Flyable + EventCollectable, clocked by
  an essential triggered device's `acq_timestamp`, one `shots` event per
  tick with the latest cached reading of every member (scalar-only
  devices, triggered scalars, `.scalars` views, the motors, `bin_number`,
  and a native-saving essential's `-nonscalar_save_path`) and the tick's
  stamp as the clock column; `complete` is done after the quota, fails
  when the clock stops.  **A member with a stamp of its own is not read
  at the tick**: its stamp lands after the clock's whenever its device is
  slower (the HASO's by ~0.9 s with saving on), and a reading taken at
  the tick is the previous shot's — so the sampler gives each such member
  `SETTLE_TIMEOUT_S` (1.5 s) for its cached stamp to fall within
  `SHOT_WINDOW_S` (0.5 s) of the clock's before reading it, and on
  timeout writes `NaN` into its numeric columns (the string save path
  stays): a stale reading never passes as data, and the missing file for
  that row is simply missing.  `ShotSampler.missed` counts them per
  step; the log says so once per member and once at the step's end.
- **`StampStream`** (`devices/sampler.py`) — a non-essential triggered
  device without a plugin, recorded for the run in its own
  `<name>_stream`: Flyable + EventCollectable over ONE device, subscribed
  to its `acq_timestamp` at `kickoff`, one event per advance (the stamp,
  the cached scalars, a native saver's save path); `complete` immediate,
  `read_configuration` the device's `drain_offset` (the join's
  correction); it never reads at a row and never fails the run.
- **`GeecsNamespace`** — every enabled device of the experiment, built from
  the DB roster (loud on failure) and connected on first use by
  `connect_on_demand`.  Triggerable (`looks_triggerable`) → `GeecsDetector`
  with `native_save` iff the DB lists `save` and `localsavingpath` (the
  detector then owns those two; they are never scan-settable children);
  otherwise `CaSnapshotReadable`.  Every served settable is a Movable child
  (`U_S1H.current`: `CaMotor` with a DB tolerance or a catalog `kind: motor`
  opt-in — the latter at the class default tolerance, WARNED for DB
  curation — else `CaSettable`), and a subscribed settable's readback is a
  column of its parent.  Namespace bindings keep GEECS spelling (`U_S1H`);
  ophyd names and event keys are `safe_name` (lowercase, one lossy encoding
  shared with the gateway's PV naming).  Collisions raise.
  `add_pseudos(catalog.variables)` (the startup profile, after the roster)
  binds every catalog `kind: pseudo` as a `CaPseudoPositioner` over those
  same children under its catalog name (`ALine_e_beam_angle_offset_x`) — a
  bad entry is ERROR-logged and skipped, never fatal to environment open.
  `telemetry()` is the background-telemetry candidate set.
- The scalar devices and children keep their contracts: `CaMotor` (readback
  convergence within the DB tolerance, no `stop()` — GEECS has no universal
  abort), `CaConfirmSettable` (writes one variable, confirms on another),
  `CaSnapshotReadable` (async readbacks sampled per row), `ScalarsView`
  (`X.scalars`, the scalars-only view every namespace device carries).
  Every CA signal carries an explicit `ca://` source (`devices/ca/_pv.py` —
  transport by import luck is the trap).
- **Every settable carries a user offset** (`CaSettable.offset`, a soft
  signal; `set_current_position(p)` redefines the user frame, ophyd's
  spelling — EPICS `.OFF`, `user = dial + offset`).  The raw GEECS value is
  the dial; `set`/`read`/`locate` stay in it.  Only the pseudo positioners
  consume the offset today; set-as-aligned + persistence + display for
  operators is an additive follow-on arc.
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
- **Pseudo positioner rulings** (settled in #904 — do not reopen; pinned
  by `tests/test_pseudo_positioner.py`):
  - A bump is a *deviation from today's alignment*, not a position:
    `mode: relative` zeroes the components' user offsets at every stage,
    so a sweep over it starts at readback 0 and the end of the scan
    restores the baselines.  Inverting a bump through one magnet and
    scanning relative to it would snap the other onto the formula's
    absolute relation at the first step — never.
  - **No reference component, no `reference:` field.** Each transform
    defines its inverse over all its components (affine: the identity
    target where one exists, else the first non-constant; otherwise the
    catalog's `inverse`); the disagreement check carries the weight, and
    its allowance includes what the inverse propagates.  Least-squares was
    rejected.
  - Disagreement *after this pseudo moved its components* fails the scan;
    before the first move a plain pseudo warns and snaps, a relative one
    fails (its deviations were just zeroed).  The restore runs at unstage
    on every exit path — end, abort, halt (the RunEngine sweeps leftover
    staged objects; on a halt without awaiting, so a failure there shows
    only in the journal).  A restore that *failed* makes the next
    `stage()` refuse; `mv <pseudo> 0`, unstaged, is the recovery, the
    only move that skips the check, and the only thing that clears the
    owed restore (a staged scan point at 0 does not).
  - Two meanings of "relative", kept apart: the *scan choice* (a relative
    trajectory, about the current readback — any movable, R56 included)
    and the *definition* (the catalog's `mode: relative`: the value is a
    deviation with no absolute meaning — the bumps).  R56 can be scanned
    relative but has no relative definition; a bump is the mirror.
  - The catalog carries the relations (targets, `forward` expressions,
    `inverse` for non-linear ones, a `description` with the geometry and
    assumptions behind a bump's coefficients); the maths is Python.
    Geometry-derived coefficients and magnet calibration are the
    physicists' job — never build them into the transforms.
  - Vocabulary is the frameworks': pseudo positioner, user offset (EPICS
    `.OFF`, ophyd `set_current_position`).  Precedent, so the follow-on
    arc does not re-derive it: spec gave every motor a user offset
    regardless of hardware; Sardana has `Offset`/`Sign` on every pool
    motor with pseudo motors on top.

## The scan path

### Acquisition: two modes, one keyword

`acquisition` (default `strict`) selects the GEECS `per_step` /
`per_shot` hook that `plans/registry.py::strict_plan` binds into the
scan verb.  `count` keeps its stock parameters; `sweep` takes the
validated trajectory payload; `optimize` takes its optimizer config.

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
must never issue an extra physical shot.  Strict holds 1 Hz on a count
and every other edge on a moving scan: the fire put (~200 ms) plus the
move overruns the ~550 ms between stamp arrival and the next edge.  The
plan layer recovers it, not `take_reading`; the gated batch is the
1 Hz mode.

**Before the bracket's first move** every bound plan runs the liveness
gate (`plans/registry.py::liveness_gate`, #852): one `CONNECTED` read for
the trigger profile's device(s) (`ShotControl.liveness_signals`), every
listed detector and every non-essential device, through the one verdict
rule in `devices/ca/liveness.py` (`read_disconnected`, fail-open — only
the exact `Disconnected` string counts).  A dead device refuses the run
with `GeecsDeviceDownError` naming every dead one, before the box is
driven and before `open_run` claims a scan number — so nothing is
claimed and no folder exists.  The rule against reading quiescence in a
scan does not apply: this is the liveness PV, read once per run.

**A failure's name**: a `FailedStatus` is given its cause's `str` (plus
notes — `exceptions.failure_cause_text`, the one rendering) as it passes
the GEECS hooks (`plans/strict.py::name_failed_status`), inside the
stock `run_wrapper` that writes `str(exc)` into the stop document, so
`ScanEndInfo` and the portal read `CANothing: <pv>: …` rather than
`<AsyncStatus …>` (#868); the settables log a failed set at ERROR
with the `:SP` PV before the status fails (`CaSettable._set_logged`).

**Gated** (`plans/gated.py`): each run bracketed OFF → STANDBY; per step
the box free-runs in SCAN while the plugin-backed essential cameras count
`shots_per_step` frames each —

```
mv(box, OFF); [first step: drain wait, arm + zero_count]
prepare(cameras, gated_trigger_info(remaining)); prepare(sampler, remaining)
declare_stream(sampler, "shots")   # once
kickoff(*cameras, sampler); mv(box, SCAN)
complete(*cameras, sampler), waited in ~1 s slices: collect(sampler) + checkpoint
mv(box, OFF); drain wait
truncate_to_quota; declare_stream(*cameras, "primary")   # once, after the first frames
collect(*cameras, "primary"); collect(sampler, "shots")
```

`primary` is declared **after the first batch's frames**, right before
its first collect, never before the kickoff: the plugin settles a
stream's geometry on the session's first fresh frame (a held frame from
before a ΔE change is re-declared then, #1023), bluesky composes the
descriptor at `declare_stream`, and the descriptor has to read that
shape — which `devices/hdf_plugin.GeecsHdfDataLogic` reads at the first
describe.  The box is OFF at the arm, so no earlier frame exists.

`primary` is a datum stream (one datum per camera per step, the frames and
their per-frame scalars in the stack); `shots` carries one event per shot
from the `ShotSampler` (the clock stamp, the motors, `bin_number`, every
non-plugin scalar).  `shots` rows go out **during** the batch (the
scanner's progress), each only once every camera holds its frame.
**Pause means pause now**: a deferred pause lands at the batch's next
checkpoint (≤ ~1.5 s), an immediate one at once; the batch holds the box
(`ShotControl.hold_for_batch`), so the pause marks it over synchronously
and the resume restores nothing — the plan keeps the shots every device
reached (`GeecsDetector.frames_this_batch` / `truncate_to`,
`ShotSampler.stop` / `keep`), records them, and continues the step with
the remaining shots.  At most the in-flight shot is lost.  The step body
is not rewindable (a resume replays nothing).  A stalled camera fails
`complete` with the GEECS timeout and the box goes OFF.  A gated step
needs an essential triggered device (the clock).

A **LabVIEW-native saving device without a file plugin may be essential**
in a gated run: the row is the stamp and its files follow by stamp,
exactly as strict treats such a device — the plugin count is a
convenience, not what makes a batch.  It is a sampler member (its
scalars, its stamp, and it may be the clock — read after its own stamp
lands, the sampler's settle above, so its files join to their own rows);
the plan prepares it **once**, at the run's first step, unbounded
(`UNBOUNDED_TRIGGER_INFO`, `gated.native_essentials`), so the device's own
lifecycle switches saving on then and off at `unstage` — run-long, never
per step: each toggle costs the device one LabVIEW loop period (~1.5 s a
camera, ~4 s the HASO) and between steps the box is OFF, so a well-behaved
device writes nothing.  Its `-nonscalar_save_path` column rides in every
`shots` row as a run-long constant.  A dropped frame from it is a missing
file, **no retake** (as the LabVIEW scanner had it); the stack check
appends a files-versus-rows line per such device to `scan.log` — each
row's stamp matched to a file by the naming contract
(`geecs_data_utils.native_files.native_file_keys`), rows without a file
and file stamps without a row counted apart (WARNING on either, never a
failure).  A plugin-backed camera in a gated run still writes no native
files (the #738 dual-write is strict-only).  As a **non-essential** such a
device gets a stream of its own (below).

**Non-essential stream** (`non_essential=[…]`, strict or gated): the
listed devices are staged, prepared unbounded, kicked off right after
`open_run` and each collected alone into `<name>_stream` before
`close_run` — `fly_during_wrapper`'s shape with the stage and prepare it
lacks, per plan, never RunEngine-level `SupplementalData.flyers`; nothing
waits on them, and from the close on every step of theirs is a
contingency that never fails the run.  A plugin-backed camera flies itself
(a datum stream, declared at the close right before its collect so the
descriptor reads the geometry the plugin settled on, #1023).  A
**triggered device without a plugin** — one saving
its own LabVIEW files (the HASO, a camera on a host without the PVA
gateway) or one with scalars only and a stamp (a power supply, a gauge),
or any detector's `.scalars` view — is a nice-to-have diagnostic that
must not hold up acquisition: it is **not** sampled at the row, which
would either wait for it or record the previous shot.  A
`devices/sampler.StampStream` records it instead — monitor-driven, one
event per stamp it publishes (`<name>-acq_timestamp`, its cached scalars
and, a native saver listed itself, its `-nonscalar_save_path`), the moment
the stamp arrives; `complete` is immediate (the run's close is the end).
Rule 2 holds: a native saver's own unbounded prepare switches its saving
on and its `unstage` off; the stream only reads, and its descriptor
carries the device's `drain_offset` for the join.  A device that
publishes nothing leaves an empty stream and a WARNING at the close.  A
device with **no** stamp (free-running) is refused at bind
(`refuse_free_running_non_essentials`, and the preflight).

### Sweep

`geecs_schemas.Sweep` carries the JSON trajectory; the client expansion
resolves catalog and `Device:Variable` names once and records references
for the preflight.  The worker resolves only expanded bindings against
its existing namespace.  `trajectory.sweep_to_cycler` is the one pure
numerical implementation, shared with the scanner's preview.

`plans/sweep.py` validates and expands before the acquisition bracket
moves the box.  It uses stock `scan_nd` through `stub_wrapper`, with
relative/reset preprocessors inside `run_wrapper` and `stage_wrapper`:
baseline capture happens after stage, reset before close and unstage.  A
reset failure fails the run.  Never put `reset_positions_wrapper` outside
`scan_nd`'s staging lifecycle.  The validated payload and ordered
metadata follow `EVENT_SCHEMA.md`; old run readers remain for history.
`count` retains its distinct noscan metadata.

### Optimize

The bound `optimize` plan (`plans/optimize.py`) runs strict acquisition in
one run, with one bin per iteration.  `OptimizerConfig` lives in
GEECS-Schemas and embeds GEST's VOCS; Xopt and GEECS-Analysis load only
inside the worker's optional optimize path (the `optimize` extra).
Validation and frame subscriptions precede the scan claim.  Measurements
use actual readbacks and timestamp-matched live frames
(`optimization/live_frames.py`).  The `optimization` stream and the JSON
config provenance follow `EVENT_SCHEMA.md`.  Optimizer expansion takes
the same configs resolver in the preflight and `submit_preset`; saving a
preset validates its authored document without expansion.  Analysis
diagnostics resolve through Data Utils' shared config-root manager and
read-only YAML reader via `ConfigsRepoResolver.resolve_diagnostic`.  The
worker compiles each diagnostic once through `geecs_analysis.compat.v2`;
scalar discovery comes from the compiled measure.  Unsupported active
processing fails before acquisition.  Live diagnostics remain
camera-only.  Per-shot results carry the matched Unix acquisition
timestamp; a per-bin mean carries no invented shot number or timestamp.
Evaluation does no config I/O and uses no ImageAnalysis.  The
reduction/min_shots rules and `measurement.scalar` names survive from the
earlier evaluator.  Relative pseudos restore on unstage; the scanner's
explicit Set to best uses the recorded physical targets, not a relative
coordinate after its zero moved.  Optimizer listings retain validation
errors through `optimizer_config_listing()`; the names-only listing
delegates to it.

### The s-file of a run with stream data

The rows are `primary`'s events when it has them and the sampler's
`shots` events otherwise, and every **datum-only** stream's per-frame
columns — and every non-essential **event** stream's events (a device
without a plugin, one "frame" per stamp) — are joined onto them by
offset-corrected stamp.  The join itself is `geecs_data_utils.shot_join`,
shared with the offline re-export so the two cannot drift, and fed
**one** drain-offsets map (from the streams' descriptor configuration)
that covers both sides of every comparison.  One row per essential shot:
a camera's per-frame scalars and its own stamp come from its stack under
the names a *strict* row uses, each row takes the nearest frame in its
own window, an orphan frame stays in the stack and in Tiled, a shot
without a frame reads `NaN`, and a column the row already carries wins.
Frames past what a stream's datums referenced are not s-file data.  Such
a run's s-file is written on a thread (a stack may only be read once the
plugin finalizes it, which happens at `unstage`, after the stop
document); a run with no datum-only stream is still written
synchronously.  `StackCheckCallback` checks a non-essential stream by
count and a *gated* stack by count **and** stamps — one frame per `shots`
row, none orphaned — and a non-essential native saver's files against its
stream's **events** (not the rows; events without a file and file stamps
without an event counted apart, files after the last event — saved
between the stream's close and the unstage — apart again; WARNING, never
failure).  A device slower than the rep rate simply leaves `NaN` on the
rows it missed; an event no row's window reaches stays in the stream and
in Tiled.

## The GEECS scan: one claim, four callbacks, the background in every row

`make_run_engine(experiment, claim=True, path_provider=…)` installs, in
this order: the `claim_scan` preprocessor (**every run claims** a scan
number on `open_run` — `scan_number`, `scan_folder`, `experiment`,
`scan_tag` into the start document, the shared `GeecsScanPathProvider`
pointed at `ScanNNN/`; a failed claim refuses the run), `scalar_headers`
(the staged devices' `Device Variable` headers into
`geecs_scalar_headers`), and last `connect_on_demand` (outermost, so it
sees the messages the others inject).  Four callbacks
(`callbacks.subscribe_scan_outputs`) write **into** the claimed folder,
never creating it, each best-effort — a failure is logged, never raised
into the RunEngine: `ScanInfoCallback` (the legacy `[Scan Info]` keys
downstream parses, `ScanEndInfo` filled at the stop), `SFileCallback`
(`ScanDataScanNNN.txt` + `analysis/sNNN.txt` from the run's own per-shot
rows joined to its stacks, for any exit status with rows — no Tiled round
trip), `ScanLogCallback` (`scan.log`, start to stop), and
`StackCheckCallback` (frames on disk versus the documents, a warning in
`scan.log`).  `tiled=True` adds the spool callback (below).  A detector's
native files go to `ScanNNN/<GEECS device>/`; `X.scalars` in the detector
list records the same columns without files (`save_images: false`).

**Background telemetry** (#1016, #929 — Master Control parity): every
scalar the experiment logs (`get='yes'`) whose device is not in the run
is read into every row as well, softly —
`devices/background.BackgroundSnapshot`, one more detector the registry
appends to every bound scan verb (strict: read per shot; gated: a sampler
member, read at the tick; `optimize` too).  Members = `namespace.telemetry()`
minus what the run records itself — no event key twice — decided in one
place: `background_wrapper` probes right before `open_run` with the run's
**own readers** (its detectors and non-essential devices; a `.scalars` view
counts as its owner) and its **movers** (the sweep resolves its axes when
the plan is built — `plans/sweep.sweep_movers`, the plan's own lookup —
and optimize holds its movables), both known to the bound plan before the
run.  Every candidate rooted at an own reader is the run's and excluded,
whatever else is scanned on that device; a mover's device is read minus the
keys the mover describes (its readback, the motor's own column), from the
stage the RunEngine already did — bluesky's `stage_wrapper` stages
`root_ancestor`, and the snapshot never stages or unstages it — so the
scanned device's other logged variables stay in the rows; a mover that does
not describe within the budget leaves its device out of the background for
that run (WARNING; the row still carries the readback — a key read twice
fails the run, a missing column does not, and master dropped the device whole
on every scan).  The probe is bounded
(`PROBE_TIMEOUT_S`, 1 s, concurrent across members): one that does not
answer — a PV the gateway does not serve, a failed connect or describe —
is left out of **that run only**, named in the log and in the start
document's `background_dropped`, and probed again at the next run (a
device back after a gateway restart returns without an environment
reopen).  The profile connects the candidates once at environment open
(`devices/background.warm_up`, bounded by `QS_CONNECT_TIMEOUT`, nothing
dropped for good), so a run's probe finds them connected — without it the
first scan after every environment open lost most of the set to the
probe's budget.  Per shot the read is a monitor-cache hit; an INVALID
reading (the gateway's mark on a dead device's stale readbacks) is `NaN`,
a member that stops answering is `NaN`, every declared key is in every
row, and `read` never raises — nothing here can fail or stall a scan.
Switch: `ExperimentDefaults.background_telemetry` (read per run, on by
default), overridden by `Preset.background_telemetry` / the plans'
`background_telemetry` keyword; the start document says which
(`background_telemetry`).  The columns carry their `Device Variable`
headers, so the s-file gets them unchanged (~300 extra columns on a full
HTU experiment, accepted).  There is **no run-level baseline stream**: the
open/close `SupplementalData` baseline was strict for every member and
failed every scan after its claim when one PV was unserved, so it was
retired.  The archiver (a later arc) is for history and trends — never a
source of s-file columns, never read in the scan path.

## The worker (`qserver/`)

`launch_re_manager.sh` (Redis + the bluesky-0MQ-proxy document stream +
`start-re-manager --keep-re`), `startup/startup.py` (imports
`geecs_bluesky` first — load-bearing, it sets `EPICS_CA_ADDR_LIST` before
libca's context exists and `EPICS_PVA_ADDR_LIST` from the `[pva]` hosts
before the first plugin signal connects; builds `RE` through
`make_run_engine(tiled=True, claim=True)` — `tiled=True` is the document
**spool**, not a writer (see "Tiled: the spool and the writer service"
below); publishes documents to the proxy; exports the namespace and the
plans of `GEECS_PLAN_NAMES` — which the manager discovers as every
generator function in the namespace, so never import a stray generator
into the profile), `user_group_permissions.yaml`, and `deploy/` (the
manager, `geecs-qserver-ready` and `geecs-tiled-writer` units + runbook).
**A running service means ready (#793)**: the readiness unit runs
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
+ `build_submission_record`).  A client submits a **plan item** by name
(`submit_plan("count", args=[["UC_Amp4_IR_input"], 10],
kwargs={"trigger_profile": "HTU-NoGas"})`) or a saved preset
(`submit_preset(preset)` → `presets.expand_preset`: device bindings,
`Device:Variable` / catalog names into `U_S1H.current`, a sweep's
trajectory payload, the provenance `md`).  The package import stays light
(PEP 562-lazy device re-exports; `bluesky-queueserver-api` behind the
`qs-client` extra).  One-shot blocking CA reads go through
`devices/ca/oneshot.py` (one persistent reader loop, never a per-call
`asyncio.run`).

## Tiled: the spool and the writer service

The engine never talks to Tiled.  Registering a run — ~250 external
datasets on a full HTU preset, one register + one data-source update
each — took ~25 s **on the engine thread** at the stop document with the
stock `TiledWriter` subscribed to the RE, ahead of unstage and the box's
standby.  Now:

- **The engine spools** (`tiled.spool.SpoolCallback`, subscribed by
  `subscribe_tiled_spool` when `config.ini` names a catalog): every
  document of a run to `<state>/spool/<start time>-<uid>.jsonl`,
  flushed per document, `fsync`ed at the stop.  Microseconds per
  document; nothing on the network.  Wrapped in `SafeDocumentCallback`,
  so a spool failure disables spooling for that run and never fails it.
- **`geecs-tiled-writer` registers** (`tiled.writer.SpoolRegistrar`, its
  own systemd unit beside the qserver's): every sweep, complete files
  (last line a `stop`) replay oldest-first through the stock
  `TiledWriter` with one substitution (next bullet; serial registration
  — the SQLite catalog commits one write at a time, so a concurrent
  variant measured no gain and was removed), then rename `.jsonl.done`
  (pruned after `--keep-days`).
- **The stream table is a Parquet file in the scan folder**
  (`tiled.parquet.GeecsRunWriter`, 0.110.0): the stock writer asks Tiled
  for an *appendable* SQL table per stream and appends rows in batches,
  the shape of a beamline streaming into Tiled mid-run; GEECS registers
  after the stop, when the table is complete, so the appendable store
  bought nothing and cost two ceilings (PostgreSQL's 8 KB tuple,
  SQLite's 2000 columns), the #1020 inference failure and a database
  for data that has a home.  The hand-over writes
  `ScanNNN/ScanDataScanNNN-<stream>.parquet` — the s-file's sibling,
  written beside its target and renamed — and registers it from
  `readable_storage` as the camera stacks are (`application/x-parquet`,
  Tiled's default table shape).  The scan folder is the record, Tiled
  the index: `read_primary_scalars` cannot tell the backends apart
  (pinned against a real in-process Tiled).  The writer **never creates
  a scan folder**: a missing one fails the registration into the backoff.
  The URI is the Tiled host's view — `config.ini` `[Paths] geecs_tiled_host_data_base_path`
  names the share as the Tiled host mounts it when that is not the
  writer's own mount (the stacks' `plugin_save_path` is the precedent).
  `--tables appendable` selects the stock path instead (an appendable SQL
  table per stream, in the server's SQL `writable_storage`).
  **Liveness is the engine's lock, not silence**: the engine holds
  `flock` on the run's file while the run is open (a paused run goes
  quiet for longer than any deadline), and only a file with no stop that
  nobody holds registers after `--orphan-after` with a synthesized `fail`
  stop.  Two failure kinds: a **corrupt file** (a malformed line, no
  start — an empty file is never a success) is set aside as
  `.jsonl.failed` at once; **everything else** (Tiled 5xx through a
  restart, a rotated key, full storage) backs off per run — the sweep
  interval doubling per attempt, capped at `--max-backoff` — for
  `--max-attempts` (15, ~75 min) before the file is set aside and its
  half-registered container removed.  Idempotent: an existing container
  for the uid (a writer that died between registering and renaming, an
  earlier failed attempt) is deleted and registered again from the spool.
  Unreachable server → nothing attempted, nothing counted as an attempt.
- **The spool is the writer's only source.**  Not the live 0MQ stream:
  best-effort by design, and a live + replay pair needs deduplication
  against `create_container(key=uid)` and partial-registration cleanup.
  The stock writer batched every table and dataset to the stop anyway,
  so a run appearing in Tiled at its close plus a few seconds — not at
  its open — costs nothing that ever worked.
- **The heartbeat is a warning, never a gate** (`<state>/heartbeat.json`:
  liveness, `tiled_reachable`, `pending`/`in_progress`/`failed`, the
  sweep's `last_error`, and `registering` — written just before each
  registration, because a registration is ~25 s of silence that
  `is_stale` must not read as death; `read_heartbeat` + the one
  `heartbeat_verdict` for every reader: the engine's open-time warning,
  the scanner's chip, `fleet_status.sh`).  With the spool a dead writer
  loses nothing, so no preflight and no plan refuses a run over it — the
  scanner shows it, `fleet_status.sh` reports it.  Ruled "service,
  spool, no refusal gate" over the in-process-thread alternative, for
  robustness (survives a worker death mid-registration, isolates the
  Tiled client, restarts alone) and for the shape: the worker is a
  document producer, every persister a consumer.
- **One directory, set explicitly in both units:** `GEECS_TILED_WRITER_STATE`
  (`tiled.spool.default_state_dir`).  The writer also honours systemd's
  `$STATE_DIRECTORY`; the engine deliberately does not (the qserver unit
  may own a state directory of its own one day, and the spool must not
  silently move with it).  Not a site value: the same path on every host
  — `/var/lib/geecs-tiled-writer`, the `StateDirectory=` both
  `qserver/deploy/geecs-tiled-writer.service` and the qserver unit
  declare (the scanner's unit sets the variable too, for its chip).
  Deploy, the hand-over from a by-hand writer, and what the heartbeat's
  words mean: `qserver/deploy/DEPLOYMENT.md` § The Tiled writer.

The s-file, ScanInfo and `scan.log` are unaffected: they never used Tiled.
A run appears in Tiled ~30 s after it ends; the cost is per run, not per
shot (the same ~230 datasets whatever the length), and the levers are
server-side: a catalog that takes parallel writes (Postgres, with the
concurrent stop brought back), or fewer datasets per stream (the plugin
registers ~9 per camera: the frame plus each per-frame attribute as its
own array).

## What stays GEECS

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
(#806) writes one HDF5 stack per scan through
`devices/hdf_plugin.GeecsHdfDataLogic` over `devices/hdf_plugin.GeecsHdfIO`
— the stock `ADHDFDataLogic`, its stream described **lazily**: the plugin
arms on the frame it holds and re-declares the geometry on the session's
first fresh frame when the device's settings moved since (#1023), so the
provider reads the main dataset's shape and dtype off the plugin's
geometry PVs at the first describe and the first stream documents (after
the first frame; strict composes its descriptor at the first `save`, gated
declares `primary` after the first batch, and a non-essential plugin
stream is declared at the run's close right before its collect) and keeps
them once a datum is out, where the stock logic froze them at `prepare`;
the run's stream
documents reference the stack and Tiled reads it with its stock adapter.  The
rule is the namespace's: a served capture stream + endpoint in
`config.ini [pva] file_plugin_addr_list` (absent = no host; never the PVA
fleet's `addr_list`).  **Which** variables a device captures is declared
per devicetype in `geecs_core.db.device_streams` (an allowlist, in capture
order: the FROG's `frogTrace`, a MagSpec camera's `Image` + `ImageInterp`,
…) and read by `namespace.capture_streams`, restricted to what the gateway
serves: the device's image variables and its served `1darray` variables
(`served_array_variables`, typed minus the devicetype's exclusions); a
declared name that is neither is a declaration error (WARNING, skipped).
A MagSpec camera therefore arms four plugins; its lineouts land as
`(N, rows, 2)` float64 stacks, axis in column 0, at native length: the row
count is the energy span over the configured ΔE, fixed for the scan like an
image's shape (the plugin settles it on the session's first fresh frame and
drops and counts a frame of another length after that), so anything that
changes it — a magnet current, a ΔE — changes between scans, never within
one.  A devicetype with no declaration keeps
the one-image guess (`primary_image_variable`: `image`, else the first
image variable) — never guess a second stream, declare it, and never
declare a variable the device does not push on every shot (the FROG's
`SpatialImage` cost an arm timeout per `prepare`).  **One folder per
stream**: the primary stream writes `<device>/<device>.h5`, a second
stream of the same device writes the sibling
`<device>-<variable>/<device>-<variable>.h5` (stream key
`<name>-<variable>`), the layout the LabVIEW-native files use for a
device's second output, so `scan_stack.find_stack_file` resolves both;
two plugins on one path would truncate each other's file
(`devices/hdf_plugin.PluginPathProvider`).  A **gated** stream (the
declaration's `gate`: a scope channel gated by its `Enable.Ch<X>`) is armed
only when the **DB** says that channel is wired — the gate variable's
configured value (`defaultvalue`, instance row over devicetype default),
read by `capture_streams` when the namespace builds the detector, so the
disabled channels simply have no plugin.  The DB and not a readback on
purpose: these enables are never set live, so the configured value *is* the
channel's state, and a PV for a variable the device does not push sits at
its initial enum value — which on `on,off` reads `on` for every channel,
wired or not.  `set` has no bearing on capture.  Consequences worth
knowing: `plugin_backed` is a **static** fact, a scope with every channel
disabled has no file plugin at all — in a gated run it is then a
native-saving essential when it has saving controls (its own files are
its record) and a plain triggered clock device otherwise — and changing
which channels are captured means editing the DB row — the worker picks
it up when its namespace is built, not per run.  A plugin-backed camera
keeps writing its native PNGs beside the stack (dual-write, the rollout's
parity evidence) until PNG retirement (#738) — **per run**, the bound
plans' `native_image_save` argument (the preset's field; unset =
`ExperimentDefaults.native_image_save`, read at every run) switches that
dual-write off for the plugin-backed cameras and nothing else
(`native_image_save_wrapper` in the registry flips
`LvNativeFileDataLogic.enabled` for the run and restores it; the controls
stay owned, so a stale `save=on` is still cleared at stage); elsewhere
per-shot data stays on the LabVIEW-native file path
(`LvNativeFileDataLogic`, named with the stamp) — the non-image
proprietary devices keep it for good, whatever the switch says.  Live
frames are the NTNDArray PVs.  A missed shot keeps its row (scalars, the
missing device's columns `NaN`, no frames) and the plan takes one more
shot, rewinding every plugin to its last referenced frame first
(`GeecsDetector.discard_uncollected`); the incomplete-shot warning and
the step's failure name each missed plugin's `WriteMessage`
(`plans/strict.py::plugin_reasons` through the bounded
`GeecsDetector.plugin_reasons()`, which `prepare`'s failure note and the
count timeouts of `trigger` and `complete` use too, #1023), so a plugin
refusing frames — a stack that would not open, a frame of another shape
than the open stack's — does not read as a camera dropping them; the
plugin clears its message on the next frame it accepts, so the reason is
current.  Natively saved files are named by
the device server and read from disk by their stamp
(`geecs_data_utils.native_files`); this package emits no Resource/Datum
documents for them, so nothing here describes their formats.

## Configuration

`~/.config/geecs_python_api/config.ini`: `[epics] ca_addr_list`, `[pva]
addr_list` + `file_plugin_addr_list` (both exported into
`EPICS_PVA_ADDR_LIST` at import, `epics_env`; the second is also the
namespace's plugin rule — the camera servers whose gateway serves the
file plugin, the rollout knob), `[tiled] uri / api_key`, `[Paths]` (data
root and the configs repo; `geecs_pva_plugin_data_base_path` = the data
root as the camera servers' file-plugin *service* sees it, UNC),
`[Database]`, `[Experiment] expt`, `[qserver]` (the client seam's manager
addresses).  Facility values have one home (root `CLAUDE.md`); the
worker's are in the host's `site.env`, rendered into the units.

## Testing

Hermetic on ophyd-async mock backends (`tests/ca_mock_helpers.py`:
`set_mock_value` on `acq_timestamp` is a shot; a setter factory stands in
for the trigger box in `tests/test_strict_plans.py`).  Run one suite at a
time, unbuffered — `poetry run python -u -m pytest tests -v`.
`tests/test_phase0_hardware.py`, `tests/test_phase1_hardware.py` and
`tests/test_806_hardware.py` **fire real shots**: they are
`hardware`-marked and gated on `GEECS_HW=1`, because an explicit `-m` on
the command line overrides the `addopts` deselect.  The startup-profile
tests run with `QS_DOC_PUBLISH_ADDR=OFF` — a peerless zmq PUB socket
blocks the context's teardown for the next test's whole timeout (#812).
`conftest.py` stops the RunEngine loop threads a test leaves behind.
`tests/test_epoch_contract.py` pins the epoch equality between Core's
wire format and Data Utils' file format here because this package already
depends on both; those foundational packages keep independent constants,
with no dependency edge between them.

## Do not

- Re-derive the scan from a request worker-side (a second description).
- Configure a device for a run from outside its lifecycle (rule 2).  The
  run-level `native_image_save` property is the named exception: a flag
  the lifecycle honours, not a PV write.
- Register a moving stock verb (`scan`, `list_scan`, `rel_scan`, …):
  `sweep` is the moving plan; the stock verbs are implementation details
  behind it.
- Read quiescence in a scan step — it costs the longest device timeout;
  it belongs in the once-run calibration or a preflight.
- Treat monitor silence as liveness — a device's timeout event carries an
  unchanged stamp, which the gateway's change suppression drops, so nothing
  is posted at all; `CONNECTED` is the signal.
- Put through `signal.set()` on a typed CA signal — go through
  `GatewaySetpointPut`.  ophyd-async 0.19.3's `SignalW.set` runs the put
  inside a stamina/tenacity retry context whose outcome travels through
  a `concurrent.futures.Future`, and the stdlib re-raises only
  `if self._exception:` — a failed `aioca.CANothing` is *falsy*, so a
  refused put comes back as success (#868).  Known exceptions still on
  `signal.set()`, owed with the upstream report: the detector's `save` /
  `localsavingpath` puts (`LvNativeFileDataLogic`) and the HDF plugin's
  signals (`GeecsHdfIO`, the stock `ADHDFDataLogic` puts) — a refused one
  reads as success there today.
- Import anything from `geecs_scanner` (the web scanner depends on this
  package; the edge is one-way, pinned by
  `tests/test_dependency_direction.py`), the portal, the logbook or the
  gateway's code.
