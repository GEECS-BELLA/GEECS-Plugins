# Clean-room rebuild: GeecsBluesky as a native Bluesky application

**Status (2026-09-11): phase 1 complete and deployed — the deletions
(#816), the plan layer (#821), the headless hardware acceptance (#822), the
PVA file plugin (#823, `06_pva_file_plugin.md`, `07_806_acceptance.md`) and
the fleet-requirements mechanism (#824) are merged into the feature branch;
the worker runs the #823 merge (#824 touched only GeecsPvaGateway, which
the worker does not install) and nine camera-server gateways serve the
plugin.  Phase 2 (§8) is
next.** Written at the end of the session that built #809 as a handoff,
then amended by the next session after the discussion recorded in §11 and
§12. Read this before `00_overview.md`, because it supersedes that
document's phase plan.

**Staleness rule.** This is the document of record for the rebuild. Any PR
that changes direction, sequencing, a §7 verdict or a §10 answer edits §2,
§8 and §10 here **in the same PR**, and the matching #807 comment points
here instead of restating. A plan of record that lags the code has already
cost one reassessment; the rule is cheaper than the next one.

Sam's framing, which is the point of the whole document:

> Let's start with the assumption that we *only* have our CA gateway, PVA
> gateway and bluesky/ophyd devices. How would we go from there? Blow up
> everything except the bluesky/ophyd fundamentals and build a solid RE
> that achieves what we want. […] Don't try to hold on to *anything* if it
> doesn't slot in completely cleanly.

And the constraint on that:

> holding on to the requisite GEECS things, like DB as source of truth,
> s-files etc.

**The governing fact, and the reason this is worth doing now** (Sam,
2026-09-09):

> me and my team are the only users of all of this code. My goal is to make
> it 'great' before trying to deploy at other facilities so that I don't
> have to deal with these issues of backward compatibility. We are in a
> unique 'clean slate' phase of development.

There is no external user, no deprecation cycle and no migration burden.
Backward compatibility is **not** a design input. Anything in this
repository may be deleted outright rather than adapted, and the cost of a
wrong turn is a rewrite, not a broken facility. Design for the shape that
is right in five years, not the one that is reachable in small steps.

---

## 1. How to use this document

1. Read §3. It is the evidence, and it is the only part that is hard-won.
   A fresh session given only the target architecture will re-litigate the
   design; a fresh session given the failure catalogue will not.
2. Treat §7 as a contract: anything marked **ASSUMED** must be verified
   before it is designed on. The previous session asserted API details from
   memory more than once and had to retract them publicly.
3. §8 records the sequencing decision (taken 2026-09-09) and the phases.
4. §11 and §12 are the facts Sam supplied about the hardware and the
   conclusions drawn from them. They constrain every design choice in §4;
   read them before proposing a change to §4 or §5.

---

## 2. Where things actually stand (verified 2026-09-11)

| thing | state |
|---|---|
| `feature/native-bluesky-plans` | integration branch off master; **#808 merged** 2026-09-09 (device namespace); **#811 merged** 2026-09-10 (phase 0, hardware-accepted — `04_phase0_measurements.md` M2/M3); **#813 merged** 2026-09-10 (test speed, #812) |
| the worker host | **on the feature branch since 2026-09-11.** Flipped in the morning at the #822 state (`~/qs-checkout`; 19 plans registered; Scan005 of 26_0911, a production preset scan through the manager with LabVIEW-native saving). Restarted at 15:00 on 0fd767fa (#823) with the nine-host plugin list: every camera on a rolled box is plugin-backed by the namespace rule, **not yet exercised through the manager** — the watch period's first camera scan is that check (`07_806_acceptance.md`; the acceptance scan ran in process from the staging clone). Rollback = `git checkout master` there, reinstall, restart. The master Console and MCP cannot submit scans meanwhile (by design, §10.5) |
| `deploy/pva-fleet-requirements` (#824) | **MERGED 2026-09-11** — GeecsPvaGateway 0.7.1: `deploy/requirements-fleet.txt` (the pinned closure, `--no-deps` both sides), `deploy/stage_wheels.sh`, the launcher's offline wheel step; the nine gateways rolled the same afternoon (`07_806_acceptance.md`, with the launcher byte-offset lesson) |
| `phase/04-pva-file-plugin` (#806) | **MERGED 2026-09-11 (#823)** — the file plugin in GeecsPvaGateway 0.7.0 (`file_plugin.py`, the areaDetector PV set + `Rewind`), `GeecsHdfIO` + `PluginPathProvider` on the worker (GeecsBluesky 0.81.0) under the **stock** `ADHDFDataLogic`, the capture daemon's runtime deleted (its unit, bootstrap entry and `site.env.example` lines wait for the end-of-branch deployment touch, §10.5); hardware-accepted on Scan007 of 26_0911 (frames == rows == PNGs, pixel-identical, 1 Hz held), Tiled read verified. Design `06_pva_file_plugin.md`, record `07_806_acceptance.md` |
| `phase/03-hardware-acceptance` (phase 1 PR 3) | **MERGED 2026-09-11 (#822)** — `tests/test_phase1_hardware.py`: the worker's wiring in process (Scans 104/105) and a `Preset` through a second RE Manager (Scans 106/108), every file asserted from disk; per-shot budget measured (`05_phase1_acceptance.md` M4–M7: strict single-shot is not the 1 Hz mode, the exposure sets the margin) |
| `phase/02-plan-layer` (phase 1 PR 2) | **MERGED 2026-09-10 (#821)** — the plan layer (GeecsBluesky 0.80.0, GEECS-Schemas 0.21.0): the registration table (`plans/registry.py` — 18 stock verbs bound strict under their own names, `trigger_profile` + `shots_per_step` keyword-only, ARMED → STANDBY bracket), the `claim_scan` preprocessor + `GeecsScanPathProvider` (every run claims), the `scalar_headers` preprocessor, the ScanInfo / s-file (from the documents) / `scan.log` callbacks, `SupplementalData` baseline telemetry, `GeecsDetector.scalars`, `Preset` v1 (save sets deleted; corpus regenerated as presets — on the configs branch `presets-v1` until the worker flip, on the configs `main` since 2026-09-11), `qs_client.submit_plan` / `submit_preset`.  Decisions in §10.7 |
| `phase/01-foundation` (phase 1 PR 1) | **MERGED 2026-09-10 (#816)** — the deletions: the `ScanRequest` funnel and named plans, free-run, `GeecsSession`, `scan_request_runner`, `preflight`, `pause_semantics`, `t0_sync`, the funnel-only devices (`CaGenericDetector`, `CaTriggerable`, `CaTelemetryReadable`, `CaTimestampedReadable`, the shot-id / contributor / nonscalar-save mixins), `ShotController` (its write machinery folded into `ShotControl`), the optimization glue (`plans/optimize`, `optimize.py`, `session_bridge`, `worker_loader`) and every test of theirs; the namespace builds `GeecsDetector` for every triggerable device (`native_save` iff the DB lists `save` + `localsavingpath`); `run_engine.make_run_engine` replaces the session; the startup profile exports the stock `bluesky.plans` verbs (`plan_names.GEECS_PLAN_NAMES`) over the namespace; `qs_client` keeps its surface (readiness + liveness preflight only) so the Console and MCP stay importable |
| #809 `phase/02-preamble-preprocessor` | **CLOSED 2026-09-11** with a pointer here, never merged. The evidence behind §3 |
| #807 | the decision log; its six-then-three phase plan is superseded by §8 here. Comments there point here |
| hardware acceptance | scans **63 and 64** on 26_0909 via #809's preprocessor: stock `bp.count` as a noscan, stock `bp.list_scan` sweeping `U_S1H:Current` −1→+1 A at 0.5 A. Both in the Tiled catalog with s-files. Scans 56–62 are disposable artifacts of a `tiled=False` run |

**What survives from the work so far, unconditionally:** the device
namespace (#808). Every device of the experiment as a long-lived
ophyd-async noun built from the DB roster, lazily connected. That is
exactly the foundation the design below needs, and it is already merged.

---

## 3. What we learned, and why it changes the plan

#809 put the GEECS scan preamble into a RunEngine preprocessor so that a
stock `bluesky.plans` verb could run a full GEECS scan. It works, on
hardware. It was then reviewed adversarially twice.

**Sixteen findings in the first pass, five more in the verification pass.**
Classified by cause, not by severity:

### Artifacts of describing one scan twice (12 of 21)

The plan says what to read and where to move. The ScanRequest says the same
things through a save set and an axis list. The preprocessor's real job had
become reconciling the two, and each of these findings is a missing
reconciliation rule:

- the plan need not read the devices the save set enables saving on, so
  saved frames had no event row to join to
- the request's shot count and the plan's point count could disagree, and
  ScanInfo recorded the request's
- the request's positions and the plan's positions could disagree
- a first attempt to check that compared membership, not the sequence
- background telemetry was prepared, connected and advertised in the start
  document, but a stock plan reads only its own detectors, so no telemetry
  column was ever emitted
- once injected, the telemetry group was read **unstaged**, which
  `GeecsBluesky/CLAUDE.md` already documents as the 0.7 s per row regression
  that made 1 Hz scans run at 0.5 Hz
- the scan motors were missing from the s-file header map
- the namespace contributed headers for settable children no event carries
- long-lived namespace devices kept the previous run's saving mode, save
  path and asset definitions, so an unrelated later plan emitted asset
  documents pointing into the previous scan's folder
- and still did, on any pre-claim failure after the save set was applied
- scan axes silently lost `confirm:` and `kind: motor` topology, because
  the namespace builds children from the DB and the catalog describes them
  separately
- `kind: motor` disagreement between the two doors is still unresolved and
  was waived

### Real defects, independent of the architecture (9 of 21)

Save-on ordering versus arming, the missing refire on the stock door, the
refire then recursing through the mutator and firing extra shots, the
stream-blind hooks, `install` re-deriving `connect_on_demand`'s kwargs, the
missing `scan.log`, the missing eager save-off, the raw request riding into
every start document, and a set of test-quality items.

### The conclusion

Twelve of twenty-one findings have one cause. Fixing them individually
produces a reconciliation engine: a growing set of rules asserting that two
descriptions of the same scan agree. The parity test added in #809 exists
precisely to detect divergence between the two doors, which is a useful
test and also an admission.

The native answer is that there is only one description. Sam:

> save sets are really just readbacks […] and the scan variables just the
> settables with aliases. […] If you encounter a weird hack or workaround
> to accommodate save sets we should think, "what is a save set doing and
> how does Bluesky solve the same problem?"

**A second leg the first pass undersold.** The leaked-state P1 (saving mode,
save path and asset definitions persisting on namespace nouns) is not a
reconciliation failure. It is **per-run device configuration done from
outside the device**: a preamble wrote `localsavingpath`/`save` and the
asset definitions onto long-lived objects. ophyd-async's answer is the
detector's own lifecycle — `stage → prepare → trigger/kickoff → unstage`,
with a `PathProvider` supplying the path — and that leak would exist even
with one description of the scan if saving were still configured by a
preamble. So the rebuild has two legs, and §4 is built on both:

1. **one description** — the client expands a request into the plan's
   `detectors` and points; nothing worker-side re-derives them;
2. **devices own their per-run state** through the standard lifecycle;
   nothing outside a device configures it for a run.

---

## 4. The target architecture

Assume only the two gateways and ophyd-async. Two rules drive every choice:
a scan has **one description** (the plan's arguments), and every per-run
fact about a device is set through **that device's own lifecycle**, never
from outside it.

### A. Devices — the namespace, extended

Every DB device is one long-lived noun (#808, kept). Two classes:

- **`StandardReadable`** for scalar-only devices; settables as children
  (`U_S1H.current`). Exists.
- **`GeecsDetector(StandardDetector)`** for anything that captures —
  cameras *and* LabVIEW-native non-scalar devices — composed of the three
  logics ophyd-async 0.19.3 defines (§7):
  - `GeecsTriggerLogic` — supports `prepare_edge` only (LabVIEW free-runs on
    external edges; nothing to program). `config_sigs` = exposure + the
    calibrated drain offset (§11.4); `get_deadtime` = exposure + offset.
    That one method lets a plan derive `exposure_timeout` per detector and
    lets the namespace **refuse** "essential at this rep rate" when
    exposure + offset exceeds the period.
  - `GeecsAcquireLogic` — near-empty: `ensure_ready` checks the gateway
    `CONNECTED` PV; the rest are no-ops. LabVIEW is always acquiring.
  - **Data logic, two implementations behind the same ABC**, chosen at
    namespace build from the DB (PVA-served image variable → plugin;
    `save_nonscalar_data` without one → LabVIEW-native):
    - `ADHDFDataLogic` over the #806 plugin PVs — **stock**, nothing of
      ours. Streamable; `collections_written` = `NumCaptured_RBV`.
    - `LvNativeFileDataLogic` — writes path + name template to
      `localsavingpath`, toggles `save`, describes the resource as
      directory + template + per-shot index (we own the names, §11.6).
      A per-event reading — there is no write-complete readback (§10.1) —
      with a bounded end-of-run check that every expected file exists and
      has stopped growing.
  - `acq_timestamp` as a readable child with the persistent monitor
    (exists, `devices/ca/triggerable.py`); the drain offset as a config
    signal from the calibration store (§4.F).
- **`ShotControl`** — one device, three protocols: `Movable` over
  `TriggerState` (the profile-defined writes per state — the existing
  abstraction, §11.1), `Pausable` (state-dependent, §10.3: in strict mode
  the RE pausing simply stops the plan firing and the box stays ARMED —
  `pause()` does nothing; in gated mode edges flow on their own, so
  `pause() → OFF` and `resume()` restores SCAN;
  the RE calls these on every Pausable it has seen in a message, §7).
  **Not a flyer** (amended 2026-09-11, `08_gated_batch.md` §3): the box
  has no counter, so a `complete` of its own could not know when N shots
  have gone — the detectors count, and in gated mode the *plan* drives
  the box SCAN after their `kickoff` and OFF after their `complete`.
  Replaces `shot_controller.py`'s plan-stub methods and
  `plans/pause_semantics.py`.

### B. Acquisition — three shapes, one mechanism each

**Strict (the default).** Stock plans with a GEECS `per_step` / `per_shot`
— the one extension point with no stock equivalent, because the fire must
sit *between* trigger and wait (§11.5 explains why free-running edges are
not an exact substitute):

```
move_per_step(step, pos_cache)
for each shot:
    trigger(essential, wait=False)    # StandardDetector.trigger re-baselines its count
    mv(shot_control, SINGLESHOT)
    wait(group, timeout=max(exposure_timeout over essential))
    create / read(essential) / save   # one event row
```

About 25 lines of pure `bps`, one of which is the non-stock idea. On a
`FailedStatus` from the wait: device confirmed down (the existing
`CONNECTED` liveness read) → `GeecsDeviceDownError`, the run aborts with
the device named; otherwise **keep the row** (every scalar the shot produced, the
frameless device's columns `NaN`, no frames) and fire once more for the
step — the partial-row semantics decided 2026-09-11 (`06_pva_file_plugin.md`
§2.1: with a file plugin an orphan frame is *not* unreferenced, the
bundler references every written frame, so the plugin is rewound to the
last referenced frame before the retake). This is `fire_and_await_shot`
(`plans/single_shot.py`), shared with the funnel's shot — built in phase 0
as `plans/strict.py`. **Why not `bps.pause()` on a dead device** (the
first draft said pause-fix-resume): a pause inside `take_reading` is
rewound on resume, and the RunEngine replays the stashed messages since the
last checkpoint — the step's trigger and fire included — before the plan
continues, so a retry after the pause would double-fire. The native
"fix it and continue" is the operator resuming a *checkpointed* step, which
needs the fire to be replay-safe; that is a plan-layer design item, not a
`take_reading` one.

**Non-essential stream.** Those detectors are *not* in `detectors`. They
are the bound plan's `non_essential=[…]` argument — `fly_during_wrapper`
**per plan** with the stage and the unbounded prepare the stock wrapper
lacks (amended 2026-09-11, `08_gated_batch.md` §3/§4.3: `SupplementalData`
is RunEngine-level state, the same for every run, and which devices are
non-essential is a fact of *this* scan):
`prepare(TriggerInfo(EXTERNAL_EDGE, number_of_events=0))` — unbounded —
`kickoff` at `open_run`, `complete`/`collect` at `close_run`, each in its
own stream, joined afterwards by offset-corrected stamp (§11.3). A 700 ms
camera or a dying device there never holds a shot and never aborts a run.
This is free-run's **second job** (§11.5), kept natively; its first job
(the rep-rate hack) dies.

**Gated batch (opt-in).** `acquisition="gated"` on the bound scan verbs
(`08_gated_batch.md` §4.2): per step `prepare(detectors,
TriggerInfo(EXTERNAL_EDGE, number_of_events=shots_per_step))`, `kickoff`,
`mv(shot_control, SCAN)`, `complete`, `mv(shot_control, OFF)`, a `Rewind`
to the quota, `collect`. Exact because the ordering is built into
`prepare → kickoff` and the frames are counted by the plugin that writes
them. Only for detectors that count (plugin-backed); the per-frame scalars
ride in the stack as NDAttributes (§4.4 there). Replaces free-run's
rep-rate role.

**Telemetry.** `SupplementalData.baseline` for every subscribed scalar of
the experiment (read at open and close): each scalar-only device whole
and each detector's scalar *signals* — never a detector itself, which is
Triggerable and would wait in ARMED for a shot that never comes
(`GeecsNamespace.telemetry()`, PR 2).  `monitors` for the changing few,
promoted from measurement over a Tiled run (§10.4) — an **experiment
config fact**, not a DB fact.  This resolves §10.5's "per event,
monitor-backed everything" against the native mechanism: per-event
telemetry for every device is the funnel's unstaged-read regression path
again; the baseline stream carries the same values at a cost of two rows.

### C. Run bookkeeping — a path provider and callbacks

- **`GeecsScanPathProvider(PathProvider)`** returns
  `PathInfo(scans/YY_MMDD/ScanNNN/<device>, filename, create_dir_depth=1)`.
  The day-scoped claim is the one GEECS thing with no native home: a
  ~60-line **`claim_scan` preprocessor** that, on `open_run`, claims the
  number, injects it into `md`, and points the provider at the run;
  releases on `close_run`. It does exactly one thing — it is **not** the
  #809 preamble, and must never grow a second job.
- **Callbacks:** ScanInfo ini on the start document (rewritten at the
  stop with `ScanEndInfo`); the s-file **from the run's own primary
  events** at the stop document (`callbacks.SFileCallback` — PR 2 moved it
  off Tiled: the files exist whether or not the catalog does, and an
  aborted run's rows are written like the legacy scanner's; the Tiled-fed
  export stays in `geecs_data_utils` as the offline re-export); the Tiled
  writer (exists — drop the `geecs://` descriptor patch in
  `tiled_integration.py` once stream documents replace those assets);
  `scan.log` as a callback.  All three read the claim's keys from the
  start document and write into the folder, never creating it.

### D. Clients

A client expands a **preset** into a stock plan item — every
`bluesky.plans` verb with a `per_step` / `per_shot` hook that a queue item
can express (`plan_names.GEECS_PLAN_NAMES`: `count`, `scan`, `list_scan`,
`grid_scan`, `log_scan`, the spirals, `x2x_scan`, their `rel_*` twins; not
`scan_nd`, not the deprecated aliases) registered once each with the
GEECS hook pre-bound and the **stock parameters preserved minus the
hook**, plus two keyword-only GEECS parameters that are facts of the scan
and belong in its one description: `trigger_profile` (the experiment
default when omitted; the bound plan brackets the run ARMED → STANDBY
through that profile's `ShotControl`) and `shots_per_step` (rows per
position, each a strict shot; the GEECS `per_step` also records
`bin_number`).  The preset (`geecs_schemas.Preset`, PR 2) is the device
group plus the plan call: each device becomes its namespace binding, or
`X.scalars` — the detector's scalars-only view, Triggerable like the
detector, writing no files — when `save_images` is off; scan variables
are `Device:Variable` or catalog-name strings the client resolves to the
Movable child (`U_S1H.current`); pseudo variables wait for phase 3.
Essential/non-essential (the flyers list) is phase 2.  The preset name
and the submission record ride in `md["geecs"]` as provenance only.
`qs_client.submit_plan` / `submit_preset` are the verbs; the Console and
GEECS-MCP are rewired onto them once (§10.5, §11.7).

### E. Deletions (whole modules, in the same PR as their replacement)

`plans/scan_request_plan.py`, `step_scan.py`, `free_run_step_scan.py`,
`named_plans.py`, `pause_semantics.py`, `t0_sync.py`, `liveness.py`,
`orchestration.py`; `session.py` and `scan_request_runner.py` (the resolver
functions that survive move to the client); `capture/` after #806;
`devices/contributor.py`, `nonscalar_save.py`, `shot_id.py`,
`ca/timestamped_readable.py`. Roughly 9k of GeecsBluesky's 24.6k lines.
**Kept:** `qs_client`, `optimization`, the `devices/ca` bases,
`namespace.py`, `action_compiler.py`, the qserver deploy tree. The refactor
is **in place** (§10, Q2 answered): the keepers are half the package and
the Console and MCP import them; a new package would re-home them for no
gain.

### F. Calibration

A standalone `measure_shot_offsets` plan (OFF → wait the longest device
timeout in the set → one fire → read every stamp) writes each device's
drain offset to the configs repo, from where the config signal reads it.
Strict runs carry the same data for free (one fire, every device waited
on, shot time known), so "sync" can also be recomputed from any recent
strict scan. Sam's validation shortcut (OFF, stalled stamps within
tolerance) becomes a `qserver_ready`-style preflight — never a scan step,
because it costs at least the longest device timeout per check (§11.2).

---

## 5. The mapping

| GEECS today | Native replacement |
|---|---|
| save set, as a device list | the plan's `detectors` argument; a client-side preset |
| save set `synchronous` flag | essential (`detectors`) vs non-essential (the bound plan's `non_essential` list, streamed per plan — `08_gated_batch.md` §4.3) |
| `save_nonscalar_data`, `localsavingpath`, `save` | the detector's data logic, opened and closed per run |
| save set explicit scalar list | the device's own readables, individually addressable |
| save-set rituals, setup/closeout | plan stubs and `finalize_wrapper` (#647) |
| `background_telemetry` | `SupplementalData.baseline` + `monitors` |
| scan variable alias | the namespace attribute (`U_S1H.current`) |
| `kind: motor`, `confirm:`, pseudo | the device class, chosen once at namespace build |
| trigger profile states | `ShotControl`: `Movable` over the states, `Pausable`; in gated mode the plan drives it SCAN/OFF around the detectors' `kickoff`/`complete` (not a flyer — `08` §3) |
| strict single shot | stock `per_step` with the fire between trigger and wait |
| free run — the rep-rate job | gated batch: fly, detectors count |
| free run — the contributor job | the non-essential stream: `non_essential=[…]` on the bound plan (`fly_during_wrapper` per plan) |
| Gate-2 save windowing | the detector's own capture window (open at prepare, close at unstage) |
| `acq_timestamp` as the shot join key | **kept** — offset-corrected, it *is* the shot id (§11.3); positional for essential detectors, by stamp for the non-essential stream |
| `shot_id`, `shot_offset`, `bin_number` | `seq_num`, the stamp, and the per-device drain offset as a config signal |
| the t0-sync ritual | a once-run calibration plan + a preflight validation (§4.F) |
| scan number and folder | a `PathProvider` plus the `claim_scan` preprocessor |
| ScanInfo ini | a start-document callback |
| s-file, Tiled catalog | callbacks (already true) |
| capture daemon | deleted by #806 |
| `ScanRequest` as a worker instruction | a client-side template that expands into a plan call |
| the funnel, named plans, the #809 preprocessor | deleted |

---

## 6. What stays GEECS

Six things have no native home, and none of them is in the scan path's
logic:

1. **The DB as the source of truth** for the device roster, types,
   tolerances and subscribed variables. The namespace builder owns this.
2. **Day-scoped scan numbering** with a multi-writer claim protocol. The
   `claim_scan` preprocessor plus the path provider.
3. **The s-file format** and its legacy `Device Variable` column headers.
   A callback, plus header metadata on the devices.
4. **ScanInfo ini.** A start-document callback.
5. **PV naming and the served-set rules.** Already owned by the two
   gateways and `geecs_core`.
6. **The fire between trigger and wait** — one line in `per_step`. It
   exists because the trigger box has no counter and the cameras stamp with
   a per-device latency (§11.5).

`ScanRequest` and its JSON Schema also survive, as the **client-side**
record of intent the GUI needs. What dies is its role as a worker-side
execution instruction.

---

## 7. Verified versus assumed

**Verified 2026-09-09 against the installed environment** (ophyd-async
**0.19.3**, bluesky **1.15.0**, tiled **0.2.9**), by reading the source,
not the docs:

- `StandardDetector` is a three-logic composition: `DetectorTriggerLogic`
  (`prepare_internal/edge/level`, `get_deadtime`, `config_sigs`,
  `default_trigger_info`), `DetectorAcquireLogic` (`ensure_ready`,
  `start_acquiring`, `wait_for_idle`, `ensure_stopped`) and
  `DetectorDataLogic`, composed by `add_detector_logics`
  (`core/_detector.py`). **#806 swaps exactly one of the three.**
- `TriggerInfo` fields: `trigger`, `livetime`, `deadtime`,
  `exposures_per_collection`, `collections_per_event`, `number_of_events`
  (**0 means unbounded**), `exposure_timeout`. `DetectorTrigger`:
  `INTERNAL`, `EXTERNAL_EDGE`, `EXTERNAL_LEVEL`. (An earlier #807 comment
  wrote `number_of_triggers`; that name does not exist in 0.19.3.)
- External triggering calls `start_acquiring()` inside `prepare`, so
  detectors are capturing before any `kickoff` — the ordering that makes
  gated mode exact.
- `kickoff()` re-reads `collections_written` on every call and honours
  `events_to_kickoff`, so `prepare(N)` then N × (kickoff → fire →
  complete) composes, and a failed `complete` can be re-kicked.
- `trigger()` re-baselines `collections_written` through
  `_update_prepare_context` on every call, so it is repeatable per step
  with `EXTERNAL_EDGE`; it refuses a context prepared with
  `number_of_events != 1`.
- `StandardFlyer`, `FlyerController` (`prepare/kickoff/complete/stop`);
  `PathProvider`, `PathInfo(directory_path, filename, create_dir_depth)`,
  `StaticPathProvider`, `AutoIncrementingPathProvider`, `YMDPathProvider`.
- `ophyd_async.epics.adcore`: `ADHDFDataLogic`, `NDFileHDF5IO`,
  `ADAcquireLogic`, `ADContAcqTriggerLogic` exist under those names.
- bluesky: `Pausable` is called on **every object the RE has seen in any
  message** (`run_engine.py:1268` for suspend, `:1531` for pause), so a
  device that appears in a `set` is paused. `SupplementalData(baseline,
  monitors, flyers)`, `fly_during_wrapper`, `monitor_during_wrapper`,
  `baseline_wrapper`, `bps.prepare/kickoff/complete/collect/
  collect_while_completing/declare_stream`, `bp.count(per_shot=)`, the
  `per_step` hook on the scan plans, `one_nd_step`, `move_per_step`.
- `TiledWriter` converts `Resource` → `StreamResource` and knows the
  `application/x-hdf5` mimetype (`callbacks/tiled_writer.py:203-238`).
- The s-file is exported **from Tiled** at the stop document
  (`sfile_callback.py`), so Tiled's ingestion is load-bearing for the
  s-file as well as the catalog.
- **Naming has moved**: the writer base is not `DetectorWriter` in this
  version, and the flyer's controller is `FlyerController`, not
  `TriggerLogic`. Older docs and blog posts will disagree.
- **Verified 2026-09-11 for phase 2** (`08_gated_batch.md` §2, with line
  numbers): `number_of_events=0` is unbounded but `kickoff` raises on a
  frame written between `prepare` and `kickoff`; `complete` is the count
  wait *then* `wait_for_idle` on the last kickoff; `exposure_timeout` is
  per update, not per batch; `fly_during_wrapper` neither stages nor
  prepares; multi-object `collect` needs a declared stream and cuts at
  the minimum index; the stock `ADHDFDataLogic` sets `NumCapture=0` and
  describes every NDAttribute the driver's XML declares; our plugin
  stores `NumCapture` without honouring it and `Rewind` moves the stale
  watermark.
- **On hardware (M2, Scan 065):** the three-logic split fits a GEECS camera
  with no areaDetector IOC — `GeecsDetector` = `GeecsTriggerLogic` +
  `GeecsAcquireLogic` + `ScalarsDataLogic` + `LvNativeFileDataLogic` under
  stock `bp.count` / `bp.list_scan` with the strict `take_reading`; the
  only GEECS line in the scan path is the fire between trigger and wait.
  Learned on the way: the shot wait lives in the acquire logic's
  `wait_for_idle` (a per-event data provider has no count to wait on); the
  baseline must be synchronous in `trigger()` (pinned by a mock race test);
  `stage()` resets the prepare context, so it must be waited on
  (`stage_all` does, a bare `bps.stage` does not). N shots per point is
  `bp.count`'s `per_shot` inside the step — fly-per-step is not needed
  for strict; fly stays for gated mode (phase 2).

**ASSUMED, must be verified before designing on it** (narrowed from the
first draft; each names the phase that retires it):

- ~~that the PVA gateway plugin can count distinct, fresh frames
  losslessly~~ **Built and pinned offline 2026-09-11** (`06_pva_file_plugin.md`;
  the stock `ADHDFDataLogic` drives the real plugin over `pva://` in
  `GeecsPvaGateway/tests/test_file_plugin.py`); on hardware with the
  rollout (§8 of 06)
- ~~that Tiled 0.2.9 reads the plugin's NDFileHDF5-layout files~~ the
  adapter path is read (`consolidators.py`, `tiled/adapters/hdf5.py`:
  `swmr=True, libver="latest"` on the closed file, verified locally with
  h5py 3.16); a Tiled read of a real run is the rollout's step 3
- ~~OFF latency / timeout-event posting~~ **Measured 2026-09-09
  (`04_phase0_measurements.md` M1):** OFF stops edges within one period
  (put 155 ms, one in-flight edge, then silence); the gateway posts **no**
  timeout events — its change suppression (`gateway.py`, "don't re-post an
  unchanged value") drops them — so liveness is the `CONNECTED` PV, never
  monitor silence. Also learned: a single shot fires on the **next external
  edge** (stamps ~1 s after the put), so 1 Hz strict needs < ~550 ms of
  software between stamp arrival and the next fire put completing.
- that per-device drain latency is constant across exposure settings, as
  §11.4 states. **First measurement** (M1): offsets span 0–220 ms across 42
  live devices — wider than the ~100 ms estimate but inside period/2.
  Constancy across shots and settings still to be shown (phase 0).

---

## 8. Sequencing: decided 2026-09-09

**Option 1½ — a phase 0 on today's hardware, then #806 and the plan layer
in parallel.** Sam chose this over the first draft's "#806 first".

The reasoning that changed the recommendation: #806 replaces **one of the
three detector logics**. The trigger logic (DG645 edge, LabVIEW free-runs)
and the acquire logic survive #806 unchanged; only the data logic swaps
from "LabVIEW native save" to `ADHDFDataLogic`. So making a GEECS camera a
genuine `StandardDetector` *now*, with today's `NonScalarSaveSupport`
relocated into a ~50-line `LvNativeFileDataLogic`, is not building twice —
it is the same class with a throwaway third logic. It kills the
preamble-configures-devices leak immediately, proves the fire between
trigger and wait on hardware in days, and gives #806 a concrete target
("replace this data logic") instead of four ASSUMED bullets. "#806 first"
would have front-loaded every schedule risk (Windows NSSM + h5py
re-bootstrap, HDF5 over SMB, emulating the areaDetector PV set in p4p) into
the least-verified component while the scan path waited.

### Phases (each a PR into `feature/native-bluesky-plans`)

0. **DONE 2026-09-09 (Scan 065, M2).** One camera as `GeecsDetector` with `LvNativeFileDataLogic` (PNG),
   `ShotControl` as a device, the strict `per_step`; stock `bp.list_scan`
   on `U_S1H` on hardware. Measures: OFF latency, whether the gateway
   posts timeout events, drain offsets across amp4in, `exposure_timeout`
   behaviour on a real trigger.
1. **In parallel:** #806 (plugin + stock `ADHDFDataLogic`) ∥ the plan
   layer, as a deletion-led PR series (§10.5): **PR 1 — the deletions**
   (#816: funnel, free-run, session, runner,
   funnel-only devices, optimization glue; `GeecsDetector` for every
   triggerable device; stock plans exported by the profile); **PR 2 —
   the plan layer** (#821: the registration
   table, the `claim_scan` preprocessor + `PathProvider`, the ScanInfo /
   s-file / `scan.log` callbacks, the baseline telemetry, `Preset` v1
   and the client seam — §10.7); **PR 3 — headless hardware acceptance**
   (#822, `05_phase1_acceptance.md`: HTU-NoGas, `U_S1H:Current`
   −1 → +1 A in 0.5 A steps, amp4in, setpoint restored), then the worker
   flipped (2026-09-11). #806 landed as #823 with its own acceptance
   (`07_806_acceptance.md`) and #824 rolled the fleet. **Phase 1 is
   complete.** Small debts carried into phase 2's warm-ups: `run_action`
   as a queue plan (done 2026-09-11, #827), the presets corpus on the configs repo's main (done
   2026-09-11), the watch period before `Compression=zlib`.
2. Gated batch + the non-essential stream — designed 2026-09-11 in
   `08_gated_batch.md` (three PRs: the plugin's per-frame scalar
   attributes; the worker's `acquisition="gated"` + `non_essential` +
   `essential` in presets; the s-file from stream data).  Free-run was
   deleted in #816.
3. The calibration plan + the preflight validation.

**On #809:** do not merge. Nothing from it is deployed; its two open P1s
need no fix if it does not ship. Close it with a pointer here once this
amendment lands, so a green PR does not tempt a later session. Salvage by
hand: `fire_and_await_shot` (the strict refire, §4.B), the document-parity
test, `GeecsNamespace.select`'s role assertions, `plan_session.py`.

**What keeps working while this happens.** Sam's team is the only user.
Master stays deployable; the feature branch rebuilds; the worker checkout
flips per hardware-accepted milestone. Nothing runs beside the funnel and
no dual-door parity is built — the parity test in #809 existed only because
two doors had to coexist.

**What the clean slate unlocks, and should be used for:**

- delete free-run, the funnel, the named plans and the capture daemon
  **eagerly**, as soon as each has a replacement, rather than in a phased
  retirement
- change the event schema, the s-file columns and the ScanRequest schema
  freely where the native shape is better; there is no reader to break
  that the team does not own
- treat "we already built it" as carrying no weight. The only question is
  whether a thing is right

---

## 9. Standing constraints (Sam, 2026-09-08/09)

- Prefer a native Bluesky or ophyd-async mechanism wherever one plausibly
  exists. Examine carefully before patching. Never close off adding a stock
  plan or a Bluesky feature later.
- Added lines must be justified. Never a second copy of a solved problem.
  Where a better copy replaces an old one, the old one goes in the same
  change.
- Do not hold on to anything that does not slot in completely cleanly.
- **Clean slate.** Sam's team is the only user. No backward compatibility,
  no deprecation cycles, no migration paths. Deleting is cheaper than
  adapting, and "we already built it" is not an argument.
- Phase PRs land into `feature/native-bluesky-plans`. Each gets an
  adversarial review pass using the `/land` three-lens brief, with every
  finding dispositioned, before Sam reviews. Master merges are
  maintainer-only.
- Hardware is available: trigger profile **HTU-NoGas**, scannable
  `U_S1H:current` −1 → +1 A in 0.5 A steps, save set **amp4in**, restore
  the setpoint. `ssh geecs-gw`; the worker checkout may be changed freely.
- Run **one** test suite at a time on the Mac, unbuffered
  (`python -u -m pytest -v`). A backgrounded pytest writing to a file looks
  stalled for minutes because stdout is block-buffered; that cost an hour
  in the last session and produced a retracted claim.
- Verify API details against the installed versions; never assert them
  from memory. §7 is the ledger.

---

## 10. Open questions

Answered 2026-09-09: Q1 sequencing → option 1½ (§8). Q2 refactor in place
(§4.E). Q3 downtime → flip the worker per milestone, no coexistence (§8).

Still open, for Sam:

1. ~~Write-complete readback on LabVIEW-native devices~~ **Answered
   2026-09-09 (Sam): there is none.** `acq_timestamp` advancing says the
   capture succeeded; nothing says the write finished. Consequence:
   `LvNativeFileDataLogic` is a **per-event reading**, and because we own
   the filenames the worker enforces the contract itself — at the run's end
   it waits, bounded, for every expected file to exist and stop changing
   size (two agreeing stats), and fails the run loudly otherwise. Once #806
   moves the cameras to the plugin, this path serves only the few non-image
   proprietary devices.
2. ~~Which amp4in devices are non-essential by default~~ **Answered
   2026-09-09 (Sam): amp4in is one camera, essential, no non-essential
   devices — it is a test preset.** Phase 0 therefore exercises the strict
   path only; the two-list model gets its first real test later, on a
   preset with a slow or optional camera.
3. ~~What `pause` drives~~ **Answered 2026-09-09 (Sam): depends on the
   mode.** Strict: nothing — the plan stops firing, the box stays ARMED;
   that is the native Bluesky pause and needs no device action. Gated
   (fly): edges flow on their own, so `pause() → OFF` and `resume()`
   restores SCAN. `ShotControl.pause()` is therefore state-dependent:
   ARMED → no-op, SCAN → OFF.
4. **Where the baseline/monitor split is recorded** — the experiment
   defaults in the configs repo is the proposal; the first list comes from
   measuring one Tiled run.
5. **Phase-1 decisions (Sam, 2026-09-10):**
   - **Deletion-led phase 1**, one PR series on the feature branch: delete
     the funnel, free-run, named plans, `session.py`, the runner, the
     funnel-only devices and their tests first; the plan layer lands in the
     same series; the worker flips once at the end. The services stay on
     `master` throughout, so the lab loses nothing until the flip.
   - **Queue-item contract:** stock plan names registered once with the
     strict `take_reading` pre-bound; `md["geecs"]` is provenance only.
   - **Presets and save sets are disposable configs; the concepts stay** —
     a preset because the same scan is run often, a save set as a grouping
     of devices. **Per-scalar selection is deprecated**: it existed to
     populate the s-file from the old console. **The s-file represents
     every scalar in the run documents.**
   - **Scan number claimed worker-side** (the `claim_scan` preprocessor).
   - **Outputs kept as callbacks:** ScanInfo ini, the s-file from Tiled,
     and **`scan.log`** (the trace of what went wrong in a scan).
   - **Namespace rule:** `GeecsDetector` for every `looks_triggerable`
     device; `native_save=True` iff the DB lists `save` and
     `localsavingpath` as **settable** for the device (only settable
     variables get a gateway `:SP`, PV_CONTRACT.md §1);
     drain offsets from the calibration file, 0.0 until measured.
   - **Telemetry in phase 1 = everything**: every subscribed scalar of the
     experiment rides in the run, per event, monitor-backed; pare back
     (baseline/monitors) later from evidence, not up front.
   - Feature-branch → master: decide after phase 1 is complete and
     exercised.
   - **#812 (test speed) goes first.**
   - **Foundation first, top layers later (Sam, 2026-09-10):** the feature
     branch is long-running and develops the complete, stable
     implementation in logical foundational steps; the Console and MCP are
     rewired *once*, when the foundation is stable — not reworked at every
     step. Phase 1's acceptance is therefore headless (RunEngine / queue
     server level), not through the Console.
   - **`GeecsSession` goes.** It was the headless engine for scans and
     tests; native patterns replace it with a small helper that builds a
     RunEngine with the namespace, preprocessors and callbacks installed.
   - **The optimization stack stays importable** (its tests keep passing);
     Xopt on the new shape gets its own hardware acceptance later.
   - **GEECS-MCP gets only the edit that keeps it importable** when the
     runner and `preflight.py` go; its real update waits for the stable
     foundation.
   - **PR 1 scoping (Sam, 2026-09-10, second round):** a **preset** is
     the device group (`device`, `save_images`, `essential`) **plus the
     plan call** (stock plan name + args/kwargs with device names as
     strings) — a saved queue item; the client expands the names against
     the namespace at submission. Setup/closeout rituals and the
     `SaveRole` enum are dropped; explicit action plans stay queue items.
     PR 2 deletes `save_set.py`, its converters and the configs-repo
     `save_sets/` tree and regenerates the corpus as presets. The
     **optimization trim is option B**: the Xopt core
     (`evaluators`, `generators`, `base_optimizer`, `config_models`,
     `inspection`) stays importable with its tests; `plans/optimize.py`,
     `optimize.py`, `session_bridge.py`, `worker_loader.py` and the
     loader hook go — optimization is broken until it is re-glued to the
     native scan path in its own phase. The **function_execute verbs**
     (`geecs_move_variable`, `geecs_describe_action`, `run_action`) go
     with the session: a manual move is a stock `mv` queue item, a
     preview is client-side resolver work. **Deployment** (site.env,
     `render_units.sh`, the units) is touched once, at the end of the
     feature branch when things freeze — not per PR.
7. **PR 2 decisions (2026-09-10, the four open questions, taken as the
   recommended defaults — Sam's brief left the answers unfilled):**
   - **Telemetry shape:** `SupplementalData` baseline at open and close
     over every scalar-only device and every detector's scalar signals
     (§4.B); monitors empty until measured.  Not per-event everything
     (§10.5's wording): triggering a detector in the baseline would wait
     for a shot ARMED never delivers, and per-event reads of every device
     are the funnel's unstaged-read regression path.
   - **When to claim:** every run the worker's RunEngine opens.  No md
     opt-out (md is provenance) and no "only with a detector" rule — a
     scalar-only magnet scan still wants its number and s-file.  The
     hermetic switch is `make_run_engine(claim=False)`.
   - **Preset v1 fields:** `name`, `description`, `trigger_profile`,
     `background`, `devices[{device, save_images}]`,
     `plan{name, args, kwargs}` — `plan` optional so the 46 Undulator
     save elements regenerate as device groups without an invented plan
     call; `essential` is phase 2.  The two GEECS keyword arguments on
     the bound plans (`trigger_profile`, `shots_per_step`) are the one
     deliberate deviation from "stock signature preserved" — both are
     facts of the scan and must be in its description, not a side
     channel.
   - **Cadence:** deferred to PR 3's measurement (the every-other-edge
     motor-scan cadence, M2).
   - **Settled from the code:** the s-file headers ride in the **start
     document** (`geecs_scalar_headers`, read by `geecs_data_utils`'s
     exporter and the browser's display names); the ScanInfo keys
     downstream parses are `Scan Parameter` (ScanAnalysis), `Start` /
     `End` / `Step size` / `Shots per step` / `ScanMode` /
     `ScanStartInfo` (the scans database), `Background`
     (`ScanPaths.is_background_scan`).
   - **The s-file comes from the documents, not Tiled** (§4.C), and is
     written for any exit status with rows.
   - **Corpus regeneration is on a configs-repo branch** (`presets-v1`),
     not main: the deployed master worker still reads `save_devices/`;
     the branch merges with the worker flip (PR 3).  **Merged to `main`
     2026-09-11**, after the flip — nothing deployed reads `save_devices/`
     any more, and two legacy documents found untracked in the share clone
     were regenerated as presets in the same push.
8. **PR 3 facts (2026-09-10/11, `05_phase1_acceptance.md` M6):** the fire
   request is asynchronous to the laser, so a cold shot's request-to-frame
   delay is uniform over one period; after the first shot the loop is
   phase-locked to the edges.  On `UC_Amp4_IR_input` (0.70 s exposure)
   edge → message ≈ 0.8 s and the fire put ≈ 108 ms, so the strict path
   has ~100 ms of per-shot margin beyond its own ~7 ms — it holds in
   process and not in the manager's worker process (2 s repeats).  At a
   1 ms exposure (M7) both losses go: 1 Hz repeats through the manager,
   a moved step on the second edge — the exposure sets the margin, the
   plan layer cannot.  A 0.5 A `U_S1H` move is a
   1.3 s blocking set, so a moved step lands on the third edge (M2's
   second was a faster move that day).  Strict single-shot is therefore
   not the 1 Hz mode; phase 2's gated batch is.  Recorded here so the
   cadence fix is scoped as "gated batch", not "a faster fire".
9. **Phase-2 design (2026-09-11, `08_gated_batch.md`)** — awaiting Sam's
   answers to its §6: scalars of non-plugin devices in a gated run
   (baseline-only v1 recommended), "N frames each" as the gated meaning
   of essential, which scalars ride in the stack (the subscribed list
   recommended), the s-file join rule, mode as a keyword, pause mid-batch
   failing the step.  Two amendments already taken as read: the box is
   not a flyer (§4.A) and the non-essential list is per plan (§4.B).
10. Two small carry-overs, unrelated to this direction: write
   `Amplitude.Ch AB: 0.5` explicitly in every state of `HTU-NoGas` so "no
   gas" stops being order-dependent, and add a check that all profiles in
   an experiment manage the same variable set.

---

## 11. Facts from Sam (2026-09-09) that constrain the design

Recorded verbatim in substance because a fresh session will not have them
and every one of them changed a design choice.

1. **Trigger states.** OFF = nothing leaves the DG645. STANDBY = the state
   the machine idles in when not scanning — almost always external rising
   edges, because the laser fires regardless and letting hardware trigger
   keeps GUIs up to date. SCAN = the same edges at data-taking amplitude.
   ARMED = strict only, source to single-shot. SINGLESHOT = the momentary
   fire. The names are an abstraction so another trigger box can carry the
   same five states with different writes. **Consequence:** OFF is the
   only quiet state; STANDBY is not quiescent and was never meant to be.
2. **Device timeouts.** A GEECS device emits a TCP event either on a
   successful acquisition or, failing that, when its own timeout expires
   (1.5 s for ~95 % of devices); the timeout event carries an *unchanged*
   `acq_timestamp`. Nothing "stalls" on its own. **Consequence:** observing
   quiescence costs at least the longest device timeout in the set, so it
   belongs in a once-run calibration or a preflight, never in a scan.
   **Measured (M1):** the timeout events never reach the CA gateway's
   `acq_timestamp` PV — its change suppression drops unchanged values, so
   monitors go silent in OFF — and they cannot serve as a miss signal
   either; liveness is the `CONNECTED` PV.
3. **`acq_timestamp` is a shot id in everything but name.** It advances
   only on a successful capture, deterministically, on domain time
   NTP-synced to ~5–10 ms. Cross-device values for one shot differ by a
   per-device constant (point 4), so after subtracting that constant all
   stamps land within NTP jitter and rounding to the nearest period has
   ~490 ms of margin at 1 Hz. Naive rounding *without* the offset fails at
   bucket boundaries as the laser's phase drifts. The re-pushed idle
   frames and stale pre-scan frames the capture daemon filters are
   **delivery** artefacts of the TCP push, not properties of the stamp.
4. **What the stamp measures.** Exposure time and trigger delay are backed
   out: `acq_timestamp` = trigger arrival + the latency of draining the
   frame, a per-camera constant (~100 ms spread between a 4 MB and a 0.5 MB
   camera). The TCP *message*, however, is sent only when the exposure
   completes — a 700 ms exposure makes the message arrive most of a second
   after the shot. **Consequence:** the stamp governs the join; the arrival
   governs how long a plan waits (`exposure_timeout`); the two are
   different quantities and the design keeps them apart.
5. **Essential vs non-essential.** Sam wants per-device control over what
   happens when camera A misses shot k: "we cannot miss shots from these
   cameras, but missing shots from those is fine; throttle acquisition for
   the first set, never for the second." A non-essential device dying
   mid-scan must not force an abort. A 700 ms camera belongs on the
   non-essential list precisely so nobody waits for it. **Consequence:**
   the non-essential set is a separate stream joined by stamp, because its
   frame for shot k may arrive during shot k+1; and deleting free-run must
   keep this role, which free-run's "contributor" devices carried. Why the
   fire is not stock: with edges free-running, `trigger()` baselines each
   detector on its last seen stamp, and stamps for one edge arrive
   50–100 ms apart across cameras; an edge inside that window while the
   trigger messages go out leaves camera A waiting for k+1 while B records
   k — roughly 5–10 % of steps misaligned at 1 Hz. Firing *after* every
   baseline is set makes each row exact by construction.
6. **Proprietary file formats.** Only the writer SDK is proprietary; the
   file names and paths are ours to choose. **Consequence:** the
   LabVIEW-native data logic can describe its resource the same way the
   plugin does; the only remaining gap is the write-complete readback
   (§10.1).
7. **The sync ritual Sam built** — withhold the trigger, watch the stamps
   stop advancing, fire one shot, watch every stamp step — measures each
   device's drain offset for one known shot. The shortcut (OFF, stalled
   stamps within ~200 ms ⇒ already synced) validates it. Sam does not want
   it baked in as-is; a "sync" button that runs it and updates the
   per-device offsets is the shape he imagines. **Consequence:** §4.F.
8. **Scope notes.** GEECS-MCP was one-shotted and can be updated later —
   not a constraint now. The Console follows the client-expansion change.
   A service on the worker host that is a browser client of the qserver
   ("the scanner through the browser") is the end point Sam has in mind;
   GEECS-DataPortal is already a FastAPI service on that host speaking to
   Tiled, so it is the natural landing — not now, but nothing here closes
   it off.

---

## 12. What we lack, honestly

Asked directly whether a Bluesky-native ecosystem can be delivered on
these foundations: yes, and not cosmetically. Stock plans, documents,
Tiled, the queueserver and the clients carry ~90 % of the leverage and do
not care how frames are counted. Two writer paths is normal — every
serious EPICS facility runs areaDetector file plugins beside detectors that
write their own formats (Eiger/Odin, Pilatus CBF, PandA HDF5), and
`DetectorDataLogic` exists precisely so the writer need not be an AD
plugin. The gaps, after §11, are three, and none is in the scan path's
logic:

1. **Lossless frame delivery.** The re-push and stale-frame behaviours are
   the TCP push's; the #806 plugin dedupes on the stamp before the
   latest-wins slot, and that is the **only** place such logic may live
   (built 2026-09-11: `GeecsPvaGateway/geecs_pva_gateway/file_plugin.py`;
   the gateway's `CLAUDE.md` names the two opposite delivery contracts).
2. **No write-complete readback on the LabVIEW-native path** (§10.1) —
   replaced by our own end-of-run file check; contained to the non-image
   proprietary devices once #806 lands.
3. **A native home for the non-essential stream** when free-run goes —
   `fly_during_wrapper` per plan behind the bound plans' `non_essential`
   argument (§4.B; `08_gated_batch.md` §4.3).

Two hazards #806 already names that deserve more weight than the issue
gives them: HDF5 written on Windows over SMB and read on Linux (disable
file locking; never read during write — a rule, not best-effort), and the
CA monitor as the shot signal (monitors can coalesce under load; fine at
1 Hz, **verify the gateway's posting guarantee before any design above
it**).

Shot identity is **not** a gap (§11.3). The first draft of this document
said it was, and was wrong.
