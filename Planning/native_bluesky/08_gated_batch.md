# Phase 2 — gated batch and the non-essential stream

**Status (2026-09-11): design, argued before code.** Phase 1 is complete
and deployed (`03_clean_room_rebuild.md` §2). This document is the
argument for phase 2 as `03` §8 lists it — the gated batch (the 1 Hz mode
`05_phase1_acceptance.md` M6/M7 say strict single-shot is not) and the
non-essential stream (free-run's second job, `03` §11.5) — with every
API claim checked against the installed source (ophyd-async 0.19.3,
bluesky 1.15.0) and the repo's own plugin.  Where it changes `03`, `03`
is amended in the same PR (its staleness rule): §4.A (the box is not a
flyer), §4.B (the non-essential stream is per plan, not RE-global), §7
(the new verified facts), §8 and §10.  The open questions at the end are
Sam's; the recommendations are mine.

Read `03` §11 first: every choice below follows from those hardware
facts, and the ones that matter most here are §11.1 (OFF is the only
quiet state), §11.3 (the stamp is the shot id), §11.5 (why free-running
edges plus per-shot triggers are not exact).

---

## 1. What phase 2 delivers

Two things, one mechanism each, and a preset field:

1. **Gated batch** — `acquisition="gated"` on the bound scan verbs.  The
   box free-runs in SCAN while the plugin-backed detectors *count* the
   frames they write; the plan drives the box OFF when every essential
   detector has its quota.  Exact by construction because arming
   (`prepare`) precedes the edges (`kickoff`, then SCAN), and the frames
   are counted by the thing that writes them.  This is the mode for a
   camera whose exposure eats the strict margin (M6: 0.70 s exposure →
   2 s repeats through the manager; M7: 1 ms → 1 Hz holds).
2. **The non-essential stream** — `non_essential=[…]` on every bound
   plan (strict or gated).  Those detectors are prepared unbounded,
   kicked off at `open_run`, completed and collected at `close_run`, in
   their own stream, joined afterwards by offset-corrected stamp.  A
   700 ms camera or a device that dies there never holds a shot and
   never aborts a run.
3. **`essential`** on `PresetDevice` (default `True`), which the client
   expands into the two lists.

Nothing is deleted: free-run went in #816.  The gateway grows one thing
(§4.4, per-frame scalar attributes), which is what makes a fly stream
carry the same columns a strict row does.

---

## 2. Verified against the installed source (2026-09-11)

Line numbers are from the installed files (`GeecsBluesky/.venv/…`).

**ophyd-async `core/_detector.py`**

- `TriggerInfo.number_of_events` is `NonNegativeInt`, "0 means infinite"
  (`:96`).  `number_of_collections = number_of_events ×
  collections_per_event` (`:107`).
- `prepare()` for `EXTERNAL_EDGE` calls `trigger_logic.prepare_edge(num=
  number_of_exposures, livetime)`, builds the prepare context
  (`_update_prepare_context`, which **reuses** the data providers when
  `collections_per_event` is unchanged — `:464-500` — so re-preparing per
  step or per shot never re-opens the plugin's file), sets
  `events_to_kickoff = number_of_events` (`:605`) and calls
  `start_acquiring()` (`:608`) — the detector is capturing **before**
  any `kickoff`.  The context records `collections_written` at prepare
  (`:511`), the baseline every later count is measured from.
- `kickoff()` (`:671-704`) re-reads `collections_written` and
  `events_to_kickoff`, requests `events_to_kickoff × cpe` collections and
  **raises** if `written + requested` exceeds `ctx.written +
  number_of_collections` (`:685-693`).  Consequences: (a) prepare(N) then
  k kickoffs summing to N composes (the `events_to_kickoff` soft signal
  is the per-kickoff quota, `:385-393`); (b) with `number_of_events=0`
  the budget is `ctx.written + 0`, so **a frame written between `prepare`
  and `kickoff` makes `kickoff` raise** — the unbounded stream must be
  kicked off right after it is prepared, while no edge can arrive.
- `complete()` (`:706-715`) is `_wait_for_index`: observe every
  provider's `collections_written_signal` until the minimum reaches
  `initial + requested`, **then `acquire_logic.wait_for_idle()` when
  this is the last kickoff** (`:548-549`, `is_last_kickoff` `:697`).  Our
  `GeecsAcquireLogic.wait_for_idle` is the stamp wait (§4.2 changes it).
  With `requested = 0` the count wait returns on the first observed value
  (`:547`).
- The wait's `timeout=trigger_info.exposure_timeout` is passed to
  `observe_signals_value` (`:534`), whose `timeout` is **per update**
  ("If an update is not produced in this time then raise", `_signal.py`),
  not a deadline for the whole batch.
- `collect_asset_docs(index)` (`:756-770`): without an index, `get_index()`
  = min `collections_written` over the detector's own providers ÷ cpe.
  `StreamResourceDataProvider.make_stream_docs` (`_data_providers.py:139-155`)
  emits one datum per resource covering `last_emitted → indices_written`
  and carries the TODO "fail if we get dropped frames".
- `trigger()` refuses a context prepared with `number_of_events != 1`
  (`:649`): strict and gated are different prepares of the same device,
  never mixed in one prepare.

**ophyd-async `epics/adcore/_data_logic.py`**

- `ADHDFDataLogic.prepare_unbounded` sets `NumCapture = 0` unconditionally
  (`prepare_file_paths`, `:160-161`, "Overwrite num_capture to go
  forever"), sets `Capture = 1` without waiting for set completion
  (`:206-208`), and describes, **beside the frame dataset, every
  NDAttribute the driver's `NDAttributesFile` XML declares** as its own
  data key at `/entry/instrument/NDAttributes/<name>` (`:104-131`,
  `:215-229`, `dtype` from the `datatype`/`dbrtype` attribute).  So a
  per-frame scalar the plugin writes under that group is a stream column
  with nothing of ours on the worker.

**bluesky 1.15.0**

- `fly_during_wrapper` (`preprocessors.py`) inserts `kickoff` + `wait`
  after `open_run` and `complete` + `wait` + `collect` before
  `close_run` — **no `stage`, no `prepare`**.  `SupplementalData` is a
  RunEngine-level preprocessor (`RE.preprocessors`), the same list for
  every run; its `flyers` cannot vary per plan.
- `collect` with more than one object requires the objects to be
  `WritesStreamAssets`, collects up to the **minimum** `get_index()`
  across them (`bundlers.py:1076-1130`), and requires a `declare_stream`
  first ("If collecting multiple objects you must predeclare a stream
  … and provide the stream name", `:1103-1107`).  With one object no
  index is passed (`:1130`), so its datum covers everything written.
- `bp.fly` is `open_run → kickoff all → complete all → collect each →
  close_run`; `collect_while_completing(flyers, dets, flush_period,
  stream_name)` completes with `wait(error_on_timeout=False)` and
  collects each period.
- A deferred pause is processed at the next `checkpoint`; a plan region
  without one is not interrupted by a deferred pause (`run_engine.py:394`);
  `rewindable_wrapper(plan, False)` stops the RE replaying messages on
  resume.  The RE calls `pause()` / `resume()` on every Pausable it has
  seen in a message (`03` §7).

**This repo**

- `GeecsPvaGateway/geecs_pva_gateway/file_plugin.py`: `NumCapture` is
  stored and echoed (`_Param("NumCapture", "i", 0, rbv=True)`) and **not
  honoured** — `_on_frame` never reads it; the session ends on
  `Capture=0` only.  `Rewind` (`_rewind`) truncates the frame and
  attribute datasets to *n*, posts `NumCaptured_RBV = n` and sets
  `stale_before = now`, so a frame arriving later with an older stamp is
  dropped as stale (`06` §2.1).  `NDAttributesFile` is served as XML
  declaring `acq_timestamp` and `recv_timestamp` (`ATTRIBUTES`,
  `NDATTRIBUTES_XML`), written per frame by `_append`.
- `server.py:_on_frame(var, update)`: the TCP push hands the plugin the
  blob and the stamp; `update` is the **whole message** — every variable
  of the device in that acquisition — so the plugin can write the
  device's per-shot scalars beside the frame without a second
  subscription.
- `GeecsBluesky/geecs_bluesky/devices/detector.py`:
  `GeecsAcquireLogic.wait_for_idle` waits for a stamp update ≠ the
  baseline and raises `GeecsTriggerTimeoutError` after `shot_timeout`;
  `GeecsDetector.prepare` is already overridden (the plugin's
  `WriteMessage` note); `discard_uncollected` rewinds every plugin to
  `provider.last_emitted`.
- `ShotControl.pause()` drives OFF from SCAN or STANDBY
  (`models/shot_control.py: QUIESCE_FROM`) and `resume()` restores the
  standing state — already the gated-mode pause `03` §10.3 asked for.
- Trigger profiles (configs `main`): `HTU-NoGas` SCAN = external rising
  edges; `HTU-LaserOFF` SCAN = **Internal** (the DG645 self-triggers), ARMED
  and OFF = single shot, STANDBY = external edges.  So with the laser off
  a gated batch *can* be exercised (SCAN flows at the box's internal
  rate) while STANDBY delivers nothing.

---

## 3. Two things `03` got wrong, and one it left open

1. **`ShotControl` is not a flyer.**  §4.A gave it a `FlyerController`
   (`prepare → OFF`, `kickoff → SCAN`, `complete → N shots then OFF`).
   The box has no counter (`03` §6.6); its `complete` cannot know when N
   shots have gone.  The detectors count.  The box stays what it is —
   `Movable` over the states, `Pausable` — and the *plan* drives it:
   SCAN after the detectors are kicked off, OFF after they complete.
2. **The non-essential stream is a plan argument, not RE state.**  §4.B
   named `SupplementalData.flyers`.  That list is installed once on the
   RunEngine and applies to every run; which devices are non-essential
   is a fact of *this* scan (a preset's `essential` flags), so it belongs
   in the plan's description like `trigger_profile` does.  `03` §7
   already allowed "or `fly_during_wrapper` per plan".  Per plan it is —
   and since `fly_during_wrapper` neither stages nor prepares, the bound
   plan's wrapper does both before kickoff (§4.3).
3. **Where the scalars of a gated run come from** was never said.  A
   fly-shaped run has no per-shot `create/read/save`; the only per-shot
   record is what the plugin writes.  §4.4 answers it natively — per-frame
   NDAttributes, the areaDetector pattern the stock data logic already
   describes — and §6 asks Sam about the devices that have no plugin.

---

## 4. The design

### 4.1 One description: `acquisition` and `non_essential` on the bound plans

The registration table (`plans/registry.py`) already appends two
keyword-only GEECS parameters to every scan verb; phase 2 appends two
more, both facts of the scan:

- `acquisition: Literal["strict", "gated"] = "strict"` — which
  `take_reading` the hook binds.  Strict is unchanged.  Gated binds
  `gated_take_reading` (§4.2) into the same `per_step` / `per_shot`, so
  `scan([cam], U_S1H.current, -1, 1, 5, shots_per_step=10,
  acquisition="gated")` is the whole description and the stock plan still
  moves, checkpoints and records metadata.
- `non_essential: Sequence[Readable] = ()` — detectors streamed for the
  run's duration (§4.3).  Names resolve like `detectors` do (the manager
  resolves device references in any argument).

Both ride in the start document beside `trigger_profile` and
`shots_per_step`.  Separate plan names (`gated_count`, `gated_scan`, …)
were considered and rejected: they double the table, and a preset that
switches mode would change its plan name rather than one field.

### 4.2 Gated batch — `gated_take_reading`

Per step (after `move_per_step`), for the essential detectors *D* (all
plugin-backed — a detector without a streamable provider fails at
`kickoff`, "not streamable", before the box moves) and the box *B*:

```
prepare(D, TriggerInfo(EXTERNAL_EDGE, number_of_events=shots_per_step,
                       exposure_timeout=per-frame budget))     # capture on, count baselined
kickoff(D, wait=True)                                          # quota = shots_per_step
mv(B, SCAN)                                                    # edges flow
complete(D, wait=True)                                         # every D counted its quota
mv(B, OFF)                                                     # edges stop (≤ 1 in flight, M1)
wait_for(D.truncate_to_quota)                                  # Rewind to baseline + quota
collect(*D, name="primary")                                    # one datum per D: the step's frames
```

- **Why prepare per step, not once.**  `prepare` baselines
  `collections_written`; a per-step prepare makes every step's quota
  relative to its own start and `is_last_kickoff` true every step, and
  it costs nothing (providers are reused, §2).  Preparing the whole scan
  once and revising `events_to_kickoff` per step works too but leaves
  the quota arithmetic in the plan; the per-step form keeps it in the
  device.
- **Why the box moves after `kickoff`, not before.**  With
  `number_of_events = N`, `kickoff` raises if frames landed between
  prepare and kickoff (§2).  OFF before prepare guarantees none did:
  the step opens in OFF (the run brackets OFF → … → STANDBY under
  `acquisition="gated"`, replacing strict's ARMED bracket).
- **Exactness.**  `complete` returns when the *slowest* essential
  detector has its quota, so the faster ones hold quota + (edges that
  passed meanwhile); after OFF, at most one more edge is in flight (M1).
  `GeecsDetector.truncate_to_quota()` — `discard_uncollected`'s sibling —
  rewinds each plugin to `ctx.collections_written + quota`; `Rewind`
  moves the stale watermark, so the in-flight frame is dropped when it
  arrives.  Every stack gains exactly `shots_per_step` frames per step;
  `collect(*D)` then sees equal indices and the datum covers the step.
  **The rewind is preferred to honouring `NumCapture`** in the plugin:
  it uses the verb #823 already built for the same purpose, keeps the
  plugin's session semantics untouched (strict re-prepares per shot in
  one session — a `NumCapture` honoured per session would end strict's
  session after one frame), and costs one truncated frame per step.
- **What "essential" means here.**  The box flows until every essential
  detector has N frames — *N frames each*, not the same N edges.  A
  camera that drops one frame in a step ends the step one edge later
  than its neighbours and its N stamps are shifted by one edge at the
  drop; the join by stamp (§4.5) shows exactly that.  Same-edge rows are
  strict's promise, and strict is one keyword away.
- **`wait_for_idle` is mode-aware.**  `complete` calls it on the last
  kickoff (§2); the stamp wait is a strict-mode concept (the shot the
  plan fired).  `GeecsDetector.prepare` records the mode from
  `number_of_events` (`1` → strict, else fly) and the acquire logic's
  `wait_for_idle` returns immediately in fly mode — the count *is* the
  completion.  No new method: the stock `prepare` already visits our
  override.
- **The row.**  A frame plus that device's per-frame attributes (§4.4)
  — the same columns a strict row carries for that device, the stamp
  included.  `bin_number` is the datum's ordinal (one collect per step);
  no `BinCounter` reading is needed, the s-file writer derives it
  (§4.5).  `shots_per_step` keeps its meaning (rows per position).
- **Timeouts.**  `exposure_timeout` is per frame (§2): one period plus
  the device's exposure and drain, the same budget strict uses
  (`DEFAULT_SHOT_TIMEOUT`).  A camera that stops producing frames for
  that long fails `complete` with a `TimeoutError` → translated (as
  `trigger` already does) into `GeecsTriggerTimeoutError`; the plan
  drives OFF in a `finalize_wrapper` and the step fails loudly.  The
  refire idea has no analogue here — the box is already delivering
  edges; a stalled essential camera is a fault, not a drop.
- **Pause.**  The step body is wrapped `rewindable_wrapper(…, False)`
  and contains no `checkpoint`, so a *deferred* pause lands between
  steps (after OFF, before the next move) — the stock stepped-scan
  behaviour.  An *immediate* pause mid-step triggers
  `ShotControl.pause()` → OFF (already built); `complete` then times out
  per frame and the step fails on resume rather than resuming a batch
  whose count baseline is stale.  Documented, not papered over: gated
  steps are short (`shots_per_step` × period), and a resume-mid-batch
  would need the plugin's count and the box's state re-baselined
  together — a phase-3 item if anyone ever needs it.

### 4.3 The non-essential stream — per-plan `fly_during` with a prepare

The bound plan wraps the stock plan (strict or gated alike):

```
stage(NE)                                                      # stage_wrapper already stages `detectors`;
prepare(NE, TriggerInfo(EXTERNAL_EDGE, number_of_events=0))    #   NE are added to it
kickoff(NE)   ← at open_run, immediately after prepare        # box is OFF/ARMED: no frame in between
… the plan …
complete(NE); collect(each NE alone, name=<device>_stream)    ← before close_run
unstage(NE)
```

- `fly_during_wrapper` is the stock insertion; the bound plan's
  `finalize`/`stage` bracket supplies the stage and the prepare it
  lacks.  Each NE is collected **alone** (one object → no index, the
  datum covers everything it wrote), in a stream named after the device:
  a joint stream would cut every camera at the slowest one's count.
- `complete` on an unbounded prepare: count wait returns at once
  (`requested = 0`), then `wait_for_idle` — a no-op in fly mode (§4.2).
  Nothing waits on a non-essential camera, ever.
- A non-essential device that dies mid-run: nothing awaits it, so
  nothing aborts; `collect` references whatever it wrote.  The stack
  check callback reports frames vs referenced per stream as it does
  for primary.
- **Non-essential requires a plugin** (a streamable provider).  A
  LabVIEW-native camera on a box without the plugin has no count and
  cannot be a flyer; the client preflight refuses it before submission
  (the manager's device tree lists the `hdf` child of every plugin-backed
  detector — the existing reference check extended by one rule).  The
  `.scalars` view cannot fly either; `essential: false` with
  `save_images: false` is refused at expansion.
- Strict runs with a non-essential list: `discard_uncollected` in the
  partial-row path rewinds only the devices *of the shot*; the streams
  are untouched (they were never referenced per event).

### 4.4 Per-frame scalars in the stack — the plugin's one addition

The stock data logic describes every NDAttribute the driver's XML lists
(§2).  The plugin already writes two.  Phase 2 extends `ATTRIBUTES` to
**the device's subscribed scalar variables** (the DB `get='yes'` list —
the same rule the namespace uses for a detector's event columns, so a
gated row and a strict row carry the same columns for that device),
taken from the `update` dict the gateway already holds at `_on_frame`,
written as `DOUBLE` datasets under `/entry/instrument/NDAttributes/`
(strings and enums as their numeric wire value where one exists,
otherwise skipped — the XML is generated per device from the DB rows the
gateway reads for the served set, so the worker knows the columns at
`prepare`).  Missing keys in a push (a variable the device did not send
that shot) are written `NaN`.

What this buys: one file per camera per run holds frames + stamp + that
camera's scalars, positionally exact because they arrived in one TCP
message; Tiled reads them as columns; nothing on the worker changes.

### 4.5 The s-file for a run with stream data

`SFileCallback` writes rows from the run's primary *events*.  A gated
run has none (its primary stream is datum-only), and a strict run with
a non-essential list has events plus streams.  The rule stays "the
s-file represents every scalar in the run documents"; the writer gains
a second source: at the stop document, after the plugins finalize (the
`StackCheckCallback` thread already waits for that), the attribute
datasets of every referenced stream resource are read and joined —
per essential stream by frame index within each datum (the bin), across
streams and with the non-essential streams by **offset-corrected stamp
rounded to the period** (`03` §11.3; the drain offsets are the
detectors' config signals, in the descriptors).  A non-essential frame
with no essential row within half a period gets its own row, blank
elsewhere — data is never dropped from the file.  The Tiled export in
`geecs_data_utils` follows the same rule offline.  This is the largest
single piece of phase 2 and is its own PR (§5).

### 4.6 Presets

`PresetDevice.essential: bool = True`.  `expand_preset`: essential
devices → `detectors`; `essential: false` → `non_essential=[…]`
(refused with `save_images: false`, §4.3).  `PlanCall.kwargs` carries
`acquisition` like it carries `shots_per_step`.  The corpus needs no
regeneration (defaults keep every preset strict and all-essential).
The client preflight adds the `hdf`-child rule for non-essential
references.

---

## 5. Sequencing — three PRs, each with its own acceptance

1. **2a — the gateway's attributes** (GeecsPvaGateway minor; Data-Utils
   patch): subscribed scalars as NDAttributes, XML per device from the
   DB rows, `NaN` for a missing key; `scan_stack` reads the attribute
   group; the offline test drives the real plugin over `pva://` as
   #823's does.  Hardware: one strict scan on `UC_Amp4_IR_input` shows
   the columns in the stack equal to the s-file's for that device.
2. **2b — the worker**: `acquisition` + `non_essential` on the bound
   plans, `gated_take_reading`, `truncate_to_quota`, the mode-aware
   `wait_for_idle`, the OFF bracket, the preflight rule, `essential` on
   `PresetDevice` (GEECS-Schemas minor) and its expansion; the s-file
   writer *skips* stream-only runs with a `scan.log` line until 2c.
   Hardware (a runbook like `05`'s, through the staging manager): a
   gated `count` and a gated `scan` on `U_S1H` with two plugin-backed
   cameras, frames == quota per step per camera, stamps one period apart,
   the box OFF between steps and STANDBY at the end; a strict scan with
   one camera non-essential, its stream's frame count ≈ the run's edges;
   a non-essential camera disconnected mid-run, the run completing.
   With the laser off: `HTU-LaserOFF` (SCAN = internal), which exercises
   everything but the laser's phase.
3. **2c — the s-file from streams** (GeecsBluesky minor; Data-Utils
   minor): §4.5, pinned by an offline test over a synthetic run and by
   re-exporting 2b's scans.

`03` §8's "free-run deleted" is already true (#816).  The optimization
re-glue and the web scanner v1 stay after phase 2 as listed.

---

## 6. Open questions for Sam

1. **Scalars of non-plugin devices in a gated run.**  A scalar-only
   device (a magnet, a gauge) or a LabVIEW-native acquirer has no
   stream.  v1 as designed records them in the **baseline** only (open
   and close).  The native per-shot alternative is `monitors`
   (`monitor_during_wrapper` over their subscribed signals — one event
   per update per signal, joined by CA timestamp in the s-file writer).
   Recommendation: baseline-only in 2b, monitors as a 2c option once the
   join code exists — a gated run is for the cameras.
2. **"N frames each, not the same N edges"** (§4.2) — acceptable as the
   meaning of essential in gated mode?  The alternative (same edges)
   is strict.
3. **Which scalars ride in the stack** (§4.4): the subscribed list
   (recommended — the row is the same in both modes) or every scalar in
   the push.
4. **The s-file join rule** (§4.5): a non-essential frame with no
   essential row within half a period gets a row of its own
   (recommended) or is dropped from the s-file (kept in the stack).
5. **Mode as a keyword** (`acquisition="gated"`, recommended) versus
   separate plan names.
6. **Pause mid-batch fails the step on resume** (§4.2) — acceptable for
   v1?

Answers go into `03` §10 as items, per its rule; this document is
amended to match.
