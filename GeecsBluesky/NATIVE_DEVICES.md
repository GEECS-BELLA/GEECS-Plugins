# Native EPICS devices in the scan path — design note

Status: **built** — `NativeCamera` (#1117), the registry and the
bind-time refusals (#1118); hardware acceptance pending. Once accepted,
the rules below move into `CLAUDE.md` and this note is deleted. §6 is a
**proposal**, not built: native cameras in gated runs and as
non-essential streams.

## Why

A device can now be served by a native EPICS IOC instead of a LabVIEW
device app (`GEECS-BELLA/geecs-epics`: a Basler on a Linux camera host
behind ADAravis, writing HDF5 through its own `NDFileHDF5` plugin). This
note is how such a device takes part in a GEECS scan.

Settled (`geecs-epics/docs/decisions.md`): **one owner per device** — the
GEECS DB or geecs-epics; a host migrates whole; IOC configuration is
EPICS-native. Integration code, the device registry included, lives on
the GEECS side (decision 6).

**The lesson of the red team:** a native camera cannot be the stock
`AravisDetector` with a few attributes added. The stock detector's
per-shot acquire model, its paths, its stack layout and its timeouts all
differ from what the scan path assumes. It must be a **GEECS camera whose
transport is areaDetector**: the same contract as `GeecsDetector`, over
stock AD records.

## Rules in `CLAUDE.md` this changes

- *"The DB as the roster's source of truth"* → the DB **plus** a native
  registry in the configs repo, disjoint by construction (§2).
- *"touches devices only through the gateway's CA PVs"* → also a native
  IOC's own CA PVs (stock areaDetector records); still never a device
  command outside EPICS.
- The #868 caveat (a refused `signal.set()` reads as success) applies to
  the native plugin's puts exactly as to the gateway's.

## Scope of the first build

In: one **native camera** as an **essential detector** in **strict**
`count` and `sweep`, frames in `ScanNNN/<device>/<device>.h5`, joined to
the rows by stamp, registered in Tiled, found by analysis.

Refused at bind with a named error (not absent — each falls through a
GEECS-only branch otherwise): gated runs, non-essential streams, the
calibration plans, `optimize` (its live frames match on gateway PVs),
scan-mode profiles (the IOC never acknowledges `ScanStart`; box-less
profiles fire no edge), a preset that asks for `X.scalars` of a native
camera. Out entirely: other device kinds, native settables in the
scanner's lists, the archive set.

## 1. The registry — in the configs repo

`<experiment>/native_devices.yaml` in the configs repo (private,
per-experiment, already pulled on the worker; no deploy key):

```yaml
schema_version: 1
devices:
  UC_ExampleCam: {kind: aravis_camera, trigger_source: software}
```

- **Names, kind and trigger source only.** The PV prefix is derived by decision 5
  (`<experiment>:<device>:`, GEECS normalization), so it cannot drift from
  `ioc.yaml`; the record suffixes are fixed per kind (`cam1:`, `HDF1:`).
  No hosts, no paths: facility values keep their one home.
- Schema: a new GEECS-Schemas model (proposed: NativeDevices in an
  epics module), v1; `kind` a discriminated union. Read through `ConfigsRepoResolver` like every configs document.
- geecs-epics CI checks every registry name has a `services/*/ioc.yaml`
  serving the derived prefix (it reads the configs repo; not the reverse).
- The test IOC (`ADTEST:CAM1`) is renamed to the convention before it is
  registered.

## 2. One owner, enforced — new code

- Migration = disable the device in the experiment **and** add it to the
  registry, in one change, then restart the gateways (they serve until
  restarted; `--include-disabled` must not be set) and re-derive the
  archive set. A checklist in geecs-epics `docs/` covers this and the
  configs that name the device's old variables (presets, catalog,
  actions, analysis).
- `GeecsNamespace` gains an explicit check: a registry name whose
  `safe_name` is also a roster device's → refuse to
  open, naming both owners. The existing check compares the roster only
  with itself.

## 3. The device — the proposed NativeCamera (a new native module under devices/)

A `StandardDetector` over stock AD IO (`AravisDriverIO`, `NDFileHDF5IO`)
with GEECS logics:

- **Trigger source** (per camera, the registry's `trigger_source`):
  `line1` / `line0` — the camera's hardware input on that line, wired to a
  DG645 output (the pulsed-event mode; a Basler's opto-isolated input is
  `Line1`, a FLIR Blackfly's `Line0` — the two behave identically here,
  only the line differs); `software` — `trigger()` executes the camera's
  `TriggerSoftware` command, so the frame is exposed when the worker says
  go, not when the laser fires. Software mode's timing is an accepted
  caveat (Sam, 2026-10-22): right for continuous or slow signals,
  shot-on-demand and dry runs; verified on the bench camera, one frame per
  trigger and none while armed and idle. Both take the same path below;
  only the source and that one command differ. ophyd-async's
  `AravisTriggerSource` knows only `Line1`, so `NativeCamera` defines its
  own sources — and types the driver's `TriggerSource` as a **string**,
  not an enum: an enum signal demands every member among the PV's choices
  at connect, and the choices are each camera's own (a Basler has no
  `Line0`, a Blackfly no `Line1`). The first prepare checks the
  configured source against the PV's choices **before the camera is
  touched** and refuses a line the camera lacks, naming its choices and the
  registry file.
- **Acquire model** (as GEECS cameras run): driver in `Continuous`,
  `TriggerMode On`, `TriggerSource` per the registry; **armed once per run** at the
  first prepare, the HDF plugin capturing in Stream mode for the run. The
  order matters: ophyd-async runs trigger logic → data logic (capture
  starts) → acquire logic, so the trigger logic's first `prepare_edge`
  **stops** a free-running camera and sets trigger mode *before* the
  capture starts (no live-view frame reaches the stack), and the acquire
  logic restarts it after (on its readback, not the put);
  `unstage` restores the driver's pre-run state (live view keeps working
  between scans). No per-shot `Acquire` writes — the stock model's
  per-shot AcquisitionStart is both too slow for 1 Hz and, in Continuous,
  a hang (`ADAcquireLogic.wait_for_idle` awaits a put that never ends).
  Strict calls `prepare` **every shot** (`strict.py`), so the trigger and
  acquire logics make every prepare after the run's first a no-op: no
  `Acquire` put, the event count ignored.
- **The run lock** (geecs-epics `run_lock.db` + `access.acf`;
  `Planning/device_panels.md` §6): the run owns the camera from the
  **first** step of the first `prepare_edge` — before the source check and
  the snapshot — until the camera is restored. `RunLockIO` sets
  `RUN:Owner` (`host:pid @HH:MM` — who and since when; the device does
  not know the scan's number at prepare) and `RUN:Lock` 1, and **waits for the
  lock's readback** (a refused put can complete as success, #868): an IOC
  that does not take it refuses the run, the camera untouched. While it is
  set the IOC refuses acquisition writes from anyone but the worker's CA
  user (panels, Phoebus, an agent). It is released in `ensure_stopped`
  only once the camera is back — after the restore succeeds, or when this
  session took the lock but never snapshotted (nothing was touched). A
  failed restore keeps it set: the camera is not back. Arming and restore
  are serialised (`_Session.guard`): an unstage sent while a prepare is
  still in flight — a sibling detector's prepare failed fast, a halt's
  sweep — waits for it, so the lock is never released mid-arming. Every camera IOC
  must serve the two records (the chain contract): a camera without them
  fails to connect.
- **A shot** = the plugin's `NumCaptured` advancing past the baseline the
  stock `trigger` task reads. With `Software` the press follows that read
  in the same task, so the order is guaranteed; with a hardware line the read
  races the plan's next message, the box fire, which is safe in practice
  (the fire put alone takes ~200 ms, then exposure and the writer's lazy
  open, against a ~1 ms read) but not by construction — unlike
  `GeecsDetector`'s synchronous stamp baseline. The wait's timeout is the
  prepared `exposure_timeout` and raises `GeecsTriggerTimeoutError`,
  setting `missed_shot`.
- **No orphan frames in v1.** The stack must hold exactly the frames the
  datums reference, in order: Bluesky's `HDF5Consolidator` sizes the array
  as the sum of datum widths and reads from frame 0, so a skipped frame
  silently shifts every later frame in Tiled and the portal by one shot.
  `GeecsDetector` avoids this by rewinding its plugin; stock `NDFileHDF5`
  cannot rewind. So, for v1:
  - a native camera that **misses a shot fails the step** (no refire —
    `discard_uncollected` raises, naming the camera); a frame it *did*
    write survives another device's miss (strict calls
    `discard_uncollected` on every plugin-backed device; with one external
    key per event the frame's datum is valid);
  - **a pause never replays a shot**: strict wraps each shot in
    `rewindable_wrapper(all_safe_rewind(devices))`, and a device without a
    `rewindable` attribute counts as safe — a pause after the frame landed
    would replay trigger + fire on resume and the append-only writer would
    hold two frames for one row (Bluesky then fails the run: "multiple
    indices in a stream_datum for one event"). `NativeCamera` declares
    `rewindable.get() → False`, so the resume continues where it stopped
    (GPT review on #1115; pinned by a mid-shot pause test);
  - a run with a native camera **and** a gateway plugin camera is refused
    at bind (a refire of the other camera would need a native rewind);
  - a stray frame (an edge outside a shot) is the same fault — hence the
    cabling rule in Acceptance.
  Refires come back when the Tiled writer registers native stacks through
  a consolidator that honours the seq_num→index map (a second
  substitution in `tiled/writer.py`, tested against real Tiled).
- **Stamp, from the stock pair** (verified on ADCore 3.14, geecs-epics#2):
  a stock IOC cannot write one Unix float per frame (no arithmetic in the
  attribute XML, no `CONST` type; ADAravis' `NDArrayTimeStamp` is the
  camera's power-up clock). The per-frame wall clock is the EPICS
  timestamp pair the writer always stores, `NDArrayEpicsTSSec` +
  `NDArrayEpicsTSnSec` (1990 epoch). So **the reader computes it**:
  `scan_stack.frame_stamps` reads the pair (`timestamps_dataset` stays GEECS-only)
  (Unix = `sec + 631152000 + nsec·1e-9`) beside the two GEECS spellings —
  any stock areaDetector stack becomes readable, and `is_stack_file`
  accepts it. `NativeCamera`'s event column reads the same pair from the
  plugin's `EpicsTSSec_RBV` / `EpicsTSNsec_RBV` and carries it in the
  **LabVIEW** epoch (+ `LABVIEW_EPOCH_OFFSET`, the `test_epoch_contract.py`
  constant) with `_column_headers`, as every row stamp does; one epoch for
  both and every join reads `NaN` without an error. Requires the IOC
  host's clock NTP-disciplined (PTP later) — the stamp is the join key.
- **Finalized** (verified, geecs-epics#2): the IOC's layout writes
  `finalized = 1` on **`/entry`** when the file closes (a root-level layout
  attribute segfaults the writer). The stack check's `_is_finalized`
  learns to read `/entry` as well as the root (#1116).
- **The HDF data logic keeps that layout** (GPT review on #1115): the
  stock `ADHDFDataLogic.prepare_unbounded` sets `XMLFileName` to `""` and
  `SWMRMode` on before the capture — the layout, and with it the marker,
  would be gone, and a SWMR file cannot take the attribute at close
  anyway. A native data logic reads the IOC's `XMLFileName` before the
  stock prepare and puts it back, with SWMR off, after it — before the
  first frame (the writer opens the file lazily at the first frame, and a
  triggered camera sends none before the first trigger).
- **Path**: a POSIX variant of `PluginPathProvider` — `PurePosixPath`, the
  worker's path mapped through `config.ini [Paths]
  geecs_native_ioc_data_base_path` (unset = the worker's own root: the IOC
  hosts mount the share at the same path, bind-mounted unchanged into the
  container). `create_dir_depth=0`: the IOC creates nothing; the provider
  makes `<device>/` inside the claimed folder, as today.
- **Liveness**: the asyn port's connection (`cam1:AsynIO.CNCT`,
  `Connect`/`Disconnect`) through a derived signal rendering the verdict's
  `Disconnected`. `devices/ca/liveness.py` stays the one rule and stays
  fail-open: with the whole IOC down the read fails and the gate passes
  it, and the run stops at the device's connect, still before the claim. (devIocStats' heartbeat only says the IOC is up, and a
  counter never reads `Disconnected`; a connected but stalled camera is
  caught by the strict timeout.)
- **Facts strict reads**: `name = safe_name(device)`,
  `_geecs_device_name`, `plugin_backed = True`, `native_save = False`,
  `missed_shot`, `plugin_reasons()` (the plugin's `WriteMessage`),
  `discard_uncollected()` (above), `count_zeroed`, `_geecs_namespace_member`.

## 4. Failure behaviour

| Fault | What happens |
|---|---|
| Camera disconnected from its IOC | liveness gate refuses before the claim, naming it |
| IOC down | the device's connect fails at stage, before the claim (the gate is fail-open) |
| Share not mounted on the IOC host | `FilePathExists_RBV = 0`; prepare fails naming the path |
| No edge for a shot | `GeecsTriggerTimeoutError` → the step fails naming the camera and the plugin's message (no refire in v1) |
| Stray frame (edge outside a shot) | stack and datums disagree → the stack check reports it; fixed by cabling, not tolerated |
| Native + gateway plugin camera in one run | refused at bind |
| Registry device enabled in the DB | namespace refuses to open |
| Registry unreadable | environment open fails (configured but broken ≠ none) |
| The IOC does not take the run lock (`RUN:Lock` stays Free) | prepare refuses, naming it, before the camera is touched |
| Restore fails at unstage | the error is raised; the run lock stays set (cleared by hand: `RUN:Lock` 0, after checking no run holds the camera) |

## 5. Where the changes land

- **GEECS-Schemas**: the registry model (proposed above) + docgen.
- **GeecsBluesky**: a new native module under devices/ (proposed), the
  POSIX path provider, the namespace merge + ownership check, the bind-time refusals, the stack
  check's `/entry` finalized read, `CLAUDE.md`.
- **GEECS-Data-Utils**: `scan_stack` reads the stock EPICS timestamp pair.
- **configs repo**: `<experiment>/native_devices.yaml`.
- **geecs-epics**: the layout + attributes XML (finalized; done in #2), the renamed
  test IOC, the migration checklist, the CI name check.
- **Deploy**: IOC hosts mount the share at the worker's path, run chrony.

## Acceptance

**Bench, done** (2026-10-22, software trigger, the device in a strict-shaped
plan from a client, not yet the queueserver): 5 shots → 5 frames, datums
`0–1 … 4–5`; the IOC's layout still selected after prepare and
`/entry.finalized == 1` after unstage; `find_stack_file` finds the stack
and every row's stamp joins its own frame; the camera back in free-run.

**Now, software trigger** (no cable): a strict 10-shot `count` of the
registered camera (`trigger_source: software`) through the queueserver
gives 10 frames in `ScanNNN/<device>/<device>.h5` that `find_stack_file`
finds, an s-file of 10 rows joined by stamp, a clean stack check, the run
in Tiled with frame *k* = shot *k*, the frames in the portal; the live
view runs before and after.

**Later, a hardware line** (`line1` on a Basler, `line0` on a Blackfly) —
**cabling rule**: the camera's input line hangs off a DG645 output **gated by the trigger
box's states**, as a GEECS camera's input does — the camera captures every
edge it sees all run long, so a free-running channel puts a stray frame
in the stack on every edge.

With the camera cabled that way: the same 10-shot run with
`trigger_source: line1` (or `line0`), and a deliberately blocked edge fails the step
naming the camera.

## 6. Gated runs — proposal (not built)

Status: **design for review**. Everything above stays true until this is
built; the refusals in "Scope of the first build" are lifted only as each
PR below lands. Names marked *proposed* do not exist yet.

### The idea in one line

A native camera in a gated batch is an **append-only, stamp-joined
essential**: the batch waits for its frame count like any plugin camera,
but its stack is never rewound and every frame it writes is referenced by
a datum in a stream of its own — so a missed or extra frame is a
**missing or orphan frame in the stamp join**, never a shift.

Why that is enough: in a gated run nothing downstream pairs frame *k*
with shot *k* by position. The s-file (`callbacks/sfile.py`) and the
offline re-export (`geecs_data_utils.tiled_export`) join a stack to the
`shots` rows by offset-corrected stamp
(`geecs_data_utils.shot_join.join_frames_to_shots`); the portal and
analysis (`geecs_data_utils.shot_files`, `stack_frame_index_map`) then
key each frame by an exact-millisecond match on the s-file's
`<device> acq_timestamp` column — so **that column is load-bearing**, and
it must survive a re-export too (PR A). An orphan frame
is already part of that contract ("the extra edge at a step's end, a
frame taken during an interrupted step" — the module docstring). The one
positional reader is Tiled's `HDF5Consolidator`, which reads frame 0 …
Σ(datum widths) − 1 of the file: it is wrong only when a frame **in the
middle** of the file is referenced by no datum. A stream whose datums
cover every frame the writer wrote, contiguously, cannot shift — whatever
the frames are.

The precedent is the gated run's **LabVIEW native-saving essentials**
(`native_essentials` in `plans/gated.py`; their check in
`callbacks/stack_check.py`): a dropped frame is a missing file, not a
retake; rows without a file and files without a row are counted
separately, a WARNING, never a failure. Native EPICS cameras get the same
policy with a stack instead of a folder of files.

### How a gated batch drives the camera

Unchanged from strict — **armed once per run**, nothing per batch:

| Setting | Value | When |
|---|---|---|
| `ImageMode` | `Continuous` | first prepare of the run (`prepare_edge`, as today) |
| `TriggerMode` / `TriggerSource` | `On` / `Line1` | first prepare, readbacks waited (as today) |
| `Acquire` | 1 | first prepare, on its readback; never toggled per batch |
| HDF `FileWriteMode` / `NumCapture` | `Stream` / `0` (unbounded) | first prepare (stock `prepare_unbounded`); one file per run |
| HDF `Capture` | 1 until `unstage` | stays open across batches, steps and pauses |

- **Per batch** the plan calls `prepare(cam, gated_trigger_info(remaining))`
  as it does for every plugin camera. ophyd-async reuses the prepare
  context when `collections_per_event` is unchanged
  (`StandardDetector._update_prepare_context`), so a batch prepare writes
  nothing to the IOC: it re-reads `NumCaptured_RBV` as the batch's
  baseline. `complete` is the stock one — done when the count passes
  baseline + remaining. `kickoff` is **not** the stock one: a frame
  landing between the prepare and the kickoff (a late in-flight frame)
  makes the stock kickoff raise "Kickoff requested … only prepared up
  to …" (`StandardDetector.kickoff`) — a gateway plugin is immune because
  its rewind set a stale watermark, a native writer has none. The native
  kickoff re-baselines on the count it reads — the **prepare** context's
  baseline too, which `frames_this_batch` and the sampler's row gates
  read, not only the kickoff's (the late frame belongs to no batch's
  quota; its datum still covers it).
- **The quota is enforced by the box, not the camera**: the box goes OFF
  once every essential holds its quota (`gated_take_reading`), exactly as
  today. Rejected alternatives: HDF `NumCapture = quota` (the writer
  closes the file at *N* — one file per batch breaks the one-stack-per-device
  layout `find_stack_file` reads); driver `ImageMode Multiple` with
  `NumImages = quota` (an `Acquire` restart per batch — the slow path the
  strict design removed — the bench camera returns empty frames in
  `Multiple` through Aravis, and it does not remove the pause surplus
  below anyway).
- **The device side** (*proposed*, in `NativeCamera`): a fly prepare
  takes the HDF logic only — the per-event stamp logic is left out, the
  `GeecsDetector._update_prepare_context` filter (a fly prepare with the
  stamp logic in would raise "Multiple collections not supported"); the
  `kickoff` and `prepare` refusals become the `trigger_source` check
  below; `complete` gets the guard `GeecsDetector._guarded_complete` has
  (a count timeout → `GeecsTriggerTimeoutError` naming the camera and
  `WriteMessage`; an abandoned batch settles quietly), with
  `mark_abandoned`, `abandon_step` and `frames_this_batch`. Those four are
  factored out of `GeecsDetector` into a shared helper both classes use,
  not copied. **No** `truncate_to`, `truncate_to_quota` or rewinds: the
  plan never calls them on a native camera.

### The plan side (`plans/gated.py`)

Today a native camera is **not** in `gated_take_reading`'s `plugin`
list — every classification there is `isinstance(d, GeecsDetector)`, and
`NativeCamera` is a `StandardDetector` — so it would fall into the
sampler's `members`. PR C classifies explicitly (`native_cameras`, the
existing helper): natives leave `members` (and the "device set changed"
check), and join the gates, `end_batch`'s abandon, the drain-offset
maximum and the settle. The batch then has two kinds of camera:

- **rewindable** — gateway plugin cameras, as today: one joint `primary`
  datum stream, trimmed to the quota, the sampler's row gates;
- **append-only** — native cameras: each prepared, kicked off and
  completed with the batch and counted as a row gate (the sampler releases
  a row only once every gate holds that many frames — duck-typed on
  `frames_this_batch`), but **never trimmed**, and collected **alone into
  its own stream**: bluesky's bundler requires every `stream_datum` of
  one `collect` to span the same index width (`bundlers.py`, "Received
  `indices` … of a different …"), and a native camera's count differs
  from the trimmed cameras' whenever it keeps a surplus frame. The stream
  is declared at the first collect (as `declare_primary` does) and
  collected wherever `primary` is — the end of a batch and the settle of
  an interrupted one — so its datums run 0…n−1 with no gap.

Stream name (*proposed*, ruling 4): `<name>_frames`, a document
constant beside `SHOTS_STREAM` and the non-essential suffix in
`geecs_data_utils.shot_join` — `_stream` keeps meaning a non-essential,
best-effort stream. The s-file needs nothing new — it joins
every datum-only stack whatever its stream (`_RunStreams.datum_only_stacks`)
— once `frame_columns_from_attributes` reads the EPICS pair (below).

### Pause in place, resume at shot K

Sam's rule (2026-09-26): a pause holds the box OFF, settles the in-flight
shot and resumes at the shot it reached. With a native camera:

1. The pause hook (`ShotControl.hold_for_batch`) marks every batch
   device abandoned — the native camera included — and the box goes OFF.
   **The camera stays armed and the HDF file stays open**: no `Acquire`,
   `Capture` or trigger-mode write. With `Line1` cabled to a box-gated
   output, OFF means no edge, so no frame, however long the pause.
2. After the drain wait, the settle computes the committed count as today,
   **with the native cameras in the minimum**:
   K = min(sampler rows, every rewindable camera's frames, every native
   camera's frames this batch). Natives are in it because they are
   essentials — a native camera that fell behind makes the others trim to
   it and the step re-takes those shots. **The settle is per-device
   tolerant** (new): today `settle()` and every collect share one
   best-effort `try` (`plans/gated.py`, the failure path), so a native
   camera whose IOC died would raise on its count read and cost every
   device's kept shots. A native whose count cannot be read is left out
   of K and its collect skipped (a WARNING naming it); the rows and the
   other cameras' frames are recorded. That alone is not enough, because
   the native camera is also a **sampler gate**: `ShotSampler.collect`
   re-reads every gate's `frames_this_batch` through one `asyncio.gather`
   before it yields a row, so the dead IOC would raise again there and
   the kept `shots` rows would never go out. The periodic progress
   collect has the same read, and an exception from it escapes the batch
   loop directly — the loop routes only `FailedStatus` into its settle
   path, so the box would not even be driven OFF (it would keep firing
   until the run's closing STANDBY). The gate handling is therefore
   part of the design (PR C):
   - **Gate reads are tolerant.** A gate whose count read raises is
     recorded as failed (device and exception) and dropped from the
     sampler's gate set for the rest of the batch; the read returns with
     the healthy gates. This holds for every gate, gateway plugin
     cameras included — today's code has the same exposure.
   - **A failed gate is a batch failure.** After each progress collect
     the plan checks the sampler's failed gates and, if there is one,
     takes the existing failure path: `end_batch`, box OFF, drain,
     settle, record the kept shots, raise.
   - **The final collection reads no gate.** The settle computes K from
     the healthy devices, `keep(K)` trims the rows, and the sampler then
     releases its rows up to the committed K without re-reading any
     gate. Every healthy gate already holds K frames by construction,
     and the failed one is out of the run's frames anyway.
   - **The original failure is what the operator sees.** The batch raises
     the dead camera's error (`GeecsTriggerTimeoutError` naming it, the
     read's exception chained), not a later error from the settle or
     the collect. The WARNING in the log says how many shots were kept.
3. The rewindable cameras are trimmed to K and the sampler keeps K rows,
   as today. A native camera that holds **more** than K keeps them: they
   are collected into its stream with the rest (datums contiguous) and
   are orphans in the stamp join — a frame of a shot no row records.
4. On resume the next batch prepares for quota − K; the native camera's
   baseline is its current count (surplus included), so the resumed step
   appends after it. Shot K+1 is shot K+1; the native file has at most
   one surplus frame per pause.

Datum indices: per stream, contiguous across batches, steps and pauses —
every collect emits `[last_emitted, NumCaptured)`. Tiled frame *j* of the
native stream is stack frame *j*; which shot it belongs to is the stamp
join's answer, as for every gated stack.

### Missed and extra frames

| Event | Detected by | The run does |
|---|---|---|
| In-flight edge after the box went OFF (≤ 1 per batch end) | its stamp joins no row | keeps it (datum, Tiled); orphan in the stack check, INFO |
| Edge between the pause and the box going OFF | same | same; at most one per pause |
| One missed `Line1` edge mid-batch | the count lags: the box runs one more edge; that row's stamp finds no frame | the **existing essential rule** (ruling 2): counted and warned, no retake — the row's native columns read `NaN`, the extra frame is an orphan, a WARNING in the stack check, as gated runs already do for LabVIEW native-saving essentials. A non-essential native's miss is a gap in its stamp join |
| Camera stops producing frames (connected) | `complete`'s per-frame `exposure_timeout` (the run's `shot_timeout`, 3 s default) | **fails** the batch through the existing failure path: the kept shots recorded, the native stream collected (every frame referenced), then `GeecsTriggerTimeoutError` naming the camera and `WriteMessage` |
| IOC dead or camera unreadable mid-batch | the count read fails — in `complete`, a progress collect's gate read, or the settle | **fails** the batch through the failure path, box OFF first; the failed gate leaves the sampler's gate set, the per-device settle leaves that camera out of K and skips its collect, and the rows up to K are released without re-reading it, so the other devices' kept shots are recorded; the error names the dead camera; its file holds frames no datum references — its Tiled view is short, never shifted |
| Stray edge (miscabling: an edge outside SCAN) | an orphan or a contested frame in the stamp join | marked in the stack check; fixed by cabling (the rule in Acceptance) |
| Late frame landing after a batch's collect | the native kickoff re-baselines past it (the stock kickoff would fail the run) | nothing to mark: the stamp join still gives it to its own row; only which batch's datum covers it changes |
| A frame after the last collect (before `unstage`) | file longer than the datums | trailing only: no shift; the stack check reports frames > datums |

**Refuse, mark or re-fire:** a gated batch never re-fires anything (the
box runs until every essential has its count); a native camera takes
the existing essential / non-essential rules, no new choice — as an
essential the batch waits on its frames, a single miss is counted and
warned, a stopped camera **fails** the batch; as a non-essential a miss is
a gap in the stamp join. The one invariant enforced at run time
is the one that keeps Tiled right: every frame in the file before the last
collect is referenced, in order.

The stack check (`callbacks/stack_check.py`) gets a fourth shape for an
append-only stream: frames = Σ datum widths (else a hole — a WARNING
saying Tiled is misaligned, the one real defect), every `shots` row joins
one native frame within its window (else a WARNING with the count), and
orphans counted as expected (ruling 5: ≤ 1 per batch end and per pause),
logged at INFO; more than that is a WARNING.

### Software trigger versus `Line1`

| Question | `Line1` | `Software` |
|---|---|---|
| What fires a frame in a batch | the box-gated DG645 edge, every laser shot | nothing — the batch fires no command; the strict design presses inside `trigger()`, which a batch never calls |
| Decision | admitted | **refused at bind** — a known gap, strict only (ruling 1; named: "a software-triggered native camera has no edge in a gated batch") |
| Pause | OFF stops edges; file open | — |
| Missed / extra | as the table above | — |

Considered and not adopted (ruling 1): a **tick-follower** pressing
`TriggerSoftware` on every advance of the shot clock — the clock's stamp
reaches the worker ~0.45 s after the edge at 1 Hz, so the frame is
exposed after the shot and its stamp sits at the edge of the ±0.5 s join
window (it can join the next row). A **free-running** camera matched to
the rows by timestamp gives a frame *about* each shot, not *of* it; that
fallback exists in the join already and is not adopted as a gated mode.

### Liveness and timeouts

- **Before the claim**: unchanged — `liveness_gate` already judges a
  native camera by its IOC (`down_native_cameras`, connected first,
  unreadable = down) for essentials and non-essentials alike.
- **During a batch**: the per-frame `exposure_timeout` (ophyd-async's
  `observe_signals_value` budget between count updates) is the run's
  `shot_timeout`. A disconnected camera or a dead IOC stops the count →
  the batch fails as above. A connected camera that misses one edge does
  not time out (the next edge comes within a period) — it is counted and warned.
- **During a pause**: no timeout runs — the batch's `complete` was
  abandoned and settles quietly; the camera may sit armed indefinitely.
  An IOC that died during the pause is found on resume, at the settle's
  read of `NumCaptured_RBV`, before the box leaves OFF: the per-device
  settle records the others' kept shots, then the batch fails naming the
  camera.
- **The drain offset** (edge → EPICS frame stamp) is **hand-set per
  camera** (ruling 6) in the namespace's `drain_offsets`, measured on the
  bench when the `Line1` cable goes in; the calibration plans keep
  refusing native cameras (`plans/calibration.py`). The EPICS frame stamp
  is the IOC host's clock at frame arrival, not the camera's clock, so a
  camera power cycle does not reset it; it shifts with camera settings
  (exposure, ROI, pixel format, packet size) and with the host clock's
  sync. So each gated run **measures it** (PR C, ~20 LOC + tests): the
  median offset over its matched frames goes into the scan metadata, and
  a WARNING fires when it differs materially from the configured value.
  A calibration plan for natives is built only if that warning shows a
  need.
- **The shot clock** stays a GEECS triggered essential (`shot_clock`): a
  gated run whose only essentials are native cameras is refused (the
  existing "nothing counts shots" error) — deferred until a run needs
  only native cameras (ruling 3).

### Non-essential streams fall out of the same design

A non-essential plugin camera is already an append-only, stamp-joined
stream: prepared unbounded after `open_run`, collected once before
`close_run` into `<name>_stream`, never waited on, never rewound,
orphans normal (`non_essential_wrapper`). A native camera with `Line1`
needs the device side above (fly prepare, kickoff, complete) **and** the
wrapper's classification: today `refuse_free_running_non_essentials`
refuses anything whose owner is not a `GeecsDetector` ("no shot stamp"),
and the wrapper would wrap such a device in a `StampStream` instead of
letting it fly — both learn native cameras (PR D). The wrapper's
contingencies and the existing count-only stack check then apply
unchanged. So: **non-essential `Line1` natives are admitted** in strict
and gated runs alike (a strict run's refire fires an extra edge → an
orphan in that stream, normal there); **`Software` natives stay refused**
as non-essentials (nothing would press). Its stack join needs the same
EPICS-pair read below, the waiver on `frame_columns_from_attributes`
noted in its docstring.

### What stays refused

Strict runs with a native **and** a gateway plugin camera (a strict
refire still needs a rewind); scan-mode profiles; the calibration plans;
`optimize`; `X.scalars` of a native camera; a gated run with a
`Software` native (a known gap, ruling 1); a gated run with no GEECS
shot clock (ruling 3). Gated runs mixing native and gateway plugin cameras are
**admitted** — gated has no refire, each kind keeps its own trim rule.

### Tests

- **Device** (`tests/test_native_camera.py`, mocked IOC): a fly prepare
  leaves the stamp logic out and a second batch prepare reuses the
  context — no put to `FilePath`, `FileName`, `Capture` or `Acquire`;
  `complete` timeout → `GeecsTriggerTimeoutError` with the camera's name
  and `WriteMessage`; an abandoned complete settles; no rewind attribute
  is ever touched.
- **Plan** (`tests/test_gated_plans.py`, the existing fake-device
  harness): (a) native + gateway plugin camera, one surplus native frame
  per batch → `primary` trimmed, the native stream's `stream_datum`
  indices contiguous 0…n−1 across three steps; (b) a pause at shot 4 of
  10 with the native camera one frame ahead → 10 rows, the resumed
  native datums start where the settle's stopped, the surplus kept;
  (c) the native camera one frame behind → K follows it and the others
  trim to it; (d) a stalled native camera → the batch fails naming it,
  its frames collected; (e) a native IOC that goes dead mid-batch,
  exercised at both reads — the progress collect's gate read and the
  settle — with the real `keep` / `collect`: the box is driven OFF, the
  K retained `shots` events **are emitted** (K rows in the documents,
  not 0), the healthy cameras' datums are emitted, and the raised error
  is the dead camera's `GeecsTriggerTimeoutError` naming it, not a
  settle or collect error; (f) a native frame
  landing between prepare and kickoff → no failure, datums contiguous,
  and `frames_this_batch` excludes it; each test **mutation-checked** (break the
  contiguity, include the native in the trim — the test fails).
- **Join** (`GEECS-Data-Utils/tests/test_shot_join.py`): a stock stack's
  EPICS pair becomes the stamp column (LabVIEW epoch); a missed frame
  reads `NaN`, a surplus frame is an orphan, the row count unchanged.
  The re-export test: an s-file rebuilt from Tiled carries the native
  `acq_timestamp` column exactly as the live one does.
- **Stack check**: the append-only shape — a hole is a WARNING naming
  Tiled, orphans are not.
- **Tiled** (`tests/test_tiled_integration.py`, real server): a native
  stream with surplus frames registers as an array of exactly the
  stack's frames, frame *j* byte-equal to the stack's.
- **Bind** (`tests/test_native_refusals.py`): gated `Line1` admitted,
  gated `Software` refused, native-only gated refused, strict mixed still
  refused; non-essential `Line1` admitted, `Software` refused.

### Hardware acceptance (`Line1` cabled to a box-gated output)

1. Gated `count` of 50: one finalized stack of 50 + ≤ 1 frames; an s-file
   of 50 rows, each joined to its own native frame (the offset-corrected
   stamps within a few ms with the hand-set offset; the run's measured
   median offset in the scan metadata, close to the configured value —
   this step is where the offset is measured and set); stack check clean (orphans ≤ 1, INFO); Tiled's array
   length = the stack's frame count and frame *j* identical to the stack's;
   the portal shows shot *k*'s frame.
2. Gated `sweep`, 3 steps × 10: ≤ 33 frames, 30 rows each joined.
3. The scanner's Pause at ~shot 20 of 50, held 60 s, resumed: 50 rows,
   shot numbers continuous, one file, ≤ 2 orphans, the live view not
   restored during the pause.
4. The edge blocked for 5 s mid-batch: the batch fails naming the camera,
   the kept shots recorded, the file finalized, the camera restored.
5. One run with a native and a gateway plugin camera: both stacks join
   every row.
6. A strict `count` with the native camera non-essential: its stream
   joins every row; the run never waits on it.

### Where the changes land, and size

| PR | Package | Production | Tests |
|---|---|---|---|
| A — the join reads the EPICS pair, live and in the Tiled re-export | GEECS-Data-Utils | ~70 | ~90 |
| B — the fly path on `NativeCamera` (re-baselining kickoff) + the shared batch-completion helper | GeecsBluesky | ~140 | ~220 |
| C — native classification and append-only streams in `gated_take_reading`, the per-device settle with tolerant sampler gates and a gate-free final collect, the stack check's fourth shape, the per-run measured drain offset, bind-time refusals lifted for gated `Line1` | GeecsBluesky | ~260 | ~400 |
| D — non-essential `Line1` natives (refusal + wrapper classification) + the scanner preflight's gated message | GeecsBluesky, GeecsScanner | ~70 | ~120 |

About 540 production + 830 test lines for A–D; A is independent, B
before C, D after B. The CLAUDE.md rule text moves with C.

The re-export half of A: the stock `ADHDFDataLogic` declares no Tiled
part for the built-in `NDArrayEpicsTS*` pair, so
`tiled_export.read_frame_columns` (which builds columns only from parts
named `<device>-hdf-<variable>-<suffix>`) finds no native stamp. Either
the native data logic declares the pair as two datasets of its stream
(preferred) **and** the reader learns it — `read_frame_columns` keeps
only parts `parse_attribute_name` accepts and wants one stamp, not a
sec/nsec pair, so the reader combines the pair through
`epics_timestamp_to_unix` (A's ~70 LOC covers both halves); or the
re-export reads the stamps from the file the stream resource names. The
build PR picks one; without either, an s-file rebuilt from Tiled loses
the column the portal keys frames on.

### Rulings (2026-10-22)

1. **Software-triggered native cameras in gated runs**: a known gap,
   refused — strict only. No tick-follower. A free-running camera matched
   by timestamp gives a frame "about this shot"; not adopted.
2. **A missed native frame**: not a new choice — a native camera takes
   the existing rules. Essential: the batch waits on its frames, a miss
   is counted and warned, no retake (as gated runs do for LabVIEW
   native-saving essentials). Non-essential: a gap in the stamp join.
3. **A native camera as the shot clock**: deferred until a run needs
   only native cameras.
4. **Stream name**: `<name>_frames`; `_stream` keeps meaning a
   non-essential, best-effort stream.
5. **Surplus frames**: ≤ 1 per batch end and per pause accepted; the
   stack check counts them as expected.
6. **Drain offset**: hand-set per camera, measured on the bench when the
   `Line1` cable goes in; each run measures the median offset from its
   matched frames into the scan metadata and warns on a material
   difference (PR C). No calibration plan unless that warning shows a
   need.

### Direction (not built here)

The append-only, stamp-joined model this design arrives at for native
cameras could later replace much of the gated run's count
reconciliation — quotas, trim-to-K, the settle — for every device: each
camera writes what it sees, every frame is referenced, and the stamp join
decides which shot it belongs to. That is a foundational change to the
gated plan and its document contract; to be discussed, not built under
this design.

## Open questions

1. **Refires for native cameras** — v1 has none (above). Worth the
   consolidator work, or is a failed step on a missed native shot
   acceptable? Decide after running with it.
2. ~~Finalized signal~~ — settled: `/entry.finalized`, written on close
   (geecs-epics#2).
3. **`Action1`** (later) — the camera's GigE Vision action-command
   trigger: a network-broadcast trigger that, on PTP-synchronized cameras,
   can be scheduled for a precise time — software-initiated with
   hardware-grade timing. Ties into the PTP plan; not in v1.

## Kinds beyond the camera — design (not built)

Status: **design for review, provisional until built.** Everything in
§7 to §11 is a starting point, not a contract: the first build of each kind
may change it, and the maintainer's answers to the open questions below
(2026-10-23) are starting points too, never a rule a later change must
argue its way past. The roster collapses to four kinds: the
camera above, a positioner, a setpoint device and an array detector.
Integration is per kind, never per device: each kind is designed here
first, developed against geecs-epics' simulators (decision 13: a
device-level fake behind the real IOC image plus a PV-level replayer, the
real prefixes on an isolated network) and accepted on its first real
device. Decision 14 fixes the framework per class: array-like devices are
areaDetector drivers, stages motor records, setpoint devices plain
records. Names marked *proposed* do not exist and are not in backticks.
Every kind maps onto the settable contract the CA gateway already teaches
every client (`GeecsCAGateway/PV_CONTRACT.md`) rather than inventing one.

### 7. Positioner — the EPICS motor record

The Newport ESP/SMC/CONEX, NewFocus and Zaber stages, the PI hexapod and
the Thorlabs stages become motor records (the motion image the catalogue
plans, `geecs-epics/docs/device-catalogue.md`). A filter wheel is a
discrete positioner: a motor record with integer positions, not a kind.

**Registry entry** (*proposed* kind epics_motor): one device, several
axes; the axes are the record names under the prefix, so the registry
still carries names only. No tolerance, units or limits here — the record
owns them (`.RDBD`, `.EGU`, `.HLM`/`.LLM`), the one home for device facts.

```yaml
U_ESP_JetXYZ: {kind: epics_motor, axes: [x, y, z]}
```

**Device**: stock ophyd-async `Motor` per axis — `user_setpoint` (`.VAL`),
`user_readback` (`.RBV`), `motor_done_move` (`.DMOV`),
`low_limit_travel`/`high_limit_travel`, `velocity`, `motor_stop` — under
one parent noun per device (*proposed* NativeMotor) bound as
`U_ESP_JetXYZ.x`, so `mv`, a sweep axis, `GeecsNamespace.variable`,
`get_settable` and an action's `set` step find it as a `CaMotor` child.
The parent carries `_geecs_device_name` and `_column_headers` (`<name>-x`
→ header `U_ESP_JetXYZ x`): the row and `EVENT_SCHEMA.md` are unchanged.

**Move completion**: the stock `Motor.set` is a put-callback on `.VAL`,
completed by the record when `.DMOV` returns to 1 — the device's own
verdict, the analogue of the GEECS executed reply — the readback observed
meanwhile, a timeout from `.VELO` and `.ACCL`. The record retries to
`.RDBD` itself, so no `within_tolerance` poll rides on top; soft limits are
checked before the put (`MotorLimitsError`, the gateway's pre-forward
rejection). Added by the thin subclass, what the GEECS path has and the
stock class lacks: a **post-move readback check** — a limit switch
(`high_limit_switch` / `low_limit_switch`) or a following error stops the
record with `.DMOV` = 1 short of the target and the put completes
"successfully" (the #868 class), so the move fails naming the switch when
the readback is off by more than `.RDBD` — and the cause in the failure
text (`_set_logged`'s job). A motor record **has** a stop: abort calls `motor_stop`.

**Scanner and preflight**: the settables list, the readback (`<axis>.RBV`)
and the move are §10's. As a detector a positioner is a per-row scalar
device (§10), so `acquisition_refusal` sees it only as a non-essential or
a gated run's sole essential; `_check_liveness` probes only the preset's
device entries (`expand_preset`'s `devices`), never a sweep axis; the
worker's gate judges it (below).

**Catalogue entry**: `target: U_ESP_JetXYZ:x`; the `kind: motor` opt-in
is redundant for a native axis, which counts as bound so
`_log_unhonoured_motor_targets` stays quiet. A pseudo over native axes
needs `offset` and `set_current_position` (`build_pseudo`'s contract),
provided as a soft signal as `CaSettable.offset` is — never the record's `.OFF`.

**Row**: the scanned axis's readback as today; other axes of a device in the preset ride as scalar columns (question 7 for telemetry).

**Liveness**: a motor record has no CONNECTED; the axis's alarm severity
is the verdict — INVALID (COMM/UDF, what an asyn motor driver sets when
the controller stops answering) reads down. **The gate's set grows by the
run's movers**: `liveness_gate` sees only detectors and non-essentials
today, while `sweep_movers` is already resolved in the same binder for
`background_wrapper`; a device without `connected_status` gets a per-kind
severity read. The IOC itself down is the connect failure at stage.

**Simulator**: motorSim, already running as `simmotor1` on the upstream
motorsim image (`geecs-epics/hosts/sim/compose.yaml`). **Acceptance** on
the first real stage:
`mv` lands within `.RDBD`; a 3 × 10 `sweep` has 30 rows with the axis at
each target; a target past `.HLM` is refused before the put; a limit
switch tripped mid-move fails the move naming it; the IOC down before the
run refuses it with nothing claimed.

### 8. Setpoint device — the EasyDriver pattern

The CAEN ELS Easy-Driver (`eztest1`, StreamDevice over its Lantronix
bridge) is the template for every scalar device with a readback and a
request. Its records already spell the gateway contract — readback
`<prefix>current`, request `<prefix>current:SP` with `DRVH`/`DRVL`, a
refused write an alarm on the setpoint record, named status bits
(`enable_output`, `fault`, `interlock`) raising MAJOR. That shape becomes
a geecs-epics contract document, as the scope's (§9); this note defines
what GEECS consumes.

**Registry entry** (*proposed* kind setpoint_device): the worker has no
DB rows to enumerate the variables, so the entry names them — settables
(each with a tolerance, the one device fact an `ao` cannot hold, and a
timeout), extra readback columns, status bits; an enum matches exactly.

```yaml
U_EZ1: {kind: setpoint_device,
        settables: {current: {tolerance: 0.005}, slewrate: {tolerance: 0.01}},
        readbacks: [voltage, temperature], status: [fault, interlock]}
```

**Set semantics** — never on a "put ok": for StreamDevice put-completion
means the device **accepted** the command (`#AK`), not that it converged
— the opposite of the gateway, where put-completion *is* convergence — so
the readback wait is mandatory. The set completes when the readback is
within the tolerance (`within_tolerance`); it fails at once when the `:SP`
record goes INVALID after the put (a `#NAK`, the SEVR rule) or a status
bit raises, and on the timeout with `GeecsConfirmTimeoutError`'s words:
`CaConfirmSettable`'s topology — set X, confirm on Y — with Y the readback
record; the device (*proposed* NativeSettable) is that logic over two IOC PVs.

**Out-of-range writes**: a native `ao` record **clamps** to `DRVH`/`DRVL`
where the gateway rejects (`PV_CONTRACT.md` §2, on purpose); clamping
would run a scan to 6 A at 5 A with every row "ok". The worker reads
`DRVH`/`DRVL` (`DBR_CTRL`) and refuses before the put, as
`Motor.check_motor_limit` does — pinned by a test asserting no put was made.

**Row**: `Device variable` readback columns for the scanned settable and
the listed readbacks, sampled per row as `CaSnapshotReadable` does; status
bits are archived (§10), not row columns. **Liveness**: a StreamDevice
readback whose device stops answering goes INVALID/TIMEOUT — any readback
at INVALID reads down, judged by the grown gate of §7. **Panel**:
readbacks, `:SP`s with limits, status bits.

**Simulator**: the protocol fake that exists — `geecs-epics/sim/easydriver_sim.py`
answering the EasyDriver's ASCII set behind the real StreamDevice image
(`eztest1-sim`), the same template and `.proto`, only the port differing.
**Acceptance** on `eztest1`: a 5-point `sweep` of `current`, each within
tolerance; a value past `DRVH` refused before the put; a `#NAK` (output
disabled) fails the set naming the alarm; the interlock bit raised
mid-move fails the set naming the bit.

### 9. Array detector — areaDetector, the camera's path end to end

**Ruled 2026-10-23 in the PicoScope session**: the scope goes the
areaDetector route with alias settings (geecs-epics decision 14; the
catalogue row is the record today). The PicoScope driver becomes an
ADDriver on ADCSimDetector's pattern (address 0 the `[channels × samples]`
NDArray per capture, addresses 1..N one array per channel) behind a
vendor-neutral scope settings contract and a documented chain (per-channel
ROI → STAT; HDF5, PVA and StdArrays on the full array). **The contract is
geecs-epics' own docs PR, not yet opened**; this note defines only what
GEECS consumes: the stack, the stamp pair, which STAT values become
columns, the refusals.

**Reuse, stated**: the kind (*proposed* NativeArrayDetector) is
`NativeCamera`'s path with the alias names in place of `cam1:`'s — armed
once per run (`NativeTriggerLogic`'s stop → trigger setup → capture
start, `NativeAcquireLogic`'s restore at unstage), the run lock taken
first (#1136, still open for the camera too), a shot =
`HDF1:NumCaptured_RBV` advancing, `Software` pressed in `trigger()` or an
external edge on the scope's trigger input, box-gated like a camera's
line; the stack `ScanNNN/<device>/<device>.h5` through `PluginPathProvider`
and `native_ioc_save_path`; the row stamp from the EPICS pair
(`_StampProvider`, `epics_timestamp_to_unix`); the `/entry` finalized
layout kept by `NativeHdfDataLogic`; `find_stack_file`, `StackCheckCallback`
and Tiled's stock consolidator unchanged; one capture per row, its own
miss fails the step, never rewindable, strict only in v1 with the camera's
refusals. The two classes share a base (open question 8) whose helpers
`native_cameras()` and `refuse_native_cameras` become.

**Where the waveform goes — decided by the reuse**: the stack, frames
`(N, channels, samples)` in `/entry/data/data`, the capture atomic across
channels, one datum per shot. Rejected: one HDF plugin per channel (N
files and arms for no gain); StdArrays into the row (a row is scalars).
The reader learns the shape: `stack_content_kind` knows the PVA gateway's
`(N, n)` waveform and `(N, n, 2)` lineout by their `wave_*` attributes
(`WAVEFORM_ATTRIBUTE_SUFFIXES`); a native scope stack carries the
contract's axis NDAttributes instead, so a third marker and a per-channel
view in `read_1d_data` are the GEECS-Data-Utils change.

**Derived scalars** (ICT charge, peak, baseline): the chain's STAT plugin
on each channel's ROI gives Total, Net, Min, Max and Centroid per capture;
a calc record applies a constant calibration. The registry names which
PVs become columns (`<name>-<column>`, header `Device Column`):

```yaml
U_UndulatorExitICT: {kind: array_detector, trigger_source: external,
                     scalars: {"Charge.ChA": "STAT1:Net_RBV"}}
```

Read **coherently**: a STAT value and the HDF count are separate plugin
callbacks with no ordering between them, so the kind reads each scalar
only when its plugin's `UniqueId_RBV` equals the written frame's (the
`ShotSampler` rule for a member with a stamp of its own, by frame id)
**and the computation that produces it is on**. NDPluginStats advances
`UniqueId_RBV` for every array it receives but recomputes Total, Net, Min
and Max only while `ComputeStatistics` is set, and the centroid family
only while `ComputeCentroid` is (profiles and the histogram have flags of
their own), so a matching frame id with the flag off retains the previous
shot's values. Each declared scalar therefore maps to its flag; the flag
is checked at prepare and read again with the value, and a scalar whose
flag is off, or whose id does not match, is `NaN` with a note — never the
previous shot's number. The camera panel already draws this distinction
(its statistics freshness is the plugin counter plus the compute flags).
The analysis-grade charge (`apply_ict_analysis`) stays post-scan over the
stack (open question 5). **Refusals** beyond the camera's: a scalar PV
outside the prefix (bind); a STAT plugin with callbacks disabled or the
declared scalar's compute flag off (prepare, "declared scalar not
computed"); `X.scalars` only when the entry declares scalars.

**Timestamp**: the EPICS pair stamps the capture's arrival, after the
post-trigger samples; the hand-set drain offset absorbs that span (ruling 6).

**Simulator**: ADCSimDetector, off the shelf, the same chain under the
real prefix; it free-runs, so the sim entry says software triggering.
**Acceptance**: the PicoScope 5242D after its driver rework — a strict
10-shot `count` gives one finalized stack of 10 `(2, samples)` frames, 10
rows each joined by stamp with the declared scalar columns coherent (a
deliberately slowed STAT yields `NaN`, never a stale value; `ComputeStatistics`
switched off mid-run while callbacks continue yields `NaN` from that shot
on, never the retained value), each channel's trace per shot in the
portal, the panel running before and after.

### 10. Cross-cutting

**Registry union and options.** `NativeDevice` gains three members,
discriminated by `kind`; `NativeDevices.devices` is unchanged; docgen follows.

| kind | options | binds as |
|---|---|---|
| `aravis_camera` (built) | `trigger_source` | `NativeCamera` |
| epics_motor | axes | a noun with one stock `Motor` per axis |
| setpoint_device | settables (tolerance, timeout), readbacks, status | one confirm-on-readback Movable per settable |
| array_detector | `trigger_source`, scalars | the camera's path over the alias records |

**One owner** is unchanged: `GeecsNamespace._add_native_devices` compares
`safe_name`s against the roster for every kind. Added: an axis or
variable name colliding with a kind's fixed record suffixes refuses to
open.

**Bind-time refusals per kind.** The camera's list stands. A positioner
or a setpoint device is admitted as a scan axis or an action target in
every mode and as a strict detector per row, exactly as a
`CaSnapshotReadable` is; refused as a non-essential — no shot stamp
(`acquisition_refusal`'s existing sentence) — and as a gated run's only
essential (the existing "nothing counts shots" rule). The array detector
takes the camera's list plus §9's. The client's native set must be
**kind-aware** for these to fire on the right devices: `expand_preset`'s
`native_devices` becomes names → kind, its `scalars_only` /
`_native_run_refusal` rules applying to detector kinds only.

**The scanner's settables list.** `DbSettables` builds `numeric_settables`
from DB rows; natives have none. A second source over `NativeDevices`
(*proposed*, beside `DbSettables` in `service/settables.py`) yields a
`NumericSettable` per motor axis and numeric setpoint-device settable,
appended after the DB list (the list-level `source` gaining a third word).
Units, min and max come from one `DBR_CTRL` read per PV (`.EGU`,
`.HLM`/`.LLM` or `DRVH`/`DRVL`), cached per process like `CachedDbRead`.
The readback (`service/readback.py`) gets a per-kind PV — `<axis>.RBV` for
a motor, the readback record for a setpoint device — instead of
`pv_name`'s gateway spelling. The move path is untouched: `MoveIn` →
`scan_variable_reference` → the worker's `mv`.

**Archive set.** `derive_candidates` reads only DB rows; a native device
has none, so the archiver would silently drop a migrated device. A
registry pass in `geecs_archiver.archive_set`, reading `NativeDevices`
through the configs-repo wrappers in `config.py` (which already load
`DerivedChannels`), derives per kind: a motor's `.RBV`, `.VAL` (under
`include_setpoints`), `.DMOV` and the limit switches (status); a setpoint
device's readbacks, `:SP`s and status bits; a detector's settings
readbacks (`cam1:AcquireTime_RBV`, the contract's `_RBV`s), `AsynIO.CNCT`
and, once #1136 lands, the run lock; devIocStats' heartbeat per IOC as
the analogue of `cagateway:heartbeat`. `exclude` globs apply; `include`
keeps its spelling rule (these are derived candidates). The
never-connected list stays the drift alarm between registry and IOCs.

**One home for the PV surface.** The native prefix is derived once today
inside `GeecsNamespace._add_native_devices`, which the archiver cannot
import (`.importlinter`: `geecs_bluesky` and `geecs_archiver` share a
layer); it moves to `geecs_core.pv_naming` beside `gateway_status_pv`. Each
kind's record suffixes — what the archive pass, the scanner and the panel
need — become data beside the kind models in `geecs_schemas.epics`.

**Liveness, per kind** — `devices/ca/liveness.py`'s shape, the gate grown by the movers (§7):

| kind | down when |
|---|---|
| camera / array detector | `AsynIO.CNCT` reads Disconnect, or the IOC does not answer (`down_native_cameras`'s rule) |
| positioner | an axis at INVALID severity (COMM/UDF) |
| setpoint device | a readback at INVALID severity (TIMEOUT/COMM) |

**Panels** (GeecsPanels, #1131–#1138 open; `Planning/device_panels.md`):
a kind whose writes move things submits through the scanner's
`POST /api/move`, serialized with scans — the positioner and a setpoint
device's numeric settables; non-motion controls (`enable_output`, `reset`)
are direct, allow-listed puts; the scope's settings are direct puts
behind the run lock, as the camera's will be (#1136). Needed from the
kind's suffix table: motor — `.RBV`, `.VAL`, `.DMOV`, `.HLM`/`.LLM`, the
switches, `.STOP`; setpoint device — readbacks, `:SP`s with limits, status
bits, `.SEVR`; array detector — the `_RBV`s, the channel arrays, STAT, the lock.

**The simulator-first acceptance ladder**, per kind: (1) GeecsBluesky
unit tests with a mocked IOC, as `tests/test_native_camera.py`; (2) the
device fake on geecs-epics' sim host, the real prefix on the isolated
network, driven by a **dev-box queueserver** — a worker whose `config.ini`
names a test experiment's configs checkout, whose `EPICS_CA_ADDR_LIST` is
the sim host and whose `[Paths]` root is a scratch tree (`claim_scan_number`
is the one creator of scan folders and must never claim in the real data
tree from a dev box); (3) the PV-level replayer for readers and panels;
(4) the first real device on the lab worker, the lists in §7–§9; (5) the
rules move into `CLAUDE.md` and the section here is deleted.

### 11. Where the changes land

- **S** (first) — GEECS-Schemas, GEECS-Core, `qs_client/presets.py`: the
  union members, option models and suffix tables, docgen; the prefix rule
  in `pv_naming`; names → kind in `expand_preset`.
- **M**, **P**, **A** — GeecsBluesky, independent after S: the positioner
  noun (post-move check, grown gate); the confirm-on-readback Movable
  (`DRVH`/`DRVL` refusal); the shared native-detector base, the scalar
  columns by frame id, the scope stack shape in GEECS-Data-Utils.
- **L** — GeecsScanner, after M and P. **R** — GeecsArchiver, after S.
  geecs-epics: the two contract docs, the motion image.

### Open questions — kinds beyond the camera

1. **Registry kind names.** (a) epics_motor / setpoint_device /
   array_detector; (b) motor / setpoint / scope; (c) by driver family.
   Recommend (a): (b) collides with the catalogue's `kind` words.
2. **Where a setpoint device's variables and tolerances live.** (a) the
   registry entry (§8); (b) an IOC manifest record read at environment
   open; (c) devicetype templates in GEECS-Schemas. Recommend (a) for v1.
3. **Units and limits in the scanner's list.** (a) one `DBR_CTRL` read
   per native PV, cached; (b) blank; (c) in the registry. Recommend (a).
4. **Filter wheels.** (a) a motor record with integer positions, under the
   positioner kind; (b) an `mbbo` setpoint device; (c) a kind of their own.
   Recommend (a).
5. **Scope derived scalars.** (a) IOC STAT/calc live plus the
   analysis-grade number post-scan; (b) post-scan only; (c) IOC only.
   Recommend (a).
6. **Out-of-range sets to a native `ao`.** (a) worker-side `DRVH`/`DRVL`
   refusal before the put; (b) a rejecting `:SP` (not stock); (c) accept
   clamping. Recommend (a).
7. **Native axes and readbacks in background telemetry.** (a) yes; (b) no.
   Recommend (a): a read costs no shot, and the baseline has every DB scalar.
8. **The native-detector base.** (a) a shared base for `NativeCamera` and
   the array detector; (b) a subclass of `NativeCamera`. Recommend (a).
9. **Archive membership for natives.** (a) a per-kind rule in
   `archive_set`; (b) policy `include` lists only; (c) EPICS `info(archive)`
   tags. Recommend (a): (b) cannot even spell them — `ArchivePolicy.include`'s
   validator admits only lowercase `a:b[:SP]`, never `.RBV` or `_RBV` —
   and (c) needs a tool the appliance does not ship.
10. **The scope's stack shape.** (a) `(N, channels, samples)` from the
    full-array HDF plugin; (b) one file per channel. Recommend (a).

### Review dispositions (PR #1140)

Fresh-context adversarial review (PR #1140), eight findings, all
**fixed**: (1) the gate never saw a mover → its set grows by `sweep_movers`;
(2) a stampless strict detector is admitted today → positioners and
setpoint devices admitted per row, refused as non-essential or sole gated
essential; (3) prefix rule and suffix sets given one home; (4) the existing
geecs-epics simulators named; (5) `_log_unhonoured_motor_targets` is the
warning at stake; (6) #1136 marked open; (7) the scope contract pointed
at, not restated; (8) `source` is list-level; residual of (2) reconciled in §7.

**Second pass (maintainer's reviewer, 2026-10-23), one P2.** Matching
`UniqueId_RBV` with callbacks enabled did not prove a STAT scalar was
computed for the shot: with `ComputeStatistics` off the plugin still
advances its frame id and keeps the old Total/Net/Min/Max, and
`ComputeCentroid` gates the centroid family separately. — Fixed in §9:
each declared scalar maps to its compute flag, checked at prepare and read
with the value, `NaN` with a note when off; the acceptance adds the
flag-off-mid-run case.
