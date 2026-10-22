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
  `line1` — the camera's hardware input, wired to a DG645 output (the
  pulsed-event mode); `software` — `trigger()` executes the camera's
  `TriggerSoftware` command, so the frame is exposed when the worker says
  go, not when the laser fires. Software mode's timing is an accepted
  caveat (Sam, 2026-10-22): right for continuous or slow signals,
  shot-on-demand and dry runs; verified on the bench camera, one frame per
  trigger and none while armed and idle. Both take the same path below;
  only the source and that one command differ. ophyd-async's
  `AravisTriggerSource` knows only `Line1`, so `NativeCamera` defines its
  own choices.
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
- **A shot** = the plugin's `NumCaptured` advancing past the baseline the
  stock `trigger` task reads. With `Software` the press follows that read
  in the same task, so the order is guaranteed; with `Line1` the read
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

**Later, `line1`** — **cabling rule**: `Line1` hangs off a DG645 output **gated by the trigger
box's states**, as a GEECS camera's input does — the camera captures every
edge it sees all run long, so a free-running channel puts a stray frame
in the stack on every edge.

With the camera cabled that way: the same 10-shot run with
`trigger_source: line1`, and a deliberately blocked edge fails the step
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
  kickoff re-baselines on the count it reads (the late frame belongs to
  no batch's quota; its datum still covers it).
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

Stream name (*proposed*, open question 4): `<name>_frames`, a document
constant beside `SHOTS_STREAM` and the non-essential suffix in
`geecs_data_utils.shot_join`. The s-file needs nothing new — it joins
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
   other cameras' frames are recorded.
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
| One missed `Line1` edge mid-batch | the count lags: the box runs one more edge; that row's stamp finds no frame | **marks**: the row's native columns read `NaN`, the extra frame is an orphan; WARNING in the stack check — no retake (as LabVIEW natives) |
| Camera stops producing frames (connected) | `complete`'s per-frame `exposure_timeout` (the run's `shot_timeout`, 3 s default) | **fails** the batch through the existing failure path: the kept shots recorded, the native stream collected (every frame referenced), then `GeecsTriggerTimeoutError` naming the camera and `WriteMessage` |
| IOC dead or camera unreadable mid-batch | the count read fails | **fails** the batch; the per-device settle leaves that camera out of K and skips its collect, so the other devices' kept shots are still recorded; its file holds frames no datum references — its Tiled view is short, never shifted |
| Stray edge (miscabling: an edge outside SCAN) | an orphan or a contested frame in the stamp join | marked in the stack check; fixed by cabling (the rule in Acceptance) |
| Late frame landing after a batch's collect | the native kickoff re-baselines past it (the stock kickoff would fail the run) | nothing to mark: the stamp join still gives it to its own row; only which batch's datum covers it changes |
| A frame after the last collect (before `unstage`) | file longer than the datums | trailing only: no shift; the stack check reports frames > datums |

**Refuse, mark or re-fire:** a gated batch never re-fires anything (the
box runs until every essential has its count); for a native camera
nothing is refused at run time either — a single miss is **marked**, a
stopped camera **fails** the batch. The one invariant enforced at run time
is the one that keeps Tiled right: every frame in the file before the last
collect is referenced, in order.

The stack check (`callbacks/stack_check.py`) gets a fourth shape for an
append-only stream: frames = Σ datum widths (else a hole — a WARNING
saying Tiled is misaligned, the one real defect), every `shots` row joins
one native frame within its window (else a WARNING with the count), and
orphans counted and logged at INFO.

### Software trigger versus `Line1`

| Question | `Line1` | `Software` |
|---|---|---|
| What fires a frame in a batch | the box-gated DG645 edge, every laser shot | nothing — the batch fires no command; the strict design presses inside `trigger()`, which a batch never calls |
| v2 decision | admitted | **refused at bind** (named: "a software-triggered native camera has no edge in a gated batch") unless open question 1 says otherwise |
| Pause | OFF stops edges; file open | — |
| Missed / extra | as the table above | — |

The software alternative (open question 1, *proposed*, a tick-follower):
during a batch the camera presses `TriggerSoftware` on every advance of
the run's shot clock (`shot_clock`'s stamp signal, already monitored by
the sampler). **Its timing is poor**: the clock's stamp reaches the
worker ~0.45 s after the edge at 1 Hz (the package `CLAUDE.md`'s ~550 ms
from stamp to next edge), so the frame is exposed ≥ 0.45 s after the
shot — it does not image the shot — and its EPICS stamp sits at the edge
of the ±0.5 s join window: without a calibrated offset it joins the
**next** row (a one-shot shift in the s-file) or nobody. Usable only for
dry runs, and only with its own offset (~60 LOC plus the offset). Its one
value: the gated path could be exercised on the bench before the `Line1`
cable exists.

### Liveness and timeouts

- **Before the claim**: unchanged — `liveness_gate` already judges a
  native camera by its IOC (`down_native_cameras`, connected first,
  unreadable = down) for essentials and non-essentials alike.
- **During a batch**: the per-frame `exposure_timeout` (ophyd-async's
  `observe_signals_value` budget between count updates) is the run's
  `shot_timeout`. A disconnected camera or a dead IOC stops the count →
  the batch fails as above. A connected camera that misses one edge does
  not time out (the next edge comes within a period) — it is marked.
- **During a pause**: no timeout runs — the batch's `complete` was
  abandoned and settles quietly; the camera may sit armed indefinitely.
  An IOC that died during the pause is found on resume, at the settle's
  read of `NumCaptured_RBV`, before the box leaves OFF: the per-device
  settle records the others' kept shots, then the batch fails naming the
  camera.
- **The drain offset** (edge → EPICS frame stamp) has no calibration
  path today: the calibration plans refuse native cameras
  (`plans/calibration.py`), so the only source is the namespace's
  hand-entered `drain_offsets` (default 0.0). At 1 Hz the half-period
  window absorbs an uncalibrated offset; at higher rates it does not.
  v2 runs with a **hand-set** value; lifting the calibration refusal for
  `Line1` natives is PR F (open question 6).
- **The shot clock** stays a GEECS triggered essential (`shot_clock`): a
  gated run whose only essentials are native cameras is refused (the
  existing "nothing counts shots" error) until open question 3 is
  answered.

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
`Software` native (unless open question 1); a gated run with no GEECS
shot clock. Gated runs mixing native and gateway plugin cameras are
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
  its frames collected; (e) a native IOC unreadable at the settle → the
  others' kept shots and the rows are still collected; (f) a native frame
  landing between prepare and kickoff → no failure, datums contiguous; each test **mutation-checked** (break the
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
   stamps within a few ms with the hand-set offset — record the spread,
   it is the calibration's first data); stack check clean (orphans ≤ 1, INFO); Tiled's array
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
| C — native classification and append-only streams in `gated_take_reading`, the per-device settle, the stack check's fourth shape, bind-time refusals lifted for gated `Line1` | GeecsBluesky | ~200 | ~300 |
| D — non-essential `Line1` natives (refusal + wrapper classification) + the scanner preflight's gated message | GeecsBluesky, GeecsScanner | ~70 | ~120 |
| (E — the software tick-follower, only if open question 1 says yes) | GeecsBluesky | ~80 | ~100 |
| (F — the drain-offset calibration for `Line1` natives, open question 6) | GeecsBluesky | ~60 | ~100 |

About 480 production + 730 test lines for A–D; A is independent, B
before C, D after B. The CLAUDE.md rule text moves with C.

The re-export half of A: the stock `ADHDFDataLogic` declares no Tiled
part for the built-in `NDArrayEpicsTS*` pair, so
`tiled_export.read_frame_columns` (which builds columns only from parts
named `<device>-hdf-<variable>-<suffix>`) finds no native stamp. Either
the native data logic declares the pair as two datasets of its stream
(preferred — Tiled then carries it like any per-frame attribute), or the
re-export reads the stamps from the file the stream resource names. The
build PR picks one; without either, an s-file rebuilt from Tiled loses
the column the portal keys frames on.

### Open questions for Sam (gated)

1. **Software trigger in gated batches**: refuse (recommended — the
   frame is ≥ 0.45 s after the shot and joins at the window's edge), or
   build the tick-follower (E) for bench dry runs before the `Line1`
   cable?
2. **A missed native frame**: mark and continue (proposed, as LabVIEW
   natives), or fail the batch on the first native miss (the strict v1
   rule)? Marking needs nothing from the writer; failing would need a
   live stamp comparison per shot.
3. **A native camera as the shot clock**: should a gated run of only
   native cameras be possible (a derived stamp signal from
   `EpicsTSSec_RBV` / `EpicsTSNsec_RBV`, ~40 LOC), or does every gated run
   keep a GEECS triggered essential?
4. **The stream name** for an essential append-only camera:
   `<name>_frames` (proposed — honest that it is not a non-essential), or
   reuse the non-essential `<name>_stream` suffix (no new document
   constant, but the start document's `non_essential` list would no
   longer say which streams were waited on)?
5. **Surplus frames in the stack**: ≤ 1 per batch end and per pause,
   kept and orphaned. Acceptable, or worth a driver-side count later
   (blocked today by the `Multiple`-mode empty-frame quirk)?
6. **The native drain offset**: run v2 with a hand-set value, or build F
   (lift the calibration plans' refusal for `Line1` natives) before gated
   runs above 1 Hz?

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
