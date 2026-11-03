# Shot synchronization — one shot identity for every device

Status: **design, for review.** Nothing here is built. Delete this file in
the PR that lands the last slice (`Planning/README.md`).

How triggered devices of every type, native EPICS IOCs and LabVIEW plugins
behind the gateway, come to agree on which acquisitions belong to the same
shot, with no hardware that delivers shot identity to the devices. It
supersedes the shot-detection part of the native shot-control note
(`GeecsBluesky/NATIVE_SHOT_CONTROL.md` §4, draft PR #1236): the DG645 seam
in that note stands; the question "how does anyone know a shot happened,
and which one" is answered here.

Cites: `file:line` against `master`; the private geecs-epics repo by path
only (`geecs-epics:<path>`, per the root `CLAUDE.md`); upstream drivers by
repository and file. Where a behaviour is not proven on hardware the text
says so.

## 1. Why

Today a shot's identity is each device's own `acq_timestamp`: the LabVIEW
host's clock when the device sent its frame, taken after the exposure
completed. The strict plan baselines each device's stamp, fires, waits for
every stamp to advance, and when one device produces nothing it rewinds the
file plugins to the last frame a row referenced and fires again
(`plans/strict.py:154-243`, `:397-445`; `devices/detector.py:957-975`,
`:1177-1190`). Frames are joined to rows by correcting each device's stamp
by a measured drain offset and matching nearest-in-time inside a per-row
window (`GEECS-Data-Utils/geecs_data_utils/shot_join.py:334-463`). This
works for LabVIEW devices because our own PVA file plugin can rewind. Stock
areaDetector writers cannot, so a native camera is refused beside a gateway
plugin camera, in gated runs, in scan-mode runs and as a non-essential
(`plans/registry.py:402-437`). The rewind is a GEECS invention; the
refusals are the price of building on it.

Three facts frame the replacement:

| Fact | Consequence |
|---|---|
| One trigger. At this facility every synchronous readback hangs off one gated output of the shot-control box, and the ARMED/STANDBY gate stops those edges (maintainer, 2026-11-02) | one witness on that output sees exactly the shots the devices see (§4.1); other facilities may need one witness per trigger domain |
| 1 Hz is the norm, 5 Hz the aspiration | the assignment margin is half a period: 500 ms, or 100 ms |
| No event system (MRF) | an EVG with an EVR per host would deliver shot identity to every device at the hardware moment: the only stronger design, ruled out on cost for this facility; nothing here precludes it later |

The policy this serves (maintainer, 2026-11-02): **every shot recorded,
gaps flagged, no retake needed.**

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **shot** | one edge of the gated trigger. Beam presence is a separate signal (an ICT); a shot with no beam is still a shot |
| **acquisition** | one capture by one device: a frame, a scope trace, a sampled scalar |
| **witness** | a hardware counter on the gated trigger with a stamp per count. A shot's identity is `(witness epoch, count)` |
| **synchronous device** | receives the edge and produces one acquisition per shot: a hardware-triggered camera, a scope in Normal trigger |
| **asynchronous device** | no edge; its value at shot time is its last sample before the shot: motors, supplies, a camera in free-run |
| **stamp kind** | where a device's timestamp was taken: *host-receipt* (the IOC host's clock when the complete acquisition arrived), *device-clock* (the camera's own tick counter), *ptp* (the camera's clock disciplined by PTP), *poll* (the witness) |
| **counter** | a per-acquisition integer: *hardware* (the camera's frame id), *driver* (the IOC's count of complete acquisitions), *announced* (the scan's shot number echoed back by a LabVIEW device) |
| **announcement** | the scan-mode shot message Master Control, and now the worker, multicasts before each shot: the software "fire" a LabVIEW device needs before it will take its next acquisition (`docs/acquisition/scan_mode.md`). Distinct from the edge: the box fires the edge after it |

Class is a property of the **configuration**, not the device type: the
IOC's trigger-mode readback says which, and it is recorded with each
acquisition. A camera switched to free-run for alignment is asynchronous
for that run.

## 3. Standard practice, briefly

- **areaDetector** gives every NDArray three identity fields: `uniqueId`
  (a counter), `timeStamp` (a double the driver fills) and `epicsTS` (an
  EPICS timestamp). Plugins and the HDF5 writer carry them per frame as
  the `NDArrayUniqueId`, `NDArrayTimeStamp` and `NDArrayEpicsTSSec` /
  `NDArrayEpicsTSnSec` attributes. The design uses exactly those; nothing
  new is added to a frame on the IOC side.
- **Bluesky's event model** stamps every reading, and a join across
  streams by timestamp is the ordinary way to relate devices that no
  hardware indexes. ophyd-async's hardware-triggered flyscans join by
  index only because a trigger source delivers the index; here nothing
  does, so time assigns and a counter audits (§4.3).
- **Timing systems** (MRF) deliver timestamps and event codes to IOCs at
  the hardware moment, which is what would make the audit unnecessary.
  Out of scope.

## 4. The model

### 4.1 The witness

**Hardware.** A Measurement Computing USB-1808X, a counter input on a BNC
tee of the gated trigger output, placed beside the shot-control box. The
same unit geecs-epics runs today as a bench DAQ instance
(`geecs-epics:services/undulator/daq/daqtest1/README.md`); a USB-CTR08 only
if several trigger domains ever matter.

**Driver facts** (epics-modules/measComp, `drvMultiFunction`): the counter
counts every edge in hardware. The IOC reads the count in its poller
thread, whose sleep defaults to 50 ms and is the `PollSleepMS` record
(`measCompApp/src/drvMultiFunction.cpp:1277`, `:2641-2651`;
`measCompApp/Db/measCompDevice.template:45-51`). The `Counts` record is
I/O Intr and processes when the count changes
(`measCompApp/Db/measCompCounter.template`), so its timestamp is the IOC
host's clock at the first poll after the edge: a lag uniform between zero
and the poll period. The counter resets at every IOC start (the `Reset`
record's PINI).

**What it publishes.** The count (`Counts`), its stamp (the record's
`TIME`) and the epoch (the IOC's start time from devIocStats, already
loaded in that instance). A shot is `(epoch, count)`, never the count
alone.

**Failure modes it owns.**

| Case | Behaviour |
|---|---|
| Poll lag | bounded by the poll period, pinned at 10 ms at boot. Assignment needs it under a quarter period minus the offset spread: trivially met at 1 Hz and 5 Hz |
| Several edges in one poll (an IOC or USB stall) | one update, count +k, one stamp: those k shots share a stamp and are flagged *witness-stamp-shared*; their acquisitions are assigned by counter order inside the group when the devices' counters agree, else flagged |
| IOC restart | the count restarts at 0 under a new epoch; nothing joins across epochs |
| Witness down during a run | rows continue, clocked by the devices against each other (today's nearest-time clustering), every row flagged *unwitnessed*; the run is not aborted |
| Doubled edge (ringing, a slow edge through the threshold) | the count runs at twice the rate every device sees and the audit flags every acquisition; the fix is at the input (level, termination). Checked at the bench before anything depends on it |
| It counts the wrong edge | the witness must see the edges the synchronous devices see: here, the gated output. A witness on an ungated output counts shots in STANDBY that no device took, and the audit flags every one. A facility whose devices hang off several gates needs a witness per gate, or a recorded gate state; out of scope, but the schema carries a trigger-domain name so it has room |

**Generality: the witness is a contract; the MCC is one implementation.**
A witness is anything that publishes, for one trigger domain, a hardware
count of the edges, a stamp per count on a disciplined clock, and an epoch
that changes whenever the count restarts. The join reads those three and
the domain's name, nothing else, so no code names the MCC:

| Implementation | Where it fits |
|---|---|
| a counter input on a DAQ IOC (the MCC here; any measComp or similar device elsewhere) | a facility with a tee and a spare counter |
| an EPICS counter record on hardware already in the chain: a PLC counter, a scaler, a delay generator with a readable trigger count | no new hardware |
| a timing system's event receiver (an MRF EVR) with its event counter and timestamp | a facility that has one; the stamp is then at the hardware moment and the audit is nearly redundant |
| the worker's own fire stamp, strict mode only: one fire, one shot | no hardware at all; no gated runs and no fire confirmation |
| none | the *unwitnessed* fallback: devices clustered against each other, every row flagged |

A chain with different hardware at different points, or several gates, is
several trigger domains: one witness per domain, each device's registry
entry names its domain, and the join assigns within a domain. Domains are
related only through a declared relation (the same source behind a fixed
delay), otherwise not at all and the row says so. This facility has one
domain.

**Why not something else.** The DG645 has no edge-stamped counter readback
and is LabVIEW-owned until it is ported last. A camera as witness, today's
gated shot clock (`plans/gated.py:86-123`), misses shots. NI DAQs have no
Linux EPICS path. The witness's value is the hardware count: it cannot
miss an edge, only stamp it late. Overturned by a measured poll-lag spread
wider than a quarter period at the target rate, or by the counter doubling
or missing edges on the real pulse.

### 4.2 Evidence stored per acquisition

| Field | Content |
|---|---|
| `device_counter`, `counter_kind` | the device's per-acquisition integer and whether it is hardware, driver or announced |
| `device_stamp`, `stamp_kind` | the device's timestamp and where it was taken (§2) |
| `sync_class` | synchronous or asynchronous, from the device's configuration readback at the time |
| `witness_epoch`, `witness_count` | the shot assigned |
| `witness_stamp` | that shot's witness stamp |
| `residual` | `device_stamp − offset − witness_stamp` after assignment |
| `announce_stamp` | scan-mode LabVIEW devices only: the worker's clock when it announced the shot this acquisition answers (§4.6) |
| `flags` | empty, or any of *missing*, *not-armed* (a LabVIEW device that was awaiting an announcement when the edge came), *extra*, *audit-mismatch*, *clock-step*, *counter-reset*, *witness-stamp-shared*, *unwitnessed*, *stale* (asynchronous) |

Stored where it is produced: event columns in the device's stream for
scalars; for frames, the stack's per-frame attributes, which already carry
the counter and the two stamps (`NDArrayUniqueId`, `NDArrayTimeStamp`,
`NDArrayEpicsTSSec` / `nSec`), plus the witness stream. The join is
re-runnable offline from these alone; no worker state is needed, and a
join run from files must reproduce the live one.

### 4.3 The join rule: time assigns, counters audit

Derived from failure modes, not preference:

| Rule | Missed trigger or dropped frame | Clock step or offset drift | Counter reset | Error propagates? |
|---|---|---|---|---|
| counters assign | silent: every later acquisition is off by one | unaffected | total loss until re-anchored by hand | **yes** |
| time assigns | the gap is visible | misassigns while the error exceeds half a period; the residual shows it first | unaffected | no |
| **time assigns, counters audit** | gap visible and confirmed by the counter delta | caught by the audit, flagged, never guessed | re-anchor at the reset, flagged | no |

- **Synchronous devices.** Each acquisition goes to the nearest witness
  shot after its device's offset, one acquisition per shot and one shot per
  acquisition, resolved globally nearest-pair-first, as the frame join does
  today (`shot_join.py:379-463`). A shot with no acquisition is a *gap*:
  NaN, flagged *missing*. An acquisition no shot claims is *extra*: kept in
  its stream, flagged; a synchronous device producing more acquisitions
  than shots is misconfigured (a scope in Auto trigger, a camera in
  free-run under a synchronous entry), and the flag says so.
- **The audit.** Between two consecutive assigned acquisitions of one
  device, the counter delta must equal the shot delta. Smaller: the gap is
  confirmed. Larger: acquisitions exist that the join did not see; both
  ends flagged *audit-mismatch*. Equal with a jump in residual: a clock
  step, flagged on the acquisition where it appears. Negative: a counter
  reset; re-anchor there, flagged. The audit never moves an acquisition.
- **Asynchronous devices.** The last sample at or before the shot's witness
  stamp (after the device's own offset, if it stamps); the sample's age is
  recorded, and a sample older than a configured limit is flagged *stale*.
  No counter audit.
- **Never guess.** A flagged acquisition stays in the data with its flags.
  Nothing is moved to a "better" shot by inference.
- **Residual tracker.** Per device, a running median and spread of
  residuals over the run. Assignment uses the *stored* offset, so a run is
  reproducible; the tracker reports drift and flags when the median leaves
  the budget (§4.4). The stored offset changes only by a recorded
  calibration, which with a witness is the median residual over the last
  N shots of any run, written as `shot_offsets.yaml` is written today
  (`GEECS-Schemas/geecs_schemas/shot_offsets.py`). Open question 6 asks
  whether that stays deliberate or becomes continuous; the recommendation
  is deliberate.

### 4.4 Stamps: a budget, not a mandate

What each device stamps today, and what moves the stamp:

| Device | Stamp | Kind | The offset contains | Jitter measured |
|---|---|---|---|---|
| LabVIEW device | `acq_timestamp` | host-receipt (a Windows host) | exposure + readout + drain: **moves with the exposure setting** | 0.5 ms SD within one scan (`geecs-epics:docs/runbooks/timing.md`); up to ~10 ms dither across hosts claimed in `shot_offsets.py` |
| native camera (ADAravis) | `NDArrayEpicsTS*` | host-receipt | exposure + readout + transfer + host scheduling | 0.1 ms frame to frame |
| PicoScope IOC | `epicsTS` at the ready poll | host-receipt | post-trigger capture + up to 2 ms poll + USB | unmeasured |
| witness | `Counts.TIME` | poll | zero to one poll period | unmeasured |

**The budget.** Per device: the residual spread (p99) under a quarter
period minus the witness lag, and the offset's drift between calibrations
under a quarter period. At 1 Hz that is about 240 ms; at 5 Hz about 40 ms.
Every row above meets the jitter half at both rates with no change. The
drift half is where the LabVIEW stamp fails at 5 Hz: an exposure scan moves
its offset inside one run. A join breaks on offset *instability*, not on
jitter, so the requirement is this budget, measured live by the tracker,
rather than "stamp at the hardware moment".

**Hardware stamps** remove the exposure and ROI dependence at the source:
the camera's own clock (`NDArrayTimeStamp`) marks the exposure in camera
ticks; with PTP enabled on the camera it is wall-clock time; without PTP, a
latch of the camera's timestamp against host time gives the offset and
drift of its tick clock (GigE Vision's timestamp control registers; aravis
reads the tick frequency at connect, and ADAravis prints it). ADAravis
carries no PTP code: enabling PTP is a GenICam feature of the camera,
exposed as a record by the auto-generated template where the camera has
it. Whether the bench cameras have it is unverified (open question 4): the
one camera checked so far, a Basler acA1280-60gm, reports no IEEE 1588;
Basler's PTP documentation says ace Classic cameras change tick duration
when PTP is enabled, which implies support on at least some of that family
(https://docs.baslerweb.com/precision-time-protocol). Slice 6, gated on
need: required for 5 Hz under exposure scans, not for 1 Hz.

**Conditions for 5 Hz**, all measurable at the bench: calibrated offsets
for every LabVIEW device; the witness poll at 10 ms; hardware stamps on
cameras whose exposure is scanned; LabVIEW stamps by worker receipt, or
their hosts' clock swing measured under 20 ms over a day (the first host:
10 to 16 ms behind the box, a few ms of swing in ten minutes, section 4.5).

### 4.5 Clocks

Measured 2026-11-03, read-only, from chrony's logs on the Linux hosts and
a passive capture of one Windows host's NTP queries at the services box
(the numbers' home: `geecs-epics:docs/runbooks/timing.md`):

| Pair | Mean | SD | Worst | Record |
|---|---|---|---|---|
| each of two camera servers vs the services box | -7 and -14 us | 33 and 43 us | within 0.18 ms | 9.4 h steady state |
| camera server vs camera server, through the box | +7 us | 43 us | 0.25 ms | 9.4 h |
| each Linux host vs campus NTP | 0 us | 6 to 13 us | within 0.09 ms | 9.4 h |
| one LabVIEW host (W32Time) vs the services box | 10 to 13 ms behind | wanders 2 ms in minutes, reversing | 16 ms | 10 min; a day's record running |

- **Linux IOC hosts**, the witness host included: chrony on campus NTP,
  set by the fleet's Ansible base role
  (`geecs-epics:docs/runbooks/ansible.md`); every IOC host also carries the
  services box as a cross-check source, so the pairwise offsets above are
  logged continuously. Drift under 4 us per hour; no step; the only
  excursions above 0.2 ms (up to 1.4 ms) sat in the minutes after a chrony
  start, which is the requirement handed to that lane: slew only after
  boot, never step during a run. The Linux side spends about a thousandth
  of the 1 Hz budget and a hundredth of the 5 Hz budget: stamp placement,
  not clock sync, is what can spend it (section 4.4).
- **Windows device hosts**: W32Time through the domain. It was built to
  keep Kerberos within five minutes and corrects the clock rate in coarse
  steps at each poll, so a host wanders and oscillates by milliseconds to
  tens of milliseconds depending on its crystal and poll interval; the
  maintainer's earlier tuning attempts found large offsets and periodic
  oscillations that differed per computer. The design does not depend on
  these clocks, in three layers: in scan mode the announce stamp, on the
  worker's clock, anchors each acquisition and the LabVIEW clock is never
  consulted (section 4.6); outside scan mode the worker's receipt stamp is
  a Windows-free alternative to `acq_timestamp`, inside the budget at 1 Hz
  and the one to use at 5 Hz; a slow wander moves the tracked offset, a
  step raises *clock-step*. Nothing is tuned: the migration retires these
  clocks a host at a time, and if a LabVIEW device's own stamp is ever
  needed at 5 Hz the answer is a real NTP client on that host (Meinberg's
  ntpd port holds about a millisecond on a LAN), a facility change on the
  facility's side.

### 4.6 Rows

- A run's rows are the witness shots between its start and its end: one
  per step in a strict run, N per step in a gated batch. A shot no device
  acquired is still a row, all NaN, flagged.
- `Shotnumber` stays the row index; a new column carries the witness count,
  with the epoch in the run's metadata. Precedent: `announced_shot`
  (`GEECS-Data-Utils/geecs_data_utils/tiled_schema.py:38`).
- **Retakes.** The join never needs one, and never rewinds: every
  acquisition keeps its own shot. A plan *may* fire one more shot at a step
  whose synchronous device missed; that is a new row with its own shot,
  plan policy, default off. "No retakes" in the policy means the join must
  not depend on them, not that a plan is forbidden one.
- **LabVIEW scan mode** is what Master Control did and what the worker's
  scan-mode profiles still do (`docs/acquisition/scan_mode.md`): a scan
  puts every LabVIEW device into scan mode, where its acquisition loop
  runs only when a new shot number arrives. On each announcement the
  device enters its acquire state, takes **one** acquisition (the next
  hardware edge, or the next frame if it free-runs), saves it under the
  announced number, and goes back to waiting for the next announcement.
  Outside scan mode the same device acquires every edge. Consequences:
  - An edge that arrives while a device is still awaiting its announcement
    is **not acquired**. The witness still counts it. For a LabVIEW device
    in scan mode, "every shot recorded" means every announced shot, and a
    witness shot it has no acquisition for is flagged *not-armed* when the
    announcement came after the edge, *missing* otherwise. The two are
    told apart by the announce stamp, so a structural gap is never read as
    a drop.
  - The wire "shot number" is the announced number echoed, reset at each
    scan (maintainer). With the announce stamp recorded, the audit for a
    scan-mode device is exact rather than a delta rule: the acquisition
    that answers announcement k must sit on the **first witness edge after
    the announce stamp plus the device's arming latency**. The join
    computes that shot from the witness stream and compares it with the
    time-assigned one; a disagreement means the device stamped a later
    edge than the announcement intended, the retake confusion of today,
    and both are flagged. Announcements may lag edges, never lead them
    twice: two acquisitions for one announcement is a duplicate, flagged.
  - A free-running camera in scan mode acquires its **next frame after
    the announcement**, not an edge. Its residual against the witness is
    spread over one frame period. Nearest-time assignment still holds
    while the frame period is under half the shot period; otherwise the
    acquisition is flagged and the row says so. The adapter records this
    class from the device's own trigger setting.
  - The **arming latency**, announcement to acquire state, is a per-device
    number that bounds how soon after the announcement the box may fire.
    Today's strict loop announces, then fires, and at 1 Hz the margin is
    wide; at 5 Hz it is a budget item, measured by the passive logger
    during ordinary scans (§8).

## 5. Device by device

### 5.1 ADAravis cameras

Read against areaDetector/ADAravis R2-3 (the image's version), unchanged on
master:

| NDArray field | ADAravis sets it to | Use here |
|---|---|---|
| `uniqueId` | the driver's own count of complete frames (`aravisApp/src/ADAravis.cpp:749-752`, `:803`); a buffer with missing packets is dropped, counted only in `AravisFailures` (`:255-256`); zeroed by any client write to `ArrayCounter`, which the strict plan does at prepare (`plans/strict.py:390`, `devices/detector.py:1121`) | the audit counter, kind *driver*; the audit works on deltas within a run and re-anchors at a reset |
| `timeStamp` → `NDArrayTimeStamp` | the camera's own clock from the frame leader, nanoseconds to seconds (`:804`): ticks since power-up, or PTP time when the camera's PTP is on | the hardware stamp, slice 6 |
| `epicsTS` → `NDArrayEpicsTSSec/nSec` | host time when the complete frame was processed (`:807`) | today's stamp, kind *host-receipt*; `NativeCamera` already turns it into the row's stamp column (`devices/native.py:456-463`) |
| unused | `arv_buffer_get_frame_id` (the GigE Vision block id, a hardware frame counter the client cannot reset; aravis 0.8.0, the image's aravis is 0.8) and `arv_buffer_get_system_timestamp` (host time at the first packet, earlier than `epicsTS`) | an upstream contribution later: a `FrameID` attribute would tell a wire drop (a block-id gap) from a missed trigger (none); not needed for v1 |

ADAravis has no chunk-data parsing, so a chunk frame counter would also be
a driver change. v1 changes nothing in the camera IOC.

### 5.2 PicoScope (GEECS-BELLA/ioc-picoscope)

`captureOnce` arms with `RunBlock`, polls `IsReady` every 2 ms, and takes
the stamp when the block is ready, after the post-trigger samples have been
captured (`picoscope/picoscopeApp/src/PS5000A.cpp:556-600`); `publish`
puts a per-publish `uniqueId` and that stamp on every array
(`:611-690`). So the stamp sits one post-trigger duration after the
trigger, which changes with `NumSamples`, `PreTrigger` and `SampleInterval`.
Slice 5 moves it to the trigger: stamp minus
`(NumSamples − PreTrigger) × SampleInterval`, a few lines in our own
driver, proven on the fake. Between `GetValues` and the next `RunBlock`
the scope is unarmed; a trigger in that gap is lost silently and appears
only as an audit gap, milliseconds wide at 1 Hz. The scope contract's
"stamped at the capture" (`geecs-epics:docs/scope-contract.md`) becomes
"stamped at the trigger".

### 5.3 The witness instance (geecs-epics)

The 1808X instance beside the shot-control box, on the nearest Linux IOC
host's USB (open question 2): stock measComp records, `PollSleepMS` pinned
to 10 at boot, a bench IOC until the counter is proven on the tee, then a
lab IOC. Its registry entry is a new kind, *shot witness*, naming the
trigger domain it counts. Decision 13 asks for a simulator: a witness fake
whose count advances on the sim bench's trigger-box fire, so the plan mode
(§6) runs end to end against the existing camera fake.

### 5.4 LabVIEW devices: the adapter

GEECS-Core already parses the wire shot number beside the values
(`GEECS-Core/geecs_core/transport/tcp_subscriber.py:389-394`,
`client/geecs_device.py:74`). The adapter turns each device push into one
evidence record: `acq_timestamp` as a *host-receipt* stamp, the shot number
as an *announced* counter, class from the device's own trigger setting,
and, in a scan-mode run, the worker's announce stamp for that number
(§4.6). It knows the regime: in scan mode one acquisition per
announcement, otherwise one per edge. One module, no new wire protocol,
deleted with the last LabVIEW device. The
LabVIEW-only strict path (baseline, retake, rewind) is **not touched**
until then and is then deleted whole.

### 5.5 Asynchronous devices

Monitor caches, the last-before rule, the age column. Nothing device-side.
A camera whose trigger-mode readback says free-run is asynchronous for the
run whatever its registry entry says, and the mismatch is reported at
prepare.

## 6. Who clocks a run

Two shapes, with their failure modes:

| | **W1: witness as join key only** (plans unchanged) | **W2: witness-clocked acquisition** (recommended) |
|---|---|---|
| How a plan knows a shot happened | each device's own signal, as today (`acq_timestamp` advancing; `NumCaptured` advancing) | the witness count advances |
| Strict fire | announce to the LabVIEW devices, fire, wait for every device, retake on a miss | announce as today, fire, see the count advance within a bounded wait (period + largest offset + margin, the constants in `plans/gated.py:78-80`), then wait the same bound for each synchronous device; a device that produces nothing gets NaN and *missing* (or *not-armed*, §4.6); no retake, no rewind |
| Gated batch | the shot clock is a camera (`plans/gated.py:86-123`) and the sampler counts its stamp advances (`devices/sampler.py:132`): a shot the clock camera misses is a row that never exists | every witness tick is a row |
| Fire confirmation | none; the native shot-control note lists it as missing | the fire is confirmed by the witness; a fire the witness never sees is reported as not fired, and the plan stops rather than fire again |
| Mixed native and LabVIEW runs | still refused: the strict retake needs a native rewind (`plans/registry.py:402-437`) | allowed: no path needs a rewind |
| Strict and gated | two machineries | one: strict is one tick per step, gated is N |
| Failure it adds | none new | the witness is a single point of failure for runs in this mode: a run without a witness falls back to W1 behaviour and is flagged *unwitnessed* throughout |
| Change | smallest: a readable and columns | a new acquisition mode beside the existing two; the LabVIEW-only strict path stays as is until deletion |

**Recommendation: W2**, built as a third acquisition mode so nothing
existing moves, becoming the default once the bench proves the witness as
a live tick source. The policy "every shot recorded, gaps flagged, no
retake" is a run-time behaviour, and W1 cannot deliver it: strict keeps
retaking and gated keeps losing the shots its clock camera misses.
Overturned by: the witness proven unreliable as a live source (USB dropouts
at a rate that would stall runs), or the per-tick wait proven incompatible
with 5 Hz.

Pause and resume under W2: pause in place at a tick boundary; rows already
closed stay closed; resume continues at the next tick. There is no rewind
to reconcile.

## 7. Where the changes land, and slices

| Slice | Repo | Content | Size (code / tests) | Hardware |
|---|---|---|---|---|
| S0 | geecs-epics | the witness instance (`/add-device` of the DAQ family) beside the shot-control box; `PollSleepMS` pinned; the bench proof: the tee'd edges counted, the poll-lag spread, no doubles; the witness fake on the sim bench | ~80 config / the check script | the tee and the USB run (maintainer); one write, the poll period |
| S1 | GEECS-Schemas | the evidence record (§4.2) and a per-device sync entry (class, stamp kind, counter kind, offset, budget, trigger domain), extending `shot_offsets.py`; the witness registry kind | ~150 / ~150 | — |
| S2 | GEECS-Data-Utils | the join: rows from the witness, assignment, audit, flags, residual statistics; the same arithmetic live and offline (`shot_join.py` grows, nothing replaced yet) | ~250 / ~300 | — |
| S3 | GeecsBluesky | the witness device (count, stamp, epoch); evidence columns for native cameras (`ArrayCounter`, the EpicsTS pair) and the scope; the LabVIEW adapter (§5.4); a passive probe in the `capture/probes` style that logs devices and witness with no scan | ~300 / ~350 | the probe runs read-only |
| S4 | GeecsBluesky | the witness-clocked acquisition mode (§6), if ruled in; the refusals of `refuse_native_run` lifted for that mode only | ~250 / ~300 | the sim bench, then a laser-off run |
| S5 | ioc-picoscope | the stamp at the trigger (§5.2) | ~15 / the fake | one capture on the bench scope |
| S6 | geecs-epics, GeecsBluesky | hardware stamps where the cameras allow: PTP on the camera links, or the timestamp latch and drift fit; stamp kind *ptp* / *device-clock* | depends on the cameras | the camera servers' links |
| S7 | GeecsBluesky, GEECS-Data-Utils | deletion at migration end (§10) | negative | — |

**Tests.** The join is pure arithmetic and gets a synthetic witness and
synthetic devices with injected misses, extra frames, doubled edges,
counter resets, clock steps, stalls that share a witness stamp and a
witness outage, one test per flag, each proven to bite. The plan mode runs
on the sim bench against the trigger-box fake, the camera fake and the
witness fake, with the fake camera made to drop a frame. The adapter is
tested against GEECS-Core's fake scan-mode device. Live and offline joins
of the same run must be identical.

## 8. Bench

Per-facility facts first (the trigger tree is this facility's; §1), then
measurements, in the order of the hardware-test runbook
(`geecs-epics:docs/runbooks/hardware-tests.md`): read-only before any write,
the maintainer's hand-off and go as the gate.

1. **Witness alone.** The tee into a counter input, the input level checked
   against the box's output before cabling (open question 3). Monitor the
   count and its stamp for an hour of ordinary running: rate against the
   rep rate, doubles, the spread of successive stamp deltas (the poll lag).
   The one write, the poll period, waits for the go.
2. **Passive logger, no scan and no beam cost.** Subscribe the LabVIEW
   devices' push stream and the witness; per device, the residual
   distribution, the gap statistics and how the wire shot number behaves
   between scans. This measures Windows clock quality at the same time.
   Left running through ordinary scans, with the worker's announce stamps,
   it also gives each device's arming latency (§4.6).
3. **Native cameras**, once the camera servers' IOCs are deployed (another
   lane owns that): `ArrayCounter` and the EpicsTS pair against the
   witness, and a read-only listing of the auto-generated GenICam records
   for IEEE 1588 and timestamp-latch features.
4. **PicoScope**: the same, before and after S5.

What the measurements decide: the stored offsets, whether the Windows
stamps meet the budget, whether S6 is needed for the target rate, and
whether W2's live wait is adequate.

## 9. Open questions for the maintainer, ranked

1. **W2.** Build the witness-clocked mode (§6), or stop at the join (W1)?
2. **Witness placement.** A small Linux IOC host beside the shot-control
   box, or a USB run to the nearest existing host; and the poll period at
   10 ms.
3. **Input compatibility.** The 1808X counter input's threshold and
   minimum pulse width against the DG645 output's level and width, before
   the tee is cabled.
4. **Camera features.** After the deploy, a read-only listing of each
   bench camera's generated records settles PTP and timestamp-latch
   support; or your knowledge of the Blackfly and ace models.
5. **The LabVIEW wire shot number outside scan mode**: in scan mode it is
   the announced number (settled, §4.6); what it counts while a device
   free-runs between scans the passive logger answers. Nothing to decide
   unless it behaves unexpectedly.
6. **Offset updates**: by recorded calibration only (recommended, for
   reproducible runs), or continuously from the tracker.
7. **Shot columns**: `Shotnumber` stays the row index with a new witness
   column (recommended, the `announced_shot` precedent), or becomes the
   witness count.
8. **5 Hz**: which of §4.4's conditions to pursue first once 1 Hz runs
   under the witness.

## 10. What this deletes at the end

When the last LabVIEW device has migrated and W2 is the default:

- the strict retake loop and the rewind: `geecs_take_reading`'s attempts,
  `discard_uncollected`, `_rewind_plugins`, the `rewindable_wrapper` use
  (`plans/strict.py`, `devices/detector.py`);
- `refuse_native_run` and its reasons (`plans/registry.py:402-437`;
  `native_scope.py`);
- the camera shot clock and the sampler's stamp-advance clock
  (`plans/gated.py:86-123`, `devices/sampler.py`);
- the LabVIEW adapter (§5.4) and the scan-mode announce with it;
- `measure_shot_offsets_plan` and `check_shot_sync_plan`
  (`plans/calibration.py:813`, `:1049`): the tracker measures offsets from
  any run;
- §4 of `GeecsBluesky/NATIVE_SHOT_CONTROL.md`, whose gaps this note closes,
  and this file.
