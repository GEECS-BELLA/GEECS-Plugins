# Shot synchronization — one shot identity for every device

Status: **design, for review.** Nothing here is built. Two phases (§6):
**Phase A**, hybrid strict scans now, with no retakes, no witness and no
gated mode;
**Phase B**, the witness model, held until gated runs with native devices
are wanted. Delete this file in the PR that lands the last slice
(`Planning/README.md`).

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
(`plans/registry.py:411-447`). The rewind is a GEECS invention; the
refusals are the price of building on it.

Three facts frame the replacement:

| Fact | Consequence |
|---|---|
| One trigger. At this facility every synchronous readback hangs off one gated output of the shot-control box, and the ARMED/STANDBY gate stops those edges (maintainer, 2026-11-02) | one witness on that output sees exactly the shots the devices see (§4.1); other facilities may need one witness per trigger domain |
| 1 Hz is the norm, 5 Hz the aspiration | the assignment margin is half a period: 500 ms, or 100 ms |
| No event system (MRF) | an EVG with an EVR per host would deliver shot identity to every device at the hardware moment: the only stronger design, ruled out on cost for this facility; nothing here precludes it later |

The policy this serves (maintainer, 2026-11-02): **every shot recorded,
gaps flagged, no retake needed.**

**Scope by phase** (maintainer, 2026-11-03). Gated runs are not used today,
even for LabVIEW devices, and will re-enter faithfully once the switch to
EPICS is complete, with no LabVIEW device in them. What is needed now is a
strict scan whose rows hold native cameras and LabVIEW devices together.
So Phase A (§6.1) is that alone: strict with no retakes, each camera's
frames joined to rows by an explicit shot, and no witness, no gated mode,
no 5 Hz and **no scan mode on native devices**. Phase B is the rest of this note: the
witness, the join with its audit, gated rows, and the stamp tiers for
5 Hz. Phase A records the evidence Phase B will read, so nothing taken in
between is lost to it.

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
| **fire permit** | a readback that says whether a useful shot can be taken now, with a count of its changes to ready; a strict plan configured with one fires only on a fresh change to ready (§6.1). Another facility's mode; its source is a facility value |
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

This section is Phase B's model. Phase A (§6.1) borrows its vocabulary,
the evidence fields of §4.2 and the drain-offset idea of §4.4, and builds
none of the rest.

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
| `device_stamp`, `stamp_kind` | the device's timestamp and where it was taken (§2). A device with two stamps, a camera's own clock and the host receipt, stores both from day one: the join names in `stamp_kind` the one it used, and the other stays for the audit and for a later fit of the camera clock (slice 6) over data taken before that slice exists |
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
  (`GEECS-Schemas/geecs_schemas/shot_offsets.py`). Open question 10 asks
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
reads the tick frequency at connect, and ADAravis prints it). Checked
2026-11-03 against each camera's own feature list, read-only: the six
native cameras on the two camera servers (four Point Grey Blackfly
BFLY-PGE, 09S2M and 13S2M; a Basler acA1600-20gm; a Basler acA2500-20gm)
all have `GevTimestampTickFrequency`, `GevTimestampControlLatch`,
`GevTimestampValue` and `ChunkTimestamp`; only the acA2500-20gm supports
IEEE 1588 (`GevSupportedIEEE1588` true, PTP off). The third camera
server's cameras are not read yet, and `arv-tool features` does not list
every feature: query a feature by name. **Decision (maintainer,
2026-11-03): skip PTP here.** Two stamp tiers at this facility:
*device-clock*, the camera's ticks fitted to host time by the latch (a
command written to the camera every few seconds, no setting changed) and
an offset-and-drift fit; and *host-receipt* as the fallback. *ptp* stays
a named kind in the contract for a facility whose cameras and switches
have it, unbuilt here. ADAravis already publishes the camera clock per
frame (section 5.1), so the latch and the fit are the whole of slice 6,
gated on need: required for 5 Hz under exposure scans, not for 1 Hz.

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
  acquisition keeps its own shot. Phase A removes the retake from strict
  altogether (§6.1, open question 1). The join does not depend on that
  removal: it would hold even if a later plan fired one more shot at a
  step, because that shot is a new row with its own shot.
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
a driver change; `ChunkTimestamp`, which all six cameras offer, is not
needed, since the frame leader already carries the same clock. v1 changes
nothing in the camera IOC.

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
host's USB (open question 6): stock measComp records, `PollSleepMS` pinned
to 10 at boot, a bench IOC until the counter is proven on the tee, then a
lab IOC. Its registry entry is a new kind, *shot witness*, naming the
trigger domain it counts. Decision 13 asks for a simulator: a witness fake
whose count advances on the sim bench's trigger-box fire, so the plan mode
(§6.2) runs end to end against the existing camera fake.

### 5.4 LabVIEW devices: the adapter

GEECS-Core already parses the wire shot number beside the values
(`GEECS-Core/geecs_core/transport/tcp_subscriber.py:389-394`,
`client/geecs_device.py:74`). The adapter turns each device push into one
evidence record: `acq_timestamp` as a *host-receipt* stamp, the shot number
as an *announced* counter, class from the device's own trigger setting,
and, in a scan-mode run, the worker's announce stamp for that number
(§4.6). It knows the regime: in scan mode one acquisition per
announcement, otherwise one per edge. One module, no new wire protocol,
deleted with the last LabVIEW device. The adapter is Phase B. Phase A
touches the LabVIEW strict path in one way only: it deletes the retake and
the strict rewind (§6.1). The baseline, the announce and the stamp wait
stay until the last LabVIEW device has migrated.

### 5.5 Asynchronous devices

Monitor caches, the last-before rule, the age column. Nothing device-side.
A camera whose trigger-mode readback says free-run is asynchronous for the
run whatever its registry entry says, and the mismatch is reported at
prepare.

## 6. Two phases: hybrid strict now, the witness when gated returns

### 6.1 Phase A: hybrid strict scans with no retakes

Evidence, 2026-11-04 (Undulator, laser off): a strict scan with a native
power supply and LabVIEW scalar devices passed (Scan002); a strict scan
with a native camera and a LabVIEW camera both saving images was refused
at plan time by the "beside a gateway plugin camera" reason
(`plans/registry.py:440-447`).

**The policy (maintainer's lean, 2026-11-04; open question 1): strict mode
takes no retakes.** One fire per step. The row closes on whatever arrived
within the timeout; a device that produced nothing reads `NaN` and is
flagged *missing*; the scan goes on. Nothing refires and nothing rewinds,
and the retake budget goes (`max_refires`, `plans/strict.py:248`,
`:397-447`). Today's loop already keeps the partial row and then takes
another shot (`EVENT_SCHEMA.md` "Partial rows"). So the change is that it
stops after the partial row; the step simply has one fewer complete row.
A pause holds in place and never replays a shot: the shot stays wrapped
in `rewindable_wrapper`, now with rewinding off for every strict run, not
only when a native camera says so (`plans/strict.py:448`,
`devices/native.py:499-506`).

**What the no-retake policy alone does not solve.** Bluesky requires that
within one event stream, every event carries a stream datum for **every**
external key or for none, all of the same width (bluesky 1.15
`bundlers.py:938-942` and `:843-850`). The stock `HDF5Consolidator` that
registers a stack in Tiled sizes it as the sum of the datum widths and
reads from frame 0, so a frame no datum references shifts every later
frame by one shot (`NATIVE_DEVICES.md:153-178`). Today's partial row
meets both rules by rewinding every gateway plugin so that no LabVIEW
camera has a datum (`plans/strict.py:401-411`,
`devices/detector.py:1177-1190`); a native keeps a frame it wrote, legal
only because it is then the run's sole external key
(`devices/native.py:676-684`). The retake supplies the complete row. Without a retake, and with a stock
native writer that cannot rewind, one stream cannot hold a row in which
one camera has a frame and another does not. **This corrects the earlier
draft of this section**, which kept extra native frames in the stack,
flagged and referenced by index. Under the stock consolidator that is the
shift above. The earlier draft also had natives pick their frame by edge
time. That is unnecessary under a box-fired strict profile: ARMED is
quiescent (`devices/shot_control.py:7-9`, `:54-60`), so, given the cabling
rule that every native trigger comes from the gated output
(`NATIVE_DEVICES.md` "Acceptance"), a native in hardware-trigger mode
takes exactly the fire's edge, as the current native strict path already
assumes. Scan-mode profiles stay refused for natives.

So the link between a row and a frame has to change. Today the link is
the row's position: frame k is row k's only because nothing ever skips.
There are two ways to make it explicit.

| | **P: one stream, placeholder frames** | **S: a datum stream per camera, an explicit frame map in the row** (recommended) |
|---|---|---|
| Shape | `primary` as today: every camera's frame is a datum in the row | each camera's stack is a datum-only stream of its own, the shape a non-essential plugin camera already has (`EVENT_SCHEMA.md:170-172`): every frame written gets a datum, contiguous, so the stock consolidator registers the stack unchanged. `primary` holds the scalars, the flags and a new `<cam>-frame_index` per camera: the stack index of the frame the plan accepted for that row, `NaN` when none |
| A LabVIEW camera misses | the PVA file plugin appends a placeholder frame on the worker's request, marked in a per-frame flag dataset (`GeecsPvaGateway/geecs_pva_gateway/file_plugin.py`, beside `_rewind` at `:1016`); every other camera keeps its frame (today all LabVIEW cameras lose theirs) | `frame_index` `NaN`, *missing*; every other camera keeps its frame |
| A native camera misses | **the step fails and the run ends**, as today (`devices/native.py:676-692`): the stock `NDFileHDF5` cannot write a placeholder | `frame_index` `NaN`, *missing*; the run continues |
| A frame arrives after its timeout | the plugin's watermark drops it (LabVIEW); a late native frame shifts the stack | it is in the stack and the stream, no row's `frame_index` names it; the stack check reports it *unreferenced*, with the shot its stamp points to |
| A stray edge outside a shot | shifts the stack (the cabling rule) | the same: *unreferenced*, no shift |
| Contract change | the per-frame placeholder flag (`EVENT_SCHEMA.md`, the stack format) | strict camera datums leave `primary` for per-camera streams, and `primary` gains `<cam>-frame_index` (`EVENT_SCHEMA.md`). The `frame_index` column and its readers are **new**; the stamp join the existing per-camera streams use (`GEECS-Data-Utils/geecs_data_utils/shot_join.py:379-463`) is the fallback and the audit |
| New GEECS mechanism | a placeholder in our plugin, deleted with LabVIEW | none in the writers or Tiled; a native camera must accept the datum-stream role (`NativeCamera.kickoff` refuses streams today, `devices/native.py:670-674`) |
| After the last LabVIEW device | the problem returns: any native miss ends a strict run | unchanged; it is also the shape Phase B's witness rows need |

A third option was considered and rejected: keep one stream and swap in a
consolidator that honours the seq_num→index map, the cure
`NATIVE_DEVICES.md:178-180` names for refires. It fixes extra frames, not
missing ones: a native miss still leaves an event with a subset of
external keys, which the bundler refuses (`bundlers.py:938-942`). It also
puts a GEECS substitution in the Tiled path. (The bundler's width check
tests the previous width for truthiness, `:844`, so a zero-width datum
passes only when it comes first. That is an accident of order, not a
contract to build on.)

**Recommendation: S.** P fails the policy for natives permanently: at
100% EPICS, any native miss still ends a strict run. Its one new
mechanism, the placeholder, lives in a plugin that is deleted with
LabVIEW. S holds the policy for every camera, keeps the stock writer and
the stock consolidator, and reuses a stream shape the run kinds already
have. Its costs are the contract change, a native datum-stream role, and
moving the strict readers from datum position to `frame_index`. The first
slice counts those readers before any code moves.

**What would overturn it:** the native drop rate measured as negligible
**and** the strict `primary` shape wanted frozen for 1.0. Then P is the
cheaper sufficient answer, and a native miss ending the run is an
accepted rarity. The rate is measurable read-only now: `AravisFailures`
and the frame counter on the deployed native cameras across ordinary
runs.

**Explicit shot identity for native frames.** Under S, `primary`'s
`<cam>-frame_index` is the recorded map from row to frame. The plan writes
it when it accepts the frame (the writer's count at acceptance), and the
stack check verifies it against the frame's stamp. That is Phase A. A shot number *inside* the native stack, per frame, is
the stronger form: areaDetector's PV attribute (an `NDAttributes` entry of
type `EPICS_PV`) stores, on each frame, a PV's value at the moment the
frame passes through. Its natural source is the witness count (§4.1), so
it is Phase B. It needs one line in the camera family's shared attribute
file in geecs-epics and no driver change. Neither form puts a GEECS
invention in the stock writer.

**Per step (S).**

1. Announce to the LabVIEW devices if the profile does, fire once, and
   await every device within its timeout, as `fire_and_await_shot` does
   today (`plans/strict.py:154-243`).
2. Close the `primary` row: real values for whatever arrived, each
   camera's `frame_index`, and `NaN` and *missing* for whatever did not.
3. Collect each camera's datum stream: one datum covering every frame
   written since the last collect, late or stray ones included, so the
   stream always matches the stack.
4. No refire. The stack check (`callbacks/stack_check.py`) asserts per
   camera that the frames on disk are the stream's datums, that the row
   map's indices are distinct and stamped near their row's fire, and it
   reports every frame no row names.

**Tests.** A fake LabVIEW camera misses with a native present: the row
keeps the native's frame, the LabVIEW stream has no record for the shot,
and the run continues. A fake native drops a frame: the same, mirrored. A
late frame and a stray edge each produce one unreferenced frame and no
shift.
A pause mid-shot resumes without replay. The Tiled registration of each
stream is checked against real Tiled (the stock consolidator, one stream
per stack). The s-file of a hybrid run with induced misses is identical
whether it is built live or offline.

**What Phase A records for Phase B**: each frame's record (`shot`,
device counter, both stamps, flags) and the announce stamp, so a witness
join can be run offline later over Phase A data.

**The fire permit: a mode to retain** (another facility's lead, via the
maintainer, 2026-11-03). At that facility a small service reads the
machine and reports *ready* or *not ready*, meaning only whether a useful
shot can be taken now, and the scanner fires a shot when the report
changes from not ready to ready, waiting ten or twenty seconds if that is
what it takes. The design keeps it as a contract and places it at one
seam:

- **Contract.** A *fire permit* is a readable with two fields: the level
  (ready or not) and a count of its changes to ready. Its source is a
  facility value in the registry: an EPICS binary record, a GEECS device
  variable through the gateway, or a thin client of a text service such
  as that facility's, each counting the transitions itself. Nothing in the
  plan names the source.
- **Seam.** The wait sits between arming and the fire, inside
  `fire_and_await_shot` (`plans/strict.py:154-243`): every device has
  baselined and the LabVIEW devices are announced and armed, then the plan
  waits for the permit, then fires, and the device timeouts start at the
  fire as they do today. So a ready window shorter than the arming latency
  is still usable, and a twenty-second wait costs no device its timeout.
- **Edge, latched, consumed.** The plan fires on a change to ready that
  happened since the last fire, read from the count, and only while the
  level is still ready at the moment of firing. Latching the count means a
  change that arrives while the plan is busy closing the previous row is
  not lost; consuming it at the fire means a level that simply stays
  ready does not fire the next step by itself. A facility whose readiness
  is a level, not an event, configures the permit as such.
- **Waiting.** No timeout by default, since the wait is the facility's
  normal; the plan reports the wait in progress and the run stays
  pausable. An optional timeout flags the step *no-permit* and pauses the
  run for the operator rather than aborting it.
- **What a long wait does to synchronization.** Nothing to the pick: fires
  seconds apart make the edge unambiguous. Hardware-triggered devices sit
  armed through the wait, as they did under Master Control. A
  free-running camera in scan mode takes its next frame right after the
  announcement, seconds before the shot, so under a permit it is treated
  as asynchronous for the run and its frame flagged *stale* when it
  precedes the fire by more than a frame period, the §4.3 rule. In Phase B
  the witness counts fires however spaced; gated runs and the permit are
  unrelated, since a gated run takes every edge.

### 6.2 Phase B: who clocks a run

Two shapes, with their failure modes. Phase A is compatible with either;
the ruling is deferred until gated runs with native devices are wanted.

| | **W1: witness as join key only** (plans unchanged) | **W2: witness-clocked acquisition** (recommended) |
|---|---|---|
| How a plan knows a shot happened | each device's own signal, as today (`acq_timestamp` advancing; `NumCaptured` advancing) | the witness count advances |
| Strict fire | announce to the LabVIEW devices, fire, wait for every device, retake on a miss | announce as today, fire, see the count advance within a bounded wait (period + largest offset + margin, the constants in `plans/gated.py:78-80`), then wait the same bound for each synchronous device; a device that produces nothing gets NaN and *missing* (or *not-armed*, §4.6); no retake, no rewind |
| Gated batch | the shot clock is a camera (`plans/gated.py:86-123`) and the sampler counts its stamp advances (`devices/sampler.py:132`): a shot the clock camera misses is a row that never exists | every witness tick is a row |
| Fire confirmation | none; the native shot-control note lists it as missing | the fire is confirmed by the witness; a fire the witness never sees is reported as not fired, and the plan stops rather than fire again |
| Mixed native and LabVIEW runs | allowed in strict by Phase A (no retakes, per-camera frame streams); gated still refused | allowed: no path needs a rewind |
| Strict and gated | two machineries | one: strict is one tick per step, gated is N |
| Failure it adds | none new | the witness is a single point of failure for runs in this mode: a run without a witness falls back to W1 behaviour and is flagged *unwitnessed* throughout |
| Change | smallest: a readable and columns | a new acquisition mode beside the existing two; the LabVIEW-only strict path stays as is until deletion |

**Recommendation for Phase B: W2**, built as a third acquisition mode so
nothing existing moves, becoming the default once the bench proves the
witness as a live tick source. The policy "every shot recorded, gaps
flagged, no retake" is a run-time behaviour, and W1 cannot deliver it for
gated runs: the clock camera keeps losing the shots it misses. Overturned
by: the witness proven unreliable as a live source (USB dropouts at a rate
that would stall runs), or the per-tick wait proven incompatible with 5 Hz.

Pause and resume under W2: pause in place at a tick boundary; rows already
closed stay closed; resume continues at the next tick. There is no rewind
to reconcile.

## 7. Where the changes land, and slices

| Slice | Repo | Content | Size (code / tests) | Hardware |
|---|---|---|---|---|
| A0 | GEECS-Plugins | this note's Phase A ruled (§9, 1 and 2); the readers of strict camera datums counted (the s-file and Parquet writers, `tiled_export.py`, the portal's Images tab, ScanAnalysis's stack reads, the stack check) so A3's size is known before it starts | reconnaissance | — |
| A1 | GeecsBluesky | strict without retakes (§6.1): one fire per step, the partial row kept and flagged, the retake loop and `max_refires` deleted; the partial row keeps today's rule (no LabVIEW camera frames, a native keeps its own) and the plugin rewind it needs for one slice; rewinding off for every strict run; the refusal stays | ~40 / ~80, net negative | none; the sim bench |
| A2 | GeecsBluesky, GEECS-Schemas | S: a datum stream per camera, `<cam>-frame_index` and the *missing* flag in `primary`, the native datum-stream role; the strict rewind and both `discard_uncollected`s deleted; `EVENT_SCHEMA.md` in the same PR | sized by A0 | none; the sim bench, real Tiled |
| A3 | GEECS-Data-Utils, GEECS-DataPortal, ScanAnalysis | the readers onto `frame_index` (`shot_join.py`, the s-file, the Images tab), the stack check per stream with *unreferenced* frames | sized by A0 | — |
| A4 | GeecsBluesky | the "beside a gateway plugin camera" refusal lifted; the bench scan of §8 item 0 | ~10 / ~30 | §8 item 0, laser off |
| A5 | GeecsBluesky, GEECS-Schemas | the fire permit (§6.1): the permit readable and its registry kind, the wait inside `fire_and_await_shot`, the *no-permit* and *stale* flags; a fake permit for the tests | ~60 / ~80 | none; a fake permit on the sim bench |

Under P instead, A2 is the plugin's placeholder (`GeecsPvaGateway`) with its
per-frame flag and the worker's request on a LabVIEW miss, A3 shrinks to
the readers of that flag, and a native miss keeps failing the step.
| S0 (Phase B) | geecs-epics | the witness instance (`/add-device` of the DAQ family) beside the shot-control box; `PollSleepMS` pinned; the bench proof: the tee'd edges counted, the poll-lag spread, no doubles; the witness fake on the sim bench | ~80 config / the check script | the tee and the USB run (maintainer); one write, the poll period |
| S1 (Phase B) | GEECS-Schemas | the evidence record (§4.2) and a per-device sync entry (class, stamp kind, counter kind, offset, budget, trigger domain), extending `shot_offsets.py`; the witness registry kind | ~150 / ~150 | — |
| S2 (Phase B) | GEECS-Data-Utils | the join: rows from the witness, assignment, audit, flags, residual statistics; the same arithmetic live and offline (`shot_join.py` grows, nothing replaced yet) | ~250 / ~300 | — |
| S3 (Phase B) | GeecsBluesky | the witness device (count, stamp, epoch); evidence columns for native cameras (`ArrayCounter`, the EpicsTS pair) and the scope; the LabVIEW adapter (§5.4); a passive probe in the `capture/probes` style that logs devices and witness with no scan | ~300 / ~350 | the probe runs read-only |
| S4 (Phase B) | GeecsBluesky | the witness-clocked acquisition mode (§6.2), if ruled in; the refusals of `refuse_native_run` lifted for that mode only | ~250 / ~300 | the sim bench, then a laser-off run |
| S5 (Phase B) | ioc-picoscope | the stamp at the trigger (§5.2) | ~15 / the fake | one capture on the bench scope |
| S6 (Phase B) | geecs-epics, GeecsBluesky | the *device-clock* tier: a periodic latch of each camera's timestamp against host time (the auto-generated `GevTimestampControlLatch` and `GevTimestampValue` records) and the offset-and-drift fit in the join; PTP unbuilt (section 4.4) | ~40 records and a sequence / ~120 fit and tests | the latch is a command to the camera, no setting changed |
| S7 (Phase B) | GeecsBluesky, GEECS-Data-Utils | deletion at migration end (§10) | negative | — |

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

0. **Phase A, no hardware write.** A hybrid strict scan on the sim bench
   (the camera fake and the LabVIEW fake, a miss injected in each), then a
   laser-off hybrid strict scan at 1 Hz with one native camera and one
   LabVIEW camera both saving images, with one dropped frame induced on
   purpose: the run completes, the drop is one *missing* cell and no
   frame shifts in Tiled or the portal, and the s-file built offline is
   identical. How the drop is induced (a LabVIEW camera's exposure past
   the timeout for one step, or a native's trigger held off) is the
   bench plan's to choose, read-only first, per the hardware-test
   runbook.
1. **Witness alone (Phase B).** The tee into a counter input, the input level checked
   against the box's output before cabling (open question 7). Monitor the
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

Phase A (1 and 2 gate every Phase A slice):

1. **No retakes in strict.** One fire per step, partial rows kept and
   flagged, the retake loop, its budget and the strict rewind deleted, for
   LabVIEW-only runs too (recommended, §6.1; the maintainer's lean). The
   alternative keeps today's retake for LabVIEW-only runs and refuses it in
   hybrid ones, which leaves two strict behaviours.
2. **S or P** (§6.1): a datum stream per camera with an explicit
   `frame_index` map in the row (recommended), or one stream with placeholder frames from the
   LabVIEW plugin, where a native miss ends the run. Overturned toward P by
   a negligible measured native drop rate plus a 1.0 freeze of strict
   `primary`.
3. **The native drop rate, measured before the ruling on 2?** Read-only:
   `AravisFailures` and the frame counter on the deployed cameras across
   ordinary runs (recommended only if P is still in play after 2).
4. **The fire permit's defaults.** Fire on a latched change to ready,
   consumed at the fire, no timeout (recommended: the other facility's
   behaviour as described), or fire on the level.

Phase B, held:

5. **W2.** Build the witness-clocked mode (§6.2), or stop at the join (W1)?
6. **Witness placement.** A small Linux IOC host beside the shot-control
   box, or a USB run to the nearest existing host; and the poll period at
   10 ms.
7. **Input compatibility.** The 1808X counter input's threshold and
   minimum pulse width against the DG645 output's level and width, before
   the tee is cabled.
8. **Camera features.** Decided 2026-11-03: PTP skipped, the
   *device-clock* tier is the latch and the fit (section 4.4). Left: the
   third camera server's cameras, read the same way.
9. **The LabVIEW wire shot number outside scan mode**: in scan mode it is
   the announced number (settled, §4.6); what it counts while a device
   free-runs between scans the passive logger answers. Nothing to decide
   unless it behaves unexpectedly.
10. **Offset updates**: by recorded calibration only (recommended, for
   reproducible runs), or continuously from the tracker.
11. **Shot columns**: `Shotnumber` stays the row index with a new witness
   column (recommended, the `announced_shot` precedent), or becomes the
   witness count.
12. **5 Hz**: which of §4.4's conditions to pursue first once 1 Hz runs
   under the witness.

## 10. What this deletes at the end

Phase A deletes at once (under S):

- the strict retake loop: `geecs_take_reading`'s attempts and
  `max_refires` (`plans/strict.py:248`, `:397-447`);
- the strict rewind: both `discard_uncollected`s
  (`devices/detector.py:1177-1190`, `devices/native.py:676-692`) and
  `NativeCamera`'s not-rewindable stand-in (`devices/native.py:499-562`);
  the `rewindable_wrapper` stays, unconditionally off. The plugin's
  `Rewind` keeps one use, zeroing the count at a fresh session;
- the native camera's fail-on-miss and the "beside a gateway plugin
  camera" refusal (`plans/registry.py:440-447`);
- strict camera datums in `primary` and the partial row's "no frames for
  the LabVIEW cameras" rule (`EVENT_SCHEMA.md` "Partial rows", which
  already overstates it as "any camera").

When the last LabVIEW device has migrated and W2 is the default:

- the plugin's `Rewind` and the PVA file plugin with it;
- `refuse_native_run` and its reasons (`plans/registry.py:411-447`;
  `native_scope.py`);
- the camera shot clock and the sampler's stamp-advance clock
  (`plans/gated.py:86-123`, `devices/sampler.py`);
- the LabVIEW adapter (§5.4) and the scan-mode announce with it;
- `measure_shot_offsets_plan` and `check_shot_sync_plan`
  (`plans/calibration.py:813`, `:1049`): the tracker measures offsets from
  any run;
- §4 of `GeecsBluesky/NATIVE_SHOT_CONTROL.md`, whose gaps this note closes,
  and this file.
