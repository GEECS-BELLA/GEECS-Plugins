# Shot synchronization — one shot identity for every device

Status: **design, ruled in direction, for review.** Nothing here is
built. The maintainer ruled on 2026-11-04 (through the hybrid-scan design
session) that the model of §6 is the preferred direction: a shot table
clocked by the witness, every per-shot device in its own stream, a join
that assigns and audits, refires without rewinds. The witness comes first
(§7, W0 to W2), then the live-join arc after its own design section and
review. Delete this file in the PR that lands the last slice
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
gaps flagged, no rewind.** A refire is allowed and is a new shot (ruled
2026-11-04).

**The ruling** (maintainer, 2026-11-04). The immediate need was a strict
scan whose rows hold native cameras and LabVIEW devices together, and the
first two days of this note (§6.5) tried to get there without the witness:
hybrid strict with no retakes, then two row shapes, P and S, then a third
that kept retakes. The maintainer liked none of them, and ruled for the
model of §6 instead, with three reasons in their words: a shot counter
makes integrating EPICS devices easier; the essential, non-essential and
`primary` machinery may be gutted if that gives a better end product; and
a `primary` browsable by stock Tiled tools is not valued, since we build
our own Tiled viewers anyway. **No scan mode on native devices** stands:
natives take the edge, and the announcement is the LabVIEW devices'
affair (§4.6).

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
| **shot table** | the run's `primary` stream under the ruling: one row per witness-counted shot, holding the shot ID, the witness stamp, the step, the setpoints and readbacks, the conditions and the flags, and no frame datums (§6.1) |
| **required** | a completeness rule on a device: a shot is complete when every required device has an acquisition assigned to it. Replaces "essential"; decides refires, never storage or the join (§6.1) |
| **firing policy** | how a step gets its shots: *on-demand* (fire, check completeness, repeat: today's strict) or *batch* (free-run N edges, check: today's gated). Two policies over one data shape (§6.1) |
| **fire permit** | a readback that says whether a useful shot can be taken now, with a count of its changes to ready; a strict plan configured with one fires only on a fresh change to ready (§6.4). Another facility's mode; its source is a facility value |
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

This is the model the ruling adopts. §6 says how plans and streams carry
it, and §7 in what order it is built.

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
| `device_stamp`, `stamp_kind` | the device's timestamp and where it was taken (§2). A device with two stamps, a camera's own clock and the host receipt, stores both from day one: the join names in `stamp_kind` the one it used, and the other stays for the audit and for a later fit of the camera clock (the *device-clock* slice, §7) over data taken before that slice exists |
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
  (`GEECS-Schemas/geecs_schemas/shot_offsets.py`). Open question 12 asks
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
frame (section 5.1), so the latch and the fit are the whole of the
*device-clock* slice (§7), gated on need: required for 5 Hz under exposure
scans, not for 1 Hz.

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
- The shot ID is `(witness epoch, count)`, carried on every row of the
  shot table (ruled 2026-11-04). Whether the s-file's `Shotnumber` stays
  the row index beside it, the `announced_shot` precedent
  (`GEECS-Data-Utils/geecs_data_utils/tiled_schema.py:38`), or becomes the
  count, is open question 13.
- **Refire, never rewind.** The join never needs a retake: every
  acquisition keeps its own shot. A plan refires a step that lacks complete
  shots (§6.1, `required`), and the refired shot is a new row with its own
  ID; nothing is rewound and no acquisition changes shot.
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
    **The announcement alone never assigns**: it predicts the next edge,
    so a device that missed that edge and took the following one would be
    filed under the wrong shot. The stamp assigns; the announced number
    is the cross-check (§6.2).
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
host's USB (open question 8): stock measComp records, `PollSleepMS` pinned
to 10 at boot, a bench IOC until the counter is proven on the tee (§8,
item 0), then a lab IOC (§7, W1). Its registry entry is a new kind,
*witness*, naming the trigger domain it counts. Decision 13 asks for a
simulator: the witness sim of §7, whose count advances on the sim bench's
trigger-box fire, so the on-demand policy (§6.1) runs end to end against
the existing camera fake.

### 5.4 LabVIEW devices: the adapter

GEECS-Core already parses the wire shot number beside the values
(`GEECS-Core/geecs_core/transport/tcp_subscriber.py:389-394`,
`client/geecs_device.py:74`). The adapter turns each device push into one
evidence record: `acq_timestamp` as a *host-receipt* stamp, the shot number
as an *announced* counter, class from the device's own trigger setting,
and, in a scan-mode run, the worker's announce stamp for that number
(§4.6). It knows the regime: in scan mode one acquisition per
announcement, otherwise one per edge. One module, no new wire protocol,
deleted with the last LabVIEW device. Under the ruling each LabVIEW device
that acquires per shot becomes a stream of its own (§6.1), which is what
the adapter produces; it lands in the arc (§7, W3). The announce stays as
long as scan-mode profiles do; the baseline and the stamp wait become the
completeness read of §6.1.

### 5.5 Asynchronous devices

Monitor caches, the last-before rule, the age column. Nothing device-side.
Under the ruling these are the shot table's *conditions* columns: the
background telemetry sampled softly against each shot's witness stamp
(§6.1), which is what the sampler's settle logic approximated. A camera
whose trigger-mode readback says free-run is asynchronous for the run
whatever its registry entry says, and the mismatch is reported at prepare.

## 6. The model in plans and streams

Ruled 2026-11-04 (§1). Six parts, then the join's sources per device, the
live-data constraint, the fire permit, and the shapes set aside on the way.

### 6.1 Six parts

1. **`primary` is the shot table.** One row per witness-counted shot: the
   shot ID `(epoch, count)`, the witness stamp, the step or bin, the motor
   setpoints and readbacks, the background telemetry sampled softly as
   conditions (§5.5), and the flags (*complete*, *missing <device>*,
   *unwitnessed*, and §4.2's). No frame datums. Gated's separate `shots`
   stream merges into it, so strict and gated have one shape. The name
   `primary` stays for stock tools.
2. **Every device that acquires per shot gets its own stream**, cameras and
   triggered LabVIEW scalar devices alike: one record per acquisition with
   its stamp, and the shot ID once the device can carry one (§6.2). This is
   today's non-essential path (`EVENT_SCHEMA.md`, "`<name>_stream`") made
   universal. It retires the sampler's settle and NaN logic
   (`devices/sampler.py`).
3. **The join assigns acquisitions to shots**: by shot ID where the device
   carries one, else by offset-corrected stamp; time assigns, counters
   audit (§4.3). A miss is a gap, an extra is flagged, nothing shifts. The
   s-file is the join's output.
4. **`required` replaces "essential", as a completeness rule only.** A shot
   is complete when every required device has an acquisition assigned to
   it. A step needs N complete shots, and the plan refires until it has
   them or its budget runs out. Live, completeness is read from each
   required device's counter or stamp advancing within a timeout.
   `required` never decides storage or the join. A refire is a new shot.
   No rewinds anywhere.
5. **Strict and gated are two firing policies over the same data**:
   *on-demand* (fire, check completeness, repeat) and *batch* (free-run N
   edges, check).
6. **Kept**: the witness as a contract (§4.1: count, stamp and epoch per
   trigger domain; no code names the MCC), and the *unwitnessed* fallback,
   the worker's fire stamp as the shot ID with every row flagged, for sims,
   development and a facility without a counter.

**The insight the ruling rests on.** "Retake" was two things, refire and
rewind. Rewind is the hard part: a GEECS invention in the file plugin, the
reason stock writers were refused beside it, and the source of every
refusal in §1. Refire is free once frames live in their own streams: a
refired shot is one more record in each stream and one more row in the
shot table, and the join files them.

**A contract change, paid once.** This changes `primary`
(`EVENT_SCHEMA.md`): strict frames leave it, and every reader that finds
strict frames by datum position moves to the join. It is the cost S would
have paid (§6.5), taken once. The readers are counted before the arc
starts (§7, W3): the s-file and Parquet writers, `tiled_export.py`, the
portal's Images tab, ScanAnalysis's stack reads, the stack check. One
shared helper beside `shot_join.py`, images aligned to rows with gaps,
serves them all, each tested on a run with an induced miss. Runs recorded
before the change keep the old shape (open question 4).

**Per shot, on-demand policy.**

1. Announce to the LabVIEW devices if the profile does; wait for the
   permit if one is configured (§6.4); fire once.
2. See the witness count advance within a bounded wait (period + largest
   offset + margin, the constants in `plans/gated.py:78-80`). A fire the
   witness never sees is reported as not fired, and the plan stops rather
   than fire again; unwitnessed, the fire stamp is the shot.
3. Within the same bound, see each required device's counter or stamp
   advance. Close the row: the shot ID, the witness stamp, the setpoints,
   the conditions, the flags. A required device that produced nothing
   makes the shot incomplete, *missing <device>*; the row stays.
4. Collect every stream's new records and datums since the last collect,
   late or stray ones included, so each stream always matches what its
   device wrote.
5. If the step still lacks complete shots and the budget allows, fire
   again. Running out of budget is open question 2.

**Batch policy.** The box passes edges (SCAN), the witness counts them,
every per-shot device records every edge in its stream, and the plan
collects until the step has N complete shots or the budget is spent; the
rows are the witness ticks, incomplete ones flagged. The camera shot clock
and the sampler's stamp-advance clock go (§10).

**Pause and resume.** In place, at a shot boundary; rows already closed
stay closed; resume continues at the next shot. There is no rewind to
reconcile, so `rewindable_wrapper` is no longer needed for its original
purpose.

### 6.2 Join sources, per device class

- **Native frames.** Preferred: the IOC records the witness `Counts` onto
  each frame through an `NDAttributes` entry of type `EPICS_PV` (one line
  in the camera family's shared attribute file in geecs-epics, no driver
  change; §7, W5). It is an observation taken as the frame passes the
  plugin, so it cannot be off by one the way a predicted number can. The
  EpicsTS stamp (§5.1) is the cross-check, and the assignment until W5 is
  deployed; a frame whose attribute and stamp disagree is flagged
  *audit-mismatch*. A frame drains 20 to 200 ms after its edge and the
  witness polls at 10 ms, so the attribute's cached value is current when
  the frame passes; the bench confirms it (§8, item 3). The scope's
  NDArrays (`geecs-epics:docs/scope-contract.md`) take the same attribute.
- **LabVIEW devices.** By stamp: `acq_timestamp`, or the worker's receipt
  stamp where the Windows clock is not trusted (§4.5), with scan mode's
  `announced_shot` as the cross-check. Never the announcement alone
  (§4.6).
- **Asynchronous devices.** §4.3's last-before rule; they are the shot
  table's conditions columns.

### 6.3 Live data

Plugin and native stacks are readable only after finalize, at unstage, so
nothing live reads a stack mid-run. Live values come from monitors: a
LabVIEW camera's scalars from the gateway PVs; a native camera's STATS
records over CA; the NTNDArray over PVA, which carries `uniqueId` and the
timestamp, and the witness count once W5 lands. The stack is the record
and wins any disagreement; the stack check reports it. Coherence: a native
camera's STATS values are grouped by the plugin's `ArrayCounter_RBV` /
`UniqueId_RBV`, never "latest of each" (GeecsPanels' precedent). Each
stream is collected per step, and the join runs as a live callback that
builds s-file rows incrementally. Fifty or more streams per scan: the
arithmetic is trivial (a `merge_asof`); what must be measured is the Tiled
writer's one Parquet file per event stream, about fifty files each
registered. Mitigations: register the stacks and the joined table only,
or group the scalar streams. The joined table is registered in Tiled as
the stock-friendly view.

### 6.4 The fire permit

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
  precedes the fire by more than a frame period, the §4.3 rule. The
  witness counts fires however spaced; the batch policy and the permit are
  unrelated, since a batch takes every edge.

### 6.5 Shapes considered on the way, 2026-11-02 to 04

Kept short, because two facts found on the way still hold the model up.

- **Hybrid strict with free-running natives and an edge-time pick**
  (2026-11-03). Wrong on two counts. First, under a box-fired strict
  profile ARMED is quiescent (`devices/shot_control.py:7-9`, `:54-60`; the
  profiles set the box to a single-shot source in ARMED), so with every
  native trigger line hanging off a box-gated output (the cabling rule,
  `NATIVE_DEVICES.md` "Acceptance") no edge reaches a native between
  fires and it takes exactly the fire's edge; the idle state STANDBY is
  what passes the edge train seen on a camera's line between runs.
  Evidence: the native path fails a run on any orphan frame
  (`NATIVE_DEVICES.md` "No orphan frames in v1"), and the 1 Hz strict
  count behind the drain offsets (Undulator 26_1022 Scan013, a native
  camera on its hardware line; `geecs-epics:docs/runbooks/timing.md`)
  passed. The cabling rule is still the assumption to confirm per camera
  (open question 5). Second, within one event stream Bluesky requires a
  stream datum for every external key or for none, all of one width
  (bluesky 1.15 `bundlers.py:938-942`, `:843-850`), and the stock
  `HDF5Consolidator` sizes a stack as the sum of datum widths and reads
  from frame 0 (`NATIVE_DEVICES.md:153-178`): a frame no datum references
  shifts every later frame in Tiled. **Hence no frame datums in the shot
  table**: frames live in their own streams, where every frame written
  gets a datum and nothing can shift.
- **No retakes, with P or S** (2026-11-04). P kept one stream and had the
  LabVIEW plugin write a placeholder frame on a miss; a native miss still
  ended the run, forever. S gave each camera its own datum stream and
  wrote a `<cam>-frame_index` map into the row; a native miss was a NaN.
  A third, R1, kept retakes and had the Tiled registration skip the
  orphan frame; the stock consolidator records such skips
  (`bluesky/consolidators.py`, `has_skips`) but exports no map, and
  Tiled's adapters know nothing of skips, so R1 needed a GEECS
  consolidator and a GEECS adapter on the Tiled server. The maintainer
  liked neither P nor S, and said why retakes had been chosen: complete
  rows keep every reader simple. The ruling takes S's stream shape for
  every per-shot device, keeps the refire, drops the rewind, and gives up
  the stock-browsable `primary`.
- **Join only, or witness-clocked.** The choice the earlier draft deferred
  to its Phase B:

| | **join only** (plans unchanged, the witness a key) | **witness-clocked** (ruled) |
|---|---|---|
| How a plan knows a shot happened | each device's own signal (`acq_timestamp` advancing; `NumCaptured` advancing) | the witness count advances |
| Batch rows | the shot clock is a camera (`plans/gated.py:86-123`), and the sampler counts its stamp advances (`devices/sampler.py:132`): a shot the clock camera misses is a row that never exists | every witness tick is a row |
| Fire confirmation | none | the witness confirms the fire; a fire it never sees stops the plan |
| Failure it adds | none new | the witness is a single point of failure: a run without one falls back to join-only behaviour, every row *unwitnessed* (open question 1) |

  Ruled witness-clocked, as the model of §6.1; the join-only behaviour
  survives as the unwitnessed fallback.

## 7. Where the changes land, and slices

Order (maintainer, 2026-11-04): W0 and J0 in parallel, then W1 and W2,
then the arc, with W5 beside it. A5 whenever the other facility needs it.

| Slice | Repo | Content | Size (code / tests) | Hardware |
|---|---|---|---|---|
| W0 | geecs-epics (bench) | the witness bench (§8, item 0): the gated output tee'd into a counter input of the bench DAQ instance, `PollSleepMS` at 10 ms, one laser-off LabVIEW-only strict scan; counts against fires, doubled edges, the poll lag | the check script | the tee, the maintainer's hands; one write, the poll period, on the maintainer's go |
| J0 | GEECS-Data-Utils | a live-join prototype replayed over existing gated runs: responsiveness per shot, and today's s-files reproduced from the streams; the seed of `shot_join.py`'s growth, or thrown away | ~200, prototype | — |
| W1 | geecs-epics | the bench DAQ instance promoted to a witness instance (`/add-device` of the DAQ family): the counter records, the 10 ms poll at boot, the epoch PV from devIocStats | ~80 config | the USB run (open question 8) |
| W2 | GeecsBluesky, GEECS-Schemas | a *witness* registry kind, waited on per shot (the fire confirmed when `Counts` advances); rows gain `witness_count`, `witness_stamp` and `witness_epoch`. Additive only, and worth it for LabVIEW-only scans: fire confirmation, and an audit of today's joins | ~150 / ~150 | the sim bench with the witness sim, then a laser-off scan |
| W3, the arc | GeecsBluesky, GEECS-Schemas, GEECS-Data-Utils, GEECS-DataPortal, ScanAnalysis | the live join as a callback; every per-shot device a stream; `required` and the completeness budget; the two firing policies over one shape; the LabVIEW adapter (§5.4); the readers onto the join through the shared helper; the deletions of §10's first list. **Gets its own design section in this note and an adversarial review before any code** | sized by its design section | the sim bench; then laser-off runs |
| W5 | geecs-epics | the witness count as an `EPICS_PV` NDAttribute in the camera family's shared attribute file, and the scope family's | one line per family / the fake | none: a config change, redeployed per IOC |
| A5 | GeecsBluesky, GEECS-Schemas | the fire permit (§6.4): the permit readable and its registry kind, the wait inside the fire step, the *no-permit* and *stale* flags; a fake permit for the tests | ~60 / ~80 | none; a fake permit on the sim bench |
| later | ioc-picoscope; geecs-epics and GeecsBluesky | the scope stamp at the trigger (§5.2); the *device-clock* tier (§4.4) when 5 Hz under exposure scans is wanted | ~15 / the fake; ~40 records and a sequence / ~120 | one capture; the latch is a command to the camera, no setting changed |

**Simulator** (the development path, maintainer, 2026-11-04): mocks for
the logic, the sim bench (`geecs-epics:docs/runbooks/sim-bench.md`) for
integration. Two pieces are built, no more: (a) a **witness sim**, a
caproto IOC beside the trigger-box fake that counts its fires and serves
`Counts`, the stamp and an epoch, with fault controls for a doubled edge,
a stall that shares one stamp across several counts, a new epoch, and
down; (b) a **skip or delay the next frame** control on the fake camera.
Not built: LabVIEW fakes beyond GEECS-Core's, a fake plugin camera, and a
recorded-roster seam (added only if VPN friction bites). The maintainer's
addition, "a simulated shot counter that can fire edges is a great proxy
for bench experiments without strict timing systems", applies to the
simulator alone: the witness sim is also the sim's edge source, pressing
the fake camera (a software press, since the fake cannot line-trigger) and
counting the press. On the bench the 1808X is a passive witness only, a
counter input on a tee of the box's gated output, never an edge source
for any device (maintainer, 2026-11-04: "I wouldn't wire an MCC to real
cameras").

**Tests.** The join is pure arithmetic and gets a synthetic witness and
synthetic devices with injected misses, extra frames, doubled edges,
counter resets, clock steps, stalls that share a witness stamp and a
witness outage, one test per flag, each proven to bite. The policies run
on the sim bench against the trigger-box fake, the witness sim and the
camera fake made to skip a frame: a required device's miss refires, a
non-required one's leaves a gap, and the run continues through both. The
adapter is tested against GEECS-Core's fake scan-mode device. Live and
offline joins of the same run must be identical, and J0 must reproduce
today's s-files from existing gated runs.

## 8. Bench

Per-facility facts first (the trigger tree is this facility's; §1), then
measurements, in the order of the hardware-test runbook
(`geecs-epics:docs/runbooks/hardware-tests.md`): read-only before any write,
the maintainer's hand-off and go as the gate.

0. **W0, the witness bench.** The input level checked against the box's
   output before cabling (open question 9); the gated output tee'd into a
   counter input of the bench DAQ instance, the maintainer's hands on the
   cable; `PollSleepMS` to 10 ms, the one write, on the go. Then one
   laser-off LabVIEW-only strict scan: the count against the fires, no
   doubled edges, and the poll lag from the spread of successive stamp
   deltas.
1. **Witness alone.** Monitor the count and its stamp for an hour of
   ordinary running: rate against the rep rate, doubles, the poll-lag
   spread.
2. **Passive logger, no scan and no beam cost.** Subscribe the LabVIEW
   devices' push stream and the witness; per device, the residual
   distribution, the gap statistics and how the wire shot number behaves
   between scans. This measures Windows clock quality at the same time.
   Left running through ordinary scans, with the worker's announce stamps,
   it also gives each device's arming latency (§4.6).
3. **Native cameras**, once the camera servers' IOCs are deployed (another
   lane owns that): `ArrayCounter` and the EpicsTS pair against the
   witness; each camera's trigger line confirmed to hang off a box-gated
   output (the cabling rule, open question 5); after W5, the witness
   attribute on each frame against the stamp assignment.
4. **PicoScope**: the same, before and after the stamp moves to the
   trigger.

What the measurements decide: the stored offsets, whether the Windows
stamps meet the budget, whether the *device-clock* tier is needed for the
target rate, and whether the per-shot witness wait is adequate.

## 9. Open questions for the maintainer, ranked

Ruled 2026-11-04: the direction (§6); witness-clocked rows over join
only; refire without rewind; the shot ID `(epoch, count)`; `required` in
place of "essential"; the simulator as the development path.

Open:

1. **The witness dies mid-run.** Flag *unwitnessed* and continue on the
   devices' own signals (recommended: the run is not lost, and the join
   still runs), or stop. Strict waits on the witness, so the choice
   decides what a live run does at that moment.
2. **A step that runs out of refires.** Flag the step and continue the
   scan (recommended), or end the scan.
3. **A live per-row flag for non-required devices.** Whether the live
   callback marks a non-required device's gap on the row as it closes,
   or only the offline join does.
4. **Runs recorded before the arc.** The readers accept both shapes for a
   while (recommended: the old shape is row *k* = frame *k* by
   construction, so the shared helper can serve it), or the old runs are
   converted once.
5. **The cabling rule, confirmed per native camera.** Every native trigger
   line hangs off a box-gated output, never the raw edge train; the
   quiescence argument of §6.5 and the lifted refusals rest on it.
6. **The bluesky cap below 1.16.** A separate small PR: 1.16 moved
   `TiledWriter` to bluesky-tiled-plugins, and `tiled/parquet.py` imports
   the private `_RunWriter` and `BATCH_SIZE`, which the shim no longer
   exports.
7. **The fire permit's defaults.** Fire on a latched change to ready,
   consumed at the fire, no timeout (recommended: the other facility's
   behaviour as described), or fire on the level.
8. **Witness placement.** A small Linux IOC host beside the shot-control
   box, or a USB run to the nearest existing host; and the poll period at
   10 ms.
9. **Input compatibility.** The 1808X counter input's threshold and
   minimum pulse width against the DG645 output's level and width, before
   the tee is cabled (W0).
10. **Camera features.** Decided 2026-11-03: PTP skipped, the
   *device-clock* tier is the latch and the fit (section 4.4). Left: the
   third camera server's cameras, read the same way.
11. **The LabVIEW wire shot number outside scan mode**: in scan mode it is
   the announced number (settled, §4.6); what it counts while a device
   free-runs between scans the passive logger answers. Nothing to decide
   unless it behaves unexpectedly.
12. **Offset updates**: by recorded calibration only (recommended, for
   reproducible runs), or continuously from the tracker.
13. **Shot columns in the s-file**: `Shotnumber` stays the row index with
   the witness count beside it (recommended, the `announced_shot`
   precedent), or becomes the count.
14. **5 Hz**: which of §4.4's conditions to pursue first once 1 Hz runs
   under the witness.

## 10. What this deletes at the end

The arc (§7, W3) deletes, under the model:

- the rewind: both `discard_uncollected`s (`devices/detector.py:1177-1190`,
  `devices/native.py:676-692`), `_rewind_plugins`, `NativeCamera`'s
  not-rewindable stand-in (`devices/native.py:498-562`) and the
  `rewindable_wrapper` use (`plans/strict.py:448`); the plugin's `Rewind`
  keeps one use, zeroing the count at a fresh session, until the plugin
  itself goes;
- the retake budget as it stands (`max_refires`, `plans/strict.py:248`,
  `:397-447`), replaced by the completeness budget of §6.1;
- frame datums in `primary`, and the partial row's "no frames" rule
  (`EVENT_SCHEMA.md` "Partial rows");
- gated's separate `shots` stream, the camera shot clock
  (`plans/gated.py:86-123`) and the sampler's stamp-advance clock with its
  settle and NaN logic (`devices/sampler.py`);
- the essential / non-essential split, replaced by `required`;
- most of `refuse_native_run`'s reasons (`plans/registry.py:411-447`):
  beside a gateway plugin camera, in gated runs, as a non-essential; the
  scan-mode refusal is decided in the arc's design section.

When the last LabVIEW device has migrated:

- the plugin's `Rewind` and the PVA file plugin with it;
- what remains of `refuse_native_run` (`native_scope.py`);
- the LabVIEW adapter (§5.4) and the scan-mode announce with it;
- `measure_shot_offsets_plan` and `check_shot_sync_plan`
  (`plans/calibration.py:813`, `:1049`): the tracker measures offsets from
  any run;
- §4 of `GeecsBluesky/NATIVE_SHOT_CONTROL.md`, whose gaps this note closes,
  and this file.
