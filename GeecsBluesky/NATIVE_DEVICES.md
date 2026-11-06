# Native EPICS devices in the scan path — design note

Status: **built** — `NativeCamera` (#1117), the registry and the
bind-time refusals (#1118); hardware acceptance pending. Once accepted,
the rules below move into `CLAUDE.md` and this note is deleted. §6 (native
cameras in gated runs) is set aside.

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

In: **native cameras** as per-shot devices — required or not, beside
gateway plugin cameras or not — in **strict** and **fly** `count` and
`sweep`, each in its own datum-only stream, frames in
`ScanNNN/<device>/<device>.h5`, assigned to the shot-table rows by stamp
(`geecs_data_utils.shot_table`), registered in Tiled, found by analysis.

Refused at bind with a named error (not absent — each falls through a
GEECS-only branch otherwise): gated runs, the
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

## 3. The device — NativeCamera (`devices/native.py`)

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
  **stops** the camera (free-running, or resting in trigger mode as an IOC
  that arms it at boot leaves it) and sets trigger mode *before* the
  capture starts (no live-view frame reaches the stack), and the acquire
  logic restarts it after (on its readback, not the put);
  `unstage` restores the driver's pre-run state (live view keeps working
  between scans). No per-shot `Acquire` writes — the stock model's
  per-shot AcquisitionStart is both too slow for 1 Hz and, in Continuous,
  a hang (`ADAcquireLogic.wait_for_idle` awaits a put that never ends).
  Strict calls `prepare` **every shot** (`strict.py`), so the trigger and
  acquire logics make every prepare after the run's first a no-op: no
  `Acquire` put, the event count ignored.
- **The stream path** (what a strict `count` / `sweep` uses: every
  native is a per-shot device in its own stream, required or not, beside
  gateway plugin cameras or not; a Software source pressed at the fire): an
  **unbounded** prepare (`number_of_events == 0`, `UNBOUNDED_TRIGGER_INFO`)
  arms the camera exactly as a strict run's first prepare does — the run
  lock, the source check and the trigger readbacks included — and the
  camera is then a datum-only stream, like a plugin camera's: `kickoff`,
  `complete` (immediate: nothing waits on the count), `describe_collect`
  and `collect_asset_docs` for the frames written so far. The datums cover
  exactly the frames the writer wrote, contiguously from frame 0: a
  dropped frame is absent from the stream, never a placeholder, and the
  stamp join assigns frames to rows offline. `kickoff` re-baselines on the
  frames already written (a frame between the prepare and the kickoff
  would make the stock kickoff raise; its datum covers it anyway).
  `trigger()` on a stream is refused; a `Software`-source camera is pressed
  at the fire by `NativeCamera.press()` (refused on a hardware line, and
  before the camera is acquiring). A **bounded** fly prepare (a gated
  batch) stays refused before the camera is touched.
- **The run lock** (geecs-epics `run_lock.db` + `access.acf`): the run owns the camera from the
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
- **No placeholder frames.** Bluesky's `HDF5Consolidator` sizes the array
  as the sum of datum widths and reads from frame 0, so the stream's datums
  cover exactly the frames the writer wrote: a skipped frame is a gap in
  the camera's own stream, never a shifted row, and the join
  (`geecs_data_utils.shot_table`) assigns frames to shot-table rows by
  stamp. A missed shot is refired like any device's (a new row); nothing
  rewinds, and a pause replays nothing (the strict slot is not
  rewindable). A stray frame (an edge outside a shot) is an extra the join
  flags — hence the cabling rule in Acceptance still.
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
  `trigger_source` and `press()` (the fire's press), the writer's frame count (the
  slot's progress counter), `count_zeroed`, `_geecs_namespace_member`.

## 4. Failure behaviour

| Fault | What happens |
|---|---|
| Camera disconnected from its IOC | liveness gate refuses before the claim, naming it |
| IOC down | the device's connect fails at stage, before the claim (the gate is fail-open) |
| Share not mounted on the IOC host | `FilePathExists_RBV = 0`; prepare fails naming the path |
| No edge for a shot | a gap in its stream: a required camera's shot is refired (a new row) up to the budget, then flagged; a non-required one's row is simply missing it |
| Stray frame (edge outside a shot) | stack and datums disagree → the stack check reports it; fixed by cabling, not tolerated |
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

## 6. Gated runs — set aside

A native camera in a gated batch was designed here (an append-only,
stamp-joined essential, pause in place, trim to K; reviewed 2026-10-22)
and set aside when gated was frozen (2026-11-04): a gated run refuses a
native camera. The design is in git history (this file before the
shot-synchronization arc's W3f). Its "non-essential streams" part is
built differently: a native camera streams in strict and fly runs,
required or not.

## Open questions

1. ~~Refires for native cameras~~ — settled: the shot table refires a
   required native camera's missed shot as any device's (a new row);
   nothing rewinds.
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

Status: **built** (2026-10-23) — the registry kind (`EpicsMotor`),
`NativeMotor` / `NativeMotorAxis` (`devices/native_motor.py`), the
namespace binding and the preset's kind-aware native set; proven on
`simmotor1`.  Not yet: the gate grown by the movers, the scanner's
settables list (§10, L — until it lands the web scanner's readback panel
reads a native axis's default field, `.VAL`, the setpoint, not `.RBV`), the
archive pass (R), a pseudo over native axes.
Found when built: the stock `Motor` already refuses an out-of-limit
target before the put (`MotorMoveLogic.check_move`), so nothing was added
for the limits; and it already carries `offset` as the record's own
`.OFF` (a config signal), so the soft-signal `offset` below for a pseudo
over native axes needs a renamed signal or a subclass, not a plain
attribute.

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
one parent noun per device (`NativeMotor`) bound as
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
text (the one failed-set log line, `log_failed_set`). A motor record
**has** a stop: abort calls `motor_stop`.

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

Status: **built** (2026-11-03) — the registry kind (`SetpointDevice`, the
family default `easydriver`), `NativeSetpoint` / `NativeSetpointDevice`
(`devices/native_setpoint.py`), the namespace binding and the run gate's
INVALID-readback judgement; proven on `eztest1-sim`. Not yet: the gate
grown by the movers (§7's, shared), enum settables, the scanner's settables
list (§10, L), the panel, a pseudo over native settables.
Found when built: a settable's setpoint record is not always `<record>:SP`
(the FAST-PS writes `current_ramp:SP` beside `current`), so the one rule
is the schema's `SetpointDevice.setpoint_records` (the family's setpoint
role for the readback, else `<record>:SP`), which the scanner's list, the
panel and the archiver can call too; the entry is validated against its
family (a status bit must be one of its fault bits; a record the family
names as something else is refused; a record it omits is allowed); the
drive limits are read as the `:SP` record's `DRVH`/`DRVL` fields (what
`DBR_CTRL` serves for an `ao`), and `DRVH` <= `DRVL` means no limits, the
record's own rule; a tripped status bit or an INVALID readback refuses the
set before the put too, not only mid-wait.

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
post-trigger samples; the drain offset, hand-set per camera and measured
on the bench (ruled 2026-10-22), absorbs that span.

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
| `witness` (built) | `counter` (default `Counter1`), `domain` (default `main`) | `Witness`: recorded, never waited on — its count, stamp and epoch in each strict row; one per domain (`EVENT_SCHEMA.md`) |

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

**Archive set** (built: `geecs_archiver.native`, the registry pass;
`GeecsArchiver/CLAUDE.md` is the rule as implemented). `derive_candidates`
reads only DB rows and a native device has none, so the registry pass reads
`native_devices.yaml` and `panel_devices.yaml` through `config.py`'s
configs-repo wrappers and archives each device's family-file `archive`
roles. For a family marking none, the per-kind rule applies: a motor's
`.RBV`, `.VAL` (under `include_setpoints`), `.DMOV` and the limit
switches; a setpoint device's readbacks, `:SP`s and bits; a detector's
settings readbacks (`cam1:AcquireTime_RBV`, the contract's `_RBV`s) and
the run lock. Under `include_status` it adds the camera's `AsynIO.CNCT`
and devIocStats' heartbeat once per IOC a device's panel data names
(`ioc:`), the analogue of `cagateway:heartbeat`. `exclude` globs apply;
`include` keeps its spelling rule (these are derived candidates). A device
both owners list, or one the pass cannot resolve, is drift. The
never-connected list stays the drift alarm between registry and IOCs.

**One home for the PV surface.** The native prefix is
`geecs_core.pv_naming.native_device_prefix` (built: the namespace,
GeecsCanopy and the archiver all call it — the archiver
cannot import the namespace, `.importlinter`: `geecs_bluesky` and
`geecs_archiver` share a layer). Each
kind's record suffixes — what the archive pass, the scanner and the panel
need — become data beside the kind models in `geecs_schemas.epics`.

**Liveness, per kind** — `devices/ca/liveness.py`'s shape, the gate grown by the movers (§7):

| kind | down when |
|---|---|
| camera / array detector | `AsynIO.CNCT` reads Disconnect, or the IOC does not answer (`down_native_cameras`'s rule) |
| positioner | an axis at INVALID severity (COMM/UDF) |
| setpoint device | a readback at INVALID severity (TIMEOUT/COMM) |

**Panels** (GeecsPanels, #1131–#1138; the package was deleted 2026-11-06,
superseded by GeecsControlRoom's device pages, whose every write goes
through the GEECS module's put route — this paragraph is its record):
a kind whose writes move things submits through the scanner's
`POST /api/move`, serialized with scans — the positioner and a setpoint
device's numeric settables (the positioner panel, #1143, writes the record
directly as a prototype until slice L gives the scanner native axes);
non-motion controls (`enable_output`, `reset`)
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
